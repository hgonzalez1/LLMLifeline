"""
Server mode — lets other devices on your home network use this
computer's model, while every device keeps its own projects and chats.

What a connected device gets is a second, separate door into this
computer: an encrypted (TLS) gateway on its own port that can do model
work and nothing else — write a chat reply, count tokens, generate an
image, caption a photo. It has no route to this computer's projects,
conversations, files, persona, beliefs or computer use; those stay behind
the main backend, which still only answers this computer itself.

Privacy, stated as exactly as it can be:
- A device's request has to be read to be answered, so this computer
  holds each request's text (or image) in memory while working on it.
  Nothing from a request or its reply is written to disk, logged or sent
  anywhere else, and it's dropped once the reply goes back.
- What this computer does keep, in server/server.json: each device's
  name, a hash of its key, when it paired, when it was last seen and
  from which address — and whether you've blocked it. Nothing else.
- Requests travel encrypted, and a device checks it's talking to this
  computer — not something pretending to be it — with a 6-digit code
  shown on both screens when it pairs (derived from this computer's
  certificate, so an impostor's code wouldn't match).
- Only addresses on your local network are answered, so a router port
  opened by mistake still doesn't expose it to the internet.
- A device can't do anything until you approve it here, and you can boot
  or block it at any time.

What code can't promise: whoever controls this computer could change
this code. People connecting are trusting whoever runs the server.
"""

import asyncio
import contextlib
import datetime
import hashlib
import hmac
import ipaddress
import json
import secrets
import socket
import ssl
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

APP_ROOT = Path(__file__).resolve().parent.parent
SERVER_DIR = APP_ROOT / "server"
SETTINGS_FILE = SERVER_DIR / "server.json"
CERT_FILE = SERVER_DIR / "cert.pem"
KEY_FILE = SERVER_DIR / "key.pem"

DEFAULT_PORT = 8100
SERVICE_TYPE = "_llmlifeline._tcp.local."
PAIRING_TTL_SECONDS = 300
MAX_PENDING = 10
CONNECTED_WINDOW_SECONDS = 45   # seen this recently counts as "connected now"
MAX_CAPTION_BYTES = 20 * 1024 * 1024

DEFAULT_SETTINGS = {
    "enabled": False,
    "port": DEFAULT_PORT,
    "name": "",               # shown to devices; defaults to this computer's name
    "allow_images": True,     # devices may generate images and caption photos
    "allow_switching": False,  # devices may change the model/checkpoint and chat/image mode
    "devices": {},            # device_id -> {name, token_hash, paired_at, last_seen, last_ip, blocked}
}


def _now() -> float:
    return time.time()


def computer_name() -> str:
    return socket.gethostname() or "LLMlifeline server"


def is_local_network_address(host: str) -> bool:
    """Private, link-local or this computer — never a public internet address."""
    try:
        ip = ipaddress.ip_address(host.split("%")[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_private or ip.is_link_local or ip.is_loopback


def lan_addresses() -> list[str]:
    """This computer's own addresses on the local network, for showing the
    person where devices can reach it (looked up locally, no network calls)."""
    found = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = info[4][0]
            if address not in found and is_local_network_address(address) and not address.startswith("127."):
                found.append(address)
    except OSError:
        pass
    return found


def certificate_fingerprint(pem: str) -> str:
    return hashlib.sha256(ssl.PEM_cert_to_DER_cert(pem)).hexdigest()


def pairing_code(fingerprint: str, request_id: str) -> str:
    """The 6 digits both screens show. Each side works it out from the
    certificate *it* sees, so a device talking to an impostor gets a
    different code than this computer shows."""
    digest = hashlib.sha256(f"{fingerprint}:{request_id}".encode()).hexdigest()
    return f"{int(digest, 16) % 1_000_000:06d}"


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def ensure_certificate() -> str:
    """This computer's own TLS certificate, made once and kept. Returns its PEM."""
    if CERT_FILE.exists() and KEY_FILE.exists():
        return CERT_FILE.read_text(encoding="ascii")
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    SERVER_DIR.mkdir(parents=True, exist_ok=True)
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"LLMlifeline server {computer_name()}"[:64])])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        # Its own trust anchor: devices pin exactly this certificate.
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=False,
                                     content_commitment=False, key_encipherment=False,
                                     data_encipherment=False, key_agreement=False,
                                     encipher_only=False, decipher_only=False), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("llmlifeline.local")]), critical=False)
        .sign(key, hashes.SHA256())
    )
    KEY_FILE.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    pem = cert.public_bytes(serialization.Encoding.PEM).decode("ascii")
    CERT_FILE.write_text(pem, encoding="ascii")
    return pem


# ---- Settings and devices ----

def load_settings() -> dict:
    try:
        raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    settings = json.loads(json.dumps(DEFAULT_SETTINGS))
    if isinstance(raw, dict):
        for key in ("enabled", "allow_images", "allow_switching"):
            if isinstance(raw.get(key), bool):
                settings[key] = raw[key]
        if isinstance(raw.get("port"), int) and 1024 <= raw["port"] <= 65535:
            settings["port"] = raw["port"]
        if isinstance(raw.get("name"), str):
            settings["name"] = raw["name"].strip()[:60]
        if isinstance(raw.get("devices"), dict):
            settings["devices"] = {k: v for k, v in raw["devices"].items() if isinstance(v, dict)}
    return settings


def save_settings(settings: dict) -> None:
    SERVER_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")


class HostHooks:
    """The model work main.py lets the gateway do. Every method is the
    same code path this computer's own chat and image generation use."""

    async def info(self) -> dict: ...
    async def chat(self, payload: dict, on_text=None) -> dict: ...  # on_text: each piece as it's written
    async def tokenize_count(self, text: str) -> int | None: ...
    async def generate_image(self, prompt: str) -> tuple[bytes, str | None, str | None]: ...
    async def caption(self, image: bytes) -> str | None: ...
    async def swap_model(self, filename: str) -> None: ...
    async def set_mode(self, mode: str) -> None: ...
    async def set_checkpoint(self, filename: str) -> None: ...


class ServerMode:
    def __init__(self):
        self.hooks: HostHooks | None = None
        self.settings = load_settings()
        self.pending: dict[str, dict] = {}      # request_id -> pairing request (memory only)
        self.in_flight: dict[str, int] = {}     # device_id -> requests being worked on
        self.waiting = 0
        self.gpu_lock = asyncio.Lock()          # one model job at a time, first come first served
        self.error: str | None = None
        self._server = None
        self._task: asyncio.Task | None = None
        self._zeroconf = None
        self.cert_pem: str | None = None
        self.fingerprint: str | None = None
        self.app = self._build_app()

    # ---- running ----

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def display_name(self) -> str:
        return self.settings["name"] or computer_name()

    async def start(self) -> None:
        if self.running:
            return
        import uvicorn

        self.error = None
        try:
            self.cert_pem = ensure_certificate()
            self.fingerprint = certificate_fingerprint(self.cert_pem)
        except Exception as e:  # noqa: BLE001 — shown to the person as-is
            self.error = f"Couldn't create this computer's certificate: {e}"
            return

        class _Embedded(uvicorn.Server):
            # Runs inside the main backend's event loop, so it mustn't
            # take over Ctrl+C handling from the main server.
            @contextlib.contextmanager
            def capture_signals(self):
                yield

        config = uvicorn.Config(self.app, host="0.0.0.0", port=self.settings["port"],
                                ssl_certfile=str(CERT_FILE), ssl_keyfile=str(KEY_FILE),
                                access_log=False, log_level="warning", lifespan="off")
        self._server = _Embedded(config)
        self._task = asyncio.create_task(self._server.serve())
        for _ in range(50):
            await asyncio.sleep(0.1)
            if self._server.started or self._task.done():
                break
        if self._task.done() or not self._server.started:
            self.error = (f"Couldn't start on port {self.settings['port']} — another program may be "
                          f"using it. Pick a different port and try again.")
            await self.stop()
            return
        await self._advertise()

    async def stop(self) -> None:
        if self._zeroconf is not None:
            with contextlib.suppress(Exception):
                await self._zeroconf.async_unregister_all_services()
                await self._zeroconf.async_close()
            self._zeroconf = None
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._task, timeout=10)
        self._server = None
        self._task = None
        self.pending.clear()

    async def _advertise(self) -> None:
        """Lets devices on the network find this computer by name (mDNS —
        local network only). Optional: devices can type the address instead."""
        try:
            from zeroconf import ServiceInfo
            from zeroconf.asyncio import AsyncZeroconf
        except ImportError:
            return
        addresses = [socket.inet_aton(a) for a in lan_addresses()]
        if not addresses:
            return
        safe = "".join(c for c in self.display_name() if c.isalnum() or c in " -_")[:40] or "LLMlifeline"
        info = ServiceInfo(SERVICE_TYPE, f"{safe} {uuid.uuid4().hex[:4]}.{SERVICE_TYPE}",
                           addresses=addresses, port=self.settings["port"],
                           properties={"name": self.display_name(), "fp": (self.fingerprint or "")[:16]})
        try:
            self._zeroconf = AsyncZeroconf()
            await self._zeroconf.async_register_service(info)
        except Exception:  # noqa: BLE001 — discovery is a convenience; the server still works
            self._zeroconf = None

    async def apply(self, changes: dict) -> None:
        """Settings changed from this computer's Network panel."""
        restart = False
        for key in ("allow_images", "allow_switching"):
            if isinstance(changes.get(key), bool):
                self.settings[key] = changes[key]
        if isinstance(changes.get("name"), str):
            self.settings["name"] = changes["name"].strip()[:60]
            restart = self.running
        if isinstance(changes.get("port"), int):
            if not 1024 <= changes["port"] <= 65535:
                raise ValueError("Pick a port between 1024 and 65535.")
            restart = restart or (self.running and changes["port"] != self.settings["port"])
            self.settings["port"] = changes["port"]
        if isinstance(changes.get("enabled"), bool):
            self.settings["enabled"] = changes["enabled"]
        save_settings(self.settings)
        if not self.settings["enabled"]:
            await self.stop()
        elif restart or not self.running:
            await self.stop()
            await self.start()

    # ---- what the Network panel shows ----

    def status(self) -> dict:
        now = _now()
        self._expire_pending()
        devices = []
        for device_id, d in self.settings["devices"].items():
            connected = (self.in_flight.get(device_id, 0) > 0
                         or (d.get("token_hash") and now - d.get("last_seen", 0) < CONNECTED_WINDOW_SECONDS))
            devices.append({
                "id": device_id, "name": d.get("name", "Unknown device"), "last_ip": d.get("last_ip"),
                "paired_at": d.get("paired_at"), "last_seen": d.get("last_seen"),
                "blocked": bool(d.get("blocked")), "paired": bool(d.get("token_hash")),
                "connected": bool(connected) and not d.get("blocked"),
            })
        devices.sort(key=lambda d: (not d["connected"], d["blocked"], -(d["last_seen"] or 0)))
        return {
            "enabled": self.settings["enabled"], "running": self.running, "error": self.error,
            "name": self.display_name(), "port": self.settings["port"],
            "addresses": lan_addresses(),
            "fingerprint": self.fingerprint,
            "allow_images": self.settings["allow_images"],
            "allow_switching": self.settings["allow_switching"],
            "pending": [{"id": rid, "device_name": p["device_name"], "ip": p["ip"],
                         "code": pairing_code(self.fingerprint or "", rid), "requested_at": p["created"]}
                        for rid, p in self.pending.items() if p["status"] == "pending"],
            "devices": devices,
            "waiting": self.waiting,
        }

    def _expire_pending(self) -> None:
        cutoff = _now() - PAIRING_TTL_SECONDS
        for rid in [rid for rid, p in self.pending.items() if p["created"] < cutoff]:
            del self.pending[rid]

    def answer_pairing(self, request_id: str, approve: bool) -> None:
        request = self.pending.get(request_id)
        if not request or request["status"] != "pending":
            raise KeyError("That request isn't waiting anymore.")
        if not approve:
            request["status"] = "denied"
            return
        token = secrets.token_urlsafe(32)
        self.settings["devices"][request["device_id"]] = {
            "name": request["device_name"], "token_hash": _hash_token(token),
            "paired_at": _now(), "last_seen": _now(), "last_ip": request["ip"], "blocked": False,
        }
        save_settings(self.settings)
        request["status"] = "approved"
        request["token"] = token  # handed to the device once, then dropped

    def boot(self, device_id: str) -> None:
        """Disconnects now. The device can ask to pair again (you'd approve it again)."""
        device = self.settings["devices"].get(device_id)
        if device is None:
            raise KeyError("No such device.")
        device["token_hash"] = None
        save_settings(self.settings)

    def block(self, device_id: str, blocked: bool) -> None:
        device = self.settings["devices"].get(device_id)
        if device is None:
            raise KeyError("No such device.")
        device["blocked"] = blocked
        if blocked:
            device["token_hash"] = None
            for p in self.pending.values():
                if p["device_id"] == device_id and p["status"] == "pending":
                    p["status"] = "denied"
        save_settings(self.settings)

    def forget(self, device_id: str) -> None:
        if self.settings["devices"].pop(device_id, None) is None:
            raise KeyError("No such device.")
        save_settings(self.settings)

    # ---- the gateway itself ----

    def _device_for(self, request: Request) -> tuple[str, dict]:
        device_id = request.headers.get("x-device-id", "")
        auth = request.headers.get("authorization", "")
        token = auth[7:] if auth.lower().startswith("bearer ") else ""
        device = self.settings["devices"].get(device_id)
        if not device or device.get("blocked"):
            raise HTTPException(status_code=403, detail="This device isn't allowed on this server.")
        if not device.get("token_hash") or not token or not hmac.compare_digest(
                device["token_hash"], _hash_token(token)):
            raise HTTPException(status_code=401, detail="This device was disconnected by the server. "
                                                        "Pair it again to reconnect.")
        device["last_seen"] = _now()
        device["last_ip"] = request.client.host if request.client else None
        return device_id, device

    @contextlib.asynccontextmanager
    async def _model_job(self, device_id: str):
        """Queued, one at a time. Nothing about the job is recorded."""
        self.in_flight[device_id] = self.in_flight.get(device_id, 0) + 1
        self.waiting += 1
        started = False
        try:
            async with self.gpu_lock:
                started = True
                self.waiting -= 1
                # Booted or blocked while it was waiting its turn.
                if self.settings["devices"].get(device_id, {}).get("token_hash") is None:
                    raise HTTPException(status_code=401, detail="This device was disconnected by the server.")
                yield
        finally:
            if not started:
                self.waiting -= 1
            self.in_flight[device_id] -= 1

    async def _streamed_chat(self, device_id: str, payload: dict):
        """A chat reply sent as it's written, one JSON object per line:
        {"text"} pieces, then {"result"} (the whole completion, as the
        unstreamed answer) or {"error": {"status", "detail"}}. A device
        that hangs up stops the job, and with it the model's work."""
        events: asyncio.Queue = asyncio.Queue()

        async def run():
            try:
                async with self._model_job(device_id):
                    result = await self.hooks.chat(payload, on_text=lambda text: events.put_nowait({"text": text}))
                events.put_nowait({"result": result})
            except HTTPException as e:
                events.put_nowait({"error": {"status": e.status_code, "detail": e.detail}})
            except Exception as e:  # noqa: BLE001 — reported to the device without its request's content
                events.put_nowait({"error": {"status": 500, "detail": f"The server failed ({type(e).__name__})."}})

        job = asyncio.create_task(run())
        try:
            while True:
                event = await events.get()
                yield json.dumps(event) + "\n"
                if "text" not in event:
                    return
        finally:
            job.cancel()  # nothing to do once it's finished; stops it if the device went away

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="LLMlifeline server", docs_url=None, redoc_url=None, openapi_url=None)
        mode = self

        @app.middleware("http")
        async def _local_network_only(request: Request, call_next):
            host = request.client.host if request.client else ""
            if not is_local_network_address(host):
                return JSONResponse({"detail": "This server only answers devices on its local network."},
                                    status_code=403)
            return await call_next(request)

        @app.get("/v1/hello")
        async def hello():
            return {"app": "LLMlifeline", "name": mode.display_name(), "pairing": True}

        @app.post("/v1/pair")
        async def request_pairing(request: Request):
            body = await request.json()
            device_id = str(body.get("device_id", ""))[:64]
            device_name = " ".join(str(body.get("device_name", "")).split())[:60] or "Unnamed device"
            secret = str(body.get("secret", ""))
            if not device_id or len(secret) < 16:
                raise HTTPException(status_code=400, detail="device_id and secret are required.")
            if mode.settings["devices"].get(device_id, {}).get("blocked"):
                raise HTTPException(status_code=403, detail="This device has been blocked by the server.")
            mode._expire_pending()
            if sum(p["status"] == "pending" for p in mode.pending.values()) >= MAX_PENDING:
                raise HTTPException(status_code=429, detail="Too many devices are waiting. Try again shortly.")
            request_id = uuid.uuid4().hex
            mode.pending[request_id] = {"device_id": device_id, "device_name": device_name,
                                        "ip": request.client.host if request.client else None,
                                        "secret_hash": _hash_token(secret), "status": "pending",
                                        "created": _now(), "token": None}
            return {"request_id": request_id, "expires_in": PAIRING_TTL_SECONDS}

        @app.post("/v1/pair/{request_id}")
        async def pairing_status(request_id: str, request: Request):
            body = await request.json()
            pairing = mode.pending.get(request_id)
            if not pairing or not hmac.compare_digest(pairing["secret_hash"], _hash_token(str(body.get("secret", "")))):
                raise HTTPException(status_code=404, detail="No such pairing request (it may have expired).")
            if pairing["status"] == "approved":
                token = pairing["token"]
                del mode.pending[request_id]
                return {"status": "approved", "token": token, "name": mode.display_name()}
            if pairing["status"] == "denied":
                del mode.pending[request_id]
            return {"status": pairing["status"]}

        @app.get("/v1/info")
        async def info(request: Request):
            mode._device_for(request)
            data = await mode.hooks.info()
            return {**data, "name": mode.display_name(), "waiting": mode.waiting,
                    "allow_images": mode.settings["allow_images"],
                    "allow_switching": mode.settings["allow_switching"]}

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            device_id, _ = mode._device_for(request)
            body = await request.json()
            payload = {"messages": body.get("messages"), "max_tokens": body.get("max_tokens")}
            if not isinstance(payload["messages"], list):
                raise HTTPException(status_code=400, detail="messages must be a list.")
            if body.get("stream") is True:
                return StreamingResponse(mode._streamed_chat(device_id, payload),
                                         media_type="application/x-ndjson")
            async with mode._model_job(device_id):
                return await mode.hooks.chat(payload)

        @app.post("/v1/tokenize/count")
        async def tokenize_count(request: Request):
            mode._device_for(request)
            body = await request.json()
            return {"count": await mode.hooks.tokenize_count(str(body.get("input", "")))}

        @app.post("/v1/images/generate")
        async def generate_image(request: Request):
            device_id, _ = mode._device_for(request)
            if not mode.settings["allow_images"]:
                raise HTTPException(status_code=403, detail="This server doesn't share image generation.")
            prompt = str((await request.json()).get("prompt", ""))[:4000]
            async with mode._model_job(device_id):
                png, caption, checkpoint = await mode.hooks.generate_image(prompt)
            import base64
            return {"image_base64": base64.b64encode(png).decode("ascii"), "caption": caption,
                    "checkpoint": checkpoint}

        @app.post("/v1/caption")
        async def caption(request: Request):
            device_id, _ = mode._device_for(request)
            if not mode.settings["allow_images"]:
                raise HTTPException(status_code=403, detail="This server doesn't share image captioning.")
            image = await request.body()
            if not image or len(image) > MAX_CAPTION_BYTES:
                raise HTTPException(status_code=400, detail="Send one image, up to 20 MB.")
            async with mode._model_job(device_id):
                return {"caption": await mode.hooks.caption(image)}

        def _require_switching():
            if not mode.settings["allow_switching"]:
                raise HTTPException(status_code=403, detail=(
                    "This server doesn't let devices change its model or mode. Whoever runs it can "
                    "allow that in its Network panel."))

        @app.post("/v1/models/swap")
        async def swap_model(request: Request):
            device_id, _ = mode._device_for(request)
            filename = str((await request.json()).get("model", ""))
            current = (await mode.hooks.info()).get("model")
            if filename and filename == current:
                return {"status": "ok"}
            _require_switching()
            async with mode._model_job(device_id):
                await mode.hooks.swap_model(filename)
            return {"status": "ok"}

        @app.post("/v1/mode")
        async def set_mode(request: Request):
            device_id, _ = mode._device_for(request)
            wanted = str((await request.json()).get("mode", ""))
            if wanted not in ("chat", "image"):
                raise HTTPException(status_code=400, detail="mode must be chat or image.")
            if (await mode.hooks.info()).get("mode") == wanted:
                return {"status": "ok"}
            _require_switching()
            async with mode._model_job(device_id):
                await mode.hooks.set_mode(wanted)
            return {"status": "ok"}

        @app.post("/v1/images/checkpoint")
        async def set_checkpoint(request: Request):
            device_id, _ = mode._device_for(request)
            filename = str((await request.json()).get("checkpoint", ""))
            if (await mode.hooks.info()).get("checkpoint") == filename:
                return {"status": "ok"}
            _require_switching()
            await mode.hooks.set_checkpoint(filename)
            return {"status": "ok"}

        return app
