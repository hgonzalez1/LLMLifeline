"""
Using a model on another computer — the client half of server mode (see
server_mode.py). Only model work goes to the server: each chat reply,
token counts, generated images and photo captions. This device's
projects, conversations, files, persona, beliefs and computer use all
stay here, and the assistant's actions (commands, file changes) run on
this device, never on the server.

The connection is pinned to the exact certificate the server had when
it was paired, so anything else answering at that address later is
refused rather than trusted.
"""

import asyncio
import json
import secrets
import socket
import ssl
import uuid
from pathlib import Path

import httpx

from server_mode import SERVICE_TYPE, certificate_fingerprint, pairing_code

APP_ROOT = Path(__file__).resolve().parent.parent
REMOTE_FILE = APP_ROOT / "remote.json"

_state: dict | None = None
_last_info: dict = {}
_last_error: str | None = None  # why the last call to the server failed, for the Network panel


class RemoteError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def state() -> dict:
    global _state
    if _state is None:
        try:
            _state = json.loads(REMOTE_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _state = {}
        _state.setdefault("active", False)
        _state.setdefault("server", None)
        _state.setdefault("pairing", None)
        _state.setdefault("device_id", uuid.uuid4().hex)
        _state.setdefault("device_name", socket.gethostname() or "LLMlifeline device")
    return _state


def _save() -> None:
    REMOTE_FILE.write_text(json.dumps(state(), indent=2), encoding="utf-8")


def reload() -> None:
    global _state, _last_info, _last_error
    _state, _last_info, _last_error = None, {}, None


def active() -> bool:
    return bool(state().get("active") and state().get("server"))


def public_state() -> dict:
    """What the Network panel shows — never the key."""
    s = state()
    server = s.get("server") or {}
    pairing = s.get("pairing") or {}
    return {
        "active": active(),
        "device_name": s["device_name"],
        "server": {k: server.get(k) for k in ("name", "host", "port", "fingerprint")} if server else None,
        "pairing": {k: pairing.get(k) for k in ("name", "host", "port", "code")} if pairing else None,
        "last_info": _last_info or None,
        "error": _last_error if server else None,
    }


def _base(host: str, port: int) -> str:
    return f"https://[{host}]:{port}" if ":" in host else f"https://{host}:{port}"


def _pinned(pem: str) -> ssl.SSLContext:
    context = ssl.create_default_context(cadata=pem)
    # The pin is the certificate itself; the address can change with DHCP.
    context.check_hostname = False
    return context


async def _fetch_certificate(host: str, port: int) -> str:
    try:
        return await asyncio.to_thread(ssl.get_server_certificate, (host, port), timeout=6)
    except (OSError, ssl.SSLError) as e:
        raise RemoteError(503, f"Couldn't reach an LLMlifeline server at {host}:{port} ({e}). Check that it's "
                               f"on, server mode is turned on, and both devices are on the same network.")


def _detail(response: httpx.Response) -> str:
    try:
        return response.json().get("detail") or response.text
    except ValueError:
        return response.text or f"HTTP {response.status_code}"


# ---- Finding and pairing with a server ----

async def discover(seconds: float = 3.0) -> list[dict]:
    """Servers announcing themselves on this network (mDNS). Best effort:
    some networks and firewalls block it, so typing the address works too."""
    try:
        from zeroconf import ServiceStateChange
        from zeroconf.asyncio import AsyncServiceBrowser, AsyncServiceInfo, AsyncZeroconf
    except ImportError:
        return []
    names: set[str] = set()

    def on_change(zeroconf, service_type, name, state_change):
        if state_change is ServiceStateChange.Added:
            names.add(name)

    found = []
    aiozc = None
    try:
        aiozc = AsyncZeroconf()
        browser = AsyncServiceBrowser(aiozc.zeroconf, [SERVICE_TYPE], handlers=[on_change])
        await asyncio.sleep(seconds)
        for name in names:
            info = AsyncServiceInfo(SERVICE_TYPE, name)
            if await info.async_request(aiozc.zeroconf, 1500) and info.parsed_addresses():
                props = {k.decode(): (v or b"").decode() for k, v in (info.properties or {}).items()}
                found.append({"name": props.get("name") or name.split(".")[0],
                              "host": info.parsed_addresses()[0], "port": info.port})
        await browser.async_cancel()
    except Exception:  # noqa: BLE001 — discovery is a convenience
        pass
    finally:
        if aiozc is not None:
            await aiozc.async_close()
    return found


async def begin_pairing(host: str, port: int) -> dict:
    host = host.strip().strip("[]")
    pem = await _fetch_certificate(host, port)
    fingerprint = certificate_fingerprint(pem)
    secret = secrets.token_urlsafe(24)
    s = state()
    async with httpx.AsyncClient(verify=_pinned(pem), timeout=15.0, trust_env=False) as client:
        try:
            hello = await client.get(_base(host, port) + "/v1/hello")
            if hello.status_code != 200 or hello.json().get("app") != "LLMlifeline":
                raise RemoteError(400, f"{host}:{port} isn't an LLMlifeline server.")
            response = await client.post(_base(host, port) + "/v1/pair", json={
                "device_id": s["device_id"], "device_name": s["device_name"], "secret": secret})
        except httpx.HTTPError as e:
            raise RemoteError(503, f"Couldn't talk to {host}:{port}: {e}")
    if response.status_code != 200:
        raise RemoteError(response.status_code, _detail(response))
    request_id = response.json()["request_id"]
    name = hello.json().get("name") or host
    s["pairing"] = {"request_id": request_id, "secret": secret, "host": host, "port": port, "name": name,
                    "cert_pem": pem, "fingerprint": fingerprint, "code": pairing_code(fingerprint, request_id)}
    _save()
    return public_state()


async def check_pairing() -> dict:
    s = state()
    pairing = s.get("pairing")
    if not pairing:
        return {"status": "none", **public_state()}
    async with httpx.AsyncClient(verify=_pinned(pairing["cert_pem"]), timeout=15.0, trust_env=False) as client:
        try:
            response = await client.post(_base(pairing["host"], pairing["port"]) + f"/v1/pair/{pairing['request_id']}",
                                         json={"secret": pairing["secret"]})
        except httpx.HTTPError as e:
            raise RemoteError(503, f"Couldn't reach {pairing['name']}: {e}")
    if response.status_code == 404:
        s["pairing"] = None
        _save()
        return {"status": "expired", **public_state()}
    if response.status_code != 200:
        raise RemoteError(response.status_code, _detail(response))
    result = response.json()
    if result["status"] == "approved":
        s["server"] = {"name": result.get("name") or pairing["name"], "host": pairing["host"],
                       "port": pairing["port"], "cert_pem": pairing["cert_pem"],
                       "fingerprint": pairing["fingerprint"], "token": result["token"]}
        s["active"] = True
        s["pairing"] = None
        _save()
    elif result["status"] == "denied":
        s["pairing"] = None
        _save()
    return {"status": result["status"], **public_state()}


def cancel_pairing() -> dict:
    state()["pairing"] = None
    _save()
    return public_state()


def set_active(on: bool) -> dict:
    s = state()
    if on and not s.get("server"):
        raise RemoteError(400, "Pair with a server first.")
    s["active"] = on
    _save()
    return public_state()


def forget() -> dict:
    s = state()
    s["active"] = False
    s["server"] = None
    s["pairing"] = None
    _last_info.clear()
    _save()
    return public_state()


# ---- Using it ----

async def call(method: str, path: str, *, json_body: dict | None = None, content: bytes | None = None,
               timeout: float = 900.0) -> dict:
    global _last_error
    try:
        result = await _call(method, path, json_body=json_body, content=content, timeout=timeout)
    except RemoteError as e:
        _last_error = e.detail
        raise
    _last_error = None
    return result


async def _call(method: str, path: str, *, json_body: dict | None, content: bytes | None,
                timeout: float) -> dict:
    server = state().get("server")
    if not server:
        raise RemoteError(400, "Not connected to a server.")
    headers = {"Authorization": f"Bearer {server['token']}", "X-Device-Id": state()["device_id"]}
    if content is not None:
        headers["Content-Type"] = "application/octet-stream"
    try:
        async with httpx.AsyncClient(verify=_pinned(server["cert_pem"]), timeout=timeout, trust_env=False) as client:
            response = await client.request(method, _base(server["host"], server["port"]) + path,
                                            json=json_body, content=content, headers=headers)
    except httpx.ConnectError as e:
        if "CERTIFICATE" in str(e).upper():
            raise RemoteError(503, f"{server['name']}'s identity doesn't match the one it paired with — it may "
                                   f"have been reinstalled, or something else is answering at {server['host']}. "
                                   f"Pair again to reconnect.")
        raise RemoteError(503, f"Can't reach {server['name']} at {server['host']}. Check that it's on, in "
                               f"server mode, and on the same network.")
    except httpx.ReadTimeout:
        raise RemoteError(504, f"{server['name']} took too long to answer.")
    except httpx.HTTPError as e:
        raise RemoteError(503, f"Lost the connection to {server['name']}: {e}")
    if response.status_code >= 400:
        raise RemoteError(response.status_code, _detail(response))
    return response.json()


async def info(timeout: float = 4.0) -> dict:
    data = await call("GET", "/v1/info", timeout=timeout)
    _last_info.clear()
    _last_info.update(data)
    return data


def cached_context_length() -> int | None:
    return _last_info.get("context_length")
