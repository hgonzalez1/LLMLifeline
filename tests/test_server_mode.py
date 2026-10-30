"""
Server mode end to end, over real TLS on this computer: pairing (with the
matching 6-digit code), chat and images through the gateway, booting,
blocking, certificate pinning and the local-network-only rule. The
model itself is scripted, so no GPU or model files are needed.

    llm-env\\Scripts\\python.exe tests\\test_server_mode.py     (Windows)
    llm-env/bin/python tests/test_server_mode.py               (macOS/Linux)
"""

import asyncio
import socket
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import computer_use  # noqa: E402
import server_mode  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="llmlifeline-server-test-"))
server_mode.SERVER_DIR = TMP / "server"
server_mode.SETTINGS_FILE = server_mode.SERVER_DIR / "server.json"
server_mode.CERT_FILE = server_mode.SERVER_DIR / "cert.pem"
server_mode.KEY_FILE = server_mode.SERVER_DIR / "key.pem"
computer_use.PERMISSIONS_FILE = TMP / "permissions.json"

import remote_client  # noqa: E402

remote_client.REMOTE_FILE = TMP / "remote.json"
remote_client.reload()

import main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

main.PROJECTS_DIR = TMP / "projects"
main.PROJECTS_DIR.mkdir()
main.GENERATED_IMAGES_DIR = TMP / "generated_images"

SECRET_PROMPT = "my private recipe for grandma's tamales uses 7 dried chiles"
seen_by_model: list = []


async def fake_completion(payload, timeout=900.0):
    seen_by_model.append(payload["messages"][-1]["content"])
    return {"choices": [{"message": {"content": "Reply from the server's model."}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 12}}


async def fake_token_count(text):
    return len(text) // 4


async def fake_status():
    return "ok"


async def fake_txt2img(prompt):
    return b"\x89PNG\r\n\x1a\nfake-image-bytes"


async def no_advertise():
    return None


async def no_title(*args):
    return None


main._local_completion = fake_completion
main._local_token_count = fake_token_count
main._local_llama_status = fake_status
main._comfy_txt2img = fake_txt2img
main._caption_bytes = lambda image: "a test picture"
main._title_background_task = no_title
main.llama_manager.is_alive = lambda: True
main.llama_manager.context_length = 4096
main.llama_manager.current_model_path = str(TMP / "fake-model.gguf")
main.image_manager.current_checkpoint = "fake.safetensors"
main.server._advertise = no_advertise

client = TestClient(main.app, base_url="http://127.0.0.1:8001")

# The gateway runs in its own event loop on a thread, like it runs inside
# the real backend's loop; this test's client side uses separate loops.
server_loop = asyncio.new_event_loop()
threading.Thread(target=server_loop.run_forever, daemon=True).start()


def on_server(coro):
    return asyncio.run_coroutine_threadsafe(coro, server_loop).result(60)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


PORT = free_port()


def pair() -> str:
    asyncio.run(remote_client.begin_pairing("127.0.0.1", PORT))
    pending = main.server.status()["pending"]
    assert len(pending) == 1, pending
    assert pending[0]["code"] == remote_client.public_state()["pairing"]["code"], "codes must match on both screens"
    main.server.answer_pairing(pending[0]["id"], True)
    result = asyncio.run(remote_client.check_pairing())
    assert result["status"] == "approved", result
    return remote_client.state()["device_id"]


def test_local_network_rule():
    for address in ("192.168.1.20", "10.0.0.5", "172.16.3.4", "127.0.0.1", "fe80::1", "fd12::1"):
        assert server_mode.is_local_network_address(address), address
    for address in ("8.8.8.8", "1.1.1.1", "::ffff:8.8.8.8", "2606:4700::1111", "not-an-ip"):
        assert not server_mode.is_local_network_address(address), address


def test_server_starts_with_its_own_certificate():
    on_server(main.server.apply({"enabled": True, "port": PORT}))
    status = main.server.status()
    assert status["running"], status["error"]
    assert server_mode.CERT_FILE.exists() and status["fingerprint"]


def test_pairing_needs_approval_and_matching_codes():
    asyncio.run(remote_client.begin_pairing("127.0.0.1", PORT))
    assert asyncio.run(remote_client.check_pairing())["status"] == "pending"
    request_id = main.server.status()["pending"][0]["id"]
    main.server.answer_pairing(request_id, False)
    assert asyncio.run(remote_client.check_pairing())["status"] == "denied"
    assert not remote_client.active()
    pair()
    assert remote_client.active()


def test_chat_runs_on_the_server_and_stays_on_the_device():
    project_id = client.post("/api/projects", json={"name": "Remote chat"}).json()["id"]
    response = client.post("/api/chat", json={"project_id": project_id,
                                              "messages": [{"role": "user", "content": SECRET_PROMPT}]})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["reply"] == "Reply from the server's model."
    assert SECRET_PROMPT in seen_by_model[-1]
    # The conversation is saved on the device...
    conversation = main._load_conversation(project_id, data["conversation_id"])
    assert any(SECRET_PROMPT in m["content"] for m in conversation["messages"])
    # ...and nothing from it is on the server's disk.
    for file in server_mode.SERVER_DIR.iterdir():
        assert "tamales" not in file.read_text(encoding="utf-8", errors="replace"), file
    health = client.get("/api/health").json()
    assert health["llama_cpp_server"] == "ok" and health["remote"]["name"]
    assert main.server.status()["waiting"] == 0


def test_images_come_back_to_the_device():
    project_id = client.post("/api/projects", json={"name": "Remote images"}).json()["id"]
    data = client.post("/api/image/generate", json={"project_id": project_id, "prompt": "a lighthouse"})
    assert data.status_code == 409, data.text  # the server's in chat mode (ComfyUI not running), and says so
    main.image_manager.is_alive = lambda: True  # now it's in image mode
    data = client.post("/api/image/generate", json={"project_id": project_id, "prompt": "a lighthouse"})
    assert data.status_code == 200, data.text
    body = data.json()
    assert body["caption"] == "a test picture"
    saved = main.GENERATED_IMAGES_DIR / project_id / Path(body["image_url"]).name
    assert saved.read_bytes().startswith(b"\x89PNG")


def test_switching_is_refused_unless_allowed():
    response = client.post("/api/models/swap", json={"model_path": "some-other-model.gguf"})
    assert response.status_code == 403 and "doesn't let devices" in response.json()["detail"]


def test_boot_disconnects_and_block_refuses_pairing():
    device_id = remote_client.state()["device_id"]
    assert main.server.status()["devices"][0]["connected"]
    main.server.boot(device_id)
    try:
        asyncio.run(remote_client.info())
        raise AssertionError("a booted device must be refused")
    except remote_client.RemoteError as e:
        assert e.status_code == 401
    main.server.block(device_id, True)
    try:
        asyncio.run(remote_client.begin_pairing("127.0.0.1", PORT))
        raise AssertionError("a blocked device must not be able to pair")
    except remote_client.RemoteError as e:
        assert e.status_code == 403 and "blocked" in e.detail
    main.server.block(device_id, False)
    pair()
    assert asyncio.run(remote_client.info())["model"] == "fake-model.gguf"


def test_an_impostor_certificate_is_refused():
    real = remote_client.state()["server"]["cert_pem"]
    other_dir = TMP / "other"
    original = (server_mode.SERVER_DIR, server_mode.CERT_FILE, server_mode.KEY_FILE)
    server_mode.SERVER_DIR, server_mode.CERT_FILE, server_mode.KEY_FILE = (
        other_dir, other_dir / "cert.pem", other_dir / "key.pem")
    impostor = server_mode.ensure_certificate()
    server_mode.SERVER_DIR, server_mode.CERT_FILE, server_mode.KEY_FILE = original
    remote_client.state()["server"]["cert_pem"] = impostor
    try:
        asyncio.run(remote_client.info())
        raise AssertionError("a different certificate must be refused")
    except remote_client.RemoteError as e:
        assert "identity" in e.detail
    finally:
        remote_client.state()["server"]["cert_pem"] = real


def test_turning_server_mode_off_stops_it():
    on_server(main.server.apply({"enabled": False}))
    assert not main.server.status()["running"]


if __name__ == "__main__":
    failures = 0
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            try:
                test()
                print(f"ok    {name}")
            except Exception as e:  # noqa: BLE001 — report every failure, not just the first
                failures += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{failures} failed" if failures else "\nall passed")
    sys.exit(1 if failures else 0)
