"""
Replies sent as they're written: reading the model server's stream
(server-sent events, as llama_cpp.server and OpenAI-style engines send
them), and what happens when a stream breaks off, an engine can't
stream, or a reply needs continuing. The model server is scripted, so
nothing is launched.

    llm-env\\Scripts\\python.exe tests\\test_streaming.py     (Windows)
    llm-env/bin/python tests/test_streaming.py               (macOS/Linux)
"""

import asyncio
import contextlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import computer_use  # noqa: E402
import server_mode  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="llmlifeline-stream-test-"))
server_mode.SERVER_DIR = TMP / "server"
server_mode.SETTINGS_FILE = server_mode.SERVER_DIR / "server.json"
server_mode.CERT_FILE = server_mode.SERVER_DIR / "cert.pem"
server_mode.KEY_FILE = server_mode.SERVER_DIR / "key.pem"
computer_use.PERMISSIONS_FILE = TMP / "permissions.json"

import remote_client  # noqa: E402

remote_client.REMOTE_FILE = TMP / "remote.json"
remote_client.reload()

import httpx  # noqa: E402
from fastapi import HTTPException  # noqa: E402

import main  # noqa: E402

main.PROJECTS_DIR = TMP / "projects"
REAL_CLIENT = httpx.AsyncClient
requests: list[dict] = []


def sse(*events) -> bytes:
    """A stream as llama_cpp.server sends it, with a keep-alive ping in it."""
    lines = [": ping - 2026-10-30 12:00:00"]
    lines += [f"data: {e if isinstance(e, str) else json.dumps(e)}" for e in events]
    return ("\n\n".join(lines) + "\n\n").encode()


def piece(text=None, finish=None) -> dict:
    delta = {"content": text} if text is not None else {}
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


@contextlib.contextmanager
def model_server(*responses):
    """The model server answers each request with the next response."""
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return queue.pop(0)

    httpx.AsyncClient = lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(handler), **kw)
    try:
        yield
    finally:
        httpx.AsyncClient = REAL_CLIENT


def streamed(body: bytes) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)


def error_of(coro) -> HTTPException:
    try:
        asyncio.run(coro)
    except HTTPException as e:
        return e
    raise AssertionError("no error")


def test_pieces_arrive_as_written_and_add_up():
    main.llama_manager.request_options = {"model": "engine-model"}
    got = []
    try:
        with model_server(streamed(sse(piece(""), piece("Hel"), piece("lo"), piece(finish="stop"),
                                       {"choices": [], "usage": {"total_tokens": 9}}, "[DONE]"))):
            result = asyncio.run(main._local_completion_stream({"messages": []}, got.append))
    finally:
        main.llama_manager.request_options = {}
    assert got == ["Hel", "lo"], got
    assert result["choices"][0]["message"]["content"] == "Hello"
    assert result["choices"][0]["finish_reason"] == "stop" and result["usage"] == {"total_tokens": 9}
    sent = requests[-1]
    assert sent["stream"] is True and sent["model"] == "engine-model", sent


def test_an_engine_that_ignores_stream_answers_in_one_piece():
    whole = {"choices": [{"message": {"content": "All at once."}, "finish_reason": "stop"}]}
    got = []
    with model_server(httpx.Response(200, json=whole)):
        content, *_ = asyncio.run(main._generate([{"role": "user", "content": "hi"}], 4096, got.append))
    assert content == "All at once." and got == ["All at once."], got


def test_an_addon_can_turn_streaming_off():
    whole = {"choices": [{"message": {"content": "Unstreamed."}, "finish_reason": "stop"}]}
    got = []
    main.llama_manager.stream = False
    try:
        with model_server(httpx.Response(200, json=whole)):
            content, *_ = asyncio.run(main._generate([{"role": "user", "content": "hi"}], 4096, got.append))
    finally:
        main.llama_manager.stream = True
    assert "stream" not in requests[-1] and content == "Unstreamed." and got == ["Unstreamed."]


def test_a_reply_cut_off_partway_is_an_error_not_an_answer():
    # What llama_cpp.server sends when it drops a stream for another request.
    with model_server(streamed(sse(piece("The first half"), "[DONE]"))):
        e = error_of(main._local_completion_stream({"messages": []}, lambda text: None))
    assert e.status_code == 502 and "before it finished" in e.detail, e.detail


def test_engine_errors_are_explained():
    too_long = httpx.Response(400, json={"error": {"message": "Requested tokens exceed context length"}})
    with model_server(too_long):
        e = error_of(main._local_completion_stream({"messages": []}, lambda text: None))
    assert e.status_code == 502 and "too long" in e.detail, e.detail

    with model_server(streamed(sse(piece("Hi"), {"error": {"message": "engine fault"}}))):
        e = error_of(main._local_completion_stream({"messages": []}, lambda text: None))
    assert e.status_code == 502 and "engine fault" in e.detail, e.detail


def test_continuations_stream_too():
    got = []
    with model_server(streamed(sse(piece("One sentence. And a half"), piece(finish="length"), "[DONE]")),
                      streamed(sse(piece("Two sentences."), piece(finish="stop"), "[DONE]"))):
        content, _, cut_short, _ = asyncio.run(
            main._generate([{"role": "user", "content": "write"}], 4096, got.append))
    assert got == ["One sentence. And a half", "Two sentences."], got
    assert content == "One sentence. Two sentences." and not cut_short, content


def test_a_server_from_before_streaming_answers_in_one_piece():
    # This device asks the paired server to stream; a v1.2 server answers with plain JSON.
    remote_client._state = {"active": True, "device_id": "this-device", "server": {
        "host": "192.168.1.20", "port": 8100, "token": "key", "name": "Old server",
        "cert_pem": server_mode.ensure_certificate()}}
    whole = {"choices": [{"message": {"content": "Old reply."}, "finish_reason": "stop"}]}
    got = []
    try:
        with model_server(httpx.Response(200, json=whole)):
            content, *_ = asyncio.run(main._generate([{"role": "user", "content": "hi"}], 4096, got.append))
    finally:
        remote_client.reload()
    assert requests[-1]["stream"] is True and content == "Old reply." and got == ["Old reply."], got


if __name__ == "__main__":
    failures = 0
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            try:
                test()
                print(f"ok    {name}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{failures} failed" if failures else "\nall passed")
    sys.exit(1 if failures else 0)
