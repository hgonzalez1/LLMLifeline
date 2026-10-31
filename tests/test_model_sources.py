"""
Where chat models come from: .gguf files in the models folder and the
folders listed in model-folders.txt, and engine add-ons in addons/ (see
backend/model_sources.py and addons/README.md). Also checks that the
backend starts an add-on's program the way the add-on asked and goes
back to llama.cpp's settings for a .gguf. Nothing is really launched.

    llm-env\\Scripts\\python.exe tests\\test_model_sources.py     (Windows)
    llm-env/bin/python tests/test_model_sources.py               (macOS/Linux)
"""

import sys
import tempfile
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import computer_use  # noqa: E402
import server_mode  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="llmlifeline-models-test-"))
server_mode.SERVER_DIR = TMP / "server"
server_mode.SETTINGS_FILE = server_mode.SERVER_DIR / "server.json"
server_mode.CERT_FILE = server_mode.SERVER_DIR / "cert.pem"
server_mode.KEY_FILE = server_mode.SERVER_DIR / "key.pem"
computer_use.PERMISSIONS_FILE = TMP / "permissions.json"

import remote_client  # noqa: E402

remote_client.REMOTE_FILE = TMP / "remote.json"
remote_client.reload()

import model_sources  # noqa: E402

APP = TMP / "app"
EXTRA = TMP / "fast drive"
model_sources.APP_ROOT = APP
model_sources.MODELS_DIR = APP / "models"
model_sources.FOLDERS_FILE = APP / "model-folders.txt"
model_sources.ADDONS_DIR = APP / "addons"
model_sources.LAST_MODEL_FILE = APP / "last-chat-model.txt"

import main  # noqa: E402

main.PROJECTS_DIR = TMP / "projects"
main.PROJECTS_DIR.mkdir()


def write(path: Path, data: str | bytes = b"") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(textwrap.dedent(data), encoding="utf-8")
    else:
        path.write_bytes(data)
    return path


write(APP / "models" / "shared.gguf", b"x" * 10)
write(APP / "models" / "big-00001-of-00002.gguf", b"x" * 5)
write(APP / "models" / "big-00002-of-00002.gguf", b"x" * 5)
write(APP / "models" / "mmproj-F16.gguf", b"x" * 3)
write(EXTRA / "shared.gguf", b"x" * 10)        # the same file, copied to a faster drive
write(EXTRA / "other.gguf", b"x" * 7)
write(APP / "model-folders.txt", f"# my SSD\n\n{EXTRA}\n{TMP / 'missing folder'}\n")

write(APP / "addons" / "good" / "addon.py", """
    NAME = "Disk models"
    calls = []

    def models():
        return [{"id": "huge/model", "name": "Huge Model", "size_gb": 186.4, "note": "streams from disk"},
                {"name": "no id, ignored"}]

    def launch(model_id, port):
        calls.append((model_id, port))
        return {"args": ["engine.exe", "--model", model_id, "--port", port], "cwd": "C:/engine",
                "env": {"THREADS": 8}, "context_length": 16384, "ready_path": "/health",
                "request_timeout": 7200, "model_titles": False,
                "request_options": {"chat_template_kwargs": {"enable_thinking": False}}}
    """)
write(APP / "addons" / "broken" / "addon.py", "raise RuntimeError('not today')\n")
write(APP / "addons" / "incomplete" / "addon.py", "def models():\n    return []\n")
write(APP / "addons" / "cannot-list" / "addon.py", """
    def models():
        raise OSError("drive missing")

    def launch(model_id, port):
        return {"args": ["x"]}
    """)
write(APP / "addons" / "no-command" / "addon.py", """
    def models():
        return [{"id": "m"}]

    def launch(model_id, port):
        return {"cwd": "C:/"}
    """)
write(APP / "addons" / "README.md", "not an add-on folder")

GOOD = "addon:good/huge/model"


def test_listed_folders_come_first_and_copies_appear_once():
    names = [(f.parent.name, f.name) for f in model_sources.gguf_files()]
    assert names == [("fast drive", "other.gguf"), ("fast drive", "shared.gguf"),
                     ("models", "big-00001-of-00002.gguf")], names


def test_a_copy_counts_as_the_active_model():
    assert model_sources.same_gguf(str(APP / "models" / "shared.gguf"), EXTRA / "shared.gguf")
    assert not model_sources.same_gguf(str(APP / "models" / "shared.gguf"), EXTRA / "other.gguf")
    assert not model_sources.same_gguf(GOOD, EXTRA / "shared.gguf")


def test_addons_that_load_offer_their_models():
    entries = model_sources.chat_models(GOOD)
    assert [e for e in entries if e["group"] == "Disk models"] == [
        {"path": GOOD, "filename": "Huge Model", "size_gb": 186.4,
         "group": "Disk models", "note": "streams from disk", "active": True}], entries
    # broken (fails to import) and incomplete (no launch) are skipped; cannot-list offers nothing
    assert set(model_sources.addons()) == {"good", "cannot-list", "no-command"}
    assert {e["group"] for e in entries} == {model_sources.GGUF_GROUP, "Disk models", "no-command"}


def test_launch_spec_fills_in_defaults():
    spec = model_sources.launch_spec(GOOD, 8000)
    assert spec["args"] == ["engine.exe", "--model", "huge/model", "--port", "8000"]
    assert spec["env"] == {"THREADS": "8"} and spec["cwd"] == "C:/engine"
    assert spec["context_length"] == 16384 and spec["ready_path"] == "/health"
    assert spec["startup_timeout"] == 600 and spec["tokenize_path"] is None


def test_unusable_addon_models_are_refused():
    for ref in ("addon:missing/x", "addon:no-command/m", "addon:good"):
        try:
            model_sources.launch_spec(ref, 8000)
        except model_sources.AddonError:
            continue
        raise AssertionError(f"{ref} was accepted")


def test_the_last_model_is_remembered_while_it_still_exists():
    model_sources.remember_model(GOOD)
    assert model_sources.remembered_model() == GOOD
    assert main._default_chat_model() == GOOD
    model_sources.remember_model("addon:good/gone")
    assert model_sources.remembered_model() is None
    model_sources.remember_model(str(TMP / "deleted.gguf"))
    assert model_sources.remembered_model() is None
    assert main._default_chat_model() == str(EXTRA / "other.gguf")  # no DEFAULT_MODEL here: the first one


class FakeProcess:
    pid = 4242

    def poll(self):
        return None


def test_the_backend_runs_an_addon_model_its_way():
    launched, stopped = [], []

    def fake_popen(args, **kwargs):
        launched.append((args, kwargs))
        return FakeProcess()

    real = main.subprocess.Popen, main._stop_process_tree, main._choose_context_length
    main.subprocess.Popen, main._stop_process_tree = fake_popen, lambda p, timeout=30: stopped.append(p.pid)
    main._choose_context_length = lambda path, doubled=False: 2048  # it asks nvidia-smi through Popen
    manager = main.LlamaServerManager()
    manager.log_path = TMP / "llama_server.log"
    try:
        manager.start(GOOD)
        args, kwargs = launched[-1]
        assert args == ["engine.exe", "--model", "huge/model", "--port", str(main.LLAMA_SERVER_PORT)]
        assert kwargs["cwd"] == "C:/engine" and kwargs["env"]["THREADS"] == "8"
        assert manager.current_model_path == GOOD and manager.context_length == 16384
        assert manager.ready_path == "/health" and manager.request_timeout == 7200
        assert manager.model_titles is False and manager.tokenize_path is None
        assert manager.request_options == {"chat_template_kwargs": {"enable_thinking": False}}
        assert model_sources.remembered_model() == GOOD

        gguf = str(EXTRA / "other.gguf")
        manager.start(gguf)
        args, _ = launched[-1]
        assert stopped == [4242]  # the add-on's program (and its children) stopped first
        assert args[1:3] == ["-m", "llama_cpp.server"] and gguf in args
        assert manager.ready_path == "/v1/models" and manager.request_timeout == 900
        assert manager.model_titles is True and manager.tokenize_path == "/extras/tokenize/count"
        assert manager.request_options == {}
    finally:
        main.subprocess.Popen, main._stop_process_tree, main._choose_context_length = real
        if manager.log_file is not None:
            manager.log_file.close()


def test_a_broken_addon_model_changes_nothing():
    manager = main.LlamaServerManager()
    try:
        manager.start("addon:no-command/m")
    except model_sources.AddonError:
        assert manager.process is None and manager.current_model_path is None
    else:
        raise AssertionError("started without a command")


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
