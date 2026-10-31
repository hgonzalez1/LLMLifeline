"""
Where chat models come from: .gguf files (run by llama_cpp.server) and
engine add-ons (models another program runs).

- .gguf files are found directly inside the models folder and inside any
  folder listed in model-folders.txt (one folder per line, # for
  comments). Listed folders are searched first, so when the same file is
  in two places (for example a copy on a faster drive) the listed copy is
  the one used.
- An engine add-on is a folder in addons/ with an addon.py. It names the
  models it can run and says how to start the program that serves them.
  That program must answer OpenAI-style /v1/chat/completions on the port
  it is given; the backend then treats it exactly like llama_cpp.server.
  See addons/README.md for the contract.

An add-on model is identified as "addon:<add-on folder>/<model id>" —
that string takes the place of a .gguf path everywhere a model is chosen.
"""

import importlib.util
import re
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = APP_ROOT / "models"
FOLDERS_FILE = APP_ROOT / "model-folders.txt"
ADDONS_DIR = APP_ROOT / "addons"
LAST_MODEL_FILE = APP_ROOT / "last-chat-model.txt"

ADDON_PREFIX = "addon:"
GGUF_GROUP = "GGUF models"
_SPLIT_PART = re.compile(r"-(\d{5})-of-\d{5}\.gguf$", re.IGNORECASE)

# What a launch spec may set, with the values llama_cpp.server gets.
LAUNCH_DEFAULTS = {
    "cwd": None,
    "env": {},
    "context_length": 4096,
    "ready_path": "/v1/models",          # answers 200 only once the model can chat
    "tokenize_path": None,               # llama_cpp.server's /extras/tokenize/count, if the engine has one
    "startup_timeout": 600,              # seconds — cold .gguf loads on a hard drive have taken ~10 min
    "request_timeout": 900,
    "model_titles": True,                # False: don't spend a model call naming each conversation
    "request_options": {},               # extra fields for every chat request, e.g. sampling settings
    "stream": True,                      # False: the engine can't send a reply as it's written
}

_addons: dict | None = None


# ---- .gguf files ----

def gguf_folders() -> list[Path]:
    folders = []
    try:
        lines = FOLDERS_FILE.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        lines = []
    for line in lines:
        line = line.split("#", 1)[0].strip().strip('"')
        if line:
            folder = Path(line).expanduser()
            folders.append(folder if folder.is_absolute() else APP_ROOT / folder)
    folders.append(MODELS_DIR)
    return folders


def _is_chat_gguf(path: Path) -> bool:
    if path.name.lower().startswith("mmproj"):  # a vision projector, not a model
        return False
    part = _SPLIT_PART.search(path.name)
    return part is None or part.group(1) == "00001"  # a split model loads from its first part


def gguf_files() -> list[Path]:
    """Every chat .gguf, listed folders first, without repeating a file
    that appears (same name and size) in more than one folder."""
    found, seen = [], set()
    for folder in gguf_folders():
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.gguf")):
            if not path.is_file() or not _is_chat_gguf(path):
                continue
            key = (path.name.lower(), path.stat().st_size)
            if key not in seen:
                seen.add(key)
                found.append(path)
    return found


def same_gguf(a: str | None, b: Path) -> bool:
    """Whether model reference a is the file b, or an identical copy of it."""
    if not a or is_addon_model(a):
        return False
    path = Path(a)
    if str(path) == str(b):
        return True
    try:
        return path.name.lower() == b.name.lower() and path.stat().st_size == b.stat().st_size
    except OSError:
        return False


# ---- Engine add-ons ----

class AddonError(Exception):
    pass


def _load_addons() -> dict:
    """Import every addons/<name>/addon.py once. A broken add-on is
    reported and skipped; it never stops the backend."""
    loaded = {}
    if not ADDONS_DIR.is_dir():
        return loaded
    for folder in sorted(ADDONS_DIR.iterdir()):
        script = folder / "addon.py"
        if not script.is_file():
            continue
        key = folder.name
        module_name = "llmlifeline_addon_" + re.sub(r"\W", "_", key)
        try:
            spec = importlib.util.spec_from_file_location(module_name, script)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            if not callable(getattr(module, "models", None)) or not callable(getattr(module, "launch", None)):
                raise AddonError("addon.py needs both models() and launch(model_id, port)")
            loaded[key] = module
        except Exception as e:  # noqa: BLE001 — someone else's code; report and carry on
            print(f"[backend] Add-on {key} could not be loaded: {type(e).__name__}: {e}")
    return loaded


def addons() -> dict:
    global _addons
    if _addons is None:
        _addons = _load_addons()
    return _addons


def reload_addons() -> None:
    global _addons
    _addons = None


def is_addon_model(ref: str | None) -> bool:
    return bool(ref) and ref.startswith(ADDON_PREFIX)


def _split_ref(ref: str) -> tuple[str, str]:
    key, _, model_id = ref[len(ADDON_PREFIX):].partition("/")
    if not key or not model_id:
        raise AddonError(f"Not an add-on model: {ref}")
    return key, model_id


def addon_models() -> list[dict]:
    """Every add-on's models, as /api/models entries."""
    entries = []
    for key, module in addons().items():
        group = str(getattr(module, "NAME", key))
        try:
            models = module.models() or []
        except Exception as e:  # noqa: BLE001
            print(f"[backend] Add-on {key} could not list its models: {type(e).__name__}: {e}")
            continue
        for m in models:
            if not isinstance(m, dict) or not m.get("id"):
                continue
            entries.append({
                "path": f"{ADDON_PREFIX}{key}/{m['id']}",
                "filename": str(m.get("name") or m["id"]),
                "size_gb": round(float(m.get("size_gb") or 0), 2),
                "group": group,
                "note": str(m.get("note") or ""),
            })
    return entries


def find_addon_model(ref: str) -> dict | None:
    return next((m for m in addon_models() if m["path"] == ref), None)


def launch_spec(ref: str, port: int) -> dict:
    """How to start add-on model ref: the add-on's launch() answer with
    defaults filled in. Raises AddonError when it can't be started."""
    key, model_id = _split_ref(ref)
    module = addons().get(key)
    if module is None:
        raise AddonError(f"The add-on {key} isn't installed (addons/{key}/addon.py).")
    try:
        spec = module.launch(model_id, port)
    except Exception as e:  # noqa: BLE001
        raise AddonError(f"The add-on {key} could not start {model_id}: {e}") from e
    if not isinstance(spec, dict) or not isinstance(spec.get("args"), list) or not spec["args"]:
        raise AddonError(f"The add-on {key} returned no command to run.")
    full = {**LAUNCH_DEFAULTS, **spec}
    full["args"] = [str(a) for a in spec["args"]]
    full["env"] = {str(k): str(v) for k, v in (spec.get("env") or {}).items()}
    full["cwd"] = str(full["cwd"]) if full["cwd"] else None
    if not isinstance(full["request_options"], dict):
        raise AddonError(f"The add-on {key}'s request_options must be a dict.")
    for name in ("context_length", "startup_timeout", "request_timeout"):
        full[name] = int(full[name])
    full["stream"] = bool(full["stream"])
    return full


def display_name(ref: str | None) -> str | None:
    """A model reference as people see it: the .gguf file name, or the
    add-on's name for its model."""
    if not ref:
        return None
    if is_addon_model(ref):
        found = find_addon_model(ref)
        return found["filename"] if found else ref.rsplit("/", 1)[-1]
    return Path(ref).name


# ---- All chat models ----

def chat_models(active: str | None) -> list[dict]:
    """The model dropdown: .gguf files, then each add-on's models."""
    entries = [{
        "filename": f.name,
        "path": str(f),
        "size_gb": round(f.stat().st_size / (1024 ** 3), 2),
        "group": GGUF_GROUP,
        "active": same_gguf(active, f),
    } for f in gguf_files()]
    for m in addon_models():
        entries.append({**m, "active": m["path"] == active})
    return entries


def remember_model(ref: str) -> None:
    try:
        LAST_MODEL_FILE.write_text(ref, encoding="utf-8")
    except OSError:
        pass


def remembered_model() -> str | None:
    """The chat model last started on this computer, if it can still run."""
    try:
        ref = LAST_MODEL_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if is_addon_model(ref):
        return ref if find_addon_model(ref) else None
    return ref if ref.lower().endswith(".gguf") and Path(ref).is_file() else None
