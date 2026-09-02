"""
LLMlifeline launcher — the Python counterpart to start.bat, compiled into
LLMlifeline.exe via build_exe.bat/PyInstaller. Does the same things
start.bat does (dependency check, torch/CUDA self-heal, launch backend,
wait for readiness, open the browser) so there's a genuine double-click
.exe, not just a batch file with a nicer icon.

IMPORTANT: this duplicates start.bat's logic on purpose rather than one
calling the other — start.bat has to keep working standalone (no Python
needed on PATH just to run a .bat file) as the fallback if the .exe
build is ever missing or broken. If you change the dependency list, the
torch/CUDA version requirement, or the startup sequence here, make the
same change in start.bat, and vice versa.

This script itself stays deliberately lightweight — it never imports
torch, ComfyUI, or anything heavy; it only ever shells out to the REAL
venv's own python.exe to do actual work, exactly like start.bat does.
That's what keeps PyInstaller's job small and reliable: this file (and
everything it imports) is pure stdlib, so freezing it is fast and won't
drag the ~15GB of ML dependencies into the exe — those stay in llm-env,
untouched by packaging. See setup-guide.md for what a fresh machine needs
before either this or start.bat can succeed.
"""

import http.client
import os
import shutil
import stat
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# PyInstaller's --onefile bootloader extracts the frozen script to a
# temp directory at runtime and runs it FROM there — __file__ inside a
# frozen build resolves to that temp extraction, not to where
# LLMlifeline.exe actually sits on disk. sys.executable is the exe's
# real path in that case; __file__ is only correct when running this
# as a plain .py script (not frozen). Confirmed the wrong branch here
# breaks everything: it can't find backend/main.py at all when frozen.
if getattr(sys, "frozen", False):
    APP_ROOT = Path(sys.executable).resolve().parent
else:
    APP_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = APP_ROOT / "llm-env" / "Scripts" / "python.exe"
BACKEND_DIR = APP_ROOT / "backend"
BACKEND_MAIN = BACKEND_DIR / "main.py"
REQUIREMENTS = BACKEND_DIR / "requirements.txt"
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8001

# ComfyUI is a real upstream project (github.com/comfyanonymous/ComfyUI),
# not code that belongs to this one, so it's never copied/zipped along
# with the rest of this project on purpose (see setup-guide.md) — it
# gets cloned fresh instead (see ensure_comfyui), pinned to the exact
# commit this project's image-generation code (backend/main.py's
# txt2img workflow + API calls) was actually built and tested against,
# rather than tracking upstream HEAD.
COMFYUI_REPO_URL = "https://github.com/comfyanonymous/ComfyUI.git"
COMFYUI_PINNED_COMMIT = "8a33128f2f8c5585c57486c07de481241e70a39c"

# Windows SW_MINIMIZE — used to launch the backend's own console window
# minimized, same as start.bat's `start ... /min`. Not exposed as a
# named subprocess constant, but the raw value works fine and needs no
# extra dependency (no pywin32) — keeps this launcher pure stdlib.
SW_MINIMIZE = 6


def _run(args, capture=False):
    return subprocess.run(args, capture_output=capture)


def _force_rmtree(path: Path) -> None:
    """
    shutil.rmtree(..., ignore_errors=True) can silently fail to fully
    remove a directory on Windows when it contains files marked
    read-only — confirmed with a real git checkout: git leaves
    .git/objects/pack/* read-only, ignore_errors swallows the resulting
    PermissionError, and the directory is left partially in place, which
    then makes the FOLLOW-UP venv-creation or git-clone fail with
    "already exists and is not empty" instead of actually recovering.
    This clears the read-only bit before retrying deletion, which is
    what actually gets rid of a git working tree (or anything else with
    read-only files) reliably. Native `rmdir /s /q` in start.bat doesn't
    need this — confirmed it handles read-only files on its own.
    """
    def _on_rm_error(func, target_path, exc_info):
        try:
            os.chmod(target_path, stat.S_IWRITE)
            func(target_path)
        except Exception:
            pass
    shutil.rmtree(path, onerror=_on_rm_error)


def _check_importable(module_name: str) -> bool:
    return _run([str(VENV_PYTHON), "-c", f"import {module_name}"], capture=True).returncode == 0


def _check_torch_cuda_current() -> bool:
    # Checks the torch VERSION, not just CUDA availability: comfy_kitchen
    # (ComfyUI's kernel library) uses custom-op type hints torch's own
    # schema inference only accepts from 2.7.0 onward — an older
    # CUDA-enabled torch still "has CUDA" but crashes ComfyUI on import.
    # Confirmed and fixed on the original dev machine by reinstalling
    # from https://download.pytorch.org/whl/cu126 (2.7.1).
    code = (
        "import torch; from packaging.version import Version; "
        "exit(0 if torch.cuda.is_available() "
        "and Version(torch.__version__.split('+')[0]) >= Version('2.7.0') else 1)"
    )
    return _run([str(VENV_PYTHON), "-c", code], capture=True).returncode == 0


def _pause_and_exit(code: int = 1):
    try:
        input("Press Enter to exit...")
    except EOFError:
        pass
    sys.exit(code)


def _find_system_python() -> list[str] | None:
    """
    Locate a Python usable for creating llm-env from scratch on a machine
    that's never run this project before. Tries the Windows `py` launcher
    with an explicit 3.10 first (most reliable when multiple Pythons are
    installed side by side), then falls back to whatever "python"
    resolves to on PATH — not a hard requirement of exactly 3.10, since
    most of this project's dependencies publish wheels for a wide version
    range; ensure_venv prints a heads-up when it falls back like this.
    """
    py_launcher = shutil.which("py")
    if py_launcher and _run([py_launcher, "-3.10", "--version"], capture=True).returncode == 0:
        return [py_launcher, "-3.10"]

    python_on_path = shutil.which("python")
    if python_on_path and _run([python_on_path, "--version"], capture=True).returncode == 0:
        return [python_on_path]

    return None


def ensure_venv():
    """
    Create llm-env from scratch if it's missing, or if it exists but is
    broken. The most common way it'd be broken: it was copied here from a
    different machine along with the rest of the project — a venv bakes
    in an absolute path to its base Python install (see
    llm-env/pyvenv.cfg), and simply does not run anywhere else. This is
    what makes handing this project to someone else as a folder/zip
    actually work without them needing to understand any of that —
    "the file exists" isn't good enough, so this actually tries to run it.
    """
    needs_fresh = not VENV_PYTHON.exists()
    if not needs_fresh:
        needs_fresh = _run([str(VENV_PYTHON), "-c", "import sys"], capture=True).returncode != 0
    if not needs_fresh:
        return

    print("No working Python virtual environment found - setting one up now.")
    print("This is a one-time step.")
    venv_dir = APP_ROOT / "llm-env"
    if venv_dir.exists():
        print("  Removing the existing llm-env folder first (it looks broken,")
        print("  most likely copied from a different machine - venvs aren't")
        print("  portable)...")
        _force_rmtree(venv_dir)

    system_python = _find_system_python()
    if system_python is None:
        print("ERROR: No Python installation found on this machine.")
        print("Install Python 3.10 from https://www.python.org/downloads/")
        print('(check "Add python.exe to PATH" during install), then run this again.')
        _pause_and_exit()

    if "-3.10" not in system_python:
        print('  NOTE: Python 3.10 specifically wasn\'t found - using whatever')
        print('  "python" resolves to on PATH instead. This project was built and')
        print("  tested against 3.10; most dependencies publish wheels for a wide")
        print("  version range, so this will often still work, but if package")
        print("  installs fail below, installing Python 3.10 itself")
        print("  (https://www.python.org/downloads/) is the safest fix.")

    result = _run(system_python + ["-m", "venv", str(venv_dir)])
    if result.returncode != 0:
        print("\nERROR: Could not create the virtual environment. See output above.")
        _pause_and_exit()
    print("  Virtual environment created.")
    print()


def ensure_comfyui():
    """See COMFYUI_REPO_URL/COMFYUI_PINNED_COMMIT above for why this clones
    rather than expecting ComfyUI to already be part of the project folder."""
    comfyui_dir = APP_ROOT / "ComfyUI"
    if (comfyui_dir / "main.py").exists():
        return

    print("ComfyUI not found - cloning it fresh from GitHub. This is a real")
    print("~6-7GB download and can take a while on a slow connection.")
    git = shutil.which("git")
    if git is None:
        print("ERROR: git is not installed or not on PATH.")
        print("Install it from https://git-scm.com/downloads, then run this again.")
        _pause_and_exit()

    if comfyui_dir.exists():
        print("  Removing incomplete ComfyUI folder...")
        _force_rmtree(comfyui_dir)

    result = _run([git, "clone", COMFYUI_REPO_URL, str(comfyui_dir)])
    if result.returncode != 0:
        print("\nERROR: Could not clone ComfyUI. See output above.")
        _pause_and_exit()
    _run([git, "-C", str(comfyui_dir), "checkout", COMFYUI_PINNED_COMMIT], capture=True)
    print("  ComfyUI cloned.")
    print()


def _warn_if_no_models():
    """
    Model files are never auto-downloadable (see setup-guide.md), so the
    best this can do is say so clearly instead of leaving someone to
    wonder why the model dropdown is empty once the app opens.
    """
    models_dir = APP_ROOT / "models"
    if not models_dir.exists() or not list(models_dir.glob("*.gguf")):
        print("NOTE: No .gguf chat models found in models\\ yet. Add at least one")
        print("there before trying to chat - see setup-guide.md.")
        print()

    checkpoints_dir = APP_ROOT / "ComfyUI" / "models" / "checkpoints"
    checkpoint_exts = (".safetensors", ".ckpt", ".pt", ".bin")
    has_checkpoint = checkpoints_dir.exists() and any(
        f.suffix.lower() in checkpoint_exts for f in checkpoints_dir.glob("*") if f.is_file()
    )
    if not has_checkpoint:
        print("NOTE: No image checkpoints found in ComfyUI\\models\\checkpoints\\ yet.")
        print("Add at least one .safetensors file there before trying Image Model.")
        print()


def ensure_dependencies():
    """
    Mirrors start.bat's dependency-check block: checks each package
    individually (not one blind `pip install -r requirements.txt`) so an
    already-complete environment starts fast, and installs the whole
    requirements.txt only when something's actually missing.
    """
    print("Checking dependencies...")
    module_to_label = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "httpx": "httpx",
        "multipart": "python-multipart (needed for image/document uploads)",
        "llama_cpp": "llama-cpp-python",
        "pypdf": "pypdf",
        "comfy_kitchen": "comfy_kitchen",
    }
    missing = [label for mod, label in module_to_label.items() if not _check_importable(mod)]
    if not _check_importable("transformers") or not _check_importable("PIL"):
        missing.append("transformers/Pillow (needed for image captioning)")

    if missing:
        for label in missing:
            print(f"  Missing: {label}")
        print()
        print("Installing missing packages. This is a one-time step and may")
        print("take several minutes, especially for llama-cpp-python.")
        print()
        result = _run([str(VENV_PYTHON), "-m", "pip", "install",
                       "--break-system-packages", "-r", str(REQUIREMENTS)])
        if result.returncode != 0:
            print("\nERROR: Dependency install failed. See output above.")
            _pause_and_exit()
        print()
        print("NOTE: if llama-cpp-python was just installed fresh above, it is")
        print("a CPU-only build by default and will run WITHOUT GPU")
        print("acceleration. Rebuilding it with CUDA support requires the")
        print("original CMAKE_ARGS build process, not this script - see setup-guide.md.")
        print()

    print("Checking torch/CUDA status...")
    if _check_torch_cuda_current():
        print("  torch CUDA: OK")
    else:
        print("  torch is missing, CPU-only, or older than the 2.7.0 comfy_kitchen")
        print("  needs. Installing a current CUDA-enabled build from PyTorch's")
        print("  cu126 index. This is a large download and may take several minutes.")
        _run([str(VENV_PYTHON), "-m", "pip", "uninstall", "torch", "torchvision",
              "torchaudio", "-y"], capture=True)
        result = _run([str(VENV_PYTHON), "-m", "pip", "install",
                       "torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1",
                       "--index-url", "https://download.pytorch.org/whl/cu126"])
        if result.returncode != 0:
            print("\nERROR: torch CUDA install failed. See output above.")
            print("ComfyUI image generation will not have GPU acceleration")
            print("until this is resolved.")
        elif not _check_torch_cuda_current():
            print("  WARNING: torch reinstalled but still does not report CUDA")
            print("  availability. This may indicate a driver or CUDA toolkit")
            print("  mismatch beyond what this script can fix automatically.")
        else:
            print("  torch CUDA install succeeded.")
    print()


def download_default_models():
    """
    Runs download_models.py via the venv's own Python (it needs
    huggingface_hub, which is only guaranteed to exist once
    ensure_dependencies() has already run). That script gates itself on
    its own marker file and is a fast no-op after the first real run —
    see its own docstring for the full reasoning.
    """
    _run([str(VENV_PYTHON), str(APP_ROOT / "download_models.py")])


def wait_for_backend(timeout_seconds: int = 180) -> bool:
    """Poll /api/health instead of guessing a fixed sleep — the same
    "wait for a real readiness signal" approach the backend itself uses
    for llama_cpp.server and ComfyUI (see _wait_for_server_ready /
    _wait_for_comfyui_ready in backend/main.py)."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            conn = http.client.HTTPConnection(BACKEND_HOST, BACKEND_PORT, timeout=2)
            conn.request("GET", "/api/health")
            if conn.getresponse().status == 200:
                return True
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass
        time.sleep(1)
    return False


def main():
    os.chdir(APP_ROOT)

    if not BACKEND_MAIN.exists():
        print(f"ERROR: Could not find {BACKEND_MAIN}")
        _pause_and_exit()

    ensure_venv()
    print(f"Using Python: {VENV_PYTHON}")
    print()

    ensure_comfyui()
    ensure_dependencies()
    download_default_models()
    _warn_if_no_models()

    print(f"Starting backend on port {BACKEND_PORT}...")
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = SW_MINIMIZE
    subprocess.Popen(
        [str(VENV_PYTHON), "-m", "uvicorn", "main:app",
         "--app-dir", str(BACKEND_DIR), "--port", str(BACKEND_PORT)],
        creationflags=subprocess.CREATE_NEW_CONSOLE,
        startupinfo=startupinfo,
    )

    print("Waiting for the backend to become ready (this can take a while on a")
    print("cold model load)...")
    url = f"http://{BACKEND_HOST}:{BACKEND_PORT}/"
    if wait_for_backend():
        print("Backend is up - opening the browser...")
        webbrowser.open(url)
    else:
        print()
        print("The backend did not respond within 3 minutes. It may still be")
        print("loading a large model - check its console window, or open")
        print(f"{url} yourself once it settles.")

    print()
    print("The backend is running in its own minimized console window.")
    print("Closing THIS window is fine - use the Stop button in the browser")
    print("page when you're done, or run stop.bat.")
    print()
    try:
        input("Press Enter to close this window...")
    except EOFError:
        pass  # no interactive console attached — nothing to wait on


if __name__ == "__main__":
    main()
