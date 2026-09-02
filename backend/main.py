"""
Layer 3 backend — owns and manages the llama_cpp.server process directly.

Responsibilities:
- Launch/stop/restart llama_cpp.server as a subprocess this backend controls
  (this is what makes model-swapping possible: llama_cpp.server itself has
  no "change model" endpoint, so swapping means kill-and-relaunch)
- List available .gguf files from the models folder
- Manage projects: real folders on disk, each with its own conversations
  and files subfolders
- Proxy chat requests to the currently-running llama_cpp.server
- Save conversation turns to disk as JSON files, scoped to a project

Disk layout:
  LLMlifeline/
    projects/
      <project-id>/
        project.json          <- name, created_at
        conversations/
          <conversation-id>.json
        files/
          (whatever the assistant reads/writes for this project later)

IMPORTANT: This backend now starts llama_cpp.server itself on startup.
Do not launch `python -m llama_cpp.server` by hand anymore — this file
owns that process's whole lifecycle.
"""

import asyncio
import json
import os
import random
import re
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---- Configuration ----

# Everything below used to be hardcoded to this one machine's absolute path
# (C:/Users/Gonzalez/LLMlifeline/...). APP_ROOT makes every path relative
# to where this file actually lives instead — backend/main.py sits one
# level inside the project root, so .parent.parent is that root. This is
# what makes the project runnable from a fresh clone/copy on someone
# else's machine, not just this exact user account on this exact PC.
APP_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = APP_ROOT / "models"
COMFYUI_PATH = APP_ROOT / "ComfyUI"
COMFYUI_CHECKPOINTS_DIR = COMFYUI_PATH / "models" / "checkpoints"
COMFYUI_PORT = 8188
GENERATED_IMAGES_DIR = APP_ROOT / "generated_images"
IMAGE_UPLOAD_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif")
DOCUMENT_UPLOAD_EXTENSIONS = (".pdf", ".txt", ".md")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50MB — uploads are read fully into memory below
PROJECTS_DIR = APP_ROOT / "projects"
PERSONA_FILE = APP_ROOT / "persona.txt"
BELIEFS_DIR = APP_ROOT / "beliefs"
BELIEFS_CACHE_DIR = APP_ROOT / "beliefs" / "_cache"
LLAMA_SERVER_HOST = "127.0.0.1"
LLAMA_SERVER_PORT = 8000
LLAMA_SERVER_URL = f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}"
DEFAULT_MODEL = MODELS_DIR / "Qwen3-30B-A3B-Q4_K_M.gguf"
LLAMA_SERVER_STARTUP_TIMEOUT = 600  # seconds — cold loads on this hardware have taken up to ~10 min

PROJECTS_DIR.mkdir(parents=True, exist_ok=True)


# ---- Image captioning (local, no external API) ----
# None of this project's chat models are vision-capable (see MODELS_DIR —
# Qwen3, gemma-3-1b-it, oh-dcft are all text-only GGUFs), so a chat reply
# about a generated or uploaded image was previously working from nothing
# but a generic placeholder message — it had no real information about
# the image and would confabulate specifics when asked. Rather than
# requiring a new multi-GB vision-capable GGUF + llama.cpp mmproj wiring,
# a small local captioning model (BLIP, ~990MB one-time download) turns
# an image into a real text description at generation/upload time, which
# then flows into chat as ordinary text — grounded, and compatible with
# every chat model already in this project.
#
# Loaded lazily on first actual use, not at module import time — importing
# transformers and pulling the model would slow down every backend
# startup even for sessions that never touch an image.
_blip_processor = None
_blip_model = None
_blip_device = None


def _load_blip() -> None:
    global _blip_processor, _blip_model, _blip_device
    if _blip_model is not None:
        return
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor

    _blip_device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[backend] Loading image captioning model (BLIP) — first use only, "
          "downloads ~990MB the very first time this ever runs, then loads "
          "from the local Hugging Face cache...")
    _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    _blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(_blip_device)
    print(f"[backend] Image captioning model ready on {_blip_device}.")


def _caption_image(image_path: Path) -> str:
    """
    Produce a real text description of an image file. Callers should
    treat a failure here as non-fatal (catch and continue with no
    caption) — losing the caption means a chat reply about that specific
    image is less grounded, not that the image itself failed to
    generate/upload.
    """
    from PIL import Image

    _load_blip()
    raw_image = Image.open(image_path).convert("RGB")
    inputs = _blip_processor(raw_image, return_tensors="pt").to(_blip_device)
    output = _blip_model.generate(**inputs, max_new_tokens=50)
    return _blip_processor.decode(output[0], skip_special_tokens=True).strip()


# ---- llama_cpp.server process management ----

def _detect_free_vram_mb() -> int | None:
    """
    Query free VRAM via nvidia-smi — the same command confirmed working
    throughout this entire build's manual testing, now used
    programmatically at startup instead of by hand. Returns None if
    nvidia-smi isn't available (no NVIDIA GPU, drivers not installed,
    or running on a machine without one) rather than raising, since a
    missing GPU should fall back to a safe default, not crash startup.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip().splitlines()[0])
    except (subprocess.SubprocessError, FileNotFoundError, ValueError, OSError):
        pass
    return None


def _detect_total_vram_mb() -> int | None:
    """
    Same approach as _detect_free_vram_mb, but the card's total capacity
    rather than what's free right now — needed to recognize when a
    model's file size alone already exceeds the whole card, not just
    the current headroom on it. Returns None under the same conditions
    _detect_free_vram_mb does.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip().splitlines()[0])
    except (subprocess.SubprocessError, FileNotFoundError, ValueError, OSError):
        pass
    return None


def _choose_context_length(model_path: str, doubled: bool = False) -> int:
    """
    Pick a context length based on free VRAM relative to the model's own
    file size on disk. This is a CONSERVATIVE HEURISTIC, not a precise
    KV-cache calculation — a real calculation needs per-model
    architecture values (layer count, KV head count, head dimension)
    that live inside each GGUF's own metadata and aren't parsed here.
    llama.cpp's own load-time logs print those exact numbers (visible in
    every model-load log this build has produced), so a genuinely
    precise version is buildable later by reading that metadata before
    launch — this heuristic is the honest, buildable-tonight version:
    if the model's weights already consume most of detected free VRAM,
    drop to a small, safe context; if there's real headroom, use a more
    generous one. Falls back to a fixed safe default (2048, this
    project's original, previously-stable value) if VRAM can't be
    detected at all.

    `doubled` applies the user's extended-context toggle: the base
    heuristic result is doubled before being returned. This does NOT
    re-check whether the doubled value actually fits in detected VRAM —
    doubling context roughly doubles KV-cache VRAM cost (see the KV
    cache formula this build's own research turned up: cache scales
    linearly with context length), so a doubled value CAN exceed what's
    actually available, especially for a model whose base heuristic
    already picked a small number specifically because headroom was
    tight. That's a real, known risk of the toggle, not a bug — a CUDA
    OOM here fails safely: the llama_cpp.server subprocess fails to
    launch and _wait_for_server_ready reports it honestly, per the
    ordinary crash-and-report path already proven throughout this
    build. It does not crash the OS or the backend itself.
    """
    free_vram_mb = _detect_free_vram_mb()
    if free_vram_mb is None:
        base = 2048  # can't detect — fall back to the value proven stable all along
    else:
        try:
            model_size_mb = Path(model_path).stat().st_size / (1024 * 1024)
        except OSError:
            return 2048

        total_vram_mb = _detect_total_vram_mb()
        if total_vram_mb is not None and model_size_mb > total_vram_mb:
            # The model's file size alone is bigger than the whole card —
            # it cannot be primarily GPU-resident no matter how much is
            # free right now, so it's already running with heavy CPU
            # offload (system RAM, not VRAM, hosts most of the weights).
            # Confirmed hitting this in practice with a 26.6GB model on an
            # 11GB card: the free-vs-model headroom math below assumes a
            # model that basically fits on the card and gets conservative
            # as that fit gets tight — applied here, headroom is just
            # deeply negative and it always bottoms out at the smallest,
            # least usable tier (1024, or 2048 doubled), regardless of the
            # extended-context toggle. A model this size is already slow
            # from CPU offload; a bigger KV cache isn't what makes it
            # unusable, so use a genuinely useful context instead of the
            # smallest tier.
            base = 4096
        else:
            headroom_mb = free_vram_mb - model_size_mb

            # Thresholds are deliberately conservative — this build's own
            # history includes a real CUDA OOM risk being flagged repeatedly
            # for models sitting close to this card's VRAM ceiling. Erring
            # toward a smaller context that definitely loads is the right
            # default when the model genuinely might fit on the GPU.
            if headroom_mb < 500:
                base = 1024
            elif headroom_mb < 2000:
                base = 2048
            elif headroom_mb < 4000:
                base = 4096
            else:
                base = 8192

    return base * 2 if doubled else base


class LlamaServerManager:
    """
    Owns the llama_cpp.server subprocess. Only one instance of the server
    runs at a time — swapping models means terminating the current process
    and launching a new one pointed at a different .gguf file.
    """

    def __init__(self):
        self.process: subprocess.Popen | None = None
        self.current_model_path: str | None = None
        self.lock = asyncio.Lock()  # prevents overlapping swap attempts
        self.log_path = APP_ROOT / "llama_server.log"
        self.log_file = None
        self.extended_context = False  # toggle: doubles the auto-selected context length
        self.load_started_at: float | None = None  # time.time() when the current/last load began
        self.context_length: int | None = None  # the real n_ctx the running server was launched with — set in start(), used by /api/chat to budget history against instead of guessing

    def start(self, model_path: str):
        """Launch llama_cpp.server as a subprocess pointed at model_path.

        Writes output to a log file rather than a PIPE — piping subprocess
        stdout on Windows under uvicorn/multiprocessing has known handle-
        inheritance issues that can cause the child process to fail to
        launch silently. A log file sidesteps that entirely and gives us
        a real record to inspect when something goes wrong.
        """
        if self.process is not None:
            self.stop()

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_path, "w", encoding="utf-8")

        chosen_context = _choose_context_length(model_path, doubled=self.extended_context)
        self.context_length = chosen_context
        self.load_started_at = time.time()
        self.process = subprocess.Popen(
            [
                sys.executable, "-m", "llama_cpp.server",
                "--model", model_path,
                "--n_gpu_layers", "-1",
                "--n_ctx", str(chosen_context),
                "--host", LLAMA_SERVER_HOST,
                "--port", str(LLAMA_SERVER_PORT),
            ],
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )
        self.current_model_path = model_path
        extended_note = " (extended context ON)" if self.extended_context else ""
        print(f"[backend] Launched llama_cpp.server (PID {self.process.pid}), "
              f"context={chosen_context}{extended_note} (auto-selected based on detected VRAM), "
              f"output -> {self.log_path}")

    def stop(self):
        """Terminate the current llama_cpp.server process, if any."""
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process = None
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None


llama_manager = LlamaServerManager()


class ImageServerManager:
    """
    Owns the ComfyUI subprocess. Structurally mirrors LlamaServerManager
    (lock, log file, start/stop/is_alive) since that pattern is proven
    correct in this build — but the actual launch mechanics are honestly
    different, not force-fit to match: ComfyUI is a standalone web
    server started from its own install directory via its own main.py,
    not a process that takes a checkpoint path on the command line.
    Checkpoint selection happens through ComfyUI's own API/workflow
    system after the server is already running, which this manager does
    NOT yet handle — this class only proves the server itself can be
    launched and reached. Model selection is a separate, later step.
    """

    def __init__(self):
        self.process: subprocess.Popen | None = None
        self.current_checkpoint: str | None = None
        self.lock = asyncio.Lock()
        self.log_path = APP_ROOT / "comfyui_server.log"
        self.log_file = None
        self.load_started_at: float | None = None

    def start(self):
        """
        Launch ComfyUI as a subprocess. Unlike llama_cpp.server, this
        takes no model argument — ComfyUI loads whichever checkpoint a
        workflow/API request specifies, after the server is already up.
        """
        if self.process is not None:
            self.stop()

        if not COMFYUI_PATH.exists():
            raise FileNotFoundError(
                f"ComfyUI not found at {COMFYUI_PATH}. Expected a ComfyUI checkout "
                f"at ./ComfyUI relative to the project root — see setup-guide.md."
            )

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_path, "w", encoding="utf-8")

        self.load_started_at = time.time()
        self.process = subprocess.Popen(
            [
                sys.executable, "main.py",
                "--port", str(COMFYUI_PORT),
            ],
            cwd=str(COMFYUI_PATH),  # ComfyUI expects to run from its own directory
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )
        print(f"[backend] Launched ComfyUI (PID {self.process.pid}), "
              f"output -> {self.log_path}")

    def stop(self):
        """
        Terminate the current ComfyUI process, if any. Deliberately does
        NOT clear current_checkpoint — checkpoint selection is a
        standing preference (like llama_manager.current_model_path),
        not process state. It should survive a stop/restart of ComfyUI
        itself; clearing it here silently un-set the active checkpoint
        on every Chat Model <-> Image Model switch, which is exactly the
        "selected but not shown as active" bug reported against this.
        """
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process = None
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None


image_manager = ImageServerManager()


async def _wait_for_server_ready(timeout: int = LLAMA_SERVER_STARTUP_TIMEOUT) -> bool:
    """Poll llama_cpp.server's /v1/models until it responds or we time out.
    Loading a multi-gigabyte model takes real time — this replaces a fixed
    sleep with an actual readiness check."""
    await asyncio.sleep(2)  # give the OS a moment to actually spawn the process
    if not llama_manager.is_alive():
        print(f"[backend] llama_cpp.server exited immediately after launch. "
              f"Check {llama_manager.log_path} for details.")
        return False

    deadline = asyncio.get_event_loop().time() + timeout
    async with httpx.AsyncClient(timeout=2.0) as client:
        while asyncio.get_event_loop().time() < deadline:
            if not llama_manager.is_alive():
                print(f"[backend] llama_cpp.server process died during startup "
                      f"(exit code {llama_manager.process.returncode if llama_manager.process else '?'}). "
                      f"Check {llama_manager.log_path} for details.")
                return False
            try:
                r = await client.get(f"{LLAMA_SERVER_URL}/v1/models")
                if r.status_code == 200:
                    return True
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout):
                pass  # not ready yet — keep polling until the outer deadline
            await asyncio.sleep(1)
    print(f"[backend] Timed out after {timeout}s waiting for llama_cpp.server to become ready.")
    return False


async def _wait_for_comfyui_ready(timeout: int = 300) -> bool:
    """
    Poll ComfyUI's /system_stats endpoint until it responds or we time
    out. Mirrors _wait_for_server_ready's structure. /system_stats is a
    real, documented ComfyUI API route used for exactly this kind of
    health check — not a guessed endpoint. Timeout default (300s) is
    shorter than the chat-side one (600s) since ComfyUI itself starts
    without loading a checkpoint — that happens later, per-request —
    so startup here should be lighter than a multi-gigabyte GGUF load.
    """
    await asyncio.sleep(2)
    if not image_manager.is_alive():
        print(f"[backend] ComfyUI exited immediately after launch. "
              f"Check {image_manager.log_path} for details.")
        return False

    deadline = asyncio.get_event_loop().time() + timeout
    async with httpx.AsyncClient(timeout=2.0) as client:
        while asyncio.get_event_loop().time() < deadline:
            if not image_manager.is_alive():
                print(f"[backend] ComfyUI process died during startup "
                      f"(exit code {image_manager.process.returncode if image_manager.process else '?'}). "
                      f"Check {image_manager.log_path} for details.")
                return False
            try:
                r = await client.get(f"http://127.0.0.1:{COMFYUI_PORT}/system_stats")
                if r.status_code == 200:
                    return True
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout):
                pass
            await asyncio.sleep(1)
    print(f"[backend] Timed out after {timeout}s waiting for ComfyUI to become ready.")
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: launch llama_cpp.server with a default model, so Chat
    # Model (the default-checked mode in the UI) is actually usable the
    # moment the backend comes up, not just after a manual dropdown pick.
    default_model = _default_gguf_path()
    if default_model:
        llama_manager.start(str(default_model))
        await _wait_for_server_ready()

    # Also pre-select a default image checkpoint — WITHOUT starting
    # ComfyUI itself, since Chat Model is the default active mode and
    # only one of the two ever runs at a time. This just means the
    # checkpoint dropdown already shows something "active" the instant
    # Image Model is switched on, instead of needing a manual selection
    # first (and, before this, a selection that a stop() call would
    # then silently wipe right back out — see ImageServerManager.stop).
    default_checkpoints = _list_checkpoint_files()
    if default_checkpoints:
        image_manager.current_checkpoint = default_checkpoints[0].name

    yield
    # Shutdown: make sure we don't leave orphaned subprocesses running.
    llama_manager.stop()
    image_manager.stop()


app = FastAPI(title="LLMlifeline Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Data models ----

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    project_id: str
    conversation_id: str | None = None
    messages: list[ChatMessage]


class CreateProjectRequest(BaseModel):
    name: str


class SwapModelRequest(BaseModel):
    model_path: str


class SetCheckpointRequest(BaseModel):
    checkpoint: str


class GenerateImageRequest(BaseModel):
    project_id: str
    conversation_id: str | None = None
    prompt: str


COMFYUI_CHECKPOINT_EXTENSIONS = (".safetensors", ".ckpt", ".pt", ".bin")


def _default_gguf_path() -> Path | None:
    """
    Pick the chat model to auto-launch at startup: DEFAULT_MODEL if that
    exact file actually exists, else the first .gguf found in
    MODELS_DIR (sorted, so it's deterministic). Falling back matters —
    DEFAULT_MODEL is a fixed filename that won't exist on every setup,
    and silently launching nothing left Chat Model unusable until a
    person manually picked something from the dropdown.
    """
    if DEFAULT_MODEL.exists():
        return DEFAULT_MODEL
    if not MODELS_DIR.exists():
        return None
    gguf_files = sorted(MODELS_DIR.glob("*.gguf"))
    return gguf_files[0] if gguf_files else None


def _list_checkpoint_files() -> list[Path]:
    if not COMFYUI_CHECKPOINTS_DIR.exists():
        return []
    return sorted(
        f for f in COMFYUI_CHECKPOINTS_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in COMFYUI_CHECKPOINT_EXTENSIONS
    )


# ---- Models folder scanning ----

@app.get("/api/models")
def list_models():
    """Return every .gguf file in the models folder, flagging which one is currently loaded."""
    if not MODELS_DIR.exists():
        raise HTTPException(status_code=500, detail=f"Models directory not found: {MODELS_DIR}")

    gguf_files = sorted(MODELS_DIR.glob("*.gguf"))
    return {
        "models": [
            {
                "filename": f.name,
                "path": str(f),
                "size_gb": round(f.stat().st_size / (1024 ** 3), 2),
                "active": str(f) == llama_manager.current_model_path,
            }
            for f in gguf_files
        ]
    }


@app.post("/api/image/start")
async def start_image_server():
    """
    Launch ComfyUI and confirm it's reachable. This is a first-pass proof
    endpoint — it does NOT yet handle checkpoint selection, the chat/
    image mutual-exclusivity from the checkbox design, or VRAM
    accounting alongside a chat model. Those are real, separate next
    steps once this proves ComfyUI itself launches correctly from this
    backend's subprocess-management pattern.
    """
    async with image_manager.lock:
        image_manager.start()
        ready = await _wait_for_comfyui_ready()

    if not ready:
        raise HTTPException(
            status_code=500,
            detail=f"ComfyUI did not become ready. Check {image_manager.log_path} for details.",
        )

    return {"status": "ok", "comfyui_port": COMFYUI_PORT}


@app.post("/api/image/stop")
async def stop_image_server():
    """Stop the ComfyUI process, if running."""
    async with image_manager.lock:
        was_running = image_manager.is_alive()
        image_manager.stop()
    return {"status": "ok", "was_running": was_running}


@app.get("/api/image/health")
async def image_health():
    """Confirm whether ComfyUI is currently running and reachable."""
    if not image_manager.is_alive():
        return {"comfyui": "not running"}

    async with httpx.AsyncClient(timeout=2.0) as client:
        try:
            r = await client.get(f"http://127.0.0.1:{COMFYUI_PORT}/system_stats")
            return {"comfyui": "ok" if r.status_code == 200 else "unreachable"}
        except (httpx.ConnectError, httpx.ReadTimeout):
            return {"comfyui": "unreachable"}


@app.get("/api/image/checkpoints")
def list_checkpoints():
    """
    Return every checkpoint file in ComfyUI's checkpoints folder, flagging
    which one is currently selected. Mirrors /api/models's shape/pattern
    exactly (filename, size, active flag) so the frontend dropdown can
    reuse the same rendering logic as the chat-model dropdown.
    """
    if not COMFYUI_CHECKPOINTS_DIR.exists():
        raise HTTPException(status_code=500, detail=f"Checkpoints directory not found: {COMFYUI_CHECKPOINTS_DIR}")

    checkpoint_files = _list_checkpoint_files()
    return {
        "checkpoints": [
            {
                "filename": f.name,
                "size_gb": round(f.stat().st_size / (1024 ** 3), 2),
                "active": f.name == image_manager.current_checkpoint,
            }
            for f in checkpoint_files
        ]
    }


@app.post("/api/image/checkpoint")
def set_checkpoint(request: SetCheckpointRequest):
    """
    Record which checkpoint filename image-generation requests should use.
    Unlike swapping a chat model, this does NOT restart ComfyUI or touch
    its process at all — ComfyUI loads whichever checkpoint a workflow
    specifies per-request, after the server is already running (see
    ImageServerManager's own docstring). generate_image (below) is what
    actually reads this selection when building a workflow.
    """
    # Resolve and confirm containment before trusting the name — same
    # boundary as _resolve_project_file, so a value like
    # "../../../whatever.safetensors" can't select a file outside this
    # folder even though it'd pass a naive exists()+suffix check.
    checkpoints_dir_resolved = COMFYUI_CHECKPOINTS_DIR.resolve()
    checkpoint_path = (COMFYUI_CHECKPOINTS_DIR / request.checkpoint).resolve()
    try:
        checkpoint_path.relative_to(checkpoints_dir_resolved)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Not a valid checkpoint file: {request.checkpoint}")

    if not checkpoint_path.is_file() or checkpoint_path.suffix.lower() not in COMFYUI_CHECKPOINT_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Not a valid checkpoint file: {request.checkpoint}")

    image_manager.current_checkpoint = checkpoint_path.name
    return {"status": "ok", "active_checkpoint": checkpoint_path.name}


def _build_txt2img_workflow(checkpoint: str, prompt: str, seed: int) -> dict:
    """
    A minimal, standard ComfyUI API-format txt2img graph: load checkpoint
    -> encode positive/negative prompt -> empty latent -> KSampler ->
    VAE decode -> save. This is the same node graph ComfyUI's own default
    workflow uses, built directly as the API's node-graph JSON (not the
    separate, richer "workflow" UI format) since that's what /prompt
    actually accepts. 1024x1024 is sized for this project's one
    checkpoint (an SDXL model, per its "XL" filename) — not adjustable
    per-request yet, since there's no UI for it.
    """
    return {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["4", 1]}},
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "llmlifeline", "images": ["8", 0]}},
    }


@app.post("/api/image/generate")
async def generate_image(request: GenerateImageRequest, background_tasks: BackgroundTasks):
    """
    Generate an image from a text prompt via ComfyUI's own /prompt API,
    using whichever checkpoint /api/image/checkpoint last selected, then
    save the turn into the same conversation file chat uses.

    Conversation persistence itself is untouched — this goes through the
    same _load_conversation/_save_conversation path as /api/chat, with
    the same message shape (role/content/timestamp) plus one new
    optional 'image_url' field on the assistant message. Old messages
    without that field are unaffected; nothing here changes how history
    is read, indexed, or retrieved.
    """
    _require_project(request.project_id)

    if not image_manager.is_alive():
        raise HTTPException(status_code=503, detail="ComfyUI is not running. Switch to Image Model first.")
    if not image_manager.current_checkpoint:
        raise HTTPException(status_code=400, detail="No checkpoint selected. Pick one from the checkpoint dropdown first.")

    conversation_id = request.conversation_id or str(uuid.uuid4())
    conversation = _load_conversation(request.project_id, conversation_id)
    conversation["project_id"] = request.project_id
    is_new_conversation = not conversation["messages"]
    conversation["messages"].append({
        "role": "user",
        "content": request.prompt,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    workflow = _build_txt2img_workflow(
        image_manager.current_checkpoint, request.prompt, random.randint(0, 2**32 - 1)
    )
    client_id = uuid.uuid4().hex

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            submit = await client.post(
                f"http://127.0.0.1:{COMFYUI_PORT}/prompt",
                json={"prompt": workflow, "client_id": client_id},
            )
            submit.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"ComfyUI rejected the generation request: {e.response.text}")
        except (httpx.ConnectError, httpx.ReadTimeout):
            raise HTTPException(status_code=503, detail="Could not reach ComfyUI to submit the generation request.")

    submit_data = submit.json()
    prompt_id = submit_data.get("prompt_id")
    if not prompt_id:
        raise HTTPException(status_code=502, detail=f"ComfyUI did not return a prompt_id: {submit_data}")

    # Poll /history for this prompt_id rather than guessing a sleep —
    # generation time varies hugely with steps/resolution/checkpoint,
    # same reasoning as _wait_for_comfyui_ready's own polling loop.
    deadline = asyncio.get_event_loop().time() + 300
    history_entry = None
    async with httpx.AsyncClient(timeout=10.0) as client:
        while asyncio.get_event_loop().time() < deadline:
            if not image_manager.is_alive():
                raise HTTPException(
                    status_code=503,
                    detail=f"ComfyUI process died while generating. Check {image_manager.log_path} for details.",
                )
            try:
                hist_resp = await client.get(f"http://127.0.0.1:{COMFYUI_PORT}/history/{prompt_id}")
                hist_resp.raise_for_status()
                hist_data = hist_resp.json()
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError):
                hist_data = {}
            if prompt_id in hist_data:
                history_entry = hist_data[prompt_id]
                break
            await asyncio.sleep(2)

    if history_entry is None:
        raise HTTPException(
            status_code=504,
            detail=f"Image generation timed out after 300s. Check {image_manager.log_path} for details.",
        )

    if history_entry.get("status", {}).get("status_str") == "error":
        raise HTTPException(
            status_code=502,
            detail=f"ComfyUI reported an error generating the image: {history_entry['status']}",
        )

    image_info = None
    for node_output in history_entry.get("outputs", {}).values():
        images = node_output.get("images")
        if images:
            image_info = images[0]
            break

    if image_info is None:
        raise HTTPException(status_code=502, detail="ComfyUI finished but produced no image output.")

    async with httpx.AsyncClient(timeout=30.0) as client:
        view_resp = await client.get(
            f"http://127.0.0.1:{COMFYUI_PORT}/view",
            params={
                "filename": image_info["filename"],
                "subfolder": image_info.get("subfolder", ""),
                "type": image_info.get("type", "output"),
            },
        )
        view_resp.raise_for_status()

    # Saved under our own control, keyed by project — separate from
    # projects/*/files/ since that folder's read endpoint only handles
    # UTF-8 text, not binary image bytes.
    project_images_dir = GENERATED_IMAGES_DIR / request.project_id
    project_images_dir.mkdir(parents=True, exist_ok=True)
    saved_filename = f"{uuid.uuid4().hex}.png"
    saved_path = project_images_dir / saved_filename
    saved_path.write_bytes(view_resp.content)

    image_url = f"/api/image/generated/{request.project_id}/{saved_filename}"

    # Caption the actual pixels, not just the prompt — the prompt is what
    # was ASKED for, not necessarily what the model actually drew. This is
    # what lets a later "describe that image" question in this same
    # conversation get a real answer instead of the model just echoing
    # the prompt back or guessing. A caption failure is non-fatal — the
    # image itself is already saved and returned either way.
    try:
        caption = _caption_image(saved_path)
    except Exception as e:
        print(f"[backend] Image captioning failed for {saved_path}: {e}")
        caption = None

    content = f'Generated an image for: "{request.prompt}"'
    if caption:
        content += f"\n\nDescription: {caption}"

    assistant_timestamp = datetime.now(timezone.utc).isoformat()
    conversation["messages"].append({
        "role": "assistant",
        "content": content,
        "image_url": image_url,
        "image_caption": caption,
        "timestamp": assistant_timestamp,
    })
    _save_conversation(conversation)

    if is_new_conversation:
        # Usually a no-op in practice: Chat and Image Model are mutually
        # exclusive, so llama_cpp.server is normally stopped whenever
        # this endpoint runs, and _generate_conversation_title bails out
        # immediately when it is. Still fired for the rare case it isn't
        # (a mode switch mid-flight) — harmless either way, and the
        # prompt text itself is already a reasonable fallback title via
        # list_conversations' plain preview.
        background_tasks.add_task(
            _title_background_task, request.project_id, conversation_id, request.prompt
        )

    return {
        "conversation_id": conversation_id,
        "project_id": request.project_id,
        "image_url": image_url,
        "caption": caption,
        "timestamp": assistant_timestamp,
        "checkpoint": image_manager.current_checkpoint,
    }


@app.get("/api/image/generated/{project_id}/{filename}")
def get_generated_image(project_id: str, filename: str):
    """
    Serve a generated image's raw bytes. project_id and filename both
    arrive as untrusted path segments, so this gets the same
    resolve-and-check-containment treatment as project files, even
    though filenames here are always backend-generated UUIDs.
    """
    project_dir_resolved = (GENERATED_IMAGES_DIR / project_id).resolve()
    candidate = (GENERATED_IMAGES_DIR / project_id / filename).resolve()
    try:
        candidate.relative_to(project_dir_resolved)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid image path.")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(candidate, media_type="image/png")


@app.post("/api/models/swap")
async def swap_model(request: SwapModelRequest):
    """
    Stop the currently-running llama_cpp.server and start a new one pointed
    at a different .gguf file. This is a real process restart — expect this
    to take anywhere from a few seconds to over a minute depending on model
    size, matching the load times you've already benchmarked directly.
    """
    model_path = Path(request.model_path)
    if not model_path.exists() or model_path.suffix != ".gguf":
        raise HTTPException(status_code=400, detail=f"Not a valid .gguf file: {request.model_path}")

    async with llama_manager.lock:
        llama_manager.start(str(model_path))
        ready = await _wait_for_server_ready()

    if not ready:
        raise HTTPException(
            status_code=500,
            detail=f"Model server did not become ready within {LLAMA_SERVER_STARTUP_TIMEOUT}s "
                    f"after swapping to {model_path.name}. Check backend console output for errors.",
        )

    return {"status": "ok", "active_model": str(model_path)}


@app.post("/api/models/stop")
async def stop_model():
    """
    Stop the currently-running llama_cpp.server without launching a
    replacement. Chat requests will fail until a model is started again
    via /api/models/swap or /api/models/restart.
    """
    async with llama_manager.lock:
        was_running = llama_manager.is_alive()
        stopped_model = llama_manager.current_model_path
        llama_manager.stop()
        llama_manager.current_model_path = None  # no model is loaded anymore

    return {
        "status": "ok",
        "was_running": was_running,
        "stopped_model": stopped_model,
    }


class ExtendedContextRequest(BaseModel):
    enabled: bool


@app.post("/api/models/extended-context")
async def set_extended_context(request: ExtendedContextRequest):
    """
    Toggle whether context length gets doubled on future model
    launches. Does NOT apply to the currently-running model — the
    active process was already launched with the previous setting.
    Takes effect on the next swap or restart, which the response makes
    explicit so the frontend can prompt for one rather than implying
    the change already happened.
    """
    llama_manager.extended_context = request.enabled
    return {
        "extended_context": llama_manager.extended_context,
        "note": "Takes effect on the next model swap or restart, not the currently-running model.",
    }


@app.post("/api/models/restart")
async def restart_model():
    """
    Restart the currently-loaded model: stop it and relaunch the same
    .gguf file. Useful for recovering from a crash or hung state without
    picking a different model — same mechanism as swap, just reusing
    whatever was already active instead of a new path.
    """
    if llama_manager.current_model_path is None:
        raise HTTPException(
            status_code=400,
            detail="No model is currently set. Use /api/models/swap to pick one first.",
        )

    model_path = llama_manager.current_model_path
    async with llama_manager.lock:
        llama_manager.start(model_path)
        ready = await _wait_for_server_ready()

    if not ready:
        raise HTTPException(
            status_code=500,
            detail=f"Restart of {Path(model_path).name} did not become ready within "
                    f"{LLAMA_SERVER_STARTUP_TIMEOUT}s. Check {llama_manager.log_path} for details.",
        )

    return {"status": "ok", "active_model": model_path}


# ---- Project management ----

def _slugify(name: str) -> str:
    """Turn a project name into a filesystem-safe folder suffix."""
    slug = re.sub(r"[^a-zA-Z0-9-]+", "-", name.strip().lower()).strip("-")
    return slug or "project"


def _project_dir(project_id: str) -> Path:
    return PROJECTS_DIR / project_id


def _project_meta_path(project_id: str) -> Path:
    return _project_dir(project_id) / "project.json"


def _require_project(project_id: str) -> dict:
    meta_path = _project_meta_path(project_id)
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
    return json.loads(meta_path.read_text(encoding="utf-8"))


@app.get("/api/projects")
def list_projects():
    """Return every project: id, name, created_at, conversation count."""
    results = []
    if not PROJECTS_DIR.exists():
        return {"projects": results}

    for project_path in sorted(PROJECTS_DIR.iterdir()):
        meta_path = project_path / "project.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        conv_dir = project_path / "conversations"
        conv_count = len(list(conv_dir.glob("*.json"))) if conv_dir.exists() else 0
        results.append({
            "id": meta["id"],
            "name": meta["name"],
            "created_at": meta.get("created_at"),
            "conversation_count": conv_count,
        })
    results.sort(key=lambda p: p.get("created_at") or "", reverse=True)
    return {"projects": results}


@app.post("/api/projects")
def create_project(request: CreateProjectRequest):
    """Create a new project folder on disk: project.json, conversations/, files/."""
    project_id = f"{_slugify(request.name)}-{uuid.uuid4().hex[:8]}"
    project_path = _project_dir(project_id)
    project_path.mkdir(parents=True, exist_ok=False)
    (project_path / "conversations").mkdir()
    (project_path / "files").mkdir()

    meta = {
        "id": project_id,
        "name": request.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _project_meta_path(project_id).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


@app.get("/api/projects/{project_id}")
def get_project(project_id: str):
    return _require_project(project_id)


@app.get("/api/projects/{project_id}/images")
def list_project_images(project_id: str):
    """
    List every generated/uploaded image saved for a project, most recent
    first, with whatever caption was stored for it. Captions live on
    conversation messages, not on the image files themselves —
    conversations stay the single source of truth (per the existing
    model-agnostic design), so this does a fresh filename -> caption
    lookup across the project's conversations on each call. Cheap enough
    for a handful of small JSON files, matching how the rest of this
    file already reads conversation data on demand rather than caching it.
    """
    _require_project(project_id)
    project_images_dir = GENERATED_IMAGES_DIR / project_id
    if not project_images_dir.exists():
        return {"images": []}

    captions_by_filename: dict[str, str | None] = {}
    conv_dir = _project_dir(project_id) / "conversations"
    if conv_dir.exists():
        for conv_file in conv_dir.glob("*.json"):
            if conv_file.name == "_keyword_index.json":
                continue
            try:
                conversation = json.loads(conv_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue  # a corrupted conversation file shouldn't break the whole gallery
            for msg in conversation.get("messages", []):
                image_url = msg.get("image_url")
                if image_url:
                    captions_by_filename[Path(image_url).name] = msg.get("image_caption")

    images = []
    for f in project_images_dir.iterdir():
        if not f.is_file():
            continue
        images.append({
            "filename": f.name,
            "url": f"/api/image/generated/{project_id}/{f.name}",
            "caption": captions_by_filename.get(f.name),
            "modified_at": datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).isoformat(),
        })

    images.sort(key=lambda i: i["modified_at"], reverse=True)
    return {"images": images}


# ---- Project file operations ----

def _project_files_dir(project_id: str) -> Path:
    return _project_dir(project_id) / "files"


def _resolve_project_file(project_id: str, relative_path: str) -> Path:
    """
    Resolve a relative path against a project's files/ folder, and refuse
    to return anything that escapes it. This is the one real security
    boundary in this whole file-tools system: without it, a path like
    '../../../Windows/System32/whatever' or an absolute path would let
    a file operation write anywhere on disk. Path.resolve() collapses
    '..' segments and symlinks before the containment check runs, so the
    check can't be fooled by a cleverly-constructed relative path.
    """
    files_dir = _project_files_dir(project_id)
    candidate = (files_dir / relative_path).resolve()
    files_dir_resolved = files_dir.resolve()

    try:
        candidate.relative_to(files_dir_resolved)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Path '{relative_path}' resolves outside this project's files folder. "
                    f"File operations are confined to the project's own files/ directory.",
        )
    return candidate


class WriteFileRequest(BaseModel):
    path: str
    content: str
    overwrite: bool = False


@app.get("/api/projects/{project_id}/files")
def list_project_files(project_id: str):
    """List every file in this project's files/ folder, recursively."""
    _require_project(project_id)
    files_dir = _project_files_dir(project_id)
    files_dir.mkdir(parents=True, exist_ok=True)  # tolerate pre-existing projects created before this feature

    results = []
    for path in sorted(files_dir.rglob("*")):
        if path.is_file():
            results.append({
                "path": str(path.relative_to(files_dir)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "modified_at": datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            })
    return {"files": results}


@app.get("/api/projects/{project_id}/files/{file_path:path}")
def read_project_file(project_id: str, file_path: str):
    """Read a specific file's content from this project's files/ folder."""
    _require_project(project_id)
    resolved = _resolve_project_file(project_id, file_path)

    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    if not resolved.is_file():
        raise HTTPException(status_code=400, detail=f"'{file_path}' is a directory, not a file")

    try:
        content = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=415,
            detail=f"'{file_path}' isn't valid UTF-8 text and can't be read as text content.",
        )

    return {
        "path": file_path,
        "content": content,
        "size_bytes": resolved.stat().st_size,
    }


@app.put("/api/projects/{project_id}/files/{file_path:path}")
def write_project_file(project_id: str, file_path: str, request: WriteFileRequest):
    """
    Create or overwrite a file in this project's files/ folder.
    Overwriting an existing file requires overwrite=true, so a create
    call can't silently clobber something that's already there unless
    that's explicitly intended.
    """
    _require_project(project_id)
    resolved = _resolve_project_file(project_id, file_path)

    if resolved.exists() and not request.overwrite:
        raise HTTPException(
            status_code=409,
            detail=f"'{file_path}' already exists. Pass overwrite=true to replace it.",
        )

    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(request.content, encoding="utf-8")

    return {
        "path": file_path,
        "size_bytes": resolved.stat().st_size,
        "created": not request.overwrite,
    }


@app.delete("/api/projects/{project_id}/files/{file_path:path}")
def delete_project_file(project_id: str, file_path: str):
    """Delete a file from this project's files/ folder."""
    _require_project(project_id)
    resolved = _resolve_project_file(project_id, file_path)

    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    if not resolved.is_file():
        raise HTTPException(status_code=400, detail=f"'{file_path}' is a directory, not a file")

    resolved.unlink()
    return {"path": file_path, "deleted": True}


# ---- Conversation persistence (scoped to a project) ----

# Bumped whenever _chunk_text's algorithm changes, so a chunk cache
# written by an older version gets treated as stale and re-chunked —
# without this, fixing the chunker wouldn't actually change anything
# for an already-cached file, since the cache is otherwise keyed only
# on the source file's mtime (see _get_belief_chunks /
# _get_project_document_chunks).
_CHUNKER_VERSION = 2

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """
    Split extracted text into passage-sized chunks for citation, cutting
    only at sentence boundaries — never mid-word or mid-sentence.

    The original version of this did a plain text[start:end] character
    slice, which regularly cut chunks off mid-word at both ends —
    confirmed in real use feeding the model passages like "ast day,
    that great day..." (a cut "last day") and "...they believed the
    scripture... But Jesus did not commit him" (cut mid-sentence, the
    rest silently missing) as if they were genuine, complete quotable
    material. The model then dutifully tried to work these broken
    fragments into its own replies, producing exactly the kind of
    garbled, sentence-fragment output reported live ("...respect for
    their beliefs …n him by force to make him king."). Whole documents
    are still useless for retrieval — you need to point at a specific
    passage, not hand the model an entire book — but every chunk now
    starts and ends on a real sentence, so whatever gets quoted is at
    least a genuine, complete thought.

    Sentences are packed greedily up to chunk_size rather than split at
    a fixed length; a single sentence longer than chunk_size on its own
    (real for dense legal/scripture text) is kept whole rather than cut
    — a passage a bit over the target length beats one broken mid-word.
    Chunks still overlap by carrying the trailing sentences of one chunk
    into the start of the next, so a passage near a boundary still
    appears intact in at least one chunk.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    sentences = [s.strip() for s in _SENTENCE_BOUNDARY_RE.split(text) if s.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        if current and current_len + len(sentence) + 1 > chunk_size:
            chunks.append(" ".join(current))
            # Carry enough trailing sentences into the next chunk to
            # cover the requested overlap.
            carried: list[str] = []
            carried_len = 0
            for s in reversed(current):
                if carried_len >= overlap:
                    break
                carried.insert(0, s)
                carried_len += len(s) + 1
            current = carried
            current_len = sum(len(s) + 1 for s in current)
        current.append(sentence)
        current_len += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


def _extract_pdf_text(path: Path) -> str:
    """Extract raw text from a PDF, page by page, concatenated."""
    import pypdf
    reader = pypdf.PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue  # a single unparseable page shouldn't kill the whole document
    return "\n".join(pages)


def _beliefs_cache_path(source_path: Path) -> Path:
    return BELIEFS_CACHE_DIR / f"{source_path.stem}.json"


def _get_belief_chunks(source_path: Path) -> list[str]:
    """
    Return cached chunks for one belief-folder file, extracting and
    caching fresh only if the source file's modification time is newer
    than the cache (or no cache exists yet). This is the actual 'don't
    recompute from raw content on every query' principle applied here —
    a multi-hundred-page Bible PDF should be parsed exactly once per
    edit, not once per chat request.
    """
    BELIEFS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _beliefs_cache_path(source_path)
    source_mtime = source_path.stat().st_mtime

    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("source_mtime") == source_mtime and cached.get("chunker_version") == _CHUNKER_VERSION:
            return cached["chunks"]

    if source_path.suffix.lower() == ".pdf":
        raw_text = _extract_pdf_text(source_path)
    elif source_path.suffix.lower() in (".txt", ".md"):
        raw_text = source_path.read_text(encoding="utf-8")
    else:
        return []  # unsupported file type in the beliefs folder — silently skipped

    chunks = _chunk_text(raw_text)
    cache_path.write_text(
        json.dumps(
            {"source_mtime": source_mtime, "chunker_version": _CHUNKER_VERSION, "chunks": chunks},
            indent=2,
        ),
        encoding="utf-8",
    )
    return chunks


def _project_document_cache_dir(project_id: str) -> Path:
    # Sibling to conversations/ and files/, not nested inside files/ —
    # files/ is listed and offered to the model's own file-action tool
    # (list_project_files, read/write/delete), and a cache dir full of
    # JSON chunk dumps has no business showing up in that listing.
    return _project_dir(project_id) / "_document_cache"


def _get_project_document_chunks(project_id: str, source_path: Path) -> list[str]:
    """
    Same cache-once-reuse-many pattern as _get_belief_chunks, scoped to
    one project's uploaded documents instead of the global beliefs
    folder — an uploaded PDF should be parsed exactly once, not on
    every chat request that happens to overlap with it.
    """
    cache_dir = _project_document_cache_dir(project_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{source_path.stem}.json"
    source_mtime = source_path.stat().st_mtime

    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("source_mtime") == source_mtime and cached.get("chunker_version") == _CHUNKER_VERSION:
            return cached["chunks"]

    if source_path.suffix.lower() == ".pdf":
        raw_text = _extract_pdf_text(source_path)
    elif source_path.suffix.lower() in (".txt", ".md"):
        raw_text = source_path.read_text(encoding="utf-8")
    else:
        return []

    chunks = _chunk_text(raw_text)
    cache_path.write_text(
        json.dumps(
            {"source_mtime": source_mtime, "chunker_version": _CHUNKER_VERSION, "chunks": chunks},
            indent=2,
        ),
        encoding="utf-8",
    )
    return chunks


def _find_project_document_passages(project_id: str, query_text: str, max_passages: int = 3) -> list[dict]:
    """
    Keyword-overlap search over this project's uploaded documents — same
    mechanism _find_relevant_context already uses for past messages,
    applied to document chunks instead. Unlike beliefs, there's no
    topic-gate here (no _is_faith_related equivalent): a document was
    deliberately uploaded to this specific project, so it's always in
    scope for retrieval here, not just when a trigger word fires.
    """
    query_keywords = set(_extract_keywords(query_text))
    if not query_keywords:
        return []

    files_dir = _project_files_dir(project_id)
    if not files_dir.exists():
        return []

    candidates = []
    for source_path in files_dir.iterdir():
        if not source_path.is_file() or source_path.suffix.lower() not in DOCUMENT_UPLOAD_EXTENSIONS:
            continue
        try:
            chunks = _get_project_document_chunks(project_id, source_path)
        except Exception:
            continue  # a malformed/unreadable document shouldn't break retrieval for everything else

        for chunk in chunks:
            overlap = query_keywords & set(_extract_keywords(chunk))
            if overlap:
                candidates.append({
                    "source": source_path.name,
                    "content": chunk,
                    "overlap_count": len(overlap),
                })

    candidates.sort(key=lambda c: c["overlap_count"], reverse=True)
    return candidates[:max_passages]


FAITH_TRIGGER_TERMS = {
    "god", "jesus", "christ", "bible", "scripture", "faith", "pray", "prayer",
    "heaven", "hell", "sin", "soul", "spirit", "gospel", "salvation", "church",
    "worship", "holy", "sacred", "believe", "belief", "religion", "religious",
    "lord", "father", "son", "trinity", "resurrection", "cross", "amen",
}


def _is_faith_related(text: str) -> bool:
    """
    Cheap trigger check — does this message contain an explicit
    faith-adjacent word at all. This exists ALONGSIDE keyword-overlap
    matching (in _find_belief_passages below), not instead of it: overlap
    alone would miss genuinely faith-relevant questions with no literal
    shared vocabulary ("why do bad things happen to good people" shares
    zero words with most scripture passages despite being exactly the
    kind of question this system exists to help with). This trigger
    check is a coarse, honest supplement — not a claim of true semantic
    understanding.
    """
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    return bool(words & FAITH_TRIGGER_TERMS)


def _find_belief_passages(query_text: str, max_passages: int = 3) -> list[str]:
    """
    Search every file in the beliefs folder for chunks relevant to the
    query. Only actually runs the (more expensive) chunk scan when
    _is_faith_related has already flagged the message as worth checking —
    most messages aren't about faith at all, and there's no reason to
    scan a multi-hundred-page cached Bible against every single chat
    message regardless of topic.
    """
    if not _is_faith_related(query_text):
        return []

    if not BELIEFS_DIR.exists():
        return []

    query_keywords = set(_extract_keywords(query_text))
    candidates = []

    for source_path in BELIEFS_DIR.iterdir():
        if not source_path.is_file() or source_path.parent == BELIEFS_CACHE_DIR:
            continue
        if source_path.suffix.lower() not in (".pdf", ".txt", ".md"):
            continue

        try:
            chunks = _get_belief_chunks(source_path)
        except Exception:
            continue  # a malformed/unreadable file shouldn't break retrieval for everything else

        for chunk in chunks:
            chunk_keywords = set(_extract_keywords(chunk))
            overlap = query_keywords & chunk_keywords
            if overlap:
                candidates.append({
                    "source": source_path.name,
                    "text": chunk,
                    "overlap_count": len(overlap),
                })

    candidates.sort(key=lambda c: c["overlap_count"], reverse=True)
    return [f"[From {c['source']}]: {c['text']}" for c in candidates[:max_passages]]


def _load_persona() -> str:
    """
    Load the global persona/behavior text. Read fresh on every call — NOT
    cached — so a person editing persona.txt in Notepad sees the change
    take effect on their very next message, with no backend restart
    required. This is the actual point of making it a file rather than a
    hardcoded string: it needs to be genuinely, immediately editable by
    anyone, not just by someone editing Python source.

    Applies globally, to every project and every conversation — not
    scoped to any one project, same as the file-tools instructions below.
    Returns an empty string (contributes nothing) if the file doesn't
    exist, so the system works normally with no persona configured.
    """
    if not PERSONA_FILE.exists():
        return ""
    try:
        return PERSONA_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _extract_keywords(text: str) -> list[str]:
    """
    Pull meaningful search terms out of a message. Deliberately crude:
    lowercase, strip punctuation, drop common short/stop words. This is
    NOT semantic search — it's literal keyword overlap. Good enough to
    find "we talked about the budget spreadsheet last week" style
    references; won't find conceptually-related content that doesn't
    share actual words (e.g. "fix the numbers in that finance doc" won't
    match "update the budget spreadsheet" — no shared terms despite being
    about the same thing). A genuine semantic layer (embedding-based,
    scoring meaning-closeness rather than word-overlap) is the documented
    next step if that gap matters in practice — not implemented here.
    Returns a list (not a set) so it can be JSON-serialized for caching.
    """
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "can", "this", "that", "these",
        "those", "i", "you", "he", "she", "it", "we", "they", "what", "which",
        "who", "whom", "whose", "when", "where", "why", "how", "and", "but",
        "or", "if", "then", "else", "for", "of", "to", "in", "on", "at",
        "by", "with", "about", "as", "from", "my", "your", "our", "me",
        "please", "hi", "hello", "hey", "thanks", "thank",
    }
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return sorted({w for w in words if len(w) > 2 and w not in stopwords})


def _keyword_index_path(project_id: str) -> Path:
    return _project_dir(project_id) / "conversations" / "_keyword_index.json"


def _load_keyword_index(project_id: str) -> dict:
    """
    Load the cached per-conversation keyword index for a project.
    This is the "embed once, reuse many" principle from the Unity
    semantic-calculator doc, applied with keyword sets instead of vector
    embeddings: rather than re-scanning every message in every other
    conversation on every chat request, each conversation's USER-message
    keywords (per the doc: score against what the user asked, not what
    the assistant answered) are extracted once and cached here. Rebuilt
    incrementally as conversations are saved, not on every read.
    """
    path = _keyword_index_path(project_id)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_keyword_index(project_id: str, index: dict) -> None:
    _keyword_index_path(project_id).write_text(json.dumps(index, indent=2), encoding="utf-8")


def _update_keyword_index_for_conversation(project_id: str, conversation: dict) -> None:
    """
    Recompute and cache this ONE conversation's keyword entries. Called
    once per save, not once per retrieval — the actual cost of keyword
    extraction is paid exactly once per user message, ever, not
    re-paid for every other conversation on every future chat request.
    """
    index = _load_keyword_index(project_id)
    entries = []
    for msg in conversation.get("messages", []):
        # An image's real description (image_caption) is indexed instead
        # of, not alongside, the message text it's attached to — an
        # uploaded image is a user-role message that ALSO carries a
        # caption, and indexing both would just have the longer, less
        # precise "message" copy of the same content compete with (and
        # crowd out of the top-N results) its own more precise "image"
        # copy. A generated image is assistant-role, so it was never
        # covered by the plain user-message branch below anyway — this
        # is what lets a LATER, unrelated conversation's question ("what
        # was that alien puppy image?") retrieve it at all.
        if msg.get("image_caption"):
            entries.append({
                "kind": "image",
                "content": msg["image_caption"],
                "image_url": msg.get("image_url"),
                "keywords": _extract_keywords(msg["image_caption"]),
            })
        elif msg.get("role") == "user" and msg.get("content"):
            entries.append({
                "kind": "message",
                "content": msg["content"],
                "keywords": _extract_keywords(msg["content"]),
            })
    index[conversation["id"]] = entries
    _save_keyword_index(project_id, index)


def _find_relevant_context(
    project_id: str,
    current_conversation_id: str | None,
    query_text: str,
    max_snippets: int = 3,
) -> list[dict]:
    """
    Search other conversations in this project for USER messages that
    share keywords with the current query — scoring against what the
    user asked in those past conversations, not what the assistant
    replied, per the doc's explicit guidance ("did they ask something
    like this before" is the retrieval target, not "did I say something
    like this before"). Reads from the cached keyword index, not raw
    conversation files, so this is a lookup against pre-computed data,
    not a full re-scan on every request.
    """
    query_keywords = set(_extract_keywords(query_text))
    if not query_keywords:
        return []

    index = _load_keyword_index(project_id)
    candidates = []
    for conv_id, entries in index.items():
        if conv_id == current_conversation_id:
            continue  # this conversation's own history is already sent in full separately
        for entry in entries:
            overlap = query_keywords & set(entry["keywords"])
            if overlap:
                candidate = {
                    "conversation_id": conv_id,
                    "kind": entry.get("kind", "message"),
                    "content": entry["content"],
                    "overlap_count": len(overlap),
                }
                if entry.get("image_url"):
                    candidate["image_url"] = entry["image_url"]
                candidates.append(candidate)

    candidates.sort(key=lambda c: c["overlap_count"], reverse=True)
    return candidates[:max_snippets]


def _conversation_path(project_id: str, conversation_id: str) -> Path:
    return _project_dir(project_id) / "conversations" / f"{conversation_id}.json"


def _load_conversation(project_id: str, conversation_id: str) -> dict:
    path = _conversation_path(project_id, conversation_id)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "id": conversation_id,
        "project_id": project_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "messages": [],
    }


def _save_conversation(conversation: dict) -> None:
    conversation["updated_at"] = datetime.now(timezone.utc).isoformat()
    path = _conversation_path(conversation["project_id"], conversation["id"])
    path.write_text(json.dumps(conversation, indent=2), encoding="utf-8")
    _update_keyword_index_for_conversation(conversation["project_id"], conversation)


@app.post("/api/projects/{project_id}/upload")
async def upload_file(
    project_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    conversation_id: str | None = Form(None),
):
    """
    Accept an image or document upload for a project and make it
    immediately usable in chat — unlike the model-driven file-action
    tool (which reads/writes one exact path the MODEL names), this is
    person-driven and grounds the upload right away: an image gets
    captioned with the same local BLIP pipeline /api/image/generate
    uses, folded straight into a saved conversation turn so a follow-up
    question in THIS conversation already has the real description in
    its history; a document gets chunked and cached the same way
    beliefs are, for the keyword-overlap retrieval
    _find_project_document_passages does across the whole project.
    """
    _require_project(project_id)

    suffix = Path(file.filename or "").suffix.lower()
    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(raw_bytes) / (1024 * 1024):.1f} MB). "
                    f"Limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    if suffix in IMAGE_UPLOAD_EXTENSIONS:
        project_images_dir = GENERATED_IMAGES_DIR / project_id
        project_images_dir.mkdir(parents=True, exist_ok=True)
        saved_filename = f"{uuid.uuid4().hex}{suffix}"
        saved_path = project_images_dir / saved_filename
        saved_path.write_bytes(raw_bytes)

        try:
            caption = _caption_image(saved_path)
        except Exception as e:
            print(f"[backend] Image captioning failed for upload {saved_path}: {e}")
            caption = None

        image_url = f"/api/image/generated/{project_id}/{saved_filename}"
        content = f"Uploaded image: {file.filename}"
        if caption:
            content += f"\n\nDescription: {caption}"

        conv_id = conversation_id or str(uuid.uuid4())
        conversation = _load_conversation(project_id, conv_id)
        conversation["project_id"] = project_id
        is_new_conversation = not conversation["messages"]
        upload_timestamp = datetime.now(timezone.utc).isoformat()
        conversation["messages"].append({
            "role": "user",
            "content": content,
            "image_url": image_url,
            "image_caption": caption,
            "timestamp": upload_timestamp,
        })
        # A user-role turn needs a paired assistant-role reply — some
        # chat templates (Gemma's, confirmed elsewhere in this file)
        # reject a request outright if two user turns land back to back
        # with nothing in between, which an upload-then-immediately-ask
        # sequence would otherwise produce.
        conversation["messages"].append({
            "role": "assistant",
            "content": (
                f'I can see the uploaded image — {caption}.' if caption
                else "I've received the uploaded image."
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        _save_conversation(conversation)

        if is_new_conversation:
            background_tasks.add_task(_title_background_task, project_id, conv_id, content)

        return {
            "type": "image",
            "conversation_id": conv_id,
            "image_url": image_url,
            "caption": caption,
            "timestamp": upload_timestamp,
        }

    if suffix in DOCUMENT_UPLOAD_EXTENSIONS:
        files_dir = _project_files_dir(project_id)
        files_dir.mkdir(parents=True, exist_ok=True)
        # Don't clobber an existing file of the same name — uploads add a
        # short unique suffix instead, since there's no overwrite
        # confirmation step in this flow the way write_project_file has.
        safe_name = Path(file.filename or "upload").name  # strip any path components
        dest = files_dir / safe_name
        if dest.exists():
            dest = files_dir / f"{dest.stem}-{uuid.uuid4().hex[:8]}{dest.suffix}"
        dest.write_bytes(raw_bytes)

        try:
            chunks = _get_project_document_chunks(project_id, dest)
        except Exception as e:
            print(f"[backend] Document extraction failed for upload {dest}: {e}")
            chunks = []

        conv_id = conversation_id or str(uuid.uuid4())
        conversation = _load_conversation(project_id, conv_id)
        conversation["project_id"] = project_id
        is_new_conversation = not conversation["messages"]
        upload_content = f"Uploaded document: {dest.name} ({len(chunks)} passages indexed for reference)"
        upload_timestamp = datetime.now(timezone.utc).isoformat()
        conversation["messages"].append({
            "role": "user",
            "content": upload_content,
            "timestamp": upload_timestamp,
        })
        # Same alternating-role requirement as the image branch above.
        conversation["messages"].append({
            "role": "assistant",
            "content": f'I\'ve indexed "{dest.name}" ({len(chunks)} passages) and can reference it going forward.',
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        _save_conversation(conversation)

        if is_new_conversation:
            background_tasks.add_task(_title_background_task, project_id, conv_id, upload_content)

        return {
            "type": "document",
            "conversation_id": conv_id,
            "filename": dest.name,
            "chunks_indexed": len(chunks),
            "timestamp": upload_timestamp,
        }

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file type: {suffix or '(no extension)'}. "
                f"Supported: images ({', '.join(IMAGE_UPLOAD_EXTENSIONS)}) "
                f"or documents ({', '.join(DOCUMENT_UPLOAD_EXTENSIONS)}).",
    )


@app.get("/api/projects/{project_id}/conversations")
def list_conversations(project_id: str):
    """Return every conversation saved under this project."""
    _require_project(project_id)
    conv_dir = _project_dir(project_id) / "conversations"
    results = []
    for path in sorted(conv_dir.glob("*.json"), reverse=True):
        if path.name == "_keyword_index.json":
            continue  # cache file, not a conversation — lives in the same folder

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue  # a genuinely corrupted file shouldn't take down the whole listing

        messages = data.get("messages") or []
        first_message_content = ""
        if messages and isinstance(messages[0], dict):
            first_message_content = messages[0].get("content", "") or ""

        results.append({
            "id": data.get("id", path.stem),
            # title is generated in the background off the first message
            # (see _generate_conversation_title) and may not exist yet on
            # a conversation that was only just started — the frontend
            # falls back to `preview` (plain truncated first message)
            # until it does.
            "title": data.get("title"),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
            "message_count": len(messages),
            "preview": first_message_content[:80],
        })
    return {"conversations": results}


@app.get("/api/projects/{project_id}/conversations/{conversation_id}")
def get_conversation(project_id: str, conversation_id: str):
    _require_project(project_id)
    path = _conversation_path(project_id, conversation_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Conversation not found")
    return _load_conversation(project_id, conversation_id)


@app.delete("/api/projects/{project_id}/conversations/{conversation_id}")
def delete_conversation(project_id: str, conversation_id: str):
    """
    Delete one conversation file and remove its entries from the
    project's keyword index — otherwise a deleted conversation could
    keep surfacing snippets through _find_relevant_context indefinitely,
    pointing at a conversation_id that no longer resolves to anything.
    """
    _require_project(project_id)
    path = _conversation_path(project_id, conversation_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Conversation not found")
    path.unlink()

    index = _load_keyword_index(project_id)
    if conversation_id in index:
        del index[conversation_id]
        _save_keyword_index(project_id, index)

    return {"id": conversation_id, "deleted": True}


async def _generate_conversation_title(first_message: str) -> str | None:
    """
    Ask the currently-running chat model for a short title summarizing
    this conversation's opening topic — what the conversation dropdown
    shows instead of a raw truncated first message. Deliberately a
    separate, plain completion call with no system prompt, persona, or
    tool instructions — this is a small utility task, not a real chat
    turn, and shouldn't be shaped by any of that.

    Best-effort only: returns None on any failure (model not running,
    timeout, empty output) so callers fall back to the plain-text
    preview list_conversations already computes on its own. Callers are
    expected to run this as a background task, not awaited inline —
    see _title_background_task — so a slow/failed title generation never
    delays the actual chat response going back to the person waiting on it.
    """
    if not llama_manager.is_alive():
        return None

    prompt = (
        "Summarize the topic of the following message in 4-6 words, "
        "title case, no punctuation, no quotation marks. Reply with "
        "ONLY the title itself, nothing else.\n\n"
        f"Message: {first_message[:500]}"
    )
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                f"{LLAMA_SERVER_URL}/v1/chat/completions",
                json={"messages": [{"role": "user", "content": prompt}], "max_tokens": 20},
            )
            response.raise_for_status()
            raw_title = response.json()["choices"][0]["message"]["content"]
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPStatusError, KeyError, IndexError, ValueError):
        return None

    # Strip a leading <think>...</think> block the same way real chat
    # replies are (see the KNOWN ISSUE note in /api/chat) — a reasoning
    # model would otherwise hand back its whole thought process as "the
    # title" instead of the short answer that follows it.
    think_match = re.match(r"^<think>.*?</think>\s*(.*)$", raw_title, re.DOTALL)
    title = (think_match.group(1) if think_match else raw_title).strip().strip('"\'')
    title = re.sub(r"\s+", " ", title)
    return title[:60] if title else None


async def _title_background_task(project_id: str, conversation_id: str, first_message: str) -> None:
    """
    Runs after the triggering response has already been sent (see
    BackgroundTasks usage in /api/chat and /api/image/generate) — that's
    why this reloads the conversation from disk rather than reusing an
    in-memory copy: the request that started it has already returned by
    the time this runs, so re-reading avoids racing whatever else might
    touch this same conversation file in between.
    """
    title = await _generate_conversation_title(first_message)
    if not title:
        return
    conversation = _load_conversation(project_id, conversation_id)
    if not conversation.get("title"):
        conversation["title"] = title
        _save_conversation(conversation)


# ---- Chat proxy ----

# Per-generation-call token budget: how many tokens of headroom
# _fit_history_to_context reserves for a reply when trimming the input,
# AND the max_tokens cap actually requested per generation call below.
# Kept as one shared constant so those two things can't drift apart —
# reserving room while trimming the prompt only matters if the request
# then actually asks for that much room back.
RESPONSE_TOKEN_RESERVE = 1024

# How many extra "keep going" calls /api/chat will make automatically
# when a reply gets cut off by hitting RESPONSE_TOKEN_RESERVE mid-answer
# (finish_reason "length") before giving up and saying so plainly. Caps
# worst-case latency/cost for a single request rather than looping
# forever on a model that just keeps generating.
MAX_CONTINUATION_ROUNDS = 4


async def _real_token_count(text: str) -> int | None:
    """
    Ask the running llama_cpp.server to tokenize text with the actual
    model's real tokenizer, via its /extras/tokenize/count endpoint
    (confirmed present in the installed llama-cpp-python version). This
    is a genuine token count for the exact model in use, not a guess.
    Returns None on any failure — connection issue, older server build
    without this endpoint, unexpected response shape — so callers fall
    back to an estimate explicitly rather than trusting a wrong number.
    """
    if not text:
        return 0
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{LLAMA_SERVER_URL}/extras/tokenize/count", json={"input": text}
            )
            r.raise_for_status()
            return r.json()["count"]
    except (httpx.HTTPError, KeyError, ValueError, TypeError):
        return None


def _estimate_token_count(text: str) -> int:
    """
    Cheap fallback when the real tokenizer endpoint isn't reachable.
    Deliberately conservative (divides by 3, not the more typical ~4
    characters/token for English) so a wrong guess errs toward trimming
    more than necessary rather than not enough.
    """
    return max(1, len(text) // 3)


async def _token_count(text: str) -> int:
    real = await _real_token_count(text)
    return real if real is not None else _estimate_token_count(text)


async def _fit_history_to_context(
    history_for_model: list[dict], context_length: int
) -> tuple[list[dict], bool]:
    """
    Trim history_for_model (a system message at index 0, if present,
    then alternating user/assistant turns) so its total token count fits
    inside the model's actual configured context window, leaving room
    for the reply itself. Drops the OLDEST user/assistant pair at a time
    — never the system message, never the most recent turn — until it
    fits. Returns (possibly-trimmed history, whether anything was
    actually dropped).

    This exists because llama_cpp.server fixes n_ctx at model-load time
    and hard-rejects any request that doesn't fit — there is no way to
    raise it per-request, and this app sends the entire saved
    conversation on every turn (see the comment above where
    history_for_model is built). Without this, a conversation that grows
    past the model's context window fails outright, every single turn,
    with no way to recover except starting a new conversation by hand.
    Trimming proactively here means a long conversation just gradually
    forgets its oldest turns instead — the same tradeoff most chat apps
    make once a conversation outgrows what a model can see at once.

    Drops in pairs (not one message at a time) specifically because at
    least one model in real use here (Gemma) hard-rejects a request
    whose messages don't strictly alternate user/assistant — dropping a
    single message from the middle of a pair would break that.
    """
    SAFETY_MARGIN = 64  # chat-template formatting adds a little overhead this doesn't capture by summing raw content alone
    budget = max(context_length - RESPONSE_TOKEN_RESERVE - SAFETY_MARGIN, 256)

    has_system = bool(history_for_model) and history_for_model[0]["role"] == "system"
    system_msg = history_for_model[0] if has_system else None
    turns = history_for_model[1:] if has_system else list(history_for_model)

    def joined_text(msgs: list[dict]) -> str:
        parts = ([system_msg["content"]] if system_msg else []) + [m["content"] for m in msgs]
        return "\n\n".join(parts)

    trimmed = False
    for _ in range(50):  # hard safety cap — should never actually take this many passes
        total = await _token_count(joined_text(turns))
        if total <= budget or len(turns) <= 1:
            break
        trimmed = True
        excess = total - budget
        avg_per_msg = total / len(turns) if turns else 1
        # Jump straight to roughly the right number of oldest messages to
        # drop instead of removing one pair and re-tokenizing every time —
        # keeps this fast even for a very long conversation. Always an
        # even number so pairing is preserved, always leaves at least one
        # turn behind.
        jump = max(2, int(excess / avg_per_msg))
        jump -= jump % 2
        jump = min(jump, len(turns) - 1)
        turns = turns[jump:]

    result = ([system_msg] if system_msg else []) + turns
    return result, trimmed


_SENTENCE_END_RE = re.compile(r"[.!?][\"'\)\]]*(?:\s|$)")


def _trim_to_last_sentence_boundary(text: str) -> str:
    """
    Cut truncated model output back to its last complete sentence,
    discarding whatever incomplete sentence trails after it. Used before
    asking the model to continue a reply that got cut off mid-generation
    — confirmed in real testing that asking a model (Gemma specifically)
    to literally "continue where you left off" from a broken mid-word or
    mid-sentence fragment often doesn't work: it starts a new sentence
    that doesn't grammatically connect to the cut point, producing
    garbled text like "...respect for their beliefs …n him by force
    to make him king." Handing it a clean sentence boundary instead
    means its continuation is a normal new sentence, which is what
    stitches together correctly.

    Falls back to the last whitespace if no sentence-ending punctuation
    is found at all (so a mid-WORD break at least never survives), and
    to the untouched text only if there's no whitespace either — nothing
    safe left to cut.
    """
    matches = list(_SENTENCE_END_RE.finditer(text))
    if matches:
        return text[: matches[-1].end()].rstrip()
    last_space = text.rfind(" ")
    if last_space > 0:
        return text[:last_space].rstrip()
    return text


def _friendly_llama_error(e: httpx.HTTPStatusError) -> str:
    """
    Turn whatever llama_cpp.server sent back into a plain-English message
    instead of surfacing its raw JSON error body to a non-technical
    person. Must never raise itself — it runs inside an exception
    handler, and a second exception there would replace a handled error
    with an unhandled one.
    """
    message = ""
    try:
        body = e.response.json()
        message = body.get("error", {}).get("message", "") or ""
    except (ValueError, AttributeError):
        pass

    if "context length" in message.lower() or "context_length_exceeded" in message.lower():
        return (
            "This conversation — plus its instructions and any related files or "
            "past messages the assistant pulled in — is too long for the "
            "current model to handle in one request, even after automatically "
            "trimming older messages. Try starting a new conversation, turning "
            "on \"2x Context\" and restarting the model, or switching to a "
            "model with a larger context window."
        )
    return (
        "The AI model reported an error and couldn't finish this response. "
        f"Check llama_server.log for the technical details. "
        f"({message or 'no further detail given'})"
    )


@app.post("/api/chat")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Proxy a chat request to the running llama_cpp.server, then save
    both the user's message and the model's reply under the given project.
    """
    _require_project(request.project_id)

    if not llama_manager.is_alive():
        # The subprocess died unexpectedly (OOM, crash, reload side-effect —
        # see notes above start()). Rather than force the person back through
        # the UI to manually re-swap, try relaunching the last-known model
        # once before giving up. Uses the same lock as manual swaps so an
        # automatic recovery can't race a user-triggered swap.
        if llama_manager.current_model_path is None:
            raise HTTPException(
                status_code=503,
                detail="No model is currently loaded. Swap to a model via /api/models/swap first.",
            )
        print(f"[backend] Model process found dead before chat request — "
              f"attempting automatic relaunch of {llama_manager.current_model_path}")
        async with llama_manager.lock:
            if not llama_manager.is_alive():  # re-check inside the lock — another request may have already fixed this
                llama_manager.start(llama_manager.current_model_path)
                recovered = await _wait_for_server_ready()
                if not recovered:
                    raise HTTPException(
                        status_code=503,
                        detail=f"Model process died and automatic relaunch of "
                                f"{Path(llama_manager.current_model_path).name} failed. "
                                f"Check {llama_manager.log_path} or try swapping models manually.",
                    )
        print("[backend] Automatic relaunch succeeded, proceeding with chat request.")

    conversation_id = request.conversation_id or str(uuid.uuid4())
    conversation = _load_conversation(request.project_id, conversation_id)
    conversation["project_id"] = request.project_id
    is_new_conversation = not conversation["messages"]

    for msg in request.messages:
        if msg.role == "user":
            conversation["messages"].append({
                "role": "user",
                "content": msg.content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    # KNOWN ISSUE (not yet solved): Qwen3's <think> reasoning block adds
    # significant tokens/latency. See project notes — two suppression
    # approaches tried, both failed, reverted. Sending plain messages for now.
    # This is also *why* the timeout below is generous: a creative request
    # with an unsuppressed reasoning block, at ~5 tok/s, can genuinely need
    # several minutes. Confirmed hitting the old 300s ceiling in testing.
    #
    # IMPORTANT: send the conversation's full saved history, not just the
    # newly-arrived message(s). Without this, every request was stateless —
    # the model had no memory of anything said earlier in this same
    # conversation, despite the UI showing an ongoing chat. Saved messages
    # carry extra fields (timestamp, reasoning) the model API doesn't want;
    # strip down to role/content only.
    history_for_model = [
        {"role": m["role"], "content": m["content"]}
        for m in conversation["messages"]
    ]

    # Combine all system-level guidance (tool instructions + any
    # cross-conversation context) into exactly ONE system message,
    # inserted once at position 0. Two separate inserts here previously
    # caused Gemma's server to reject the request outright with
    # "Conversation roles must alternate user/assistant/user/assistant" —
    # confirmed in testing. A single system message at the very front is
    # what the model's chat template actually tolerates; stacking system
    # messages, or inserting one on every turn of a growing multi-turn
    # conversation, breaks that structure.
    tool_instructions = (
        "You can create, read, or delete files in this project's files "
        "folder. When the user asks you to do one of these, you must "
        "respond with a block in this exact format — do not just describe "
        "what you would do, actually output this block:\n\n"
        "To CREATE or OVERWRITE a file:\n"
        "```file-action\n"
        "action: create\n"
        "path: relative/path/to/file.ext\n"
        "content: the full file content goes here\n"
        "```\n"
        "Example — user says \"create a file called notes.txt with the "
        "text hello\":\n"
        "```file-action\n"
        "action: create\n"
        "path: notes.txt\n"
        "content: hello\n"
        "```\n\n"
        "To READ a file (you'll then be given its real content to answer with):\n"
        "```file-action\n"
        "action: read\n"
        "path: relative/path/to/file.ext\n"
        "```\n"
        "Example — user says \"what does notes.txt say\":\n"
        "```file-action\n"
        "action: read\n"
        "path: notes.txt\n"
        "```\n\n"
        "To DELETE a file — only when the user clearly and explicitly "
        "asks you to delete or remove a specific file:\n"
        "```file-action\n"
        "action: delete\n"
        "path: relative/path/to/file.ext\n"
        "```\n"
        "Example — user says \"delete notes.txt\":\n"
        "```file-action\n"
        "action: delete\n"
        "path: notes.txt\n"
        "```\n\n"
        "For normal conversation with no file request, just reply "
        "normally with no file-action block."
    )

    persona = _load_persona()
    system_message_parts = ([persona] if persona else []) + [tool_instructions]

    latest_user_message = next(
        (m.content for m in reversed(request.messages) if m.role == "user"), None
    )
    if latest_user_message:
        related = _find_relevant_context(
            request.project_id, conversation_id, latest_user_message
        )
        if related:
            context_lines = []
            for r in related:
                if r["kind"] == "image":
                    # A real description, not a placeholder — this is the
                    # actual fix for the reported bug: a chat model asked
                    # about a generated/uploaded image from another
                    # conversation now gets its genuine caption instead of
                    # confabulating specifics it never had.
                    context_lines.append(
                        f"[An image from a related earlier conversation, described as]: {r['content']}"
                    )
                else:
                    context_lines.append(f"[From a related earlier conversation, user said]: {r['content']}")
            system_message_parts.append(
                "The following excerpts from other conversations in this project "
                "may be relevant to the current question:\n\n" + "\n\n".join(context_lines)
            )

        document_passages = _find_project_document_passages(request.project_id, latest_user_message)
        if document_passages:
            doc_lines = [f'[From an uploaded document "{p["source"]}"]: {p["content"]}' for p in document_passages]
            system_message_parts.append(
                "The following passages from documents uploaded to this project "
                "may be relevant to the current question:\n\n" + "\n\n".join(doc_lines)
            )

        belief_passages = _find_belief_passages(latest_user_message)
        if belief_passages:
            system_message_parts.append(
                "The following passages from your beliefs folder may be relevant "
                "to what's being discussed. Reference them naturally if they genuinely "
                "fit — cite the source file, don't just recite the passage flatly:\n\n"
                + "\n\n".join(belief_passages)
            )

    combined_system_message = "\n\n---\n\n".join(system_message_parts)

    # Insert at position 0 ONLY if there isn't already a system message
    # there — this is what actually prevents accumulation across turns
    # in the same conversation, rather than blindly inserting every time.
    if history_for_model and history_for_model[0]["role"] == "system":
        history_for_model[0] = {"role": "system", "content": combined_system_message}
    else:
        history_for_model.insert(0, {"role": "system", "content": combined_system_message})

    # Budget against the ACTUAL context window the running model was
    # launched with (see LlamaServerManager.context_length), not a
    # guess — trims oldest turns first so a long conversation degrades
    # gracefully instead of hard-failing. Falls back to the original,
    # pre-fix default (2048) only if a model was somehow never started
    # through start() at all, which shouldn't happen in practice.
    context_length = llama_manager.context_length or 2048
    history_for_model, history_was_trimmed = await _fit_history_to_context(
        history_for_model, context_length
    )

    # Generate the reply. llama_cpp.server defaults an unset max_tokens to
    # "fill whatever's left of the context window" — which, combined with
    # never checking finish_reason, is what let a cut-off answer through
    # silently before this fix (a reply is only "done" when the model
    # actually stopped on its own; hitting the token cap mid-sentence is
    # a different, distinguishable outcome — OpenAI-style APIs report it
    # via finish_reason == "length"). When that happens, ask the model to
    # keep going from exactly where it stopped and stitch the pieces
    # together, up to MAX_CONTINUATION_ROUNDS times, rather than ever
    # showing a silently-truncated answer as if it were complete.
    current_messages = list(history_for_model)
    assistant_pieces: list[str] = []
    finish_reason = None
    cumulative_usage: dict = {}

    async with httpx.AsyncClient(timeout=900.0) as client:
        for _round in range(MAX_CONTINUATION_ROUNDS + 1):
            try:
                response = await client.post(
                    f"{LLAMA_SERVER_URL}/v1/chat/completions",
                    json={"messages": current_messages, "max_tokens": RESPONSE_TOKEN_RESERVE},
                )
                response.raise_for_status()
            except httpx.ReadTimeout:
                raise HTTPException(
                    status_code=504,
                    detail="The model is still generating a response after 15 minutes. "
                            "This can happen with long/complex requests on slower models, "
                            "especially with reasoning enabled. The generation may still "
                            "complete on the model server even though this request gave up — "
                            "check llama_server.log.",
                )
            except httpx.ConnectError:
                raise HTTPException(
                    status_code=503,
                    detail="Cannot reach llama_cpp.server even though the process is alive. "
                            "It may still be loading the model.",
                )
            except httpx.HTTPStatusError as e:
                raise HTTPException(status_code=502, detail=_friendly_llama_error(e))

            result = response.json()
            choice = result["choices"][0]
            piece = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
            assistant_pieces.append(piece)
            for k, v in result.get("usage", {}).items():
                if isinstance(v, (int, float)):
                    cumulative_usage[k] = cumulative_usage.get(k, 0) + v

            if finish_reason != "length":
                break  # the model actually finished on its own — done

            # Don't hand the model (or the final stitched reply) a broken
            # mid-sentence fragment to build on — cut back to the last
            # complete sentence first, discarding the incomplete tail, so
            # the next round starts a clean new sentence instead of being
            # asked to complete a graft point it may just ignore.
            kept = _trim_to_last_sentence_boundary(piece)
            if kept:
                assistant_pieces[-1] = kept
            else:
                kept = piece  # nothing safe to cut — keep the raw piece rather than lose it entirely

            current_messages = current_messages + [
                {"role": "assistant", "content": kept},
                {"role": "user", "content": (
                    "Continue your response with what comes next. Do not "
                    "repeat anything you already said, and do not add any "
                    "new preamble or greeting."
                )},
            ]
            # The growing continuation transcript can itself outgrow the
            # context budget — re-fit before every further call, same as
            # the original request was.
            current_messages, _ = await _fit_history_to_context(current_messages, context_length)

    # Plain "".join() glues pieces together with nothing between them —
    # confirmed in testing that a hard max_tokens cutoff can land
    # mid-word, producing "...vibrant fish" + "to the inky..." joined
    # into "fishto...". Insert a single space at a seam only when
    # neither side already has one, so a cut that happened to land on a
    # real word/space boundary isn't touched.
    raw_content = ""
    for piece in assistant_pieces:
        if raw_content and piece and not raw_content[-1].isspace() and not piece[0].isspace():
            raw_content += " "
        raw_content += piece
    response_cut_short = finish_reason == "length"
    if response_cut_short:
        # Still truncated after every automatic continuation attempt —
        # say so plainly rather than saving/showing a partial answer with
        # nothing indicating it isn't the whole thing.
        raw_content += (
            "\n\n*[This reply was cut short even after being continued "
            "automatically several times — it kept filling the model's "
            "available context. Try asking in smaller pieces, start a new "
            "conversation, or turn on \"2x Context\" and restart the model.]*"
        )

    # Strip a leading <think>...</think> block if present. This does NOT
    # reduce generation time or token cost — the model still generates the
    # full reasoning internally either way (see KNOWN ISSUE note above,
    # both suppression-at-generation attempts failed). This only cleans up
    # what gets stored and returned, so raw <think> tags don't leak into
    # saved conversations or the API response.
    think_match = re.match(r"^<think>(.*?)</think>\s*(.*)$", raw_content, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1).strip()
        assistant_content = think_match.group(2).strip()
    else:
        reasoning = None
        assistant_content = raw_content

    # Detect and execute a file-action block, per the tool-use instructions
    # taught in the system prompt above. Regex is deliberately tolerant of
    # whitespace variation — a 1B model won't reproduce the exact format
    # byte-for-byte every time — but a genuinely malformed or missing
    # field fails visibly (a real error message replaces the block) rather
    # than silently doing nothing, so the person can see something was
    # attempted and why it didn't work.
    # Detect and execute a file-action block, per the tool-use instructions
    # taught in the system prompt above. Regex is deliberately tolerant of
    # whitespace variation — a 1B model won't reproduce the exact format
    # byte-for-byte every time — but a genuinely malformed or missing
    # field fails visibly (a real error message replaces the block) rather
    # than silently doing nothing, so the person can see something was
    # attempted and why it didn't work.
    #
    # 'action' defaults to 'create' when absent, so every file-action
    # block generated under the original (pre-extension) system prompt
    # still parses and works exactly as before.
    file_action_match = re.search(
        r"```file-action\s*\n"
        r"(?:action:\s*(\w+)\s*\n)?"
        r"path:\s*(.+?)\s*\n"
        r"(?:content:\s*(.*?)\n)?"
        r"```",
        assistant_content,
        re.DOTALL,
    )
    if file_action_match:
        action = (file_action_match.group(1) or "create").strip().lower()
        action_path = file_action_match.group(2).strip()
        action_content = file_action_match.group(3) or ""

        if action == "create":
            try:
                resolved = _resolve_project_file(request.project_id, action_path)
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved.write_text(action_content, encoding="utf-8")
                confirmation = f"✅ Created file: `{action_path}`"
            except HTTPException as e:
                confirmation = f"⚠️ Could not create file `{action_path}`: {e.detail}"
            assistant_content = (
                assistant_content[:file_action_match.start()]
                + confirmation
                + assistant_content[file_action_match.end():]
            ).strip()

        elif action == "delete":
            # One-way and destructive — report honestly rather than
            # pretending success if the file was never real to begin with.
            try:
                resolved = _resolve_project_file(request.project_id, action_path)
                if not resolved.exists():
                    confirmation = f"⚠️ Could not delete `{action_path}`: file not found."
                elif not resolved.is_file():
                    confirmation = f"⚠️ Could not delete `{action_path}`: it's a directory, not a file."
                else:
                    resolved.unlink()
                    confirmation = f"🗑️ Deleted file: `{action_path}`"
            except HTTPException as e:
                confirmation = f"⚠️ Could not delete `{action_path}`: {e.detail}"
            assistant_content = (
                assistant_content[:file_action_match.start()]
                + confirmation
                + assistant_content[file_action_match.end():]
            ).strip()

        elif action == "read":
            # Two-step: fetch the real content, then make a SECOND model
            # call with that content included, so the final reply is
            # grounded in what's actually in the file rather than the
            # model guessing/hallucinating contents it never saw.
            try:
                resolved = _resolve_project_file(request.project_id, action_path)
                if not resolved.exists():
                    file_content_for_model = f"[File not found: {action_path}]"
                elif not resolved.is_file():
                    file_content_for_model = f"[{action_path} is a directory, not a file]"
                else:
                    file_content_for_model = resolved.read_text(encoding="utf-8")
            except (HTTPException, UnicodeDecodeError) as e:
                file_content_for_model = f"[Could not read {action_path}: {e}]"

            follow_up_messages = history_for_model + [
                {"role": "assistant", "content": assistant_content},
                {
                    "role": "user",
                    "content": f"[System: contents of {action_path}]\n\n{file_content_for_model}\n\n"
                                f"Now answer the original question using this file content.",
                },
            ]
            # The file's real content can itself be large enough to blow
            # the same context budget the original request was just fit
            # to — re-trim rather than assume history_for_model already
            # being in-budget still holds once a whole file is appended.
            follow_up_messages, _ = await _fit_history_to_context(
                follow_up_messages, context_length
            )
            follow_up_payload = {"messages": follow_up_messages, "max_tokens": RESPONSE_TOKEN_RESERVE}
            try:
                async with httpx.AsyncClient(timeout=900.0) as follow_up_client:
                    follow_up_response = await follow_up_client.post(
                        f"{LLAMA_SERVER_URL}/v1/chat/completions", json=follow_up_payload
                    )
                    follow_up_response.raise_for_status()
                follow_up_choice = follow_up_response.json()["choices"][0]
                assistant_content = follow_up_choice["message"]["content"].strip()
                if follow_up_choice.get("finish_reason") == "length":
                    # Same truncation guarantee as the main reply — this
                    # secondary call doesn't get the multi-round
                    # continuation loop (it's already a follow-up), so
                    # just say plainly that it was cut short.
                    response_cut_short = True
                    assistant_content += (
                        "\n\n*[This reply was cut short — reading the file back left "
                        "too little room in the model's context for a full answer. "
                        "Try a shorter file, a new conversation, or turn on "
                        "\"2x Context\" and restart the model.]*"
                    )
            except httpx.HTTPStatusError as e:
                assistant_content = (
                    f"{assistant_content}\n\n[Could not read {action_path} back to you: "
                    f"{_friendly_llama_error(e)}]"
                )
            except (httpx.ConnectError, httpx.ReadTimeout):
                assistant_content = (
                    f"{assistant_content}\n\n[Could not reach the model to read "
                    f"{action_path} back to you — it may still be busy or have "
                    f"crashed. Check llama_server.log.]"
                )

    assistant_timestamp = datetime.now(timezone.utc).isoformat()
    conversation["messages"].append({
        "role": "assistant",
        "content": assistant_content,
        "reasoning": reasoning,
        "timestamp": assistant_timestamp,
    })
    _save_conversation(conversation)

    if is_new_conversation:
        first_user_message = next((m.content for m in request.messages if m.role == "user"), None)
        if first_user_message:
            background_tasks.add_task(
                _title_background_task, request.project_id, conversation_id, first_user_message
            )

    return {
        "conversation_id": conversation_id,
        "project_id": request.project_id,
        "reply": assistant_content,
        "reasoning": reasoning,
        "timestamp": assistant_timestamp,
        "usage": cumulative_usage,
        "history_trimmed": history_was_trimmed,
        "response_cut_short": response_cut_short,
    }


# ---- Health check ----

@app.get("/api/health")
async def health():
    """
    Confirm both this backend and the upstream llama_cpp.server are reachable.

    A short timeout here specifically distinguishes "busy generating a
    response" from "actually down": llama_cpp.server runs a single worker,
    so while it's producing tokens for a chat request, a concurrent health
    check can't get an immediate answer. That's not a failure — it's the
    server correctly doing the one thing it was asked to do. A ReadTimeout
    here is reported as "busy", not "unreachable".
    """
    llama_status = "unreachable"
    if llama_manager.is_alive():
        async with httpx.AsyncClient(timeout=1.5) as client:
            try:
                r = await client.get(f"{LLAMA_SERVER_URL}/v1/models")
                llama_status = "ok" if r.status_code == 200 else "unreachable"
            except httpx.ReadTimeout:
                llama_status = "busy"
            except httpx.ConnectError:
                llama_status = "unreachable"

    loading_elapsed_seconds = None
    if llama_status != "ok" and llama_manager.load_started_at is not None:
        loading_elapsed_seconds = round(time.time() - llama_manager.load_started_at, 1)

    return {
        "backend": "ok",
        "llama_cpp_server": llama_status,
        "active_model": llama_manager.current_model_path,
        "extended_context": llama_manager.extended_context,
        "context_length": llama_manager.context_length,
        "loading_elapsed_seconds": loading_elapsed_seconds,
    }


@app.post("/api/shutdown")
async def shutdown_everything(background_tasks: BackgroundTasks):
    """
    Full teardown, unlike /api/models/stop (which only ever touched
    llama_cpp.server): stop both managed subprocesses, then terminate
    this backend process itself. This is what the frontend's Stop
    button calls — one button, everything actually goes away, not just
    the chat model.

    The process exit is deliberately deferred a beat via a background
    task rather than done inline here: os._exit() kills the process
    immediately, and if that ran before this function returned, the
    HTTP response would never reach the browser — the request would
    just look like a dropped connection instead of a clean success.
    """
    async with llama_manager.lock:
        llama_manager.stop()
    async with image_manager.lock:
        image_manager.stop()

    def _exit_soon():
        time.sleep(0.5)  # give the response time to actually flush to the client
        os._exit(0)

    background_tasks.add_task(_exit_soon)
    return {"status": "ok", "message": "All processes stopped. This server is shutting down."}


# ---- Frontend (must be mounted LAST — a mount at "/" only catches
# whatever request no earlier, more specific @app.route above it already
# matched, so every /api/... route above still works normally) ----

app.mount("/", StaticFiles(directory=APP_ROOT / "frontend", html=True), name="frontend")