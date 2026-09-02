# LLMlifeline — Setup Guide

*A local, offline chatbot with file tools, project memory, scripture citation, and image generation — running entirely on your own PC.*

> **Status:** Actively developed. This reflects the real, current state of the build as of tonight — not a plan for the future.

> Looking for how to actually **use** the app — projects, chatting, image generation, customizing the personality — once it's installed? That's **[README.md](README.md)**. This document covers installation, new-machine setup, and technical troubleshooting.

---

## What this program does

- Runs large language models entirely on your own computer, fully offline
- Drop any compatible `.gguf` model into a folder and it becomes selectable
- Chat with the model, with real memory of the current conversation
- Organize conversations into projects, each with its own file workspace the model can create, read, and delete files in
- Retrieves relevant context from other conversations in the same project automatically
- A global, user-editable persona file controls how the assistant speaks and behaves
- Cites real passages from PDF/text documents in a "beliefs" folder when relevant
- Generates images via a local Stable Diffusion pipeline (ComfyUI), as an alternative to the chat model — not simultaneously with it

---

## Before you start: hardware check

**Minimum:**
- Windows 10 or 11 (64-bit)
- An NVIDIA graphics card (strongly recommended — CPU-only works but is much slower)
- 16GB+ system RAM (32GB+ recommended)
- Real free storage — model files run 1–30GB each; image generation adds several more GB for checkpoints

**Check what you have:**
```powershell
dxdiag
```
System tab shows CPU and RAM. Then:
```powershell
nvidia-smi
```
Shows your GPU and VRAM if you have an NVIDIA card.

---

## One-time setup

The project root is `LLMlifeline\` — on the original build machine that's `C:\Users\<you>\LLMlifeline`, but every path in the app is now relative to wherever this folder actually lives, so it works from a fresh copy on a different machine too (see "Handing this to someone else" below). Everything lives inside it — models, projects, persona, beliefs, and the code itself.

### Step 1 — What you need already installed

`start.bat` (or `LLMlifeline.exe`) sets up everything else on its own — the virtual environment, ComfyUI, every Python package — the first time you run it. What it can't install for you, because these need to already exist on the machine before anything Python-related can even start:
- **Python 3.10**, with "Add python.exe to PATH" checked during install ([python.org](https://www.python.org/downloads/)) — a different 3.x version will often still work, but 3.10 is what this project was actually built and tested against
- **git** ([git-scm.com](https://git-scm.com/downloads)) — needed once, to pull in ComfyUI
- **An NVIDIA GPU with a current driver** — run `nvidia-smi` in a terminal to confirm

That's it. Just make sure the project folder itself (`backend\`, `frontend\`, `start.bat`, this guide) is somewhere on disk, and move on to Step 2.

### Step 2 — Run the startup script

```
start.bat
```

(Double-click it, or run it from a terminal in the project root.) A double-click-friendly `LLMlifeline.exe` also exists — see "The .exe" below.

**What happens the first time (on a completely fresh machine — none of this repeats once it's done):**
- If `llm-env\` (the Python virtual environment) is missing, or present but broken — the most common way that happens is the whole project folder was copied or zipped from a different machine, since a venv bakes in an absolute path to its own Python install and simply doesn't run anywhere else — it's removed if broken and created fresh, automatically, using whatever Python is on this machine
- If `ComfyUI\` is missing, it's cloned fresh from its real upstream repository (`github.com/comfyanonymous/ComfyUI`), pinned to the exact commit this project was built and tested against — this step alone is a real ~6-7GB download and can take a while
- It checks for every required Python package — `fastapi`, `uvicorn`, `httpx`, `python-multipart`, `llama-cpp-python`, `pypdf`, `comfy_kitchen`, `transformers`/`Pillow` (for image captioning), plus `torch` and ComfyUI's other dependencies
- Anything missing gets installed automatically — this can take several minutes, especially `llama-cpp-python` (which compiles from source) and `torch` (which is a large download)
- It checks that `torch` reports real GPU/CUDA support **and** is recent enough (2.7.0+) for ComfyUI's kernel library, reinstalling a current CUDA build from PyTorch's own index if not — confirmed necessary on this exact setup: an older CUDA-enabled torch still "has CUDA" but crashes ComfyUI on import
- It launches the backend (which also serves the app itself — see Step 4), waits for it to actually respond, then opens your default browser automatically

This is what makes handing the whole project folder (or a zip of it) to someone else's PC actually work — they don't need to know what a virtual environment or ComfyUI even is, or run a single command by hand.

**A real, known limitation, stated honestly:** the automatic install does **not** guarantee a GPU-accelerated build of `llama-cpp-python` — a from-scratch install lands on a CPU-only build by default, silently, with no error or warning. Fixing that means either finding a prebuilt CUDA wheel matching your exact CUDA version, or building it locally:
```powershell
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
llm-env\Scripts\pip install llama-cpp-python --force-reinstall --no-cache-dir
```
which needs a C++ compiler — the free [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) ("Desktop development with C++" workload) plus a matching CUDA Toolkit from NVIDIA. Not something the startup script attempts for you — it's genuinely machine-specific.

### Step 3 — Models

On a genuinely fresh setup (`models\` and `ComfyUI\models\checkpoints\` both empty), `download_models.py` fetches a default set automatically the first time you run `start.bat`/`LLMlifeline.exe` — three chat models and one image checkpoint, listed in `download_models.py` itself, ~49GB total. This happens **once, ever** — it writes `models\.default_models_downloaded` when done, and never runs again after that, even if you later delete everything it downloaded. Press Ctrl+C during that step if you'd rather skip it and pick your own from the start.

To use different models — instead of the defaults, or in addition to them — drop `.gguf` chat model files into:
```
LLMlifeline\models\
```
and `.safetensors` image checkpoints into:
```
LLMlifeline\ComfyUI\models\checkpoints\
```
Both folders already have a `put_checkpoints_here`-style placeholder marking where they go. Chat models appear in the model dropdown; checkpoints appear in the Image Model checkpoint dropdown — both refresh automatically, no restart needed.

Anything beyond the automatic defaults is never bundled or downloaded for you — model files run tens of gigabytes each, so adding more is always a manual, deliberate step.

### Step 4 — Confirm it's running

`start.bat` opens your browser automatically once the backend is ready — the app is served directly from the backend now, at `http://127.0.0.1:8001/`, so there's no separate `frontend\index.html` file to open by hand anymore. If it doesn't open on its own, go to that URL yourself. You should see:
- A status dot with text next to it — "Ready" (green), "Busy" (yellow), or "Not Ready" (red)
- A project sidebar
- A model dropdown showing your `.gguf` files

### The `.exe`

`LLMlifeline.exe` is a compiled version of the same startup logic (`launcher.py`), for a genuine double-click launch with no `.bat` file involved. It does **not** bundle torch/CUDA/ComfyUI/model files — those still live in, and get installed into, `llm-env` on first run, exactly like `start.bat`. Rebuild it after any change to `launcher.py` with `build_exe.bat`. It's unsigned, so Windows SmartScreen/Defender will likely flag it the first time it runs on a machine that didn't build it — "More info" → "Run anyway" gets past that; it's expected for any new unsigned binary, not a sign of a problem.

### Handing this to someone else / setting up on a different machine

Zip up or copy the project folder and hand it over — you don't need to strip anything out first. `llm-env\` won't actually work on the new machine (see Step 2), but `start.bat`/`LLMlifeline.exe` detects that on its own and replaces it automatically; same for `ComfyUI\` if it's missing entirely, and same for model files, via `download_models.py` (see Step 3) — the recipient doesn't need to source any of that themselves unless they want something other than the defaults. The only things that genuinely can't be automated, per Step 1: Python, git, and an NVIDIA GPU + driver need to already be on the new machine.

If you're building the zip yourself and want it smaller/faster to transfer, you can leave `llm-env\`, `ComfyUI\`, and everything under `models\`/`ComfyUI\models\checkpoints\` out on purpose — same outcome, just a smaller file, since all of it gets recreated/redownloaded fresh on first run either way.

**Pushing this to GitHub specifically:** a `.gitignore` is already in place, excluding `llm-env\`, `ComfyUI\`, model weight files, and your own personal data (`projects\`, `generated_images\`, `persona.txt`, `beliefs\`) — none of that belongs in a shared repo, whether for size limits or privacy. `LLMlifeline.exe` and `build\` are excluded too, since `build_exe.bat` regenerates both from `launcher.py`; if you want a built exe available to people who clone the repo, attach it to a GitHub Release instead of committing it.

---

## Everyday use

**Starting up:** run `start.bat` (or `LLMlifeline.exe`). That's the only step for a normal session — see **[README.md](README.md)** for everything the app itself can do: projects, chatting, image generation, uploads, and customizing the persona/beliefs/models.

---

## Understanding what's happening under the hood (optional)

**VRAM and context length:** the app automatically checks your available graphics card memory at every model load and picks a safe context length based on it — you don't need to calculate this yourself. There's also a "2x Context" toggle in the app if you want to try doubling it; this uses meaningfully more VRAM and can fail to load on some models, but fails safely (the model just won't load) rather than causing any real problem elsewhere.

**Why responses can be slow to start:** some models "think" internally before answering, which can add real time and isn't currently something this build can turn off — it's a known, open limitation. The thinking is shown collapsed in the interface so it doesn't clutter the actual answer.

**Image generation:** uses a separate pipeline (ComfyUI) from the chat models, and only one — chat or image — is active at a time, selected via a checkbox in the app. This is a genuinely different kind of model, not something layered on top of the chat models.

---

## Glossary

- **GGUF** — a file format for AI models built to run efficiently on regular computers.
- **VRAM** — memory built into your graphics card. Faster than system RAM, but smaller.
- **Quantization** — shrinking a model's size by storing its numbers less precisely, with a small, usually unnoticeable quality tradeoff.
- **Context length** — how much conversation (and instructions, and retrieved reference material) the model can consider at once, measured in tokens.
- **Tokens per second (tok/s)** — how fast a model generates its response. Roughly 1 token ≈ ¾ of a word.
