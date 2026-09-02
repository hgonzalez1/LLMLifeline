"""
Downloads this project's default models on a genuinely fresh setup.

GGUF/safetensors model files are too large for GitHub, so instead of
shipping them, this fetches them the first time the app is ever
started — called from both start.bat and launcher.py, right after
dependencies are installed (huggingface_hub, used below, is one of
those dependencies, so it isn't guaranteed to exist before then).

Runs at most once, ever: a completed run writes FIRST_RUN_MARKER, which
is checked before doing anything else. Someone who deletes an
auto-downloaded model later, or swaps in their own choice, is making a
deliberate decision this script has to respect — it must never
re-download something that was intentionally removed. If any single
file fails partway through, the marker is deliberately NOT written, so
the whole thing is retried (already-downloaded files are skipped via
the same exists-check used for someone's own manually-placed files) on
the next launch instead of leaving the setup half-finished forever.

Every repo_id/filename pair below was verified directly against
Hugging Face with hf_hub_download(..., dry_run=True) — for the three
chat models, the reported file_size matched this project's own
already-in-use files byte-for-byte. All four are ungated: no Hugging
Face account or token needed for any of them.
"""

import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
MODELS_DIR = APP_ROOT / "models"
CHECKPOINTS_DIR = APP_ROOT / "ComfyUI" / "models" / "checkpoints"
FIRST_RUN_MARKER = MODELS_DIR / ".default_models_downloaded"

# (repo_id, filename, destination directory, approx size in GB for the
# heads-up message below)
DEFAULT_DOWNLOADS = [
    ("bartowski/gemma-2-9b-it-abliterated-GGUF", "gemma-2-9b-it-abliterated-Q8_0.gguf", MODELS_DIR, 9.8),
    ("mradermacher/oh-dcft-v3.1-gpt-4o-mini-GGUF", "oh-dcft-v3.1-gpt-4o-mini.f16.gguf", MODELS_DIR, 16.1),
    ("JonathanColetti/Qwen3.8-27B-Uncensored-GGUF", "Qwen3.8-27B-Uncensored-noMTP-Q4_K_M.gguf", MODELS_DIR, 16.5),
    ("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors", CHECKPOINTS_DIR, 6.9),
]


def main():
    if FIRST_RUN_MARKER.exists():
        return  # already handled, ever - never re-trigger even if models\ is empty now

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("NOTE: huggingface_hub isn't installed yet - skipping the default")
        print("model download for now. It'll be tried again on the next launch,")
        print("or add your own .gguf/.safetensors files manually in the meantime.")
        return

    total_gb = sum(size for *_, size in DEFAULT_DOWNLOADS)
    print()
    print(f"First run: downloading {len(DEFAULT_DOWNLOADS)} default models (~{total_gb:.0f}GB total,")
    print("one-time only). Press Ctrl+C now if you'd rather supply your own -")
    print("this won't be offered again either way; just drop your own files into")
    print("models\\ and ComfyUI\\models\\checkpoints\\ instead.")
    print()

    for repo_id, filename, dest_dir, size_gb in DEFAULT_DOWNLOADS:
        dest_path = dest_dir / filename
        if dest_path.exists():
            print(f"  Already have {filename}, skipping.")
            continue

        print(f"  Downloading {filename} (~{size_gb:.1f}GB) from {repo_id}...")
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(dest_dir))
            print(f"    Done: {downloaded_path}")
        except Exception as e:
            print(f"    ERROR downloading {filename}: {e}")
            print("    Skipping for now - you can add this model manually, or just")
            print("    run the launcher again later to retry it.")
            return  # marker NOT written - a future run retries whatever's still missing

    FIRST_RUN_MARKER.parent.mkdir(parents=True, exist_ok=True)
    FIRST_RUN_MARKER.write_text(
        "Marks that the default model download already ran once.\n"
        "Delete this file if you want it offered again (e.g. after removing\n"
        "all your models and wanting the defaults back).\n"
    )
    print("Default models ready.")
    print()


if __name__ == "__main__":
    main()
