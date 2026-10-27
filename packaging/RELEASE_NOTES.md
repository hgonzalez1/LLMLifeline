A private AI chat and image generator that runs on your own computer. No cloud, no subscription, no account.

None of these downloads include the AI models or the big Python packages. The first launch installs those itself: Python packages, ComfyUI (~6-7 GB) and the default models (~49 GB, skippable with Ctrl+C). After that, launches are fast.

All builds are **unsigned**. The steps below show how to get past each OS's warning.

## macOS (Apple Silicon): `LLMlifeline-macos-arm64.dmg`

1. Open the .dmg and drag **LLMlifeline** into **Applications**.
2. Install first: Apple's Command Line Tools (`xcode-select --install` in Terminal), and Python 3.10–3.13 from [python.org](https://www.python.org/downloads/macos/). The built-in macOS python3 is too old.
3. Not signed with an Apple Developer ID. Run this once in Terminal:
   `xattr -dr com.apple.quarantine /Applications/LLMlifeline.app`
   Or use System Settings → Privacy & Security → **Open Anyway** after the first blocked launch.
4. Double-click LLMlifeline. A Terminal window shows setup progress, and your browser opens when it's ready. Your data lives in `~/LLMlifeline`.

Mac notes: Intel Macs aren't supported by this build. Chat uses Metal GPU acceleration. The context-length auto-sizing only knows NVIDIA cards, so Macs use the safe 2048-token default. Image generation runs on Metal and is slower than on an NVIDIA card.

## Windows: `LLMlifeline-windows-x64.zip`

Unzip anywhere and double-click `LLMlifeline.exe` (or `start.bat`). Needs Python 3.10, git, and ideally an NVIDIA GPU. SmartScreen will warn about the unsigned exe: **More info → Run anyway**.

## Linux (x86_64): `LLMlifeline-linux-x86_64.tar.gz`

```bash
tar -xzf LLMlifeline-linux-x86_64.tar.gz
cd LLMlifeline
./LLMlifeline
```

Needs `python3` (3.10–3.13), `python3-venv`, `git` and a C/C++ compiler (`build-essential`) for llama-cpp-python. An NVIDIA GPU is optional. The chat engine installs CPU-only. For CUDA, see setup-guide.md.

See [setup-guide.md](https://github.com/hgonzalez1/LLMLifeline/blob/main/setup-guide.md) for details and troubleshooting.
