A private AI chat and image generator that runs on your own computer. No cloud, no subscription, no account.

**1.1.1** fixes a startup crash in 1.1.0. Its backend wouldn't start when LLMlifeline's Python environment was Python 3.11 or newer, which covers most Macs. If you installed 1.1.0, update to this one the same way as described below.

## New in 1.1: computer use

The assistant can now work on your computer to build things, the way coding assistants do. It can read and write folders you share, run commands, and download web pages. It's **off until you turn it on** under **🔐 Computer use** in the sidebar.

- **You grant every permission.** Each request shows exactly what will happen (the command, the web address or the file) with **Allow once**, **Always allow** and **Deny**.
- **You can revoke anything at any time** in the Permissions panel.
- Commands can be off, ask before each one, or run without asking. Web access is download-only and asks before each new site.

**It never uploads your data.** This isn't a setting. LLMlifeline enforces it in code, under every permission:

- The assistant can only make plain downloads (no sending, no logins, no cookies).
- Any web address that would carry text from your files, chats or command output is refused.
- Commands that can send data off your computer are refused: curl, scp, git push, publish and deploy tools, and scripts with network-sending code.
- LLMlifeline's own folder (your conversations and settings) is off limits.
- The model runs on your computer, so your conversations never leave it.

What it can't promise: a program can do anything once it runs. LLMlifeline checks each command and any script it can read, but it can't see inside compiled programs. That's why commands ask first by default.

**Also fixed:** any website open in your browser could previously read LLMlifeline's local server, including your conversations. LLMlifeline now answers only its own page on your computer. ComfyUI's online "API nodes" are switched off.

## Updating from 1.0.0 or 1.1.0

Your conversations, persona, beliefs and models are never touched by these updates.

- **Mac:** open the new `.dmg` and drag **LLMlifeline** into **Applications**. Choose **Replace**. Run the `xattr` command below once more, since it's a new app. Your data in `~/LLMlifeline` stays as it is, and the new code is copied in on first launch.
- **Windows:** quit LLMlifeline (Stop in the app). Unzip `LLMlifeline-windows-x64-update.zip`, then copy what's inside its `LLMlifeline` folder over your existing LLMlifeline folder, choosing **Replace**.
- **Linux:** stop LLMlifeline, then from the folder that contains your `LLMlifeline` folder run `tar -xzf LLMlifeline-linux-x86_64-update.tar.gz`.
- **From git:** `git pull`.

The update packages leave out `persona.txt` and `beliefs/`. The full downloads below include defaults for those, so use them only for a fresh install.

## Fresh install

None of these downloads include the AI models or the big Python packages. The first launch installs those itself: Python packages, ComfyUI (~6-7 GB) and the default models (~49 GB, skippable with Ctrl+C). All builds are **unsigned**; the steps below show how to get past each system's warning.

### macOS (Apple Silicon): `LLMlifeline-macos-arm64.dmg`

1. Open the .dmg and drag **LLMlifeline** into **Applications**.
2. Install first: Apple's Command Line Tools (`xcode-select --install` in Terminal), and Python 3.10–3.13 from [python.org](https://www.python.org/downloads/macos/). The built-in macOS python3 is too old.
3. It isn't signed with an Apple Developer ID. Run this once in Terminal:
   `xattr -dr com.apple.quarantine /Applications/LLMlifeline.app`
   Or try to open it, then use System Settings → Privacy & Security → **Open Anyway**.
4. Double-click LLMlifeline. A Terminal window shows setup progress, and your browser opens when it's ready. Your data lives in `~/LLMlifeline`.

Mac notes: Intel Macs aren't supported by this build. Chat uses Metal GPU acceleration. Context-length auto-sizing only knows NVIDIA cards, so Macs use the safe 2048-token default.

### Windows: `LLMlifeline-windows-x64.zip`

Unzip anywhere and double-click `LLMlifeline.exe` (or `start.bat`). It needs Python 3.10, git, and ideally an NVIDIA GPU. SmartScreen will warn about the unsigned exe: click **More info → Run anyway**.

### Linux (x86_64): `LLMlifeline-linux-x86_64.tar.gz`

```bash
tar -xzf LLMlifeline-linux-x86_64.tar.gz
cd LLMlifeline
./LLMlifeline
```

It needs `python3` (3.10–3.13), `python3-venv`, `git` and a C/C++ compiler (`build-essential`) for llama-cpp-python. An NVIDIA GPU is optional, and the chat engine installs CPU-only. For CUDA, see setup-guide.md.

See [setup-guide.md](https://github.com/hgonzalez1/LLMLifeline/blob/main/setup-guide.md) for details and troubleshooting.
