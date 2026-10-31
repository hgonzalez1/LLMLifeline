A private AI chat and image generator that runs on your own computer. No cloud, no subscription, no account.

## New in 1.3: replies as they're written, and more models to choose from

- **Replies appear as they're written,** word by word, so you can start reading right away instead of waiting for the whole answer. A model's reasoning shows in a collapsed **Reasoning** section. With computer use, each step appears as the assistant takes it. A reply still finishes and is saved if you reload or close the page.
- **Server mode streams too.** A device using a computer that runs 1.3 sees replies as they're written. A sharing computer still on 1.2 answers in one piece, as before.
- **Models kept in other folders:** create `model-folders.txt` in the LLMlifeline folder and list folders of `.gguf` models, one per line, for example a fast SSD. They join the model dropdown, which now groups models under headings.
- **Engine add-ons** let LLMlifeline use models another program runs, such as a model too big for your memory that streams from disk. See `addons/README.md` for how to write or install one. Only install add-ons you trust.
- **Your last model loads next time.** LLMlifeline starts with the chat model you used last.
- **Fixed:** stopping or switching a model now ends every process it started, so its memory is freed straight away.

## New in 1.2: share one model across your home network

Open **🖧 Network** in the sidebar. One computer, usually the one with the strongest graphics card, shares its model, and your other computers use it. **Every computer keeps its own projects, chats, files and settings.** Only the model's work (writing replies, drawing images, describing photos) happens on the sharing computer.

- **Approve every device.** A device asking to connect shows a 6-digit code, and so does the sharing computer. Approve only if they match. That proves the device is talking to your real computer, not an impostor.
- **Devices list:** see who's connected now. **Boot** disconnects a device, which can ask to pair again. **Block** refuses it until you unblock it.
- **Encrypted and local only:** connections are encrypted, and only devices on your home network are answered.
- **What the sharing computer keeps:** only each device's name and when and from where it was last seen. It holds a request in memory just long enough to answer it, and saves, logs and forwards nothing from it.
- **Find computers automatically,** or type the address.
- **Lightweight install:** a computer without a good graphics card can now skip the heavy install. On a fresh setup, LLMlifeline asks where its model will run. "On another computer" means no models, no image tools and no GPU libraries, under 1 GB in total.
- **Your choice:** whether devices may generate images, and whether they may change the sharing computer's model (off by default, since it affects everyone).

The sharing computer needs to be powerful, or should run a smaller, faster model. Everyone shares one model, so requests take turns. Windows or macOS will ask once whether to allow incoming connections; on Windows, allow private networks only.

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

## Updating from an earlier version

Your conversations, persona, beliefs and models are never touched by these updates. On its next start, LLMlifeline installs the two small packages server mode needs (`cryptography` and `zeroconf`) by itself.

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
