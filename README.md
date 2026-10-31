# LLMlifeline

*A private AI chat + image generator that runs entirely on your own PC — no cloud, no subscription, no account.*

Chat with a local AI model, organize conversations into projects, generate images, and shape how the assistant talks and what it knows — all through one simple screen.

**New here?** This page covers day-to-day use. For installing it the first time, see **[setup-guide.md](setup-guide.md)** — the short version is: run `start.bat` (or double-click `LLMlifeline.exe`), and everything else, including your first set of AI models, sets itself up automatically. On a Mac or Linux, grab the `.dmg` or `.tar.gz` from the [Releases page](https://github.com/hgonzalez1/LLMLifeline/releases).

Video Description
https://www.instagram.com/p/DczP_zttc8v/
---

## Using LLMlifeline

Once it's running, your browser opens to the app on its own. Here's what everything does.

### Projects

Everything lives inside a **project** — think of it like a folder for one topic, one person, or one purpose. Click **+ New Project** in the top-left to create one; click any project in the list to switch to it. Each project keeps its own conversations completely separate from every other project's.

### Chatting

Type in the box at the bottom and hit **Send**, or Enter. The reply appears as the assistant writes it, word by word, so you can start reading right away; a blinking cursor at its end means more is coming. If the model reasons before answering, its reasoning shows as a collapsed **Reasoning** section you can open. Each message shows the time it was sent. The little colored dot at the top tells you the assistant's status — green means ready, yellow means it's busy answering, red means it's not currently running.

If you reload or close the page while a reply is being written, the reply still finishes and is saved: open the conversation again in a moment to see it.

**Past conversations** — the dropdown in the left sidebar under "Conversations" lists every past conversation in the current project, automatically titled by topic (it takes a few seconds after your first message for the real title to appear — it starts as a placeholder). Pick one to load it back up, complete with its full history. The trash icon next to it deletes the conversation you're currently viewing — this can't be undone. **New Chat** starts a fresh one.

### Generating images

Check the **Image Model** box at the top to switch modes — this automatically pauses the chat model and starts up the image generator (the two never run at the same time; your hardware only has to do one job at once). Pick a checkpoint from the dropdown if you have more than one, then type a description and hit Send, same as chatting. The image appears right in the conversation, with a real caption describing what was actually drawn. Uncheck the box (or check **Chat Model**) to switch back — your last active chat model picks back up automatically.

### Choosing a chat model

The model dropdown at the top lists every chat model this computer can run, under headings: **GGUF models** (the `.gguf` files LLMlifeline runs itself) and one heading for each installed engine add-on. Pick one, and LLMlifeline stops the current model and loads that one. Your projects, conversations and settings stay as they are. The model you used last is loaded again the next time LLMlifeline starts.

- **Models kept in other folders.** Create `model-folders.txt` in the LLMlifeline folder and put one folder per line, for example a folder of models on a faster drive. The `.gguf` files directly inside those folders join the dropdown. Those folders are searched before `models\`, so if the same file is in both places, the copy from your listed folder is used.
- **Engine add-ons.** An add-on lets LLMlifeline use models that another program runs, for example a model far too big for your memory that a separate engine streams from disk. Installing one means putting its folder in `addons\`; see [addons/README.md](addons/README.md). Everything else in LLMlifeline works the same with an add-on's models, but some add-on engines are much slower: the status dot shows "Not Ready" while one loads, and a reply can take minutes. With a slow engine most of the wait comes before the first word, while the engine reads the conversation; after that the reply appears as it's written.

### Asking the assistant to work with files

Just ask naturally — "create a file called notes.txt with this content," "what does that file say," "delete the old draft." Each project has its own private `files/` workspace, and the assistant genuinely creates, reads, and deletes real files there on request — not simulated, not just describing what it would do. Unless you turn on computer use (below), it's confined to that one project's workspace and can't touch anything else on your computer.

### Computer use — letting the assistant build things on your computer

Click **🔐 Computer use** in the sidebar and tick the box. The assistant can then work like a coding assistant: read and write folders you share, run commands (build, test, install packages), and download web pages and files. It works in steps, shown in the chat as small 🔧 notes you can expand.

**You're in charge of every permission:**
- Anything it hasn't been given yet shows an approval card first, with exactly what it wants to do: the command, the web address or the file. Choose **Allow once**, **Always allow** (for that command, site or folder) or **Deny**.
- The Permissions panel lists everything you've granted. **Revoke** removes any one item, and **Revoke everything** turns it all off.
- Commands can be off, ask before each one, or run without asking. Web access is download-only and asks before each new site.

**It never uploads your data.** That isn't a setting, and nothing in the panel can turn it off:
- The assistant can only make plain downloads (no sending, no logins, no cookies).
- Any web address that would carry text from your files, chats or command output is refused.
- Commands that can send data off your computer are refused: curl, scp, git push, publish and deploy tools, and scripts with network-sending code.
- LLMlifeline's own folder (your conversations and settings) is always off limits.

What it can't promise: a program can do anything once it runs. LLMlifeline checks each command and any script it can read, but it can't see inside a compiled program. That's why commands ask first by default.

Small models follow the step format less reliably. A bigger chat model will build things far better.

### Sharing one model across your home network

Click **🖧 Network** in the sidebar. One computer, usually the one with the strongest graphics card, shares its model, and the other computers on your network use it. Every computer keeps its own projects, chats, files, persona, beliefs and computer-use permissions. The computer actions the assistant takes always run on the computer you're using.

**On the computer sharing its model:**
- Tick **"Let devices on my network use this computer's model"**. The panel shows the address other computers connect to.
- If Windows asks whether to allow LLMlifeline on networks, allow **private networks only**.
- When a device asks to connect, it appears in the panel and on the sidebar button ("1 waiting"). It shows a 6-digit code. Approve it only if the device shows the same code.
- The Devices list shows who's connected now. **Boot** disconnects a device; it can ask to pair again, and you'd approve it again. **Block** refuses it until you unblock it. **Forget** removes it.
- Two options: whether devices can generate images and describe photos (only while this computer is in Image mode), and whether they can change its model or switch it between chat and images. Switching is off by default, because it changes the model for everyone.

**On a computer using another's model:** click **Find computers on my network**, or type the other computer's address, then **Connect**. A code appears; check that the other computer shows the same one, and have it approve. The status bar then shows "Ready · <that computer's name>". **Use this computer's own model** switches back, and **Forget** unpairs. If the other computer boots it, the panel offers **Pair again**.

**What the sharing computer sees and keeps:**
- To answer a request it has to read it: the recent chat, persona and any file text in it, or a photo to describe. It holds that only in memory while working.
- Nothing from a request or reply is saved, logged or sent anywhere else.
- It keeps each device's name, when it paired, and when and from which address it was last seen.
- Connections are encrypted, and only devices on your local network are answered.
- The 6-digit code proves to each device that it's talking to the real computer and not an impostor.

The one thing code can't promise: whoever controls the sharing computer could change its software. Use it with people who trust that computer.

**A computer without a good graphics card** can skip the heavy install entirely. On a fresh setup, LLMlifeline asks where its model will run. Choosing "on another computer" makes a lightweight install of under 1 GB, with no models, no image tools and no GPU libraries. Everything waits on one shared model, so expect a queue when several people use it at once. The sharing computer either needs to be powerful or should run a smaller, faster model.

### Uploading images and documents

Click the 📎 button next to the message box to upload a photo (it gets captioned automatically, so you can ask about it) or a document — PDF, .txt, or .md (it gets indexed, so the assistant can find and quote relevant passages later, even in a different conversation).

### Project Images

The bottom-left sidebar shows every image that's been generated or uploaded in the current project, as thumbnails. Click one to view it full-size, and click the full-size image to zoom in further.

### Stopping LLMlifeline

Click the **Stop** button at the top. This shuts everything down cleanly — the chat model, the image generator, all of it — and tells you when it's safe to close the browser tab. (If that's ever not reachable — a crash, a closed tab — running `stop.bat` does the same cleanup manually.)

---

## Making it yours

LLMlifeline is meant to be edited, not just used. Here's what's actually in the project folder and what each part is for.

| Folder / file | What it's for |
|---|---|
| `persona.txt` | The assistant's personality — see below |
| `beliefs\` | Faith/scripture reference documents (PDF/text/markdown) the assistant can cite from — see below |
| `projects\` | Every project's saved conversations — generated automatically, not for hand-editing |
| `generated_images\` | Every image ever generated or uploaded, organized by project |
| `models\` | Chat model files (`.gguf`) |
| `model-folders.txt` | Optional: more folders with `.gguf` chat models, one per line — see "Choosing a chat model" |
| `addons\` | Optional engine add-ons that bring their own models — see [addons/README.md](addons/README.md) |
| `ComfyUI\models\checkpoints\` | Image model files (`.safetensors`) |

### `persona.txt` — the assistant's personality

This is a plain text file at the top of the project folder. Whatever you write here shapes how the assistant talks and behaves in **every** project and **every** conversation — it's global, not per-project. Open it in any text editor (Notepad is fine), write a description of how you want it to act — its tone, what it should focus on, what it should avoid — save the file, and that's it. **No restart needed** — the very next message you send already uses the new version.

If the file doesn't exist yet, or you leave it empty, the assistant just behaves normally with no special personality layered on.

*Example:* `You are a calm, encouraging tutor. Explain things simply, check for understanding before moving on, and never make the person feel bad for not knowing something.`

### `beliefs\` — faith/scripture reference material it can quote from

Drop PDF, `.txt`, or `.md` files into this folder — scripture, doctrine, anything faith-related you want the assistant to cite from directly instead of answering from memory alone. It's specifically faith-triggered: when your message contains a faith-related word (God, Jesus, prayer, scripture, sin, and similar), it automatically searches these files and pulls in real passages, naming the source file. A message with no faith-related wording won't trigger it, even if a file in here happens to be relevant — this folder isn't a general-purpose knowledge base (for that, upload a document to a project instead, via the 📎 button — see above). Add or remove files anytime; there's no button to press, it's picked up automatically.

### `projects\` — your actual conversation history and files

This is where every conversation you've ever had actually lives on disk, organized by project — along with each project's own `files\` subfolder, the private workspace the assistant reads and writes in when you ask it to (see "Asking the assistant to work with files," above). You don't need to open this folder day-to-day — everything here is fully manageable from inside the app. It's documented here mainly so it's clear where your data actually is, and that it never leaves your computer.

### `generated_images\` and model folders

`generated_images\` fills up on its own as you generate or upload images — nothing to configure. `models\` and `ComfyUI\models\checkpoints\` are where AI model files themselves live; LLMlifeline starts you off with a working default set automatically (see setup-guide.md), and adding your own is as simple as dropping a compatible file into the right one — it shows up in the app's dropdowns on its own, no restart needed.

---

## Going further

- **Installing on a new machine, GPU/driver requirements, rebuilding the `.exe`, troubleshooting:** [setup-guide.md](setup-guide.md)
- Everything in this project — your conversations, your persona, your beliefs, your models — stays on your own computer. Nothing is uploaded anywhere.
