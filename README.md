# LLMlifeline

*A private AI chat + image generator that runs entirely on your own PC — no cloud, no subscription, no account.*

Chat with a local AI model, organize conversations into projects, generate images, and shape how the assistant talks and what it knows — all through one simple screen.

**New here?** This page covers day-to-day use. For installing it the first time, see **[setup-guide.md](setup-guide.md)** — the short version is: run `start.bat` (or double-click `LLMlifeline.exe`), and everything else, including your first set of AI models, sets itself up automatically.

Video Description
https://www.instagram.com/p/DczP_zttc8v/
---

## Using LLMlifeline

Once it's running, your browser opens to the app on its own. Here's what everything does.

### Projects

Everything lives inside a **project** — think of it like a folder for one topic, one person, or one purpose. Click **+ New Project** in the top-left to create one; click any project in the list to switch to it. Each project keeps its own conversations completely separate from every other project's.

### Chatting

Type in the box at the bottom and hit **Send**, or Enter. Each message shows the time it was sent. The little colored dot at the top tells you the assistant's status — green means ready, yellow means it's busy answering, red means it's not currently running.

**Past conversations** — the dropdown in the left sidebar under "Conversations" lists every past conversation in the current project, automatically titled by topic (it takes a few seconds after your first message for the real title to appear — it starts as a placeholder). Pick one to load it back up, complete with its full history. The trash icon next to it deletes the conversation you're currently viewing — this can't be undone. **New Chat** starts a fresh one.

### Generating images

Check the **Image Model** box at the top to switch modes — this automatically pauses the chat model and starts up the image generator (the two never run at the same time; your hardware only has to do one job at once). Pick a checkpoint from the dropdown if you have more than one, then type a description and hit Send, same as chatting. The image appears right in the conversation, with a real caption describing what was actually drawn. Uncheck the box (or check **Chat Model**) to switch back — your last active chat model picks back up automatically.

### Asking the assistant to work with files

Just ask naturally — "create a file called notes.txt with this content," "what does that file say," "delete the old draft." Each project has its own private `files/` workspace, and the assistant genuinely creates, reads, and deletes real files there on request — not simulated, not just describing what it would do. It's confined to that one project's workspace and can't touch anything else on your computer.

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
