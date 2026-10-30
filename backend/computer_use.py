"""
Computer use — lets the assistant work beyond its project's own files/
folder, the way Claude Code or Codex do: read and write folders you
grant, run commands, and download web pages. Each of those is a
permission you grant and can revoke at any time (permissions.json in the
project root, edited from the app's Permissions panel). Nothing here is
on until you turn it on; with it off, chat behaves exactly as before.

ONE RULE IS NOT A PERMISSION: the assistant never uploads your data to
anyone. That's enforced in this file, in code, under every permission
setting — not by asking the model nicely:

- Web access is download-only. fetch_url() makes plain GET requests with
  no body, no cookies, no stored credentials and no proxy settings, and
  url_problem()/url_leak() refuse any web address that carries text from
  your chats, files, persona or command output, or a blob that looks
  like encoded data.
- Commands that can send data off this computer — curl/wget, scp/ssh,
  git push, publish/deploy tools, cloud CLIs, tunnels, and scripts or
  inline code that open network connections — are refused outright by
  screen_command(), even in "run without asking" mode.
- LLMlifeline's own folder (your conversations, other projects, models,
  this permissions file) is off limits to the assistant's file tools.

What this can't promise, stated plainly: a program can do whatever it
likes once it runs. screen_command() checks the command line and any
script it can read; it can't see inside a compiled program or a package
it installs. That's why commands ask first by default.
"""

import asyncio
import ipaddress
import json
import os
import re
import socket
import subprocess
import sys
import uuid
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, unquote_plus, urljoin, urlsplit

import httpx

APP_ROOT = Path(__file__).resolve().parent.parent
PERMISSIONS_FILE = APP_ROOT / "permissions.json"

COMMAND_MODES = ("off", "ask", "auto")
# No "any site without asking" mode on purpose: a site you haven't
# approved only ever gets fetched after you've seen its address.
INTERNET_MODES = ("off", "ask")

DEFAULT_PERMISSIONS = {
    "enabled": False,      # master switch — off means today's project-files-only behavior
    "folders": [],         # [{"path": absolute folder, "write": bool}]
    "commands": "ask",     # off | ask | auto (applies only while enabled)
    "command_rules": [],   # command prefixes that run without asking in "ask" mode
    "internet": "ask",     # off | ask (applies only while enabled)
    "domains": [],         # sites downloaded from without asking
}

MAX_RESULT_CHARS = 4000        # per action, what the model gets back
COMMAND_TIMEOUT_SECONDS = 300
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
USER_AGENT = "LLMlifeline/1.1 (download-only)"

# Handed to every command the assistant runs, so tools that phone home
# by default don't.
NO_TELEMETRY_ENV = {
    "DO_NOT_TRACK": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "NEXT_TELEMETRY_DISABLED": "1",
    "GATSBY_TELEMETRY_DISABLED": "1",
    "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
    "POWERSHELL_TELEMETRY_OPTOUT": "1",
    "ASTRO_TELEMETRY_DISABLED": "1",
    "NUXT_TELEMETRY_DISABLED": "1",
    "STORYBOOK_DISABLE_TELEMETRY": "1",
    "VCPKG_DISABLE_METRICS": "1",
}


# ---- Permissions file ----

def _is_within(path: Path, folder: Path) -> bool:
    try:
        path.relative_to(folder)
        return True
    except ValueError:
        return False


EXAMPLE_FOLDER = r"C:\Users\you\Projects" if sys.platform == "win32" else "/Users/you/Projects"


def normalize_folder(raw: str) -> str:
    """Absolute folder path, fully resolved — raises ValueError with a
    message the Permissions panel can show as-is. Doesn't require the
    folder to exist yet: "Always allow" can grant the folder a new
    project is about to be created in."""
    text = (raw or "").strip().strip('"')
    if not text:
        raise ValueError("Enter a folder path.")
    path = Path(os.path.expanduser(text))
    if not path.is_absolute():
        raise ValueError(f"'{text}' isn't a full path. Use something like {EXAMPLE_FOLDER}.")
    return str(path.resolve())


def _normalize_domain(raw: str) -> str | None:
    text = (raw or "").strip().lower()
    if "://" in text:
        text = urlsplit(text).hostname or ""
    text = text.split("/")[0].lstrip("*.").rstrip(".")
    if text.startswith("www."):
        text = text[4:]
    return text if re.fullmatch(r"[a-z0-9-]+(\.[a-z0-9-]+)+", text) else None


def validate_permissions(raw, strict: bool = False) -> dict:
    """
    Normalizes permissions read from disk or sent by the UI. Anything
    unknown or malformed falls back to the safe default — except with
    strict=True (a save from the UI), where a bad folder raises
    ValueError so the person sees why it wasn't added.
    """
    perms = json.loads(json.dumps(DEFAULT_PERMISSIONS))
    if not isinstance(raw, dict):
        return perms
    perms["enabled"] = raw.get("enabled") is True
    if raw.get("commands") in COMMAND_MODES:
        perms["commands"] = raw["commands"]
    if raw.get("internet") in INTERNET_MODES:
        perms["internet"] = raw["internet"]
    for entry in raw.get("folders") or []:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            continue
        try:
            path = normalize_folder(entry["path"])
        except ValueError:
            if strict:
                raise
            continue
        if not any(f["path"] == path for f in perms["folders"]):
            perms["folders"].append({"path": path, "write": entry.get("write") is True})
    for rule in raw.get("command_rules") or []:
        if isinstance(rule, str) and rule.split():
            rule = " ".join(rule.split())
            if rule not in perms["command_rules"]:
                perms["command_rules"].append(rule)
    for domain in raw.get("domains") or []:
        domain = _normalize_domain(domain) if isinstance(domain, str) else None
        if domain and domain not in perms["domains"]:
            perms["domains"].append(domain)
    return perms


def load_permissions() -> dict:
    try:
        raw = json.loads(PERMISSIONS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    return validate_permissions(raw)


def save_permissions(perms: dict) -> dict:
    perms = validate_permissions(perms)
    PERMISSIONS_FILE.write_text(json.dumps(perms, indent=2), encoding="utf-8")
    return perms


def remember(perms: dict, grant: dict) -> dict:
    """Applies an approval card's "Always allow" choice."""
    perms = json.loads(json.dumps(perms))
    if grant.get("command_rule"):
        perms["command_rules"].append(grant["command_rule"])
    if grant.get("domain"):
        perms["domains"].append(grant["domain"])
    if grant.get("folder"):
        folder = grant["folder"]
        existing = next((f for f in perms["folders"] if f["path"] == folder["path"]), None)
        if existing:
            existing["write"] = existing["write"] or folder["write"]
        else:
            perms["folders"].append(folder)
    return save_permissions(perms)


# ---- Parsing the model's action blocks ----

_ACTION_BLOCK_RE = re.compile(r"```(file-action|run-command|fetch-url)[ \t]*\n(.*?)```", re.DOTALL)
_FIELD_RE = re.compile(r"^(action|path|content|cwd|command|url|save):[ \t]?(.*)$")


def parse_actions(text: str) -> list[dict]:
    """
    Every action block in a reply, in order. Tolerant like the original
    single-block regex (a small model won't reproduce the format
    byte-for-byte): `content:` takes everything after it, multi-line;
    `command:` takes following lines that aren't another field.
    'action' defaults to 'create', so file-action blocks written under
    the original system prompt still work exactly as before.
    """
    actions = []
    for match in _ACTION_BLOCK_RE.finditer(text):
        kind, body = match.group(1), match.group(2)
        fields: dict[str, str] = {}
        lines = body.split("\n")
        i = 0
        while i < len(lines):
            field = _FIELD_RE.match(lines[i].strip()) if lines[i].strip() else None
            if not field:
                i += 1
                continue
            key, value = field.group(1), field.group(2)
            if key == "content":
                rest = [value] + lines[i + 1:]
                fields["content"] = "\n".join(rest).rstrip("\n")
                break
            if key == "command":
                parts = [value]
                while i + 1 < len(lines) and lines[i + 1].strip() and not _FIELD_RE.match(lines[i + 1].strip()):
                    i += 1
                    parts.append(lines[i])
                value = "\n".join(parts)
            fields[key] = value.strip()
            i += 1
        action = {"id": str(uuid.uuid4()), "span": [match.start(), match.end()]}
        if kind == "file-action":
            action.update(type="file", action=(fields.get("action") or "create").lower(),
                          path=fields.get("path", ""), content=fields.get("content", ""))
        elif kind == "run-command":
            action.update(type="command", command=fields.get("command", ""), cwd=fields.get("cwd", ""))
        else:
            action.update(type="fetch", url=fields.get("url", ""), save=fields.get("save", ""))
        actions.append(action)
    return actions


def describe(action: dict) -> str:
    if action["type"] == "file":
        return f"{action['action']} {action['path']}"
    if action["type"] == "command":
        return f"run `{action['command']}`"
    return f"download {action['url']}" + (f" to {action['save']}" if action.get("save") else "")


# ---- Paths ----

def resolve_path(raw: str, workspace: Path) -> Path:
    """Relative paths stay in the project's files folder, as always;
    absolute (or ~) paths mean somewhere else on this computer."""
    text = (raw or "").strip().strip('"').strip("`")
    expanded = Path(os.path.expanduser(text))
    return (expanded if expanded.is_absolute() else workspace / text).resolve()


def check_path(perms: dict, path: Path, write: bool, workspace: Path) -> tuple[str, str | None]:
    if _is_within(path, workspace.resolve()):
        return "allow", None
    if _is_within(path, APP_ROOT):
        return "deny", ("LLMlifeline's own folder (your conversations, other projects, models "
                        "and settings) is off limits to the assistant.")
    if not perms["enabled"]:
        return "deny", ("That's outside this project's files folder. Working elsewhere on this "
                        "computer is off — turn on computer use in Permissions.")
    for folder in perms["folders"]:
        if _is_within(path, Path(folder["path"])) and (folder["write"] or not write):
            return "allow", None
    return "ask", None


# ---- The no-upload screen for commands ----

# Programs whose job is (or includes) sending data somewhere else.
# Downloads go through fetch-url instead, which can't upload.
UPLOAD_TOOLS = {
    "curl", "wget", "wget2", "httpie", "xh", "scp", "sftp", "ftp", "tftp", "lftp", "ssh",
    "rsync", "nc", "ncat", "netcat", "socat", "telnet", "invoke-webrequest", "iwr",
    "invoke-restmethod", "irm", "start-bitstransfer", "bitsadmin", "certutil", "mshta",
    "send-mailmessage", "sendmail", "mail", "mailx", "mutt", "msmtp", "gh", "hub", "glab",
    "aws", "az", "gcloud", "gsutil", "bq", "rclone", "s3cmd", "azcopy", "b2", "vercel",
    "netlify", "surge", "heroku", "flyctl", "doctl", "ngrok", "cloudflared", "localtunnel",
}
# Fine for building and downloading, but not with these subcommands.
PUBLISHERS = {
    "git", "npm", "npx", "yarn", "pnpm", "bun", "cargo", "poetry", "flit", "hatch", "twine",
    "gem", "docker", "podman", "dotnet", "nuget", "hf", "huggingface-cli", "ollama", "conda",
    "anaconda", "helm", "wrangler", "firebase", "fly", "mvn", "gradle", "gradlew", "deno",
    "vsce", "ovsx", "eas", "fastlane",
}
PUBLISH_VERBS = {"push", "publish", "upload", "deploy", "send-email", "request-pull", "login"}
INTERPRETERS = {"python", "python3", "py", "node", "deno", "bun", "ruby", "perl", "php",
                "powershell", "pwsh", "bash", "sh", "zsh", "go"}
SCRIPT_EXTENSIONS = {".py", ".js", ".mjs", ".cjs", ".ts", ".rb", ".pl", ".php", ".ps1",
                     ".sh", ".bash", ".zsh", ".bat", ".cmd", ".go"}
SHELL_SCRIPT_EXTENSIONS = {".sh", ".bash", ".zsh", ".bat", ".cmd", ".ps1"}

# Code that opens outbound connections, in the languages a project here
# is likely to be written in.
NETWORK_CODE_PATTERNS = [
    r"\b(import|from)\s+(requests|httpx|aiohttp|urllib3|socket|smtplib|ftplib|paramiko|boto3|"
    r"telnetlib|websockets?|pycurl|xmlrpc\.client|http\.client|urllib\.request)\b",
    r"\burllib\.request\b", r"\burlopen\s*\(",
    r"\bfetch\s*\(", r"\bXMLHttpRequest\b", r"\bsendBeacon\b", r"\bnew\s+WebSocket\b",
    r"\b(require|import)\s*\(\s*['\"](node:)?(https?|net|tls|dgram|node-fetch|axios|got|request|"
    r"undici|ws|nodemailer|ftp|ssh2)['\"]",
    r"\bfrom\s+['\"](node:)?(https?|net|tls|dgram|node-fetch|axios|got|undici|ws|nodemailer|ssh2)['\"]",
    r"(?i)\b(Invoke-WebRequest|Invoke-RestMethod|Start-BitsTransfer|Send-MailMessage)\b",
    r"(?i)\bNet\.WebClient\b", r"(?i)\bSystem\.Net\.(Http|Sockets|WebClient|Mail)\b", r"(?i)\bTcpClient\b",
    r"\bNet::(HTTP|FTP|SMTP)\b", r"\bLWP::", r"\bIO::Socket\b", r"\bcurl_exec\b", r"\bfsockopen\b",
    r"\"net/http\"", r"\"net\"", r"\bjava\.net\.", r"\breqwest::",
]
_NETWORK_CODE_RE = re.compile("|".join(f"(?:{p})" for p in NETWORK_CODE_PATTERNS))
_SEGMENT_SPLIT_RE = re.compile(r"&&|\|\||[;&|\n]")
# Also splits on subshells ($(...), backticks, parentheses) so a program
# hidden inside one is still the first word of its own segment.
_SCREEN_SEGMENT_RE = re.compile(r"&&|\|\||\$\(|[;&|\n`()]")
_WORD_SPLIT_RE = re.compile(r"[\s;&|()<>`'\"]+")
# Words that just run the next word as the real program.
_WRAPPERS = {"sudo", "env", "nohup", "time", "exec", "start", "call", "xargs", "npx", "bunx",
             "pnpx", "cmd", "/c", "/k", "-c", "-command", "-noprofile", "-noninteractive",
             "powershell", "pwsh", "bash", "sh", "zsh", "doas", "nice", "timeout", "watch"}


def _program_name(token: str) -> str:
    name = re.split(r"[\\/]", token.strip("'\""))[-1].lower()
    return re.sub(r"\.(exe|cmd|bat|com|ps1)$", "", name)


def _programs(command: str) -> list[tuple[str, list[str]]]:
    """(program, the words after it) for each part of a chained command —
    only the word actually being run, so a folder called "mail" or "hub"
    isn't mistaken for a mail or GitHub client."""
    found = []
    for segment in _SCREEN_SEGMENT_RE.split(command):
        words = [_program_name(t) for t in _WORD_SPLIT_RE.split(segment) if t]
        while words and (words[0] in _WRAPPERS or re.fullmatch(r"[a-z_][a-z0-9_]*=.*", words[0])):
            words = words[1:]
        if len(words) >= 3 and words[0] in INTERPRETERS and words[1] == "-m":
            words = words[2:]  # python -m twine upload → twine
        if words:
            found.append((words[0], words[1:]))
    return found


def _network_code_in(text: str) -> str | None:
    match = _NETWORK_CODE_RE.search(text)
    if not match:
        return None
    line_no = text.count("\n", 0, match.start()) + 1
    line = text.splitlines()[line_no - 1].strip() if text.splitlines() else match.group(0)
    return f"line {line_no}: {line[:120]}"


def _read_small(path: Path, limit: int = 2 * 1024 * 1024) -> str | None:
    try:
        if path.is_file() and path.stat().st_size <= limit:
            return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    return None


_URL_IN_TEXT_RE = re.compile(r"https?://[^\s'\"<>|)`]+", re.IGNORECASE)


def screen_command(command: str, cwd: Path, private_texts: list[str] = (), chat_texts: list[str] = (),
                   _depth: int = 0) -> str | None:
    """
    Returns why a command is refused, or None if it passes. This is the
    hard no-upload rule for commands — no permission setting skips it.
    Web addresses inside a command (git clone, pip install <url>, start
    <url>...) get the same checks as a fetch-url download.
    """
    if "permissions.json" in command.lower():
        return "Commands can't touch LLMlifeline's permissions file."
    for url in _URL_IN_TEXT_RE.findall(command):
        problem = url_problem(url)
        if problem:
            return f"{problem} ({url[:80]})"
        if url_leak(url, list(private_texts), list(chat_texts)):
            return (f"That command's web address contains text from your chats or files. "
                    f"LLMlifeline never sends your data to websites.")
    all_words = [_program_name(t) for t in _WORD_SPLIT_RE.split(command) if t]
    if any(w in ("powershell", "pwsh") for w in all_words) and any(
            w in ("-e", "-ec", "-enc", "-encodedcommand") for w in all_words):
        return "Encoded PowerShell commands can't be checked, so LLMlifeline won't run them."
    for program, rest in _programs(command):
        if program in UPLOAD_TOOLS:
            return (f"`{program}` can send data off this computer, so LLMlifeline won't run it. "
                    f"To download something, use a fetch-url block instead.")
        if program in PUBLISHERS:
            verb = next((v for v in rest[:3] if v in PUBLISH_VERBS), None)
            if verb:
                return f"`{program} {verb}` uploads to an outside service, so LLMlifeline won't run it."
    words = all_words
    found = _network_code_in(command)
    if found:
        return f"This command contains code that opens a network connection ({found}), so LLMlifeline won't run it."

    if _depth >= 2:
        return None
    # Scripts it's about to run, plus the files beside them that the
    # script could import.
    for token in _WORD_SPLIT_RE.split(command):
        suffix = Path(token).suffix.lower() if token else ""
        if suffix not in SCRIPT_EXTENSIONS:
            continue
        script = (cwd / token).resolve() if not Path(token).is_absolute() else Path(token)
        siblings = [script]
        try:
            siblings += [p for p in script.parent.iterdir() if p.suffix.lower() == suffix and p != script][:200]
        except OSError:
            pass
        for file in siblings:
            text = _read_small(file)
            if text is None:
                continue
            found = _network_code_in(text)
            if found:
                return (f"{file.name} contains code that opens a network connection ({found}). "
                        f"LLMlifeline won't run it — you can run it yourself if you trust it.")
            if file.suffix.lower() in SHELL_SCRIPT_EXTENSIONS:
                for line in text.splitlines():
                    problem = screen_command(line, file.parent, private_texts, chat_texts, _depth + 1)
                    if problem:
                        return f"{file.name}: {problem}"
    # npm/yarn/pnpm/bun scripts run whatever package.json says.
    if words and words[0] in ("npm", "yarn", "pnpm", "bun") and len(words) > 1:
        name = words[2] if words[1] in ("run", "run-script") and len(words) > 2 else words[1]
        package_text = _read_small(cwd / "package.json")
        try:
            scripts = json.loads(package_text).get("scripts", {}) if package_text else {}
        except ValueError:
            scripts = {}
        for key in (f"pre{name}", name, f"post{name}"):
            if isinstance(scripts.get(key), str):
                problem = screen_command(scripts[key], cwd, private_texts, chat_texts, _depth + 1)
                if problem:
                    return f"package.json script \"{key}\": {problem}"
    return None


def _command_prefix(command: str) -> str:
    words = command.split()
    if len(words) >= 2 and words[0].lower() in ("npm", "npx", "yarn", "pnpm", "git", "pip", "pip3",
                                                "cargo", "go", "dotnet", "docker", "uv", "poetry"):
        return " ".join(words[:2])
    if len(words) >= 3 and words[1] == "-m":
        return " ".join(words[:3])
    return words[0] if words else ""


def _matches_rule(command: str, rules: list[str]) -> bool:
    """Every part of a chained command (a && b; c | d) has to match a rule
    on its own, so an allowed prefix can't carry something else along."""
    segments = [" ".join(s.split()) for s in _SEGMENT_SPLIT_RE.split(command) if s.strip()]
    return bool(segments) and all(
        any(seg == rule or seg.startswith(rule + " ") for rule in rules) for seg in segments
    )


# ---- The no-upload screen for web addresses ----

PRIVATE_WINDOW = 20   # characters in a row from your files/command output that can't appear in a URL
CHAT_WINDOW = 32      # ...from what you typed in this chat
DIGIT_RUN = 6         # digit runs this long from either (phone numbers, IDs, card numbers)
MAX_URL_LENGTH = 600
MAX_QUERY_LENGTH = 300
MAX_OPAQUE_PIECE = 64  # one path segment or query value longer than this looks like encoded data


def _normalize_text(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", unquote_plus(text).lower()).split())


def url_problem(url: str) -> str | None:
    """Shape rules for any address LLMlifeline will download from."""
    if len(url) > MAX_URL_LENGTH:
        return "That web address is too long to be a plain download."
    try:
        parts = urlsplit(url)
    except ValueError:
        return "That isn't a valid web address."
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return "Only http:// and https:// web addresses can be downloaded."
    if parts.username or parts.password:
        return "Web addresses with a username or password in them aren't allowed."
    host = parts.hostname.lower()
    if host == "localhost" or host.endswith(".localhost"):
        return "Addresses on this computer itself aren't downloadable."
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_loopback or ip.is_link_local or ip.is_unspecified or ip.is_multicast:
            return "Addresses on this computer itself aren't downloadable."
    except ValueError:
        pass
    if len(parts.query) > MAX_QUERY_LENGTH:
        return "That web address carries too much data in it to be a plain download."
    pieces = [p for p in parts.path.split("/") if p]
    for values in parse_qs(parts.query, keep_blank_values=True).values():
        pieces += values
    for piece in pieces:
        if len(piece) > MAX_OPAQUE_PIECE and " " not in unquote_plus(piece):
            return "Part of that web address looks like encoded data, so LLMlifeline won't send it."
    return None


def url_leak(url: str, private_texts: list[str], chat_texts: list[str]) -> str | None:
    """
    The text a URL would hand a website, if it matches something of
    yours: a run of PRIVATE_WINDOW characters from files/command output,
    CHAT_WINDOW from your messages, or a DIGIT_RUN-digit number from
    either. A URL that appears word-for-word in your own text is fine —
    it's a link you already have, not data being smuggled out.
    """
    bare = url.split("#")[0]
    if any(bare in text for text in private_texts + chat_texts):
        return None
    parts = urlsplit(url)
    target = _normalize_text(f"{parts.path} {parts.query} {parts.fragment}")

    def windows(size: int) -> set[str]:
        return {target[i:i + size] for i in range(len(target) - size + 1)} if len(target) >= size else set()

    for size, texts in ((PRIVATE_WINDOW, private_texts), (CHAT_WINDOW, chat_texts)):
        wanted = windows(size)
        for text in texts:
            normalized = _normalize_text(text)
            for w in wanted:
                if w in normalized:
                    return w
    for digits in re.findall(rf"\d{{{DIGIT_RUN},}}", target.replace(" ", "")):
        if any(digits in re.sub(r"\D", "", t) for t in private_texts + chat_texts):
            return digits
    return None


def _domain_allowed(host: str, domains: list[str]) -> bool:
    host = host.lower().rstrip(".")
    return any(host == d or host.endswith("." + d) for d in domains)


# ---- Decisions ----

def decide(perms: dict, action: dict, workspace: Path, private_texts: list[str],
           chat_texts: list[str]) -> dict:
    """
    {"status": "allow" | "ask" | "deny", "reason": str, and for "ask":
    "title"/"detail" for the approval card plus "remember" — what
    "Always allow" grants — and its "remember_label"}.
    """
    kind = action["type"]
    if kind == "file":
        if action["action"] not in ("create", "read", "delete", "list"):
            return {"status": "deny", "reason": f"Unknown file action '{action['action']}'."}
        if not action.get("path"):
            return {"status": "deny", "reason": "The file action had no path."}
        path = resolve_path(action["path"], workspace)
        write = action["action"] in ("create", "delete")
        status, reason = check_path(perms, path, write, workspace)
        if status != "ask":
            return {"status": status, "reason": reason}
        folder = path if action["action"] == "list" else path.parent
        verb = {"create": "write", "delete": "delete", "read": "read", "list": "look in"}[action["action"]]
        return {
            "status": "ask",
            "title": f"The assistant wants to {verb} a file outside this project" if action["action"] != "list"
                     else "The assistant wants to look in a folder outside this project",
            "detail": str(path),
            "remember": {"folder": {"path": str(folder), "write": write}},
            "remember_label": f"Always allow {'reading and writing' if write else 'reading'} {folder}",
        }

    if kind == "command":
        command = action.get("command", "").strip()
        if not command:
            return {"status": "deny", "reason": "The run-command block had no command."}
        if not perms["enabled"] or perms["commands"] == "off":
            return {"status": "deny", "reason": "Running commands is turned off in Permissions."}
        cwd = resolve_path(action["cwd"], workspace) if action.get("cwd") else workspace.resolve()
        if not cwd.is_dir():
            return {"status": "deny", "reason": f"The folder {cwd} doesn't exist."}
        cwd_status, cwd_reason = check_path(perms, cwd, True, workspace)
        if cwd_status == "deny":
            return {"status": "deny", "reason": cwd_reason}
        problem = screen_command(command, cwd, private_texts, chat_texts)
        if problem:
            return {"status": "deny", "reason": problem}
        if cwd_status == "allow" and (perms["commands"] == "auto" or _matches_rule(command, perms["command_rules"])):
            return {"status": "allow", "reason": None, "cwd": str(cwd)}
        prefix = _command_prefix(command)
        remember = {"command_rule": prefix}
        label = f"Always allow commands starting with `{prefix}`"
        if cwd_status == "ask":
            remember["folder"] = {"path": str(cwd), "write": True}
            label += f" and working in {cwd}"
        return {"status": "ask", "title": "The assistant wants to run a command",
                "detail": command, "cwd": str(cwd), "remember": remember, "remember_label": label}

    url = action.get("url", "").strip().strip("<>")
    action["url"] = url
    if not perms["enabled"] or perms["internet"] == "off":
        return {"status": "deny", "reason": "Internet access is turned off in Permissions."}
    problem = url_problem(url)
    if problem:
        return {"status": "deny", "reason": problem}
    leak = url_leak(url, private_texts, chat_texts)
    if leak:
        return {"status": "deny", "reason": (
            f"Refused: that web address contains text from your chats or files (\"{leak}\"). "
            f"LLMlifeline never sends your data to websites.")}
    save_status = "allow"
    if action.get("save"):
        save_status, save_reason = check_path(perms, resolve_path(action["save"], workspace), True, workspace)
        if save_status == "deny":
            return {"status": "deny", "reason": save_reason}
    host = urlsplit(url).hostname.lower()
    if save_status == "allow" and _domain_allowed(host, perms["domains"]):
        return {"status": "allow", "reason": None}
    domain = _normalize_domain(host) or host
    remember = {"domain": domain}
    if save_status == "ask":
        save_folder = resolve_path(action["save"], workspace).parent
        remember["folder"] = {"path": str(save_folder), "write": True}
    return {"status": "ask", "title": f"The assistant wants to download from {host}",
            "detail": url + (f"\n→ save to {resolve_path(action['save'], workspace)}" if action.get("save") else ""),
            "remember": remember, "remember_label": f"Always allow downloads from {domain}"}


# ---- Doing the work ----

def _clip(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n[... {len(text) - limit} more characters not shown]"


def run_file_action(action: dict, workspace: Path) -> str:
    path = resolve_path(action["path"], workspace)
    shown = action["path"]
    try:
        if action["action"] == "create":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(action.get("content", ""), encoding="utf-8")
            return f"✅ Created file: `{shown}`"
        if action["action"] == "delete":
            if not path.exists():
                return f"⚠️ Could not delete `{shown}`: file not found."
            if not path.is_file():
                return f"⚠️ Could not delete `{shown}`: it's a directory, not a file."
            path.unlink()
            return f"🗑️ Deleted file: `{shown}`"
        if action["action"] == "read":
            if not path.exists():
                return f"[File not found: {shown}]"
            if not path.is_file():
                return f"[{shown} is a directory, not a file]"
            return f"[Contents of {shown}]\n{_clip(path.read_text(encoding='utf-8', errors='replace'))}"
        if not path.is_dir():
            return f"[Folder not found: {shown}]"
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        lines = [f"{p.name}/" if p.is_dir() else p.name for p in entries[:300]]
        more = f"\n[... {len(entries) - 300} more]" if len(entries) > 300 else ""
        return f"[Folder {shown}]\n" + ("\n".join(lines) or "(empty)") + more
    except OSError as e:
        return f"⚠️ Could not {action['action']} `{shown}`: {e}"


def _run_command_sync(command: str, cwd: str) -> str:
    env = {**os.environ, **NO_TELEMETRY_ENV}
    try:
        result = subprocess.run(
            command, shell=True, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=COMMAND_TIMEOUT_SECONDS,
            encoding="utf-8", errors="replace",
        )
        output = result.stdout.strip() or "(no output)"
        return f"[Ran `{command}` in {cwd} — exit code {result.returncode}]\n{_clip(output)}"
    except subprocess.TimeoutExpired as e:
        partial = (e.stdout or "") if isinstance(e.stdout, str) else ""
        return (f"[`{command}` was stopped after {COMMAND_TIMEOUT_SECONDS // 60} minutes]\n"
                f"{_clip(partial.strip()) or '(no output)'}")
    except OSError as e:
        return f"⚠️ Could not run `{command}`: {e}"


async def run_command(command: str, cwd: str) -> str:
    # A worker thread rather than asyncio subprocesses: those need the
    # Proactor event loop on Windows, which uvicorn doesn't always use.
    return await asyncio.to_thread(_run_command_sync, command, cwd)


class _PageText(HTMLParser):
    """Visible text and links of an HTML page — enough for the model to
    read docs and search results without the markup."""

    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.text: list[str] = []
        self.links: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "svg"):
            self._skip += 1
        if tag == "a":
            href = dict(attrs).get("href")
            if href and not href.startswith(("#", "javascript:", "mailto:")):
                link = urljoin(self.base_url, href)
                # DuckDuckGo's results point at its own redirector;
                # unwrap to the real page.
                parts = urlsplit(link)
                if parts.hostname and parts.hostname.endswith("duckduckgo.com") and parts.path.startswith("/l/"):
                    link = parse_qs(parts.query).get("uddg", [link])[0]
                if link not in self.links:
                    self.links.append(link)
        if tag in ("p", "br", "div", "li", "h1", "h2", "h3", "h4", "tr", "pre"):
            self.text.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.text.append(data)


async def _resolves_outside_this_computer(host: str) -> bool:
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None)
    except socket.gaierror:
        return True  # let the request itself report the DNS failure
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if ip.is_loopback or ip.is_link_local or ip.is_unspecified or ip.is_multicast:
            return False
    return True


async def fetch_url(url: str, approved_host: str | None, perms: dict, private_texts: list[str],
                    chat_texts: list[str], save_path: Path | None) -> str:
    """
    Download-only GET. No body, no cookies kept between hops, no proxy or
    .netrc credentials (trust_env=False), and every redirect is checked
    by the same rules as the first address — a redirect to a site you
    haven't allowed stops there.
    """
    current = url
    async with httpx.AsyncClient(follow_redirects=False, trust_env=False, timeout=30.0,
                                 headers={"User-Agent": USER_AGENT, "Accept": "*/*"}) as client:
        for _hop in range(6):
            host = urlsplit(current).hostname or ""
            problem = url_problem(current) or (url_leak(current, private_texts, chat_texts) and
                                               "that address contains text from your chats or files")
            if problem:
                return f"⚠️ Stopped before downloading {current}: {problem}"
            if host.lower() != (approved_host or "").lower() and not _domain_allowed(host, perms["domains"]):
                return (f"⚠️ {url} redirected to {current}, a site that isn't on your allowed list. "
                        f"Ask the assistant to fetch it directly if you want it.")
            if not await _resolves_outside_this_computer(host):
                return f"⚠️ Stopped: {host} points at this computer itself."
            client.cookies.clear()
            try:
                async with client.stream("GET", current) as response:
                    if response.is_redirect and response.headers.get("location"):
                        current = urljoin(current, response.headers["location"])
                        continue
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body += chunk
                        if len(body) > MAX_DOWNLOAD_BYTES:
                            return f"⚠️ {current} is larger than {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB — not downloaded."
                    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
                    status = response.status_code
            except httpx.HTTPError as e:
                return f"⚠️ Could not download {current}: {e}"
            header = f"[Downloaded {current} — HTTP {status}, {content_type or 'unknown type'}, {len(body)} bytes]"
            if save_path is not None:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_path.write_bytes(bytes(body))
                return f"{header}\nSaved to {save_path}"
            if content_type == "text/html" or (not content_type and body[:200].lstrip().startswith(b"<")):
                page = _PageText(current)
                page.feed(body.decode("utf-8", errors="replace"))
                text = re.sub(r"\n\s*\n+", "\n\n", "".join(page.text)).strip()
                links = "\n".join(page.links[:30])
                if host.lower().endswith("duckduckgo.com"):
                    # A results page is mostly menus; the links are what matter.
                    return f"{header}\nResults:\n{links}\n\n{_clip(text, 1500)}"
                return f"{header}\n{_clip(text)}" + (f"\n\nLinks on the page:\n{links}" if links else "")
            if content_type.startswith("text/") or content_type in ("application/json", "application/xml",
                                                                     "application/javascript"):
                return f"{header}\n{_clip(body.decode('utf-8', errors='replace'))}"
            return f"{header}\n(Not text — add a save: line to the fetch-url block to save it as a file.)"
    return f"⚠️ {url} redirected too many times."


async def execute(action: dict, decision: dict, perms: dict, workspace: Path,
                  private_texts: list[str], chat_texts: list[str]) -> str:
    """Carries out an action that's been allowed (by a rule or by the
    person on its approval card)."""
    if action["type"] == "file":
        return run_file_action(action, workspace)
    if action["type"] == "command":
        cwd = decision.get("cwd") or (str(resolve_path(action["cwd"], workspace)) if action.get("cwd")
                                      else str(workspace.resolve()))
        return await run_command(action["command"], cwd)
    save_path = resolve_path(action["save"], workspace) if action.get("save") else None
    return await fetch_url(action["url"], urlsplit(action["url"]).hostname, perms,
                           private_texts, chat_texts, save_path)


# ---- What the model is told ----

def tool_instructions(perms: dict, base_instructions: str) -> str:
    if not perms["enabled"]:
        return base_instructions
    if sys.platform == "win32":
        shell = "Windows cmd.exe"
    elif sys.platform == "darwin":
        shell = "macOS /bin/sh"
    else:
        shell = "Linux /bin/sh"
    folders = "\n".join(f"- {f['path']} ({'read and write' if f['write'] else 'read only'})"
                        for f in perms["folders"]) or "- none yet (you can still ask; the person approves)"
    parts = [
        base_instructions,
        "You can also work on this computer to build things for the person. Write one or more "
        "action blocks; each is carried out and you'll be given the results. Some need the "
        "person's approval first. When the task is done, reply normally with no block.",
        "For files outside this project, use the file-action block with a full path. "
        f"Folders the person has shared:\n{folders}\n"
        f"To list a folder:\n```file-action\naction: list\npath: {EXAMPLE_FOLDER}\n```",
    ]
    if perms["commands"] != "off":
        parts.append(
            f"To run a command ({shell}; it runs in this project's files folder unless you set cwd):\n"
            f"```run-command\ncwd: {EXAMPLE_FOLDER}\ncommand: npm run build\n```\n"
            "Don't use curl, wget, ssh, git push or other upload tools — they're blocked."
        )
    if perms["internet"] != "off":
        parts.append(
            "To download a web page or file (download only — nothing can be sent):\n"
            "```fetch-url\nurl: https://example.com/page\n```\n"
            "Add a line `save: path/to/file` to save the download as a file instead. "
            "To search the web, fetch https://html.duckduckgo.com/html/?q=a+few+search+words"
        )
    parts.append(
        "Never put the person's private information, file contents or chat text into a web "
        "address or command. LLMlifeline blocks it."
    )
    return "\n\n".join(parts)
