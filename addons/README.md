# Engine add-ons

LLMlifeline runs `.gguf` chat models itself, with llama.cpp. An **engine add-on** lets it also run models that another program serves: a different inference engine, a model too big for memory that streams from disk, or anything else that can answer chat requests. Add-on models appear in the model dropdown under the add-on's own heading, and you switch to them like any `.gguf`. Projects, persona, beliefs, file tools, computer use, Image Model and server mode all work the same with them.

**Only install add-ons you trust.** An add-on is Python code that LLMlifeline runs, and it starts programs on your computer.

## Installing one

Put the add-on's folder here, as `addons/<name>/`, with its `addon.py` directly inside. (A shortcut-style directory junction or symlink to a folder elsewhere works too.) Restart LLMlifeline. If an add-on fails to load, the backend window says why and everything else keeps working.

## Writing one

`addon.py` defines two functions and, optionally, a name:

```python
NAME = "My engine"   # the heading in the model dropdown (default: the folder name)


def models():
    """The models this add-on can run. Called whenever the model list is shown,
    so keep it quick (cache anything slow)."""
    return [
        {"id": "big-model", "name": "Big Model 400B", "size_gb": 210, "note": "streams from disk"},
    ]


def launch(model_id, port):
    """How to start the program that serves model_id."""
    return {
        "args": ["C:/my-engine/server.exe", "--model", model_id, "--port", str(port)],
        "cwd": "C:/my-engine",          # optional
        "env": {"MY_ENGINE_THREADS": "8"},  # optional, added to LLMlifeline's own environment
        "context_length": 16384,        # tokens the model can see; long chats are trimmed to fit
        "ready_path": "/health",        # answers 200 only once the model can chat
        "startup_timeout": 900,         # seconds to wait for ready_path
        "request_timeout": 3600,        # seconds the engine may go quiet: before the first word, or between pieces
        "model_titles": False,          # True: also ask the model to title each conversation
        "tokenize_path": None,          # a llama.cpp-style /extras/tokenize/count, if it has one
        "request_options": {"temperature": 0.7},  # added to every chat request
        "stream": True,                 # False if the engine can't send a reply as it's written
    }
```

The program it starts must:

- listen on `127.0.0.1` at the `port` it is given (the same port LLMlifeline's own llama.cpp server uses, since only one chat model runs at a time);
- answer OpenAI-style `POST /v1/chat/completions` with `messages` and `max_tokens`, returning `choices[0].message.content` and `finish_reason` (`"length"` when cut off by `max_tokens`, so LLMlifeline can ask it to continue);
- with `"stream": true` in the request, send the reply as it's written: server-sent events (`data: {...}` lines) whose `choices[0].delta.content` carry the text, the last one with its `finish_reason`, then `data: [DONE]`. A stream that ends without a `finish_reason` counts as cut off, and the reply is reported as an error instead of being saved. An engine that answers a streamed request with ordinary JSON still works (the reply then appears all at once); one that rejects `stream` needs `"stream": False`;
- answer `GET <ready_path>` with status 200 once it is ready.

LLMlifeline writes the program's output to `llama_server.log`. Stopping the model, switching models or switching to Image Model stops the program **and every process it started**.

Everything in `launch()` other than `args` is optional. The defaults are the values for llama.cpp: a 4,096-token context, `/v1/models` as the ready path, 10 minutes to start, 15 minutes of silence allowed per reply, model-written conversation titles, no extra request fields and streamed replies. Without a `tokenize_path`, LLMlifeline estimates token counts (about 3 characters per token) when it trims long chats.
