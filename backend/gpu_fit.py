"""
How much of a .gguf model fits on this computer's graphics cards.

llama_cpp.server is told how many of a model's layers to put on the GPU.
Asking for all of them (-1) is right when the model fits, but on Windows a
model bigger than the card doesn't fail to load: the NVIDIA driver quietly
moves the overflow into shared system memory, which the card then reads over
PCIe. Measured on an 11 GB RTX 2080 Ti: Qwen3-30B-A3B Q4 (17.5 GB) wrote
5.5 tokens/s that way and 17-18 with only the layers that fit; a dense 27B
Q4 0.6 against 2.8. So before each launch the backend reads the model's
layer sizes from its file and the cards' free memory (main._gpu_memory_mb)
and asks for as many layers as fit, every layer when the whole model does.

The buffers llama.cpp adds on the card (compute scratch, a hybrid model's
recurrent state) depend on the model and can only be guessed before it
loads. Its log states them once it has, so the backend checks that log after
every load (see LlamaServerManager.refit), reloads once if the guess was
wrong either way, and remembers the real figure for that model on this
computer in gpu-fit.json, so the next load gets it right the first time.

Only the file's header is read: names, shapes and positions of its tensors
and the few settings that size the KV cache. Tensor sizes come from the gaps
between their positions, so no table of quantization formats is needed.
"""

import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

MIB = 1024 * 1024

# Compute and recurrent-state buffers before a model's own figure is known.
# Measured with llama-cpp-python 0.3.35 (CUDA): 301 MiB for Qwen3-30B-A3B,
# 505-509 MiB plus 72-150 MiB of recurrent state for a 27B hybrid model.
BUFFERS_GUESS_MIB = 512
# The CUDA context on each card, which llama.cpp's log doesn't report:
# 30-165 MiB beyond the logged buffers on an RTX 2080 Ti.
CUDA_CONTEXT_MIB = 128
# Left free so a browser or the desktop growing a little doesn't push the
# model into shared memory after it loads. Measured with gemma-2-9b Q8:
# 42 of 43 layers left ~250 MiB free and wrote 24.3 tokens/s; all 43
# spilled ~0.5 GB and wrote 17.3; 40 wrote 21.9.
SAFETY_MARGIN_MIB = 192

MEMORY_FILE: Path | None = None  # set by main: APP_ROOT / "gpu-fit.json"

_LOADED = re.compile(r"CUDA(\d+) (model|KV|RS|compute) buffer size\s*=\s*([\d.]+) MiB")

_SPLIT_PART = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
_BLOCK = re.compile(r"blk\.(\d+)\.")

# GGUF value types: struct format and size of the fixed-size ones.
_FIXED = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9


@dataclass
class Layout:
    block_bytes: list[int]   # weights of each repeating layer, in order
    output_bytes: int        # the output layer (a copy of the token embeddings when they're shared)
    kv_bytes_per_token: list[int]  # each layer's f16 KV cache per token of context
    vocab_size: int

    @property
    def block_count(self) -> int:
        return len(self.block_bytes)


class _Reader:
    def __init__(self, f):
        self.f = f

    def unpack(self, fmt: str):
        size = struct.calcsize(fmt)
        data = self.f.read(size)
        if len(data) != size:
            raise ValueError("file ends inside its header")
        return struct.unpack(fmt, data)[0]

    def string(self) -> str:
        length = self.unpack("<Q")
        if length > 1 << 24:
            raise ValueError("implausible string length")
        return self.f.read(length).decode("utf-8", "replace")

    def value(self, kind: int):
        if kind in _FIXED:
            return self.unpack(_FIXED[kind])
        if kind == _STRING:
            return self.string()
        if kind == _ARRAY:
            item_kind, count = self.unpack("<I"), self.unpack("<Q")
            if item_kind in _FIXED and count > 64:
                # Long numeric arrays (token types, scores) aren't needed: skip them.
                self.f.seek(count * struct.calcsize(_FIXED[item_kind]), 1)
                return _Skipped(count)
            if item_kind == _STRING and count > 64:
                for _ in range(count):  # the vocabulary: only its length matters here
                    self.f.seek(self.unpack("<Q"), 1)
                return _Skipped(count)
            return [self.value(item_kind) for _ in range(count)]
        raise ValueError(f"unknown GGUF value type {kind}")


class _Skipped:
    """A long array that was read past; only its length is kept."""

    def __init__(self, count: int):
        self.count = count

    def __len__(self) -> int:
        return self.count


def _read_part(path: Path) -> tuple[dict, dict[str, int]]:
    """One .gguf file's settings and the size in bytes of each of its tensors."""
    with open(path, "rb") as f:
        r = _Reader(f)
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        if r.unpack("<I") < 2:
            raise ValueError("GGUF version 1 isn't supported")
        tensor_count, kv_count = r.unpack("<Q"), r.unpack("<Q")
        settings = {}
        for _ in range(kv_count):
            key = r.string()
            settings[key] = r.value(r.unpack("<I"))
        offsets = []
        for _ in range(tensor_count):
            name = r.string()
            dims = r.unpack("<I")
            f.seek(8 * dims, 1)
            r.unpack("<I")  # its quantization type: the size comes from the offsets instead
            offsets.append((r.unpack("<Q"), name))
        alignment = settings.get("general.alignment", 32)
        alignment = alignment if isinstance(alignment, int) and alignment > 0 else 32
        data_start = -(-f.tell() // alignment) * alignment
    data_size = path.stat().st_size - data_start
    offsets.sort()
    sizes = {}
    for i, (offset, name) in enumerate(offsets):
        end = offsets[i + 1][0] if i + 1 < len(offsets) else data_size
        sizes[name] = max(0, end - offset)
    return settings, sizes


def _parts(path: Path) -> list[Path]:
    """Every file of a model split into parts (named -00001-of-0000N.gguf)."""
    m = _SPLIT_PART.search(path.name)
    if not m:
        return [path]
    total = int(m.group(2))
    return [path.with_name(path.name[:m.start()] + f"-{i:05d}-of-{total:05d}.gguf") for i in range(1, total + 1)]


def read_layout(model_path: str | Path) -> Layout | None:
    """The layer sizes of a .gguf model, or None if the file can't be read."""
    try:
        settings, sizes = {}, {}
        for part in _parts(Path(model_path)):
            part_settings, part_sizes = _read_part(part)
            settings = settings or part_settings  # the first part holds the model's settings
            sizes.update(part_sizes)
        arch = settings["general.architecture"]
        count = int(settings[f"{arch}.block_count"])
        if count <= 0:
            return None
        blocks = [0] * count
        for name, size in sizes.items():
            m = _BLOCK.match(name)
            if m and int(m.group(1)) < count:  # extra blocks (draft heads) aren't loaded as layers
                blocks[int(m.group(1))] += size
        output = sizes.get("output.weight", sizes.get("token_embd.weight", 0)) + sizes.get("output_norm.weight", 0)

        def per_layer(key, default):
            value = settings.get(f"{arch}.{key}", default)
            return [int(v) for v in value] if isinstance(value, list) else [int(value)] * count

        heads = per_layer("attention.head_count", 1)
        kv_heads = per_layer("attention.head_count_kv", heads[0])
        embedding = int(settings.get(f"{arch}.embedding_length", 0))
        head_dim = embedding // max(heads[0], 1)
        key_len = int(settings.get(f"{arch}.attention.key_length", head_dim))
        value_len = int(settings.get(f"{arch}.attention.value_length", head_dim))
        interval = int(settings.get(f"{arch}.full_attention_interval", 1) or 1)
        kv = [kv_heads[i] * (key_len + value_len) * 2 if (i + 1) % interval == 0 else 0 for i in range(count)]
        vocab = settings.get("tokenizer.ggml.tokens")
        vocab_size = len(vocab) if vocab is not None else int(settings.get(f"{arch}.vocab_size", 0))
        return Layout(blocks, output, kv, vocab_size)
    except (OSError, ValueError, KeyError, TypeError, struct.error):
        return None


def gpu_layers(layout: Layout | None, free_mib: list[int], context_length: int,
               buffers_mib: float = BUFFERS_GUESS_MIB) -> int:
    """
    The --n_gpu_layers to launch llama_cpp.server with: -1 (every layer)
    when the model fits in the cards' free memory, else how many do, 0 when
    none do. -1 too when there's no NVIDIA card to measure (on a Mac the GPU
    shares the computer's memory) or the file can't be read, which is what
    every load did before this.

    llama.cpp counts the output layer as the last of block_count + 1 and
    offloads from the end, so k layers are the output layer and the last
    k - 1 repeating ones (its log: "offloaded 32/65", layers 33-64).
    """
    if layout is None or not free_mib:
        return -1
    budget = (sum(free_mib) - buffers_mib - (CUDA_CONTEXT_MIB + SAFETY_MARGIN_MIB) * len(free_mib)) * MIB

    def need(k: int) -> int:
        layers = range(layout.block_count - (k - 1), layout.block_count)
        return layout.output_bytes + sum(layout.block_bytes[i] + layout.kv_bytes_per_token[i] * context_length
                                         for i in layers)

    if need(layout.block_count + 1) <= budget:
        return -1
    for k in range(layout.block_count, 0, -1):
        if need(k) <= budget:
            return k
    return 0


def loaded_buffers(log_text: str) -> dict[str, float]:
    """What llama.cpp put on the cards, in MiB, from its load log: the
    model's weights, KV cache, recurrent state and compute scratch."""
    totals: dict[str, float] = {}
    for _, kind, mib in _LOADED.findall(log_text):
        totals[kind] = totals.get(kind, 0.0) + float(mib)
    return totals


def _memory_key(model_path: str, context_length: int) -> str:
    path = Path(model_path)
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return f"{path.name}|{size}|{context_length}"


def _memory() -> dict:
    try:
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8")) if MEMORY_FILE else {}
    except (OSError, ValueError):
        return {}


def remembered_buffers(model_path: str, context_length: int) -> float | None:
    """The compute and recurrent buffers this model took the last time it
    loaded on this computer with this context, if it has."""
    value = _memory().get(_memory_key(model_path, context_length))
    return float(value) if isinstance(value, (int, float)) else None


def remember_buffers(model_path: str, context_length: int, mib: float) -> None:
    if MEMORY_FILE is None:
        return
    memory = _memory()
    memory[_memory_key(model_path, context_length)] = round(mib, 1)
    try:
        MEMORY_FILE.write_text(json.dumps(memory, indent=1), encoding="utf-8")
    except OSError:
        pass  # only costs a reload next time
