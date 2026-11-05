"""
How much of a .gguf goes on the graphics card (backend/gpu_fit.py): its
layer sizes read from the file, the number of layers that fit in the cards'
free memory, what llama.cpp's log says it really loaded, and the figure
remembered per model. Uses small .gguf files written here; no GPU needed.

    llm-env\\Scripts\\python.exe tests\\test_gpu_fit.py     (Windows)
    llm-env/bin/python tests/test_gpu_fit.py               (macOS/Linux)
"""

import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import gpu_fit  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="llmlifeline-gpu-fit-test-"))
MIB = gpu_fit.MIB


def _string(text: str) -> bytes:
    data = text.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _value(value) -> bytes:
    if isinstance(value, str):
        return struct.pack("<I", 8) + _string(value)
    if isinstance(value, list) and value and isinstance(value[0], str):
        return struct.pack("<IIQ", 9, 8, len(value)) + b"".join(_string(v) for v in value)
    if isinstance(value, list):
        return struct.pack("<IIQ", 9, 5, len(value)) + b"".join(struct.pack("<i", v) for v in value)
    return struct.pack("<II", 4, value)


def write_gguf(path: Path, settings: dict, tensors: list[tuple[str, int]], alignment: int = 32) -> Path:
    """A .gguf with these settings and tensors of these sizes (zero-filled)."""
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(settings))
    header += b"".join(_string(key) + _value(value) for key, value in settings.items())
    offset, infos, data = 0, b"", b""
    for name, size in tensors:
        infos += _string(name) + struct.pack("<IQIQ", 1, size, 0, offset)
        padded = -(-size // alignment) * alignment
        data += b"\0" * padded
        offset += padded
    body = header + infos
    body += b"\0" * (-len(body) % alignment)
    path.write_bytes(body + data)
    return path


def model_settings(blocks: int = 4, **extra) -> dict:
    return {"general.architecture": "llama", "llama.block_count": blocks, "llama.embedding_length": 512,
            "llama.attention.head_count": 8, "llama.attention.head_count_kv": 2,
            "tokenizer.ggml.tokens": [f"t{i}" for i in range(100)],  # long enough to be skipped
            "tokenizer.ggml.token_type": [1] * 100, **extra}


LAYER = 64 * 1024  # bytes of each tensor below; a layer has two
TENSORS = ([("token_embd.weight", 3 * LAYER)] + [(f"blk.{i}.{part}", LAYER) for i in range(4)
                                                 for part in ("attn_q.weight", "ffn_up.weight")]
           + [("output_norm.weight", 32), ("output.weight", 3 * LAYER)])


def test_layer_sizes_come_from_the_file():
    layout = gpu_fit.read_layout(write_gguf(TMP / "model.gguf", model_settings(), TENSORS))
    assert layout.block_bytes == [2 * LAYER] * 4
    assert layout.output_bytes == 3 * LAYER + 32  # padding between tensors counts toward the one before
    assert layout.kv_bytes_per_token == [2 * (64 + 64) * 2] * 4  # 2 KV heads of 64 dims, K and V, f16
    assert layout.vocab_size == 100


def test_shared_embeddings_count_as_the_output_layer():
    tied = [t for t in TENSORS if t[0] != "output.weight"]
    layout = gpu_fit.read_layout(write_gguf(TMP / "tied.gguf", model_settings(), tied))
    assert layout.output_bytes == 3 * LAYER + 32  # llama.cpp copies the embeddings to the card


def test_hybrid_models_only_keep_kv_for_attention_layers():
    settings = model_settings(**{"llama.full_attention_interval": 2, "llama.attention.key_length": 256,
                                 "llama.attention.value_length": 256})
    layout = gpu_fit.read_layout(write_gguf(TMP / "hybrid.gguf", settings, TENSORS))
    assert layout.kv_bytes_per_token == [0, 2048, 0, 2048]


def test_a_model_in_parts_is_read_whole():
    first, second = TENSORS[:5], TENSORS[5:]
    write_gguf(TMP / "split-00001-of-00002.gguf", model_settings(), first)
    write_gguf(TMP / "split-00002-of-00002.gguf", {}, second)
    layout = gpu_fit.read_layout(TMP / "split-00001-of-00002.gguf")
    assert layout.block_bytes == [2 * LAYER] * 4 and layout.output_bytes == 3 * LAYER + 32


def test_unreadable_files_give_no_layout():
    (TMP / "junk.gguf").write_bytes(b"not a model")
    assert gpu_fit.read_layout(TMP / "junk.gguf") is None
    assert gpu_fit.read_layout(TMP / "missing.gguf") is None
    assert gpu_fit.read_layout(write_gguf(TMP / "no-arch.gguf", {"x": 1}, TENSORS)) is None


def layout(blocks: int = 10, block_mib: int = 100, output_mib: int = 50) -> gpu_fit.Layout:
    return gpu_fit.Layout([block_mib * MIB] * blocks, output_mib * MIB, [0] * blocks, 1000)


def free_for(weights_mib: float, buffers: float = 0, cards: int = 1) -> int:
    """Free memory that holds exactly weights_mib plus the overheads."""
    return int(weights_mib + buffers + (gpu_fit.CUDA_CONTEXT_MIB + gpu_fit.SAFETY_MARGIN_MIB) * cards)


def test_a_model_that_fits_goes_entirely_on_the_card():
    assert gpu_fit.gpu_layers(layout(), [free_for(1050)], 4096, buffers_mib=0) == -1


def test_a_bigger_model_puts_the_output_layer_and_its_last_layers_on_the_card():
    # llama.cpp's k layers are the output layer and the last k - 1: 50 + 3 x 100.
    assert gpu_fit.gpu_layers(layout(), [free_for(350)], 4096, buffers_mib=0) == 4
    assert gpu_fit.gpu_layers(layout(), [free_for(349)], 4096, buffers_mib=0) == 3
    assert gpu_fit.gpu_layers(layout(), [free_for(1049)], 4096, buffers_mib=0) == 10


def test_buffers_and_kv_cache_take_their_share():
    assert gpu_fit.gpu_layers(layout(), [free_for(350, buffers=300)], 4096, buffers_mib=300) == 4
    assert gpu_fit.gpu_layers(layout(), [free_for(350)], 4096, buffers_mib=300) == 1
    kv = gpu_fit.Layout([100 * MIB] * 10, 50 * MIB, [1024] * 10, 1000)  # 4 MiB per layer at 4096
    assert gpu_fit.gpu_layers(kv, [free_for(350 + 12)], 4096, buffers_mib=0) == 4
    assert gpu_fit.gpu_layers(kv, [free_for(350)], 4096, buffers_mib=0) == 3


def test_two_cards_add_up():
    assert gpu_fit.gpu_layers(layout(), [free_for(525), free_for(525)], 4096, buffers_mib=0) == -1


def test_nothing_fits_or_nothing_to_measure():
    assert gpu_fit.gpu_layers(layout(), [100], 4096) == 0
    assert gpu_fit.gpu_layers(layout(), [], 4096) == -1  # no NVIDIA card: as before
    assert gpu_fit.gpu_layers(None, [10000], 4096) == -1  # unreadable file: as before


LOG = """load_tensors: offloaded 32/65 layers to GPU
load_tensors:   CPU_Mapped model buffer size =  7934.64 MiB
load_tensors:        CUDA0 model buffer size =  7835.71 MiB
llama_kv_cache:      CUDA0 KV buffer size =   128.00 MiB
llama_memory_recurrent:      CUDA0 RS buffer size =    71.70 MiB
sched_reserve:      CUDA0 compute buffer size =   509.01 MiB
sched_reserve:  CUDA_Host compute buffer size =    42.13 MiB
sched_reserve:      CUDA1 compute buffer size =   100.00 MiB
"""


def test_what_was_loaded_is_read_from_the_log():
    loaded = gpu_fit.loaded_buffers(LOG)
    assert loaded == {"model": 7835.71, "KV": 128.0, "RS": 71.7, "compute": 609.01}
    assert gpu_fit.loaded_buffers("CPU only") == {}


def test_each_models_buffers_are_remembered_per_context():
    model = write_gguf(TMP / "remembered.gguf", model_settings(), TENSORS)
    gpu_fit.MEMORY_FILE = None
    gpu_fit.remember_buffers(str(model), 4096, 600)  # nowhere to keep it: no error
    assert gpu_fit.remembered_buffers(str(model), 4096) is None
    gpu_fit.MEMORY_FILE = TMP / "gpu-fit.json"
    gpu_fit.remember_buffers(str(model), 4096, 580.71)
    assert gpu_fit.remembered_buffers(str(model), 4096) == 580.7
    assert gpu_fit.remembered_buffers(str(model), 8192) is None
    gpu_fit.MEMORY_FILE.write_text("{broken", encoding="utf-8")
    assert gpu_fit.remembered_buffers(str(model), 4096) is None


if __name__ == "__main__":
    failures = 0
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            try:
                test()
                print(f"ok    {name}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{failures} failed" if failures else "\nall passed")
    sys.exit(1 if failures else 0)
