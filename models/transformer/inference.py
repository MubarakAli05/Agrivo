"""Deterministic CPU decoding from verified synthetic-baseline checkpoints.

Raw generated text is deliberately not presented as an agricultural answer.
Retrieval/answer validation must independently decide whether evidence supports
an answer. Greedy decoding masks non-text control IDs; EOS remains allowed.
Incomplete or invalid UTF-8 is replaced, preserving all valid surrounding text.
"""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from agri.config import load_config
from tokenizer.tokenizer import BPETokenizer, SPECIAL_TOKENS
from tokenizer.train_tokenizer import tokenizer_path
from training.trainer import CAVEATS, checkpoint_path, cpu_session, load_checkpoint


MAX_NEW_TOKENS = 128
_FIELDS = ("input_ids", "structured_ids", "numeric_features", "numeric_values", "attention_mask")


def _budget(max_new_tokens: int) -> None:
    if type(max_new_tokens) is not int or not 0 <= max_new_tokens <= MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must be an integer between 0 and {MAX_NEW_TOKENS}")


def _decode_bytes(text: BPETokenizer, ids: list[int]) -> tuple[str, bool]:
    """Use the public byte vocabulary rather than decoding tokens individually."""
    vocabulary = json.loads(text.artifacts()["vocab.json"])
    byte_tokens = {index: bytes.fromhex(value) for value, index in vocabulary["byte_tokens"].items()}
    chunks = []
    for token in ids:
        if type(token) is not int or not 0 <= token < text.vocab_size:
            raise ValueError("Generated ID is outside the actual tokenizer vocabulary")
        if token in byte_tokens:
            chunks.append(byte_tokens[token])
        elif token != text.token_to_id("<eos>"):
            raise ValueError("Unexpected generated non-text control ID")
    raw = b"".join(chunks)
    try:
        return raw.decode("utf-8"), False
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace"), True


def _empty_channels(ids: list[int]) -> dict[str, Any]:
    return {"input_ids": ids, "structured_ids": [0] * len(ids),
            "numeric_features": [-1] * len(ids), "numeric_values": [[0.0] * 7 for _ in ids],
            "attention_mask": [True] * len(ids)}


def _generate(root: Path, text: BPETokenizer, prefix: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
    import torch

    config = load_config(root / "configs" / "agri-mini.json")
    count = len(prefix["input_ids"])
    context_length = config["model"]["context_length"]
    if not count or count > context_length:
        raise ValueError("Prompt must fit context_length; silent truncation is forbidden")
    if any(not isinstance(prefix.get(name), list) or len(prefix[name]) != count for name in _FIELDS):
        raise ValueError("Packed input channels must have identical lengths")
    for name in ("input_ids", "structured_ids", "numeric_features"):
        if any(type(token) is not int for token in prefix[name]):
            raise ValueError(f"{name} must contain integer IDs, not booleans")
    if any(value is not True for value in prefix["attention_mask"]):
        raise ValueError("Generation requires an intact unpadded prompt")
    state = deepcopy(prefix)
    generated: list[int] = []
    capacity = min(max_new_tokens, context_length - count)
    stop_reason = "context_limit" if capacity < max_new_tokens else "max_new_tokens"
    with cpu_session(config["seed"]):
        model, report = load_checkpoint(root)
        if model.config.vocab_size != text.vocab_size:
            raise ValueError("Checkpoint actual vocabulary disagrees with tokenizer")
        eos = text.token_to_id("<eos>")
        with torch.no_grad():
            for _ in range(capacity):
                batch = {name: torch.tensor([state[name]], dtype=(torch.float32 if name == "numeric_values" else
                         torch.bool if name == "attention_mask" else torch.long)) for name in _FIELDS}
                logits = model(**batch)[0, -1]
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("Nonfinite generation logits")
                scores = logits.clone()
                scores[:len(SPECIAL_TOKENS)] = float("-inf")
                scores[eos] = logits[eos]
                token = int(scores.argmax().item())
                generated.append(token)
                if token == eos:
                    stop_reason = "eos"
                    break
                state["input_ids"].append(token)
                state["structured_ids"].append(0)
                state["numeric_features"].append(-1)
                state["numeric_values"].append([0.0] * 7)
                state["attention_mask"].append(True)
    output, invalid = _decode_bytes(text, generated)
    return {"phase": 7, "status": "GREEN", "text": output,
            "model": report["model"], "training": report["training"],
            "source_status": report["source_status"],
            "next_required_step": "Integrate retrieved evidence and independently validate supported answers",
            "blockers": list(report["blockers"]),
            "metrics": {"generated_tokens": len(generated), "prompt_tokens": count,
                        "invalid_utf8": invalid, "answer_quality": None},
            "resource_usage": {"scope": "Inference; token counts measured, wall time and peak RAM not measured",
                               "device": "cpu", "cpu_threads": 1, "gpu_bytes": 0,
                               "peak_ram_bytes": None, "wall_seconds": None, "storage_bytes_written": 0},
            "generated_token_ids": generated, "generated_tokens": len(generated), "prompt_tokens": count,
            "max_new_tokens": max_new_tokens, "stop_reason": stop_reason,
            "checkpoint": str(checkpoint_path(root)), "synthetic": True, "trained": True,
            "training_optimizer_steps": report["metrics"]["optimizer_steps"],
            "decoding": "greedy", "invalid_utf8": invalid,
            "caveats": list(CAVEATS), "answer_quality_validated": False}


def generate(root: Path, prompt: str, max_new_tokens: int = 32) -> dict[str, Any]:
    """Generate an untrusted continuation with an empty context and no supplied answer."""
    _budget(max_new_tokens)
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    root = root.resolve()
    config = load_config(root / "configs" / "agri-mini.json")
    if len(prompt) > 16 * config["model"]["context_length"]:
        raise ValueError("Prompt exceeds the bounded input character budget")
    text = BPETokenizer.load(tokenizer_path(root))
    ids = [text.token_to_id("<bos>"), text.token_to_id("<question>"), *text.encode(prompt),
           text.token_to_id("<context>"), text.token_to_id("<answer>")]
    return _generate(root, text, _empty_channels(ids), max_new_tokens)


def generate_from_packed(root: Path, packed: dict[str, Any], max_new_tokens: int = 32) -> dict[str, Any]:
    """Continue ``pack_example(..., answer='')`` after removing its terminal EOS.

    Reject a nonempty answer rather than accidentally evaluating a teacher-forced
    answer. Structured IDs/numeric channels are retained byte-for-byte in the
    copied prompt, and the caller's packed dictionary is never mutated.
    """
    _budget(max_new_tokens)
    root = root.resolve()
    text = BPETokenizer.load(tokenizer_path(root))
    if not isinstance(packed, dict):
        raise ValueError("Expected pack_example output with an empty answer")
    ids, spans = packed.get("input_ids"), packed.get("text_spans")
    if not isinstance(ids, list) or len(ids) < 5 or not isinstance(spans, dict):
        raise ValueError("Invalid packed prompt or text spans")
    boundary = len(ids) - 1
    answer = spans.get("answer")
    if (not isinstance(answer, list) or len(answer) != 2 or any(type(x) is not int for x in answer)
            or answer != [boundary, boundary] or ids[-1] != text.token_to_id("<eos>")
            or ids[-2] != text.token_to_id("<answer>") or ids[0] != text.token_to_id("<bos>")):
        raise ValueError("Packed generation requires an empty answer span and terminal EOS; never feed answer text")
    if any(not isinstance(packed.get(name), list) or len(packed[name]) != len(ids) for name in _FIELDS):
        raise ValueError("Packed input channels must have identical lengths")
    prefix = {name: deepcopy(packed[name][:-1]) for name in _FIELDS}
    result = _generate(root, text, prefix, max_new_tokens)
    result["source_ids"] = deepcopy(packed.get("source_ids", []))
    result["evidence_ids"] = deepcopy(packed.get("evidence_ids", []))
    return result
