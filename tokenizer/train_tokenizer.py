"""Offline training and immutable, provenance-linked tokenizer releases."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

from agri.config import load_config
from agri.data_schema import SPLITS
from agri.synthetic_data import release_path, validate_data
from tokenizer.structured import StructuredSoilTokenizer
from tokenizer.tokenizer import BPETokenizer, SPECIAL_TOKENS


VERSION = "agri-tokenizer-v1"
MODEL_FILES = {"vocab.json", "merges.txt", "config.json", "structured.json"}


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate tokenizer manifest key: {key}")
        result[key] = value
    return result


def tokenizer_path(root: Path) -> Path:
    config = load_config(root / "configs" / "agri-mini.json")
    return root / "tokenizer" / "releases" / f"{VERSION}-seed{config['seed']}-vocab{config['model']['vocab_size']}"


def _inputs(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    validate_data(root)
    config = load_config(root / "configs" / "agri-mini.json")
    size = config["model"]["vocab_size"]
    if not len(SPECIAL_TOKENS) + 256 <= size <= 8192:
        raise ValueError("Phase 4 text vocabulary budget must be between 267 and 8192")
    data = release_path(root)
    soil = [json.loads(line) for line in (data / "soil.jsonl").read_text(encoding="utf-8").splitlines()]
    qa = {split: [json.loads(line) for line in (data / "qa" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
          for split in SPLITS}
    train_soil = [row for row in soil if row["split"] == "train"]
    provenance = {"version": VERSION, "seed": config["seed"], "requested_vocab_size": size,
                  "training_split": "train", "training_soil_records": len(train_soil),
                  "training_qa_examples": len(qa["train"]),
                  "training_text_sha256": hashlib.sha256(_bytes(_texts(qa["train"]))).hexdigest(),
                  "training_soil_sha256": hashlib.sha256(_bytes(train_soil)).hexdigest(),
                  "source": "agrimini_synthetic_fixture", "synthetic": True,
                  "text_fields": ["question", "context", "answer"], "structured_n_bins": 16,
                  "all_missing_policy": "Preserve raw observed value with a distinct uncalibrated token",
                  "pretrained": False}
    return config, soil, qa, provenance


def _texts(examples: list[dict[str, Any]]) -> list[str]:
    return [example[field] for example in examples for field in ("question", "context", "answer")]


def encode_example(tokenizer: BPETokenizer, example: dict[str, Any]) -> list[int]:
    ids = [tokenizer.token_to_id("<bos>")]
    for field in ("question", "context", "answer"):
        ids.append(tokenizer.token_to_id(f"<{field}>"))
        ids.extend(tokenizer.encode(example[field]))
    ids.append(tokenizer.token_to_id("<eos>"))
    return ids


def _evaluate(text: BPETokenizer, structured: StructuredSoilTokenizer, soil: list[dict[str, Any]],
              qa: dict[str, list[dict[str, Any]]], context_length: int) -> dict[str, Any]:
    metrics = {}
    for split, examples in qa.items():
        lengths = []
        for example in examples:
            for value in _texts([example]):
                ids = text.encode(value)
                if text.decode(ids) != text.normalize(value):
                    raise ValueError(f"Text roundtrip failed in {split}")
                if text.token_to_id("<unk>") in ids:
                    raise ValueError("Unexpected unknown ID with full byte fallback")
            lengths.append(len(encode_example(text, example)))
        for row in (r for r in soil if r["split"] == split):
            structured.encode(row)
        metrics[split] = {"examples": len(examples), "roundtrip_failures": 0, "unknown_text_tokens": 0,
                          "min_sequence_tokens": min(lengths), "max_sequence_tokens": max(lengths),
                          "mean_sequence_tokens": round(sum(lengths) / len(lengths), 2),
                          "examples_over_context_length": sum(n > context_length for n in lengths),
                          "truncated_examples": 0}
    return metrics


def train_tokenizer(root: Path, dry_run: bool = False) -> dict[str, Any]:
    root = root.resolve()
    config, soil, qa, provenance = _inputs(root)
    destination = tokenizer_path(root)
    if not dry_run and destination.exists():
        return validate_tokenizer(root)
    train_soil = [row for row in soil if row["split"] == "train"]
    # A bounded preflight precedes full training and never writes artifacts.
    tiny = BPETokenizer(vocab_size=min(320, config["model"]["vocab_size"])).fit(_texts(qa["train"][:4]))
    tiny_structured = StructuredSoilTokenizer().fit(train_soil[:4])
    for value in _texts(qa["train"][:4]):
        if tiny.decode(tiny.encode(value)) != tiny.normalize(value):
            raise ValueError("Tiny tokenizer dry-run roundtrip failed")
    tiny_structured.encode(train_soil[0])
    if dry_run:
        return {"phase": 4, "status": "GREEN", "dry_run": True, "artifacts_written": False,
                "training_qa_examples": 4, "training_soil_records": 4,
                "text_vocab_size": tiny.vocab_size, "gpu": "Not used", "pretrained": False}
    text = BPETokenizer(vocab_size=config["model"]["vocab_size"]).fit(_texts(qa["train"]))
    structured = StructuredSoilTokenizer().fit(train_soil)
    _evaluate(text, structured, soil, qa, config["model"]["context_length"])
    payloads = {**text.artifacts(), **structured.artifacts()}
    manifest = {**provenance, "created_at": datetime.now(timezone.utc).isoformat(),
                "files": {name: {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}
                          for name, content in payloads.items()}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tokenizer-staging-", dir=destination.parent) as temp:
        staging = Path(temp) / "release"
        staging.mkdir()
        for name, content in {**payloads, "manifest.json": _bytes(manifest)}.items():
            (staging / name).write_bytes(content)
        BPETokenizer.load(staging)
        StructuredSoilTokenizer.load(staging / "structured.json")
        staging.rename(destination)
    return validate_tokenizer(root)


def validate_tokenizer(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config, soil, qa, provenance = _inputs(root)
    destination = tokenizer_path(root)
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"), object_pairs_hook=_unique)
    if not isinstance(manifest, dict) or set(manifest) != set(provenance) | {"created_at", "files"}:
        raise ValueError("Invalid tokenizer manifest")
    if _bytes({key: manifest[key] for key in provenance}) != _bytes(provenance):
        raise ValueError("Tokenizer training provenance/configuration mismatch; refusing reuse")
    created = manifest["created_at"]
    if not isinstance(created, str) or datetime.fromisoformat(created).tzinfo is None:
        raise ValueError("Tokenizer manifest requires timezone-aware creation time")
    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != MODEL_FILES:
        raise ValueError("Tokenizer artifact inventory mismatch")
    if {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()} != MODEL_FILES | {"manifest.json"}:
        raise ValueError("Missing or unexpected tokenizer artifacts")
    for name, expected in files.items():
        content = (destination / name).read_bytes()
        actual = {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}
        if _bytes(expected) != _bytes(actual):
            raise ValueError(f"Tokenizer checksum/size mismatch: {name}; refusing overwrite")
    text = BPETokenizer.load(destination)
    structured = StructuredSoilTokenizer.load(destination / "structured.json")
    metrics = _evaluate(text, structured, soil, qa, config["model"]["context_length"])
    return {"phase": 4, "status": "GREEN", "scope": "Custom tokenizer only; not model quality evaluation",
            "release_path": str(destination), "text_vocab_size": text.vocab_size,
            "requested_vocab_size": config["model"]["vocab_size"], "structured_vocab_size": structured.vocab_size,
            "structured_tokens_per_record": 41, "training_qa_examples": provenance["training_qa_examples"],
            "training_soil_records": provenance["training_soil_records"], "metrics": metrics,
            "data_used": "Synthetic Phase 3 training split only; held-out splits used for roundtrip checks only",
            "model": "Not implemented", "training": "BPE merges and soil quantiles/categories only; no neural training",
            "source_status": "No external ingestion; existing approvals unchanged",
            "storage_bytes": sum(p.stat().st_size for p in destination.iterdir() if p.is_file()),
            "context_length": config["model"]["context_length"],
            "warnings": ["Full QA contexts exceeding the model context budget are reported, never truncated; packing is required before neural training"],
            "next_required_step": "Review Phase 4, then Phase 5: custom transformer and explicit context-budget handling"}
