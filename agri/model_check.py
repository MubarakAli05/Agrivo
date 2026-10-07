"""Read-only readiness validation and bounded, random-weight CPU model checks."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from agri.config import load_config
from agri.data_schema import SPLITS
from agri.synthetic_data import release_path
from models.transformer.packing import pack_example
from tokenizer.structured import StructuredSoilTokenizer
from tokenizer.tokenizer import BPETokenizer
from tokenizer.train_tokenizer import tokenizer_path, validate_tokenizer


REPORT_PATH = Path("models") / "transformer" / "reports" / "phase5.json"
CODE_ROOT = Path(__file__).resolve().parent.parent
MODEL_SOURCES = ("models/transformer/model.py", "models/transformer/packing.py", "agri/model_check.py")


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate model report key: {key}")
        result[key] = value
    return result


def model_report_path(root: Path) -> Path:
    return root / REPORT_PATH


def _prepare(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    config = load_config(root / "configs" / "agri-mini.json")
    model = config["model"]
    if config["device"] != "cpu":
        raise ValueError("Phase 5 checks are CPU-only; select device=cpu")
    limits = {"context_length": 1536, "layers": 4, "hidden_size": 256,
              "attention_heads": 8, "feed_forward_size": 1024, "vocab_size": 8192}
    if any(model[name] > limit for name, limit in limits.items()):
        raise ValueError("Configuration exceeds the bounded Phase 5 CPU check limits")
    validate_tokenizer(root)
    directory = tokenizer_path(root)
    text = BPETokenizer.load(directory)
    structured = StructuredSoilTokenizer.load(directory / "structured.json")
    packed = {}
    metrics = {}
    for split in SPLITS:
        path = release_path(root) / "qa" / f"{split}.jsonl"
        examples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        packed[split] = [pack_example(text, structured, row, model["context_length"]) for row in examples]
        lengths = [len(row["input_ids"]) for row in packed[split]]
        metrics[split] = {"examples": len(lengths), "min_positions": min(lengths),
                          "max_positions": max(lengths), "truncated_examples": 0}
    architecture = {key: model[key] for key in ("layers", "hidden_size", "attention_heads", "feed_forward_size",
                                                "context_length", "dropout")}
    architecture.update(vocab_size=text.vocab_size, structured_vocab_size=structured.vocab_size)
    provenance = {"seed": config["seed"], "architecture": architecture,
                  "data_manifest_sha256": hashlib.sha256((release_path(root) / "manifest.json").read_bytes()).hexdigest(),
                  "tokenizer_manifest_sha256": hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest(),
                  "code_sha256": {name: hashlib.sha256(CODE_ROOT.joinpath(*name.split("/")).read_bytes()).hexdigest()
                                  for name in MODEL_SOURCES}}
    base = {"phase": 5, "status": "GREEN", "scope": "Custom transformer mechanics only; random weights",
            "data_used": "Synthetic fixtures only; all splits checked for packing, full-example forwards restricted to train; tiny probe uses constructed IDs",
            "source_status": "External datasets remain disabled and require separate approval",
            "model": "AgriTransformer-v0; no pretrained or trained weights", "training": "Not started; zero optimizer steps",
            "provenance": provenance, "packing_metrics": metrics,
            "next_required_step": "Review Phase 5, then Phase 6: tiny training run",
            "blockers": ["No trained model or agricultural quality evidence", "External-source approvals pending"]}
    return config, base, packed


def _estimates(architecture: dict[str, Any]) -> dict[str, Any]:
    return {"cpu": "One thread; tiny backward check then one batch-one full-length forward (usually seconds)",
            "ram": "Budget up to 2 GiB process RAM; estimate only, not a measured/enforced memory limit",
            "gpu": "0 bytes; CPU-only, no GPU operations",
            "one_attention_matrix_bytes": 4 * architecture["attention_heads"] * architecture["context_length"] ** 2,
            "storage": "JSON report only (under 20 KiB); no weights/checkpoints or dataset downloads",
            "training_epochs": 0, "optimizer_steps": 0, "checkpoint_frequency": "None; no training",
            "data_sufficiency": "Sufficient only for mechanics; not real agricultural generalization"}


def check_model(root: Path, dry_run: bool = False) -> dict[str, Any]:
    root = root.resolve()
    started, cpu_started = time.perf_counter(), time.process_time()
    config, base, packed = _prepare(root)
    try:
        import torch
        from models.transformer.model import AgriTransformer, TransformerConfig, causal_lm_loss
        from models.transformer.packing import collate_examples
    except ImportError as exc:
        raise ValueError('Model dependency unavailable; install the project extra: pip install -e ".[model]"') from exc
    architecture = base["provenance"]["architecture"]
    estimates = _estimates(architecture)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        # Isolate random initialization/dropout without touching a CUDA generator.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config["seed"])
            tiny = TransformerConfig(vocab_size=architecture["vocab_size"],
                                     structured_vocab_size=architecture["structured_vocab_size"],
                                     context_length=16, layers=1, hidden_size=16, attention_heads=2,
                                     feed_forward_size=32, dropout=0)

            def backward_check(model: Any) -> None:
                ids = torch.full((1, 16), 2, dtype=torch.long)
                ids[0, 2] = 0
                structured_ids = torch.zeros_like(ids)
                structured_ids[0, 2] = 1
                features = torch.full_like(ids, -1)
                features[0, 2] = 0
                values = torch.zeros((1, 16, 7))
                values[0, 2, 0] = 1.0
                labels = ids.clone()
                labels[0, 2] = -100
                logits = model(ids, structured_ids=structured_ids, numeric_features=features, numeric_values=values)
                loss = causal_lm_loss(logits, labels)
                if not torch.isfinite(logits).all().item() or not torch.isfinite(loss).item():
                    raise ValueError("Nonfinite random-weight smoke output/loss")
                loss.backward()
                if any(parameter.grad is None or not torch.isfinite(parameter.grad).all().item()
                       for parameter in model.parameters() if parameter.requires_grad):
                    raise ValueError("Missing or nonfinite model gradient")
                model.zero_grad(set_to_none=True)

            tiny_model = AgriTransformer(tiny)
            backward_check(tiny_model)
            if dry_run:
                return {**base, "dry_run": True, "artifacts_written": False, "tiny_forward_backward": "passed",
                        "tiny_parameters": sum(p.numel() for p in tiny_model.parameters()), "resource_estimates": estimates}
            torch.random.default_generator.manual_seed(config["seed"])
            model = AgriTransformer(TransformerConfig(**architecture))
            if architecture["context_length"] < 16:
                raise ValueError("Phase 5 model check requires at least 16 positions")
            backward_check(model)
            model.eval()
            longest = max(packed["train"], key=lambda row: len(row["input_ids"]))
            batch = collate_examples([longest])
            with torch.no_grad():
                logits = model(**{key: value for key, value in batch.items() if key != "labels"})
            if logits.shape != (1, len(longest["input_ids"]), architecture["vocab_size"]) or not torch.isfinite(logits).all().item():
                raise ValueError("Full training-example forward check failed")
            report = {**base, "dry_run": False, "artifacts_written": True,
                      "created_at": datetime.now(timezone.utc).isoformat(), "torch_version": str(torch.__version__),
                      "checks": {"tiny_forward_backward": "passed", "configured_short_forward_backward": "passed",
                                 "full_training_example_forward": "passed", "optimizer_steps": 0,
                                 "forward_example_id": longest["example_id"], "forward_positions": len(longest["input_ids"]),
                                 "parameters": sum(p.numel() for p in model.parameters()),
                                 "fp32_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters())},
                      "resource_estimates": estimates,
                      "resource_usage": {"wall_seconds": round(time.perf_counter() - started, 3),
                                         "cpu_seconds": round(time.process_time() - cpu_started, 3),
                                         "ram": "Native peak RAM not measured", "gpu": "Not used",
                                         "storage": "Local JSON validation report only; no weights"}}
    finally:
        torch.set_num_threads(threads)
    destination = model_report_path(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = _bytes({"report": report, "sha256": hashlib.sha256(_bytes(report)).hexdigest()})
    with tempfile.TemporaryDirectory(prefix=".model-check-", dir=destination.parent) as temp:
        staging = Path(temp) / "phase5.json"
        staging.write_bytes(content)
        staging.replace(destination)
    return validate_model(root)


def validate_model(root: Path) -> dict[str, Any]:
    """Validate a recorded check against current inputs/code; never initialize a model."""
    root = root.resolve()
    _, base, _ = _prepare(root)
    envelope = json.loads(model_report_path(root).read_text(encoding="utf-8"), object_pairs_hook=_unique)
    if not isinstance(envelope, dict) or set(envelope) != {"report", "sha256"}:
        raise ValueError("Invalid model check report")
    report = envelope["report"]
    if not isinstance(report, dict) or hashlib.sha256(_bytes(report)).hexdigest() != envelope["sha256"]:
        raise ValueError("Model check report checksum mismatch")
    if _bytes({key: report.get(key) for key in base}) != _bytes(base):
        raise ValueError("Model check provenance mismatch; rerun check-model explicitly")
    extra = {"dry_run", "artifacts_written", "created_at", "torch_version", "checks", "resource_estimates", "resource_usage"}
    if set(report) != set(base) | extra or report["dry_run"] is not False or report["artifacts_written"] is not True:
        raise ValueError("Invalid model check result schema")
    if not isinstance(report["created_at"], str) or datetime.fromisoformat(report["created_at"]).tzinfo is None:
        raise ValueError("Model check requires a timezone-aware timestamp")
    checks = report["checks"]
    if (not isinstance(checks, dict) or type(checks.get("optimizer_steps")) is not int or checks["optimizer_steps"] != 0
            or any(checks.get(name) != "passed" for name in ("tiny_forward_backward", "configured_short_forward_backward",
                                                            "full_training_example_forward"))):
        raise ValueError("Model check did not pass without training")
    return {**report, "report_path": str(model_report_path(root)), "storage_bytes": model_report_path(root).stat().st_size}
