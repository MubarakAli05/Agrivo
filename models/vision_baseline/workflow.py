"""Phase 14: offline synthetic CNN mechanics, never an agronomic model.

vision_plan is allocation-free and read-only. check_vision(..., dry_run=True)
returns that plan without training or writing. Normal checks perform two CPU SGD
steps and publish a checksum envelope plus a content-addressed scratch checkpoint
under the already-ignored checkpoints tree. validate_vision only reads/hashes;
it never initializes, loads, trains, or diagnoses with a model. SHA-256 detects
accidental changes, not malicious replacement of an artifact and its checksum.
"""

import hashlib
from importlib.metadata import version
import io
import json
import math
import os
from pathlib import Path
import platform
import re
from typing import Any
from uuid import uuid4

from agri.source_registry import inspect_sources


ARTIFACT_DIR = Path("checkpoints") / "vision_baseline"
REPORT_PATH = ARTIFACT_DIR / "phase14.json"
_CODE_ROOT = Path(__file__).resolve().parents[2]
_CLASSES = ["SYNTHETIC_PATTERN_HORIZONTAL", "SYNTHETIC_PATTERN_VERTICAL", "SYNTHETIC_PATTERN_DIAGONAL"]
_SCOPE = "Generated geometric tensor smoke only; not an agronomic model or plant-disease evidence"
_BUDGET = {"device": "cpu", "threads": 1, "optimizer": "SGD", "learning_rate": 0.05,
           "optimizer_steps": 2, "batch_size": 6, "image_shape": [3, 16, 16], "parameter_count": 507}


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate vision JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f"Nonfinite JSON number: {value}")


def _json(content: bytes) -> Any:
    return json.loads(content, object_pairs_hook=_unique, parse_constant=_reject_constant)


def _hash_file(path: Path) -> str | None:
    return _sha(path.read_bytes()) if path.is_file() else None


def vision_report_path(root: Path) -> Path:
    return root / REPORT_PATH


def _source_snapshot(root: Path) -> dict[str, Any]:
    paths = {"source_registry_sha256": root / "data" / "source_registry.yaml",
             "license_registry_sha256": root / "licenses" / "registry.json"}
    fingerprints = {name: _hash_file(path) for name, path in paths.items()}
    if any(value is None for value in fingerprints.values()):
        return {**fingerprints, "status": "UNKNOWN", "sources": {},
                "reason": "Source/license registry missing; approval cannot be established"}
    inspected = inspect_sources(root)
    entries = {}
    for source_id, entry in inspected["sources"].items():
        if entry["type"] == "plant_vision":
            entries[source_id] = {
                "enabled": entry["enabled"], "source_version": entry["source_version"],
                "inspection_status": entry["inspection"]["status"],
                "approval": entry["license"]["approval"], "license_status": entry["license"]["status"],
                "research_training": entry["ingestion"]["research_training"],
            }
    if any(_hash_file(paths[name]) != value for name, value in fingerprints.items()):
        raise ValueError("Source registries changed during read-only inspection")
    return {**fingerprints, "status": "INSPECTED_METADATA_ONLY", "sources": entries,
            "reason": "Approval metadata is not image availability or a trained approved model"}


def vision_plan(root: Path) -> dict[str, Any]:
    """Read-only resource/provenance plan required before any workflow training.

    The project seed/device are respected; transformer budgets do not configure
    this fixed, separately bounded smoke. No image files or adapters are read.
    """
    root = root.resolve()
    config_bytes = (root / "configs" / "agri-mini.json").read_bytes()
    config = _json(config_bytes)
    if not isinstance(config, dict) or config.get("device") != "cpu":
        raise ValueError("Vision baseline requires configured device=cpu")
    seed = config.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Vision seed must be an integer in [0, 2**32)")
    if config.get("offline") is not True:
        raise ValueError("Vision baseline requires offline=true; no external image access")
    code_paths = sorted(Path(__file__).parent.glob("*.py")) + [_CODE_ROOT / "agri" / "source_registry.py"]
    effective_config = {**_BUDGET, "seed": seed, "classes": list(_CLASSES)}
    return {
        "phase": 14, "status": "YELLOW", "scope": _SCOPE,
        "real_image_integration": "BLOCKED", "real_image_training": False,
        "agronomic_model": False, "classes": list(_CLASSES),
        "blockers": ["No approved trained real-image model or evaluated image integration",
                     "Existing plant fixtures are metadata only, not image training/evaluation data",
                     "Dataset access and intended-use approvals must be satisfied separately"],
        "resource_estimates": {
            **_BUDGET, "input_tensor_bytes": 6 * 3 * 16 * 16 * 4,
            "parameter_bytes": 507 * 4, "gradient_bytes": 507 * 4,
            "gpu_bytes": 0, "checkpoint_frequency": "Once after two synthetic optimizer steps",
            "artifact_budget_bytes": 1_048_576,
            "process_ram": "Estimate up to 1 GiB including Torch; not an enforced process limit",
            "cpu_time": "Seconds expected for two tiny steps; not a timing guarantee",
            "sufficiency": "Mechanics only; no validation/test split and no generalization evidence",
        },
        "provenance": {
            "project_config_sha256": _sha(config_bytes), "effective_config": effective_config,
            "effective_config_sha256": _sha(_bytes(effective_config)),
            "code_sha256": {str(path.relative_to(_CODE_ROOT)): _sha(path.read_bytes()) for path in code_paths},
            "source_approval_snapshot": _source_snapshot(root),
            "initialization": "Scratch random weights; CPU RNG seeded and restored",
            "synthetic_data": "Six generated RGB 16x16 horizontal/vertical/diagonal arrays plus seeded uniform noise; no images read",
            "data_kind": "SYNTHETIC_PATTERN", "pretrained": False,
            "python_version": platform.python_version(), "torch_distribution_version": version("torch"),
        },
        "quality": {"plant_disease_accuracy": None, "plant_disease_confidence": None,
                    "validation_loss": None, "test_loss": None, "field_generalization": None,
                    "reason": "No real-image evaluation; synthetic training-batch losses are mechanics only"},
    }


def diagnose_image(image: Any = None, *, root: Path | None = None) -> dict[str, Any]:
    """Abstain for every image; no decoding, fetching, checkpoint loading, or inference."""
    return {"status": "UNKNOWN", "unknown": True, "diagnosis": None, "confidence": None,
            "evidence": [], "agronomic_model": False,
            "reason": "No trained approved real-image model; synthetic pattern smoke cannot diagnose plants"}


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{uuid4().hex}.part")
    try:
        with staging.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def check_vision(root: Path, dry_run: bool = False) -> dict[str, Any]:
    """Plan, optionally run two synthetic steps, and atomically publish ignored artifacts."""
    if type(dry_run) is not bool:
        raise ValueError("dry_run must be boolean")
    root = root.resolve()
    plan = vision_plan(root)
    base = {**plan, "schema_version": 1, "dry_run": dry_run, "artifacts_written": False,
            "synthetic_model_mechanics": "NOT_RUN", "checks": {"optimizer_steps": 0}, "checkpoint": None}
    if dry_run:
        return base

    import torch
    from .model import run_synthetic_smoke

    checks, state = run_synthetic_smoke(plan["provenance"]["effective_config"]["seed"])
    if _bytes(vision_plan(root)) != _bytes(plan):
        raise ValueError("Vision provenance changed during synthetic training; no report published")
    checkpoint = io.BytesIO()
    torch.save({"state_dict": state, "provenance": plan["provenance"], "checks": checks,
                "classes": list(_CLASSES), "scope": _SCOPE, "agronomic_model": False}, checkpoint)
    content = checkpoint.getvalue()
    checksum = _sha(content)
    name = f"synthetic-{checksum}.pt"
    report = {**base, "artifacts_written": True, "synthetic_model_mechanics": "GREEN", "checks": checks,
              "checkpoint": {"name": name, "sha256": checksum, "size_bytes": len(content)}}
    envelope = _bytes({"report": report, "sha256": _sha(_bytes(report))})
    if len(content) + len(envelope) > plan["resource_estimates"]["artifact_budget_bytes"]:
        raise ValueError("Vision artifact budget exceeded")
    # Publish an immutable checkpoint first, then atomically switch the report pointer.
    _atomic_write(root / ARTIFACT_DIR / name, content)
    _atomic_write(vision_report_path(root), envelope)
    return report


def _hash_string(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_vision(root: Path) -> dict[str, Any]:
    """Validate stored hashes and current provenance without fitting or writing.

    Missing reports are explicitly NOT_RUN/YELLOW. Corrupt or stale artifacts
    raise ValueError and are never repaired automatically. Checkpoints are hashed
    as opaque bytes rather than unpickled during status checks.
    """
    root = root.resolve()
    plan = vision_plan(root)
    path = vision_report_path(root)
    if not path.exists():
        return {**plan, "schema_version": 1, "dry_run": False, "artifacts_written": False,
                "synthetic_model_mechanics": "NOT_RUN", "checks": {"optimizer_steps": 0}, "checkpoint": None}
    try:
        if path.stat().st_size > 1_048_576:
            raise ValueError("Oversized vision report")
        envelope = _json(path.read_bytes())
        if not isinstance(envelope, dict) or set(envelope) != {"report", "sha256"}:
            raise ValueError("Invalid vision report envelope")
        report = envelope["report"]
        if not isinstance(report, dict) or envelope["sha256"] != _sha(_bytes(report)):
            raise ValueError("Vision report checksum mismatch")
        if set(report) != set(plan) | {"schema_version", "dry_run", "artifacts_written", "synthetic_model_mechanics", "checks", "checkpoint"}:
            raise ValueError("Invalid vision report fields")
        if _bytes({key: report[key] for key in plan}) != _bytes(plan):
            raise ValueError("Vision report provenance or scope mismatch")
        if (type(report["schema_version"]) is not int or report["schema_version"] != 1
                or report["dry_run"] is not False or report["artifacts_written"] is not True
                or report["synthetic_model_mechanics"] != "GREEN"):
            raise ValueError("Invalid persisted synthetic mechanics status")
        checks = report["checks"]
        fixed = {"status": "GREEN", "optimizer_steps": 2, "threads": 1, "device": "cpu", "examples": 6,
                 "parameter_count": 507, "finite_gradients": True, "weights_changed": True}
        hash_keys = {"initial_state_sha256", "final_state_sha256", "images_sha256", "targets_sha256"}
        if not isinstance(checks, dict) or set(checks) != set(fixed) | hash_keys | {"training_batch_losses"}:
            raise ValueError("Invalid synthetic mechanics checks")
        if _bytes({key: checks[key] for key in fixed}) != _bytes(fixed):
            raise ValueError("Invalid synthetic mechanics budget or result")
        losses = checks["training_batch_losses"]
        if (not isinstance(losses, list) or len(losses) != 3
                or any(type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0 for loss in losses)
                or any(not _hash_string(checks[key]) for key in hash_keys)
                or checks["initial_state_sha256"] == checks["final_state_sha256"]):
            raise ValueError("Invalid synthetic loss/weight evidence")
        checkpoint = report["checkpoint"]
        if not isinstance(checkpoint, dict) or set(checkpoint) != {"name", "sha256", "size_bytes"}:
            raise ValueError("Invalid synthetic checkpoint record")
        if (not _hash_string(checkpoint["sha256"])
                or checkpoint["name"] != f"synthetic-{checkpoint['sha256']}.pt"
                or type(checkpoint["size_bytes"]) is not int or not 0 < checkpoint["size_bytes"] <= 1_048_576):
            raise ValueError("Invalid synthetic checkpoint identity")
        checkpoint_path = root / ARTIFACT_DIR / checkpoint["name"]
        if checkpoint_path.stat().st_size != checkpoint["size_bytes"]:
            raise ValueError("Vision checkpoint size mismatch")
        if _sha(checkpoint_path.read_bytes()) != checkpoint["sha256"]:
            raise ValueError("Vision checkpoint checksum mismatch")
        return report
    except (OSError, TypeError, KeyError, OverflowError, UnicodeError) as exc:
        raise ValueError("Invalid or missing vision artifact") from exc
