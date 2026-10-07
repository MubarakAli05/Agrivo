"""Non-destructive project scaffolding; no downloads or training."""

import json
from pathlib import Path
from typing import Any

from agri.config import default_config, load_config
from agri.source_registry import (
    LICENSE_PATH, SOURCE_PATH, initialize_registries, inspect_sources,
)
from agri.synthetic_data import release_path, validate_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIRECTORIES = (
    "app/api", "app/dashboard",
    "data/raw", "data/processed", "data/indexed", "data/cache", "data/releases",
    "data/metadata", "data/quarantine",
    "sources/soilgrids", "sources/isric", "sources/ssurgo", "sources/data_gov",
    "sources/plantvillage", "sources/plantdoc", "sources/plant_pathology",
    "sources/ai_challenger",
    "tokenizer/tests", "models/transformer", "models/soil_baseline",
    "models/vision_baseline", "retrieval", "qa/raw", "qa/generated",
    "qa/verified", "qa/tests", "training", "evaluation", "configs",
    "checkpoints", "tests", "docs", "licenses",
)


def setup(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = root / "configs" / "agri-mini.json"
    if config_path.exists():
        load_config(config_path)
    for directory in DIRECTORIES:
        path = root.joinpath(*directory.split("/"))
        if path.exists() and not path.is_dir():
            raise ValueError(f"Expected directory: {path}")
    for directory in DIRECTORIES:
        path = root.joinpath(*directory.split("/"))
        path.mkdir(parents=True, exist_ok=True)
        marker = path / ".gitkeep"
        if not marker.exists():
            marker.touch(exist_ok=False)
    if not config_path.exists():
        with config_path.open("x", encoding="utf-8") as stream:
            json.dump(default_config(), stream, indent=2, allow_nan=False)
            stream.write("\n")
    initialize_registries(root)
    return status(root)


def status(root: Path) -> dict[str, Any]:
    root = root.resolve()
    missing = [name for name in DIRECTORIES
               if not root.joinpath(*name.split("/")).is_dir()]
    config_path = root / "configs" / "agri-mini.json"
    config = load_config(config_path) if config_path.exists() else None
    ready = config is not None and not missing
    registry_files = [(root / path).exists() for path in (SOURCE_PATH, LICENSE_PATH)]
    if any(registry_files) and not all(registry_files):
        raise ValueError("Incomplete source/license registry pair; run setup to validate or restore")
    registry = inspect_sources(root) if all(registry_files) else None
    report: dict[str, Any] = {
        "phase": 1,
        "status": "GREEN" if ready else "RED",
        "scope": "Repository layout and configuration only",
        "root": str(root),
        "config_path": str(config_path),
        "config_valid": config is not None,
        "missing_directories": missing,
        "data_used": "None; no source inspected or ingested",
        "model": "Not implemented; architecture settings only",
        "training": "Not started",
        "metrics": None,
        "source_status": "Not inspected; registry is Phase 2",
        "next_required_step": "Phase 2: source registry (review required)",
        "blockers": [] if ready else ["Run setup with a valid configuration"],
        "resource_usage": {
            "cpu": "Setup/status only; no training",
            "ram": "Not measured",
            "gpu": "Not used",
            "storage": "Scaffold/configuration/source metadata only; no datasets or weights",
        },
    }
    report["phase1_status"] = "GREEN" if ready else "RED"
    report["phase2_status"] = registry["status"] if registry else "RED"
    if registry is not None:
        report.update({
            "phase": 2,
            "status": registry["status"] if ready else "RED",
            "scope": "Repository configuration and validated source/license registry",
            "data_used": "Official source-page metadata only; no dataset ingestion",
            "source_status": {
                "registry_valid": True,
                "source_count": registry["source_count"],
                "declared_license_count": registry["declared_license_count"],
                "unresolved_license_count": registry["unresolved_license_count"],
                "registry_sha256": registry["registry_sha256"],
                "license_registry_sha256": registry["license_registry_sha256"],
            },
            "next_required_step": registry["next_required_step"],
        })
        if registry["status"] != "GREEN":
            report["blockers"].append("External ingestion requires source-specific access/license/intended-use approval")
    if config is not None and release_path(root).exists():
        dataset = validate_data(root)
        report.update({
            "phase": 3, "phase3_status": dataset["status"],
            "status": ("RED" if not ready or registry is None else registry["status"]),
            "scope": "Validated synthetic fixtures; external source review remains separate",
            "data_used": dataset["data_used"], "synthetic_dataset": dataset,
            "next_required_step": dataset["next_required_step"],
        })
        report["resource_usage"]["storage"] = f"Synthetic release: {dataset['storage_bytes']} bytes; no model weights"
    return report
