"""Tiny deterministic fixtures, not agricultural observations or predictions."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import tempfile
from typing import Any

from agri.config import load_config
from agri.data_schema import DATASET_VERSION, SOIL_FEATURES, SOURCE_ID, SPLITS, validate_soil_record
from agri.source_registry import load_registries


GENERATOR_VERSION = "1"
FILE_NAMES = ("soil.jsonl", "plants.jsonl", "qa/train.jsonl", "qa/validation.jsonl", "qa/test.jsonl", "schema.json")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def release_path(root: Path) -> Path:
    seed = load_config(root / "configs" / "agri-mini.json")["seed"]
    return root / "data" / "releases" / f"{DATASET_VERSION}-seed{seed}"


def _split_groups(rng: random.Random, count: int, train: int, validation: int) -> dict[int, str]:
    indices = list(range(count))
    rng.shuffle(indices)
    return {index: "train" if rank < train else "validation" if rank < train + validation else "test"
            for rank, index in enumerate(indices)}


def build_fixture(seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = random.Random(seed)
    soil_splits = _split_groups(rng, 12, 8, 2)
    plant_splits = _split_groups(rng, 6, 4, 1)
    soil, plants, qa = [], [], []
    for site in range(12):
        for depth_index, (top, bottom) in enumerate(((0, 5), (5, 15))):
            sand, silt = rng.randint(20, 50), rng.randint(15, 30)
            ph = None if (site, depth_index) in ((3, 0), (11, 1)) else round(rng.uniform(5, 8), 2)
            record = {
                "record_id": f"synthetic-soil-{site:02d}-{depth_index}",
                "location_id": f"synthetic-site-{site:02d}",
                "latitude": None, "longitude": None,
                "depth": {"top": top, "bottom": bottom, "unit": "cm"},
                "ph": ph, "organic_carbon": round(rng.uniform(5, 25), 2),
                "nitrogen": 0.0 if site == 0 else round(rng.uniform(0.1, 2), 2),
                "phosphorus": None,
                "potassium": None if site % 3 == 0 else round(rng.uniform(30, 150), 2),
                "bulk_density": round(rng.uniform(1, 1.5), 2),
                "sand": sand, "silt": silt, "clay": 100 - sand - silt,
                "cec": round(rng.uniform(5, 20), 2),
                "moisture": None if depth_index else round(rng.uniform(10, 30), 2),
                "ec": None if site % 2 else round(rng.uniform(0.1, 1), 2),
                "crop": f"SYNTHETIC_CROP_{site % 3}", "region": f"SYNTHETIC_REGION_{site % 2}",
                "source": SOURCE_ID, "source_version": DATASET_VERSION,
                "timestamp": None, "synthetic": True, "value_origin": "SYNTHETIC",
                "split": soil_splits[site],
                "units": {name: spec["unit"] for name, spec in SOIL_FEATURES.items()},
                "uncertainty": {name: None for name in SOIL_FEATURES},
            }
            for name in SOIL_FEATURES:
                record[name + "_missing"] = record[name] is None
            if ph is not None:
                record["uncertainty"]["ph"] = {
                    "lower": round(ph - 0.2, 2), "upper": round(ph + 0.2, 2),
                    "method": "synthetic_interval_not_calibrated",
                }
            validate_soil_record(record)
            soil.append(record)
    for index in range(6):
        plants.append({
            "record_id": f"synthetic-plant-{index:02d}", "group_id": f"synthetic-plant-group-{index:02d}",
            "dataset_name": "AgriMini synthetic plant metadata fixture", "dataset_version": DATASET_VERSION,
            "image_path": None, "image_checksum": None, "plant_id": None,
            "crop": f"SYNTHETIC_CROP_{index % 3}", "disease": f"SYNTHETIC_CONDITION_{index % 2}",
            "label": f"SYNTHETIC_CLASS_{index}", "source": SOURCE_ID, "source_url": None,
            "license": None, "split": plant_splits[index], "synthetic": True,
            "metadata": {"purpose": "Schema/QA tests only; no image, real disease label, or prediction"},
        })

    def example(group: str, split: str, kind: str, question: str, context: dict[str, Any],
                answer: str, evidence: list[str], unknown: bool = False) -> None:
        qa.append({
            "example_id": f"{group}-{kind}", "group_id": group, "split": split, "kind": kind,
            "question": question, "context": _json(context), "answer": answer,
            "source_ids": [SOURCE_ID], "evidence_ids": evidence,
            "dataset_version": DATASET_VERSION, "quality": "synthetic_verified",
            "synthetic": True, "unknown": unknown,
        })

    for site in range(12):
        first, second = soil[site * 2:site * 2 + 2]
        group, split = first["location_id"], first["split"]
        context = {"synthetic": True, "records": [first, second],
                   "definitions": {"ph": "In this fixture schema, ph is a unitless numeric field."}}
        one, both = [first["record_id"]], [first["record_id"], second["record_id"]]
        value = first["ph"]
        example(group, split, "property", f"What is pH at 0-5 cm for {group}?", context,
                "UNKNOWN: pH is missing from this synthetic fixture." if value is None else
                f"The synthetic fixture gives pH {value} at 0-5 cm; it is not a measured or estimated field value.",
                one, value is None)
        example(group, split, "unit", f"Which nitrogen unit is used for {group}?", context,
                f"The synthetic fixture uses {first['units']['nitrogen']} for nitrogen.", one)
        example(group, split, "missing", f"What is the exact phosphorus value for {group}?", context,
                "UNKNOWN: no phosphorus value exists in this synthetic fixture.", one, True)
        interval = first["uncertainty"]["ph"]
        example(group, split, "uncertainty", f"What pH uncertainty is given for {group}?", context,
                "UNKNOWN: no pH uncertainty interval exists in this synthetic fixture." if interval is None else
                f"The synthetic interval is {interval['lower']} to {interval['upper']}; it is not calibrated uncertainty.",
                one, interval is None)
        example(group, split, "source", f"Is the value for {group} measured, and what is its source?", context,
                f"It is synthetic, not measured or estimated. Source: {SOURCE_ID}; version: {DATASET_VERSION}.", one)
        missing = first["ph"] is None or second["ph"] is None
        example(group, split, "comparison", f"Compare pH at both depths for {group}.", context,
                "UNKNOWN: at least one depth has no pH value in this synthetic fixture." if missing else
                f"Synthetic pH is {first['ph']} at 0-5 cm and {second['ph']} at 5-15 cm. No crop recommendation is implied.",
                both, missing)
        example(group, split, "definition", f"How does the schema define ph for {group}?", context,
                context["definitions"]["ph"], one)
        example(group, split, "location_unknown", f"What real coordinates correspond to {group}?", context,
                "UNKNOWN: this fictional location has no real coordinates.", one, True)
    for plant in plants:
        context = {"synthetic": True, "records": [plant]}
        group, split = plant["group_id"], plant["split"]
        example(group, split, "dataset", f"Which class and crop appear in {plant['record_id']}?", context,
                f"Synthetic class {plant['label']} and crop {plant['crop']}; these are test labels, not real disease evidence.",
                [plant["record_id"]])
        example(group, split, "image_unknown", f"Can you diagnose the image for {plant['record_id']}?", context,
                "UNKNOWN: no image is present, so no plant disease prediction is possible.",
                [plant["record_id"]], True)
    return soil, plants, qa


def _payloads(seed: int) -> tuple[dict[str, bytes], dict[str, Any]]:
    soil, plants, qa = build_fixture(seed)
    rows = {"soil.jsonl": soil, "plants.jsonl": plants}
    rows.update({f"qa/{split}.jsonl": [q for q in qa if q["split"] == split] for split in SPLITS})
    payloads = {name: ("\n".join(_json(row) for row in records) + "\n").encode("utf-8")
                for name, records in rows.items()}
    schema = {
        "schema_version": 1, "soil_features": SOIL_FEATURES,
        "missingness": "null value plus feature_missing=true; zero is not missing",
        "coordinates": "null; locations and region/crop names are fictional",
        "uncertainty": "null or synthetic interval; never calibrated confidence",
        "plant_images": "none; metadata-only placeholders, not vision training data",
        "qa_quality": "synthetic_verified means consistency with generated fixtures, not agricultural factuality",
    }
    payloads["schema.json"] = (_json(schema) + "\n").encode("utf-8")
    manifest = {
        "schema_version": 1, "dataset_version": DATASET_VERSION, "generator_version": GENERATOR_VERSION,
        "seed": seed, "source": SOURCE_ID, "original_url": None, "downloaded_at": None,
        "synthetic": True, "license": None,
        "usage": "Original local fixtures for requested tests/proof-of-working training; redistribution/commercial terms not reviewed",
        "transformation_history": ["Deterministic local generation", "Assign whole location/plant groups to splits",
                                   "Construct QA strictly from fixture records", "Validate schema and freeze release"],
        "files": {name: {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content),
                         "record_count": len(rows[name]) if name in rows else None}
                  for name, content in payloads.items()},
        "counts": {"soil_records": len(soil), "plant_metadata_records": len(plants), "images": 0,
                   "qa_examples": len(qa), "unknown_examples": sum(q["unknown"] for q in qa)},
        "splits": {split: {"soil_records": sum(r["split"] == split for r in soil),
                           "plant_metadata_records": sum(p["split"] == split for p in plants),
                           "qa_examples": sum(q["split"] == split for q in qa)} for split in SPLITS},
    }
    return payloads, manifest


def _check_consistency(soil: list[dict[str, Any]], plants: list[dict[str, Any]], qa: list[dict[str, Any]]) -> None:
    records = {record["record_id"]: record for record in soil + plants}
    if len(records) != len(soil) + len(plants):
        raise ValueError("Duplicate record ID")
    groups: dict[str, str] = {}
    for record in soil + plants:
        if record in soil:
            validate_soil_record(record)
        group = record.get("location_id", record.get("group_id"))
        if group in groups and groups[group] != record["split"]:
            raise ValueError("Group leakage across data splits")
        groups[group] = record["split"]
    seen = set()
    for example in qa:
        if example["example_id"] in seen:
            raise ValueError("Duplicate QA example")
        seen.add(example["example_id"])
        if groups.get(example["group_id"]) != example["split"]:
            raise ValueError("QA split differs from its record group")
        if example["source_ids"] != [SOURCE_ID] or not example["evidence_ids"]:
            raise ValueError("QA provenance missing")
        context = json.loads(example["context"])
        if not context.get("synthetic") or example["synthetic"] is not True:
            raise ValueError("QA must be explicitly synthetic")
        for record in context["records"]:
            original = records.get(record["record_id"])
            if record != original or record["split"] != example["split"]:
                raise ValueError("QA context is altered or leaks across splits")
        for record_id in example["evidence_ids"]:
            if record_id not in {r["record_id"] for r in context["records"]}:
                raise ValueError("QA citation does not resolve inside its context")


def generate_synthetic(root: Path) -> dict[str, Any]:
    root = root.resolve()
    load_registries(root)
    seed = load_config(root / "configs" / "agri-mini.json")["seed"]
    destination = release_path(root)
    if destination.exists():
        return validate_data(root)
    payloads, manifest = _payloads(seed)
    _check_consistency(*build_fixture(seed))
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".synthetic-staging-", dir=destination.parent) as temp:
        staging = Path(temp) / "release"
        staging.mkdir()
        for name, content in payloads.items():
            path = staging.joinpath(*name.split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        (staging / "manifest.json").write_text(_json(manifest) + "\n", encoding="utf-8")
        staging.rename(destination)
    return validate_data(root)


def validate_data(root: Path) -> dict[str, Any]:
    root = root.resolve()
    seed = load_config(root / "configs" / "agri-mini.json")["seed"]
    destination = release_path(root)
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Invalid synthetic manifest")
    created_at = manifest.get("created_at")
    if not isinstance(created_at, str) or datetime.fromisoformat(created_at).tzinfo is None:
        raise ValueError("Manifest requires a timezone-aware generation timestamp")
    payloads, expected_manifest = _payloads(seed)
    if _json({key: value for key, value in manifest.items() if key != "created_at"}) != _json(expected_manifest):
        raise ValueError("Manifest differs from the versioned synthetic recipe")
    expected_paths = {destination.joinpath(*name.split("/")) for name in FILE_NAMES} | {destination / "manifest.json"}
    if {path for path in destination.rglob("*") if path.is_file()} != expected_paths:
        raise ValueError("Synthetic release file inventory is incomplete or unexpected")
    for name, expected in payloads.items():
        content = destination.joinpath(*name.split("/")).read_bytes()
        if content != expected:
            raise ValueError(f"Synthetic file changed or is corrupt: {name}; refusing overwrite")
    soil = [json.loads(line) for line in (destination / "soil.jsonl").read_text(encoding="utf-8").splitlines()]
    plants = [json.loads(line) for line in (destination / "plants.jsonl").read_text(encoding="utf-8").splitlines()]
    qa = [json.loads(line) for split in SPLITS
          for line in (destination / "qa" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    _check_consistency(soil, plants, qa)
    missing = {name: sum(record[name + "_missing"] for record in soil) for name in SOIL_FEATURES}
    return {
        "phase": 3, "status": "GREEN", "scope": "Synthetic fixture consistency only; not model or agronomic evaluation",
        "dataset_version": DATASET_VERSION, "release_path": str(destination), "seed": seed,
        "data_used": "Original local synthetic soil and plant metadata; no external data or real coordinates",
        "counts": manifest["counts"], "splits": manifest["splits"],
        "missing_soil_values": missing, "qa_kinds": dict(Counter(q["kind"] for q in qa)),
        "duplicate_records": 0, "cross_split_group_leakage": 0,
        "model": "Not implemented", "training": "Not started",
        "source_status": "External ingestion approvals unchanged; fixtures cite their own local manifest",
        "model_hallucination_test": "Not run; UNKNOWN examples are test fixtures only",
        "manifest_sha256": hashlib.sha256((destination / "manifest.json").read_bytes()).hexdigest(),
        "storage_bytes": sum(path.stat().st_size for path in expected_paths),
        "next_required_step": "Review Phase 3, then Phase 4: custom tokenizer; fit on training split only",
    }
