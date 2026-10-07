"""Read-only readiness inspection and approval-first local snapshot ingestion.

All adapter imports are lazy so Phase 10 can be committed independently. A local
snapshot is still external ingestion: disabling network access is not approval.
"""

from copy import deepcopy
import importlib
from pathlib import Path
import shutil
from typing import Any

from agri import source_registry
from sources._common import (
    ADAPTER_VERSION, decode_json, encoded, envelope_schema, safe_path, sha256,
    unique_ids, validate_envelope,
)

_ADAPTERS = {
    "plantvillage": (10, "sources.plantvillage"),
    "plantdoc": (10, "sources.plantdoc"),
    "soilgrids": (11, "sources.soilgrids"),
    "isric": (11, "sources.isric"),
    "data_gov": (12, "sources.data_gov"),
    "ssurgo": (13, "sources.ssurgo"),
}


def _adapter(source_id: str) -> Any:
    if source_id not in _ADAPTERS:
        raise ValueError(f"No snapshot adapter for source: {source_id}")
    module = importlib.import_module(_ADAPTERS[source_id][1])
    if not hasattr(module, "_normalize"):
        raise ValueError(f"Adapter implementation not installed: {source_id}")
    return module


def adapter_schema(source_id: str) -> dict[str, Any]:
    """Return the explicit accepted snapshot contract without accessing data."""
    module = _adapter(source_id)
    return {
        "source_id": source_id, "phase": _ADAPTERS[source_id][0],
        "adapter_version": ADAPTER_VERSION, "envelope": envelope_schema(),
        **deepcopy(module._schema()),
    }


def inspect_adapter(root: Path, source_id: str) -> dict[str, Any]:
    """Inspect stored metadata only. GREEN never means real-data validation."""
    report = source_registry.inspect_sources(Path(root), source_id)
    entry = report["sources"][source_id]
    schema = adapter_schema(source_id)
    blockers = deepcopy(entry["ingestion"])
    version = entry["source_version"]
    for gate in blockers.values():
        if not version or version.casefold() in {"latest", "unknown", "unversioned", "pending"}:
            gate["blockers"].append("Pin an exact reviewed snapshot source_version")
            gate["allowed"] = False
    return {
        "source_id": source_id, "phase": schema["phase"], "status": "YELLOW",
        "adapter_status": "READY", "real_data_quality": "UNVERIFIED",
        "ingestion": blockers, "source_metadata": entry, "schema": schema,
        "registry_sha256": report["registry_sha256"],
        "license_registry_sha256": report["license_registry_sha256"],
        "next_required_step": "Resolve listed source/license/intended-use blockers; pin an exact dataset version; only then ingest and independently validate real data.",
    }


def _blocked(source_id: str, error: PermissionError) -> dict[str, Any]:
    return {"source_id": source_id, "status": "YELLOW", "blocked": True,
            "ingested": False, "real_data_quality": "UNVERIFIED",
            "reasons": str(error).split("; "),
            "next_required_step": "Obtain explicit dataset/source and intended-use approval in the registries; no output has been written."}


def ingest_snapshot(root: Path, source_id: str, snapshot: Path,
                    purpose: str = "research_training") -> dict[str, Any]:
    """Gate before any snapshot read/stat, validate completely, publish exclusively.

    Approval failures return an actionable YELLOW report with no writes. Invalid
    snapshots raise ValueError; they are not silently quarantined or normalized.
    Existing releases are verified byte-for-byte and never overwritten.
    """
    root = Path(root)
    try:
        approved_source = source_registry.require_ingestion_approval(root, source_id, purpose)
    except PermissionError as exc:
        return {**_blocked(source_id, exc), "snapshot_read": False}
    module = _adapter(source_id)
    root = safe_path(root)
    snapshot = safe_path(snapshot)
    if snapshot.suffix.lower() != ".json" or not snapshot.is_file():
        raise ValueError("snapshot must be an existing local .json file")
    registry_report = source_registry.inspect_sources(root, source_id)
    entry = registry_report["sources"][source_id]
    if approved_source != {key: entry[key] for key in approved_source}:
        raise ValueError("Source registry changed during approval; retry after review")
    terms = entry["license"]
    raw = snapshot.read_bytes()
    payload = decode_json(raw)
    validate_envelope(payload, source_id, approved_source, terms)
    records, history = module._normalize(payload)
    if not records:
        raise ValueError("Snapshot produced no records")
    unique_ids(records)
    provenance = {
        "source": source_id, "dataset_id": payload["dataset_id"],
        "source_version": payload["source_version"], "original_url": payload["source_url"],
        "retrieved_at": payload["retrieved_at"], "citations": payload["citations"],
        "license": payload["license"], "approved_license": terms,
        "data_origin": payload["data_origin"], "snapshot_sha256": sha256(raw),
        "snapshot_file_size": len(raw), "adapter_version": ADAPTER_VERSION,
        "registry_sha256": registry_report["registry_sha256"],
        "license_registry_sha256": registry_report["license_registry_sha256"],
        "purpose": purpose, "real_data_quality": "UNVERIFIED",
    }
    transformations = [{"operation": "validate_snapshot", "adapter_version": ADAPTER_VERSION}, *history]
    shared_steps = [step for step in transformations if "record_id" not in step]
    per_record_steps: dict[str, list[dict[str, Any]]] = {}
    for step in transformations:
        if "record_id" in step:
            per_record_steps.setdefault(step["record_id"], []).append(step)
    for record in records:
        record["provenance"] = {**deepcopy(provenance), "transformation_history": deepcopy(
            shared_steps + per_record_steps.get(record["record_id"], []))}
    output = b"".join(encoded(record) for record in records)
    schema = adapter_schema(source_id)
    manifest = {
        "schema_version": 1, **provenance,
        "status": "YELLOW", "record_count": len(records), "schema": schema,
        "transformation_history": transformations,
        "files": {"snapshot.json": {"sha256": sha256(raw), "bytes": len(raw)},
                  "records.jsonl": {"sha256": sha256(output), "bytes": len(output)}},
        "limitations": schema["limitations"],
    }
    manifest_bytes = encoded(manifest)
    release_id = sha256(manifest_bytes)
    destination = safe_path(root / "data" / "releases" / "external" / source_id / release_id)
    files = {"snapshot.json": raw, "records.jsonl": output, "manifest.json": manifest_bytes}
    # Recheck approval/fingerprints before publication; concurrent review changes fail closed.
    try:
        source_registry.require_ingestion_approval(root, source_id, purpose)
    except PermissionError as exc:
        return {**_blocked(source_id, exc), "snapshot_read": True}
    current = source_registry.inspect_sources(root, source_id)
    if any(current[key] != registry_report[key] for key in ("registry_sha256", "license_registry_sha256")):
        raise ValueError("Approval registries changed during parsing; nothing published")
    reused = destination.exists()
    if reused:
        if not destination.is_dir() or {p.name for p in destination.iterdir()} != set(files):
            raise ValueError("Existing immutable release is incomplete or modified")
        for name, content in files.items():
            path = safe_path(destination / name)
            if not path.is_file() or path.read_bytes() != content:
                raise ValueError("Existing immutable release checksum mismatch; refusing overwrite")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.mkdir(exist_ok=False)
        try:
            for name, content in files.items():
                with (destination / name).open("xb") as stream:
                    stream.write(content)
        except BaseException:
            shutil.rmtree(destination)
            raise
    return {"source_id": source_id, "status": "YELLOW", "blocked": False,
            "ingested": True, "real_data_quality": "UNVERIFIED", "reused": reused,
            "record_count": len(records), "release_id": release_id,
            "manifest": str(destination / "manifest.json"),
            "records": str(destination / "records.jsonl"),
            "manifest_sha256": sha256(manifest_bytes),
            "next_required_step": "Validate real-data quality and upstream rights independently; a local parser run does not certify the external phase."}
