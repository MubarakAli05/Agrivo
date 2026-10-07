"""Strict snapshot primitives. Internal parsers are called only after the gate."""

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any
from urllib.parse import urlsplit

ADAPTER_VERSION = "1"
ENVELOPE_FIELDS = {
    "schema_version", "source_id", "dataset_id", "source_version", "source_url",
    "retrieved_at", "citations", "license", "data_origin", "format", "records",
}
LICENSE_FIELDS = {"name", "url", "attribution", "upstream_rights"}


def exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label}: expected exactly {sorted(fields)}")
    return value


def text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label}: expected nonempty, trimmed text")
    if any(ord(c) < 32 for c in value):
        raise ValueError(f"{label}: control characters forbidden")
    return value


def identifier(value: Any, label: str) -> str:
    value = text(value, label)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}", value) or ".." in value:
        raise ValueError(f"{label}: unsafe identifier")
    return value


def number(value: Any, label: str, minimum: float | None = None,
           maximum: float | None = None) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label}: expected JSON number, not bool or numeric text")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label}: non-finite number") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum) or (
            maximum is not None and result > maximum):
        raise ValueError(f"{label}: invalid or out-of-range number")
    return result


def url(value: Any, label: str) -> str:
    value = text(value, label)
    parsed = urlsplit(value)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or any(c.isspace() for c in value) or "\\" in value):
        raise ValueError(f"{label}: expected credential-free HTTPS URL")
    return value


def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _bad_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def decode_json(data: bytes) -> dict[str, Any]:
    result = json.loads(data.decode("utf-8"), object_pairs_hook=unique_pairs,
                        parse_constant=_bad_constant)
    if not isinstance(result, dict):
        raise ValueError("Snapshot must be a JSON object")
    return result


def encoded(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_path(path: Path) -> Path:
    """Reject traversal and links, including Windows junctions and UNC paths."""
    path = Path(path)
    if ".." in path.parts or str(path).startswith(("\\\\", "//")):
        raise ValueError("Path traversal/UNC paths are forbidden")
    path = path.absolute()
    for part in path.parts:
        if part == path.anchor:
            continue
        if ":" in part or part.endswith((".", " ")) or PureWindowsPath(part).is_reserved():
            raise ValueError("Reserved names, alternate data streams and ambiguous Windows paths are forbidden")
    for item in (path, *path.parents):
        if item.is_symlink() or getattr(item, "is_junction", lambda: False)():
            raise ValueError("Symlinks/junctions are forbidden in snapshot/output paths")
    return path


def image_path(value: Any) -> str:
    value = text(value, "image_path")
    posix = PurePosixPath(value.replace("\\", "/"))
    windows = PureWindowsPath(value)
    if (posix.is_absolute() or windows.drive or windows.root or ".." in posix.parts
            or any(c in value for c in (":", "\x00")) or not posix.name
            or any(part in ("", ".", "..") or part.endswith((".", " ")) or PureWindowsPath(part).is_reserved()
                   for part in value.replace("\\", "/").split("/"))):
        raise ValueError("image_path must be a safe relative path (metadata only)")
    return posix.as_posix()


def rows(value: Any, label: str = "records") -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value or not all(isinstance(v, dict) for v in value):
        raise ValueError(f"{label}: expected nonempty array of objects")
    return value


def unique_ids(records: list[dict[str, Any]], key: str = "record_id") -> None:
    seen = set()
    for row in records:
        value = identifier(row[key], key)
        if value in seen:
            raise ValueError(f"Duplicate {key}: {value}")
        seen.add(value)


def validate_envelope(snapshot: dict[str, Any], source_id: str,
                      source: dict[str, Any], terms: dict[str, Any]) -> None:
    exact(snapshot, ENVELOPE_FIELDS, "snapshot")
    if type(snapshot["schema_version"]) is not int or snapshot["schema_version"] != 1:
        raise ValueError("snapshot schema_version must be integer 1")
    if snapshot["source_id"] != source_id:
        raise ValueError("Snapshot source_id does not match requested source")
    identifier(snapshot["dataset_id"], "dataset_id")
    version = text(snapshot["source_version"], "source_version")
    if version.casefold() in {"latest", "unknown", "unversioned", "pending"}:
        raise ValueError("A pinned source_version is required, not a moving alias")
    if source["source_version"] is None or version != source["source_version"]:
        raise ValueError("Pin this exact snapshot source_version in the reviewed registry first")
    if url(snapshot["source_url"], "source_url") != source["url"]:
        raise ValueError("source_url must match the exact reviewed dataset URL")
    timestamp = text(snapshot["retrieved_at"], "retrieved_at")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})", timestamp):
        raise ValueError("retrieved_at must be an ISO-8601 timestamp with timezone")
    datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    citations = snapshot["citations"]
    if not isinstance(citations, list) or not citations:
        raise ValueError("Nonempty citations are required")
    for citation in citations:
        text(citation, "citation")
    rights = exact(snapshot["license"], LICENSE_FIELDS, "snapshot license")
    for key in LICENSE_FIELDS:
        text(rights[key], f"license.{key}")
    url(rights["url"], "license.url")
    if rights["name"] != terms["license"] or rights["url"] != terms["license_url"]:
        raise ValueError("Snapshot license must match reviewed dataset terms")
    if snapshot["data_origin"] not in ("synthetic_fixture", "external_snapshot"):
        raise ValueError("data_origin must explicitly identify fixture vs external snapshot")
    if snapshot["format"] not in ("json", "csv"):
        raise ValueError("format must be json or csv")


def envelope_schema() -> dict[str, Any]:
    return {
        "transport": "One UTF-8 JSON file, even when records contains CSV text. No URLs are fetched.",
        "additional_fields": False,
        "required_fields": sorted(ENVELOPE_FIELDS),
        "fields": {
            "schema_version": "integer 1 (not bool)", "source_id": "exact adapter ID",
            "dataset_id": "stable reviewed dataset/resource identifier",
            "source_version": "nonempty pinned version, must equal approved registry source_version",
            "source_url": "HTTPS, must equal exact reviewed registry dataset URL",
            "retrieved_at": "ISO-8601 timestamp with timezone; supplied acquisition time, not verified",
            "citations": "nonempty array of nonempty bibliographic strings",
            "license": {"required_fields": sorted(LICENSE_FIELDS), "additional_fields": False,
                        "name": "must match reviewed license", "url": "must match reviewed HTTPS license URL",
                        "attribution": "preserved attribution text", "upstream_rights": "preserved upstream rights/scope notes; not a rights guarantee"},
            "data_origin": ["synthetic_fixture", "external_snapshot"],
            "format": "json; csv only for data_gov", "records": "adapter-specific payload below",
        },
        "numeric_policy": "JSON numbers only, never bool/string/NaN/Infinity. Missing numeric values are null. CSV explicitly parses decimal text.",
        "identifier_policy": "[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}, no '..'; primary record/table IDs must be unique; grouping and foreign-key IDs may repeat",
        "approval_policy": "require_ingestion_approval runs BEFORE supplied snapshot access; catalogs must be replaced by individually reviewed dataset metadata, never blanket-approved",
        "output_policy": "Immutable, checksum-verified content-addressed release; original snapshot retained unchanged; no image/raster reads, downloads or model training",
    }
