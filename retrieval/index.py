"""Deterministic, split-isolated retrieval over the validated local synthetic release.

Only soil and plant metadata records are indexed, never QA questions or answers.
The fixed local index is replace-atomic and every read revalidates its source bytes.
"""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from agri.data_schema import SOIL_FEATURES, SPLITS
from agri.synthetic_data import release_path, validate_data


INDEX_PATH = Path("data") / "indexed" / "synthetic-v1.json"
MAX_QUESTION_CHARS = 512
MAX_HITS = 30
MAX_INDEX_BYTES = 2_000_000
VERSION = "agrimini-synthetic-index-v1"


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate index key: {key}")
        result[key] = value
    return result


def _local(root: Path, path: Path) -> Path:
    if not path.resolve().is_relative_to(root):
        raise ValueError("Index and source paths must remain inside the workspace")
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError("Linked index/source paths are not supported")
    return path


def index_path(root: Path) -> Path:
    """Return the single ignored output location; no caller-selected files are read."""
    root = root.resolve()
    return _local(root, root / INDEX_PATH)


def _expected(root: Path) -> dict[str, Any]:
    _local(root, root / "configs" / "agri-mini.json")
    directory = _local(root, release_path(root))
    for path in directory.rglob("*"):
        _local(root, path)
    report = validate_data(root)
    manifest_bytes = (directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    evidence = []
    for name, kind in (("soil.jsonl", "soil"), ("plants.jsonl", "plant_metadata")):
        source_path = directory / name
        content = source_path.read_bytes()
        for line, raw in enumerate(content.splitlines(), 1):
            record = json.loads(raw)
            version = record.get("source_version", record.get("dataset_version"))
            if (record.get("synthetic") is not True or record.get("source") != manifest["source"]
                    or version != manifest["dataset_version"] or record.get("split") not in SPLITS):
                raise ValueError("Missing or inconsistent synthetic evidence provenance")
            evidence.append({
                "record_id": record["record_id"], "kind": kind, "split": record["split"],
                "source": record["source"], "source_version": version,
                "source_path": str(source_path.relative_to(root)), "source_line": line,
                "source_sha256": _sha(content), "record_sha256": _sha(_bytes(record)),
                "manifest_sha256": _sha(manifest_bytes),
                "manifest_path": str((directory / "manifest.json").relative_to(root)),
                "original_url": manifest["original_url"], "license": manifest["license"],
                "usage": manifest["usage"], "timestamp": record.get("timestamp"),
                "location_id": record.get("location_id"), "depth": record.get("depth"),
                "units": record.get("units", {}), "uncertainty": record.get("uncertainty", {}),
                "synthetic": True, "record": record,
            })
    evidence.sort(key=lambda item: item["record_id"])
    return {
        "version": VERSION, "synthetic": True, "dataset_version": report["dataset_version"],
        "manifest_sha256": _sha(manifest_bytes),
        "manifest_path": str((directory / "manifest.json").relative_to(root)),
        "source": manifest["source"], "original_url": manifest["original_url"],
        "license": manifest["license"], "usage": manifest["usage"],
        "transformation_history": manifest["transformation_history"],
        "records": evidence,
    }


def _report(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {"phase": 8, "status": "GREEN", "synthetic": True,
            "scope": "Local synthetic evidence integrity only; no real agricultural facts",
            "index_path": str(index_path(root)), "index_sha256": _sha(_bytes(payload)),
            "manifest_sha256": payload["manifest_sha256"], "records": len(payload["records"]),
            "counts": dict(Counter(row["kind"] for row in payload["records"])),
            "splits": {split: sum(row["split"] == split for row in payload["records"]) for split in SPLITS},
            "qa_answers_indexed": 0, "external_sources_used": 0, "default_split": "train",
            "limitations": ["Fictional locations only", "Plant metadata only; no images",
                            "External sources disabled pending license approval"]}


def build_index(root: Path) -> dict[str, Any]:
    """Build the fixed ignored index atomically; refuse to repair corrupt/stale artifacts."""
    root = root.resolve()
    destination = index_path(root)
    if destination.exists():
        return validate_index(root)
    payload = _expected(root)
    content = _bytes({"payload": payload, "sha256": _sha(_bytes(payload))})
    if len(content) > MAX_INDEX_BYTES:
        raise ValueError("Index exceeds the synthetic storage bound")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = _local(root, destination.with_name(f".synthetic-index-{uuid4().hex}.staging"))
    try:
        with staging.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, destination)
    finally:
        staging.unlink(missing_ok=True)
    return validate_index(root)


def _load(root: Path) -> dict[str, Any]:
    path = index_path(root)
    if path.stat().st_size > MAX_INDEX_BYTES:
        raise ValueError("Index exceeds the synthetic storage bound")
    envelope = json.loads(path.read_bytes(), object_pairs_hook=_unique)
    if not isinstance(envelope, dict) or set(envelope) != {"payload", "sha256"}:
        raise ValueError("Invalid index envelope")
    payload = envelope["payload"]
    if envelope["sha256"] != _sha(_bytes(payload)):
        raise ValueError("Index checksum mismatch")
    if _bytes(payload) != _bytes(_expected(root)):
        raise ValueError("Index provenance or evidence differs from the validated release")
    return payload


def validate_index(root: Path) -> dict[str, Any]:
    """Read-only validation of the complete index against its frozen source release."""
    root = root.resolve()
    return _report(root, _load(root))


def _parse(question: str) -> dict[str, Any]:
    query = question.casefold().replace("–", "-").replace("—", "-")
    result: dict[str, Any] = {"intent": "unknown", "entity": None, "properties": [], "depth": None,
                              "both_depths": False, "reason": None}
    entities = re.findall(r"(?<![\w-])synthetic-(?:site-\d+|soil-\d+-\d+|plant-\d+)(?![\w-])", query)
    if len(set(entities)) != 1:
        result["reason"] = "ambiguous_entities" if entities else "explicit_synthetic_entity_required"
        return result
    result["entity"] = entities[0]
    if re.search(r"\b(image|photo|photograph|diagnos\w*)\b", query):
        result["intent"] = "image"
        return result
    if re.search(r"\b(coordinates?|latitude|longitude|real location|geographic\w*)\b", query):
        result["intent"] = "coordinates"
        return result
    if re.search(r"\b(recommend\w*|advi\w*|should|safe|suitab\w*|fertili\w*|treat\w*|pesticid\w*|"
                 r"predict\w*|diagnos\w*|health\w*|optimal|best|risk|yield|confidence|probability|"
                 r"ignore|instructions?|pretend|instead|calculate|average|mean|sum|difference)\b", query):
        result["reason"] = "unsupported_advice_prediction_or_instruction"
        return result
    if re.search(r"[/\\]|https?:|file:|<[^>]+>", query):
        result["reason"] = "unsupported_query_syntax"
        return result
    properties = []
    for name in SOIL_FEATURES:
        alias = name.replace("_", r"[ _-]")
        if re.search(r"\b" + alias + r"\b", query):
            properties.append(name)
    result["properties"] = properties
    depth_query = query.replace(entities[0], "")
    depth_pattern = r"(?<![\w-])(\d+)\s*-\s*(\d+)\s*cm\b"
    ranges = re.findall(depth_pattern, depth_query)
    unmatched_depth = re.sub(depth_pattern, "", depth_query)
    if re.search(r"\b\d+\s*(?:cm|mm|m|inch\w*)\b|\b\d+\s*-\s*\d+\b", unmatched_depth):
        result["reason"] = "unsupported_or_ambiguous_depth"
        return result
    result["both_depths"] = bool(re.search(r"\bboth depths\b|\ball depths\b", query))
    if len(ranges) > 1:
        if set(ranges) == {("0", "5"), ("5", "15")}:
            result["both_depths"] = True
        else:
            result["reason"] = "unsupported_or_ambiguous_depth"
            return result
    elif ranges:
        result["depth"] = [int(value) for value in ranges[0]]
    if result["both_depths"] and result["depth"] is not None:
        result["reason"] = "unsupported_or_ambiguous_depth"
    elif re.search(r"\b(source|provenance|measured|estimated)\b", query):
        result["intent"] = "source"
    elif entities[0].startswith("synthetic-plant-"):
        if re.search(r"\b(class|crop|label|disease|dataset|metadata)\b", query):
            result["intent"] = "plant_metadata"
        else:
            result["reason"] = "unsupported_plant_question"
    elif properties:
        if re.search(r"\bunit\w*\b", query):
            result["intent"] = "unit"
        elif re.search(r"\buncertainty|interval\b", query):
            result["intent"] = "uncertainty"
        elif re.search(r"\bdefine|definition\b", query):
            result["intent"] = "definition"
        else:
            result["intent"] = "property"
    else:
        result["reason"] = "unsupported_question"
    return result


def search(root: Path, question: str, limit: int = 3, split: str = "train") -> dict[str, Any]:
    """Return complete exact-entity evidence in one explicit split, or an explicit no-hit reason.

    Ranking is deterministic specificity (entity/depth/property), then record ID;
    it is not a probability or confidence. A limit overflow fails, never truncates.
    """
    if not isinstance(question, str) or not question.strip() or len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"question must contain 1..{MAX_QUESTION_CHARS} characters")
    if type(limit) is not int or not 1 <= limit <= MAX_HITS:
        raise ValueError(f"limit must be an integer in 1..{MAX_HITS}")
    if not isinstance(split, str) or split not in SPLITS:
        raise ValueError("split must explicitly be train, validation or test")
    root = root.resolve()
    payload = _load(root)
    parsed = _parse(question)
    result: dict[str, Any] = {"status": "YELLOW", "question": question, "split": split, "synthetic": True, "hits": [],
                             "reason": parsed["reason"], "query": parsed, "total_matches": 0,
                             "index_sha256": _sha(_bytes(payload)),
                             "manifest_sha256": payload["manifest_sha256"],
                             "ranking": "Exact synthetic entity, depth and property; record ID tie-break"}
    if parsed["reason"]:
        return result
    entity = parsed["entity"]
    rows = [row for row in payload["records"]
            if entity in (row["record_id"], row["location_id"])]
    if not rows:
        result["reason"] = "unknown_synthetic_entity"
        return result
    rows = [row for row in rows if row["split"] == split]
    if not rows:
        result["reason"] = "entity_not_in_requested_split"
        return result
    if parsed["depth"] is not None:
        rows = [row for row in rows if row["depth"] is not None
                and [row["depth"]["top"], row["depth"]["bottom"]] == parsed["depth"]]
    if not rows:
        result["reason"] = "depth_not_available"
        return result
    if parsed["both_depths"] and len(rows) != 2:
        result["reason"] = "incomplete_depth_evidence"
        return result
    result["total_matches"] = len(rows)
    if len(rows) > limit:
        result["reason"] = "evidence_overflow"
        return result
    for row in rows:
        row["rank_score"] = 100 + 20 * (parsed["depth"] is not None) + 5 * len(parsed["properties"])
        row["rank_basis"] = ["exact_synthetic_entity", "explicit_split"]
    result["hits"] = sorted(rows, key=lambda row: (-row["rank_score"], row["record_id"]))
    result["status"] = "GREEN"
    result["reason"] = None
    return result
