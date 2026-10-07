"""Evidence-first QA: retrieve, validate, condition the custom model, then render facts.

Neural output is an unverified draft, never agricultural advice or factual authority.
Only complete local synthetic records support the deterministic final renderer.
"""

import json
from pathlib import Path
from typing import Any

from agri.config import load_config
from agri.data_schema import SOIL_FEATURES
from models.transformer.packing import pack_example
from retrieval.index import search
from tokenizer.structured import StructuredSoilTokenizer
from tokenizer.tokenizer import BPETokenizer
from tokenizer.train_tokenizer import tokenizer_path, validate_tokenizer


LIMITATIONS = [
    "Synthetic fixtures only: no measured soil, real geography or agricultural ground truth.",
    "No recommendations, diagnosis, computed predictions or calibrated confidence.",
    "Plant metadata has no images; external sources remain disabled pending license approval.",
    "Neural drafts are not verified facts and are withheld from the factual answer.",
]


def _validate_checkpoint(root: Path) -> dict[str, Any]:
    from training.trainer import validate_training
    return validate_training(root)


def _generate(root: Path, packed: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
    from models.transformer.inference import generate_from_packed
    return generate_from_packed(root, packed, max_new_tokens=max_new_tokens)


def _pack_query(root: Path, example: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
    validate_tokenizer(root)
    directory = tokenizer_path(root)
    text = BPETokenizer.load(directory)
    structured = StructuredSoilTokenizer.load(directory / "structured.json")
    context_length = load_config(root / "configs" / "agri-mini.json")["model"]["context_length"]
    packed = pack_example(text, structured, example, context_length)
    # Inference removes the empty-answer EOS, but must retain every evidence position.
    if len(packed["input_ids"]) - 1 + max_new_tokens > context_length:
        raise ValueError("Complete evidence plus generation exceeds context_length; no truncation allowed")
    return packed


def _render(retrieved: dict[str, Any]) -> tuple[str | None, str | None]:
    query, hits = retrieved["query"], retrieved["hits"]
    intent, properties = query["intent"], query["properties"]
    if intent == "image":
        return None, "no_images_or_diagnostic_model"
    if intent == "coordinates":
        return None, "no_real_coordinates"
    lines = []
    for hit in hits:
        record, citation = hit["record"], f"[{hit['record_id']}]"
        if intent == "source":
            lines.append(f"{citation} Source: {hit['source']}; version: {hit['source_version']}. "
                         "This is synthetic, not measured or estimated field data.")
        elif intent == "plant_metadata":
            if any(record.get(name) is None for name in ("label", "crop", "disease", "dataset_name")):
                return None, "missing_plant_metadata"
            lines.append(f"{citation} Synthetic metadata: class {record['label']}; crop {record['crop']}; "
                         f"condition {record['disease']}; dataset {record['dataset_name']}. "
                         "These are fictional test labels, not disease evidence.")
        elif intent in ("property", "unit", "uncertainty", "definition"):
            if not properties or hit["kind"] != "soil":
                return None, "unsupported_question"
            depth = record["depth"]
            location = f"{record['location_id']} at {depth['top']}-{depth['bottom']} {depth['unit']}"
            for name in properties:
                unit = record["units"].get(name)
                if unit is None:
                    return None, "missing_units"
                if intent == "unit":
                    fact = f"{name} unit: {unit}"
                elif intent == "definition":
                    if name != "ph":
                        return None, "unsupported_definition"
                    fact = f"ph is a numeric field in this synthetic schema; unit: {SOIL_FEATURES[name]['unit']}"
                elif record.get(name) is None or record.get(name + "_missing") is not False:
                    return None, "missing_value"
                elif intent == "uncertainty":
                    interval = record["uncertainty"].get(name)
                    if interval is None:
                        return None, "missing_uncertainty"
                    fact = (f"{name} synthetic interval: {interval['lower']} to {interval['upper']} {unit}; "
                            f"method: {interval['method']}; not calibrated confidence")
                else:
                    fact = f"{name}: {record[name]} {unit}"
                lines.append(f"{citation} Synthetic {location}: {fact}.")
        else:
            return None, "unsupported_question"
    return "\n".join(lines) + "\nThese are fixture values only, not real geographic or agricultural facts.", None


def answer_query(root: Path, question: str, max_new_tokens: int = 32) -> dict[str, Any]:
    """Answer from train-only evidence and a validated checkpoint, otherwise abstain.

    ``answer_origin`` distinguishes the evidence renderer from the untrusted neural
    draft. Full records and checksum-bound citations remain available on abstention.
    GREEN means a supported synthetic answer, YELLOW an expected abstention, and
    RED a corrupt dependency or execution failure; none denotes agronomic quality.
    The model prompt contains an EMPTY answer and never any fixture QA answer.
    """
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 128:
        raise ValueError("max_new_tokens must be an integer in 1..128")
    root = root.resolve()
    result: dict[str, Any] = {
        "phase": 9, "status": "YELLOW", "question": question,
        "answer": "UNKNOWN: verified evidence is unavailable.",
        "data_used": "None; no verified local evidence retrieved.",
        "sources": [], "evidence": [], "unknown": True, "synthetic": True, "model_used": False,
        "model_version": None, "model_checkpoint": None,
        "answer_origin": "abstention", "neural_draft": None, "confidence": None,
        "limitations": list(LIMITATIONS), "reason": None, "split": "train",
    }

    def unknown(reason: str, status: str = "YELLOW") -> dict[str, Any]:
        result["reason"] = reason
        result["status"] = status
        result["answer"] = f"UNKNOWN: {reason.replace('_', ' ')}."
        return result

    try:
        retrieved = search(root, question, split="train")
    except (OSError, ValueError) as exc:
        # Input bounds remain explicit API errors rather than looking like a source outage.
        if not isinstance(question, str) or not question.strip() or len(question) > 512:
            raise ValueError("question must contain 1..512 characters") from exc
        if isinstance(exc, FileNotFoundError):
            return unknown("source_or_index_missing")
        return unknown("source_or_index_invalid", "RED")
    result["retrieval"] = {key: value for key, value in retrieved.items() if key != "hits"}
    result["evidence"] = retrieved["hits"]
    if retrieved["hits"]:
        result["data_used"] = (f"Retrieved {len(retrieved['hits'])} validated local synthetic fixture records "
                               "from the train split; no external data or real observations.")
    result["sources"] = [{key: hit[key] for key in (
        "source", "source_version", "record_id", "source_path", "source_line", "source_sha256",
        "record_sha256", "manifest_sha256", "manifest_path", "original_url", "license", "usage",
        "timestamp", "synthetic", "split")}
        for hit in retrieved["hits"]]
    if retrieved["reason"] is not None or not retrieved["hits"]:
        return unknown(retrieved["reason"] or "no_evidence")
    answer, reason = _render(retrieved)
    if reason is not None:
        return unknown(reason)
    records = [hit["record"] for hit in retrieved["hits"]]
    group = records[0].get("location_id", records[0].get("group_id"))
    if not group or any(row.get("location_id", row.get("group_id")) != group for row in records):
        return unknown("ambiguous_evidence_groups")
    example = {
        "example_id": "retrieval-query", "group_id": group, "split": "train", "question": question,
        "context": json.dumps({"synthetic": True, "records": records}, sort_keys=True, ensure_ascii=False,
                              allow_nan=False),
        "answer": "", "source_ids": sorted({row["source"] for row in records}),
        "evidence_ids": [row["record_id"] for row in records], "synthetic": True,
    }
    try:
        _validate_checkpoint(root)
    except (ImportError, FileNotFoundError):
        return unknown("checkpoint_missing_or_unavailable")
    except (OSError, ValueError, RuntimeError):
        return unknown("checkpoint_invalid", "RED")
    try:
        packed = _pack_query(root, example, max_new_tokens)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        if "truncation" in str(exc):
            return unknown("evidence_context_overflow")
        if isinstance(exc, (ImportError, FileNotFoundError)):
            return unknown("tokenizer_or_context_unavailable")
        return unknown("tokenizer_or_context_invalid", "RED")
    try:
        generated = _generate(root, packed, max_new_tokens)
        if (not isinstance(generated, dict) or generated.get("status") != "GREEN"
                or generated.get("synthetic") is not True or generated.get("trained") is not True):
            return unknown("invalid_model_result", "RED")
    except (ImportError, FileNotFoundError):
        return unknown("model_generation_unavailable")
    except (OSError, ValueError, RuntimeError):
        return unknown("model_generation_failed", "RED")
    result.update(answer=answer, status="GREEN", unknown=False, model_used=True, reason=None,
                  model_version="AgriTransformer-v0", model_checkpoint=generated.get("checkpoint"),
                  answer_origin="deterministic_evidence_renderer",
                  neural_draft={"status": "withheld_unverified", "used_as_factual_answer": False},
                  model_role="Evidence-conditioned neural draft; deterministic evidence is the factual authority")
    return result
