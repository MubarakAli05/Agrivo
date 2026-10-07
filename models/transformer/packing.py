"""Intact synthetic QA streams with separate structured IDs and numeric channels."""

from copy import deepcopy
import json
import math
from typing import Any

from agri.data_schema import SOIL_FEATURES, SPLITS, validate_soil_record
from tokenizer.structured import StructuredSoilTokenizer
from tokenizer.tokenizer import BPETokenizer


NUMERIC_CHANNELS = ("value", "lower", "upper", "missing", "uncertainty_missing",
                    "uncalibrated", "outside_training_range")


def _signed_log(value: float | None) -> float:
    return 0.0 if value is None else math.copysign(math.log1p(abs(value)), value)


def pack_example(text: BPETokenizer, structured: StructuredSoilTokenizer,
                 example: dict[str, Any], context_length: int) -> dict[str, Any]:
    """Preserve full normalized text and raw evidence; reject overflow instead of truncating."""
    if type(context_length) is not int or context_length < 1:
        raise ValueError("context_length must be a positive integer")
    required = {"question", "context", "answer", "example_id", "group_id", "split",
                "source_ids", "evidence_ids", "synthetic"}
    if not isinstance(example, dict) or not required <= example.keys():
        raise ValueError("Expected a complete synthetic QA example")
    if example["synthetic"] is not True or example["split"] not in SPLITS:
        raise ValueError("Only explicitly split synthetic examples are supported")
    if any(not isinstance(example[key], str) for key in ("question", "context", "answer")):
        raise ValueError("Question, context and answer must be text")
    context = json.loads(example["context"])
    if not isinstance(context, dict) or context.get("synthetic") is not True or not isinstance(context.get("records"), list):
        raise ValueError("Expected explicit synthetic context records")
    records = context["records"]
    record_ids = set()
    for row in records:
        if not isinstance(row, dict) or not isinstance(row.get("record_id"), str):
            raise ValueError("Invalid context record")
        if row["record_id"] in record_ids:
            raise ValueError("Duplicate context record")
        record_ids.add(row["record_id"])
        if (row.get("synthetic") is not True or row.get("split") != example["split"]
                or row.get("location_id", row.get("group_id")) != example["group_id"]):
            raise ValueError("Context record crosses synthetic group/split boundaries")
        if row.get("source") not in example["source_ids"]:
            raise ValueError("Context source is not declared by the QA example")
    if not example["evidence_ids"] or not set(example["evidence_ids"]) <= record_ids:
        raise ValueError("Missing cited evidence in context")

    ids: list[int] = []
    structured_ids: list[int] = []
    numeric_features: list[int] = []
    numeric_values: list[list[float]] = []
    labels: list[int] = []
    spans: dict[str, list[int]] = {}
    raw_structured = []

    def append(token: int = 0, structured_id: int = 0, feature: int = -1,
               values: list[float] | None = None) -> None:
        ids.append(token)
        structured_ids.append(structured_id)
        numeric_features.append(feature)
        numeric_values.append(values if values is not None else [0.0] * 7)
        labels.append(token if token else -100)

    def control(name: str) -> None:
        append(text.token_to_id(name))

    def field(name: str) -> None:
        start = len(ids)
        for token in text.encode(example[name]):
            append(token)
        spans[name] = [start, len(ids)]

    control("<bos>")
    control("<question>")
    field("question")
    control("<context>")
    for row in records:
        if "location_id" not in row:
            continue
        validate_soil_record(row)
        encoded = structured.encode(row)
        raw_structured.append({"record_id": row["record_id"], **encoded})
        control("<source>")
        for index, token in enumerate(encoded["token_ids"]):
            feature, values = -1, None
            if index < 36 and index % 3 == 0:
                feature = index // 3
                numeric = encoded["numeric"][list(SOIL_FEATURES)[feature]]
                interval = numeric["uncertainty"]
                values = [_signed_log(numeric["value"]),
                          _signed_log(interval["lower"]) if interval else 0.0,
                          _signed_log(interval["upper"]) if interval else 0.0,
                          float(numeric["missing"]), float(interval is None),
                          float(not numeric["calibrated"]), float(numeric["outside_training_range"] is True)]
            elif index == 40:
                feature = 12
                values = [0.0, _signed_log(row["depth"]["top"]), _signed_log(row["depth"]["bottom"]),
                          0.0, 0.0, 0.0, 0.0]
            append(structured_id=token, feature=feature, values=values)
        control("<sep>")
    field("context")
    control("<answer>")
    field("answer")
    control("<eos>")
    if len(ids) > context_length:
        raise ValueError(f"{example['example_id']}: {len(ids)} positions exceed context_length={context_length}; no truncation allowed")
    return {"input_ids": ids, "structured_ids": structured_ids, "numeric_features": numeric_features,
            "numeric_values": numeric_values, "attention_mask": [True] * len(ids), "labels": labels,
            "text_spans": spans, "raw_structured": raw_structured,
            "example_id": example["example_id"], "split": example["split"], "group_id": example["group_id"],
            "source_ids": deepcopy(example["source_ids"]), "evidence_ids": deepcopy(example["evidence_ids"])}


def collate_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Right-pad one split without mixing group identities or supervising structured IDs."""
    import torch

    if not examples:
        raise ValueError("Cannot collate an empty batch")
    if len({example["split"] for example in examples}) != 1:
        raise ValueError("A batch must contain only one split")
    length = max(len(example["input_ids"]) for example in examples)
    result = {}
    for name, fill, dtype in (("input_ids", 0, torch.long), ("structured_ids", 0, torch.long),
                              ("numeric_features", -1, torch.long), ("numeric_values", [0.0] * 7, torch.float32),
                              ("attention_mask", False, torch.bool), ("labels", -100, torch.long)):
        rows = [example[name] + [fill] * (length - len(example["input_ids"])) for example in examples]
        result[name] = torch.tensor(rows, dtype=dtype)
    return result
