"""Canonical units and validation for Phase 3 synthetic soil fixtures only."""

import math
from typing import Any


SOIL_FEATURES = {
    "ph": {"unit": "1", "minimum": 0, "maximum": 14},
    "organic_carbon": {"unit": "g/kg", "minimum": 0, "maximum": None},
    "nitrogen": {"unit": "g/kg", "minimum": 0, "maximum": None},
    "phosphorus": {"unit": "mg/kg", "minimum": 0, "maximum": None},
    "potassium": {"unit": "mg/kg", "minimum": 0, "maximum": None},
    "bulk_density": {"unit": "kg/dm3", "minimum": 0, "maximum": None},
    "sand": {"unit": "%", "minimum": 0, "maximum": 100},
    "silt": {"unit": "%", "minimum": 0, "maximum": 100},
    "clay": {"unit": "%", "minimum": 0, "maximum": 100},
    "cec": {"unit": "cmol(+)/kg", "minimum": 0, "maximum": None},
    "moisture": {"unit": "% volume", "minimum": 0, "maximum": 100},
    "ec": {"unit": "dS/m", "minimum": 0, "maximum": None},
}
SPLITS = ("train", "validation", "test")
SOURCE_ID = "agrimini_synthetic_fixture"
DATASET_VERSION = "synthetic-v0.1"


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def validate_soil_record(record: Any) -> None:
    metadata = {"record_id", "location_id", "latitude", "longitude", "depth", "crop",
                "region", "source", "source_version", "timestamp", "synthetic", "value_origin",
                "split", "units", "uncertainty"}
    expected = metadata | set(SOIL_FEATURES) | {name + "_missing" for name in SOIL_FEATURES}
    if not isinstance(record, dict) or set(record) != expected:
        raise ValueError("Invalid synthetic soil schema")
    for name in ("record_id", "location_id", "crop", "region"):
        if not isinstance(record[name], str) or not record[name].strip():
            raise ValueError(f"{name}: expected nonempty text")
    if (record["source"] != SOURCE_ID or record["source_version"] != DATASET_VERSION
            or record["synthetic"] is not True or record["value_origin"] != "SYNTHETIC"):
        raise ValueError("Synthetic origin must be explicit; no real-source attribution allowed")
    if any(record[key] is not None for key in ("latitude", "longitude", "timestamp")):
        raise ValueError("Fictional fixtures must not claim real coordinates or observation times")
    if record["split"] not in SPLITS:
        raise ValueError("Invalid split")
    depth = record["depth"]
    if (not isinstance(depth, dict) or set(depth) != {"top", "bottom", "unit"}
            or depth["unit"] != "cm" or not _number(depth["top"])
            or not _number(depth["bottom"]) or not 0 <= depth["top"] < depth["bottom"]):
        raise ValueError("Invalid depth interval")
    if record["units"] != {key: spec["unit"] for key, spec in SOIL_FEATURES.items()}:
        raise ValueError("Unexpected soil units; no implicit conversion is supported")
    uncertainty = record["uncertainty"]
    if not isinstance(uncertainty, dict) or set(uncertainty) != set(SOIL_FEATURES):
        raise ValueError("Each property requires explicit uncertainty missingness")
    for name, spec in SOIL_FEATURES.items():
        value = record[name]
        if type(record[name + "_missing"]) is not bool or record[name + "_missing"] != (value is None):
            raise ValueError(f"{name}: missingness mask disagrees with value")
        if value is not None:
            if (not _number(value) or value < spec["minimum"]
                    or (spec["maximum"] is not None and value > spec["maximum"])):
                raise ValueError(f"{name}: invalid numeric value")
        interval = uncertainty[name]
        if interval is not None:
            if (value is None or not isinstance(interval, dict)
                    or set(interval) != {"lower", "upper", "method"}
                    or interval["method"] != "synthetic_interval_not_calibrated"
                    or not _number(interval["lower"]) or not _number(interval["upper"])
                    or not interval["lower"] <= value <= interval["upper"]):
                raise ValueError(f"{name}: invalid synthetic uncertainty interval")
    texture = [record[name] for name in ("sand", "silt", "clay")]
    if all(value is not None for value in texture) and not math.isclose(sum(texture), 100, abs_tol=1e-6):
        raise ValueError("Complete texture fractions must sum to 100 percent")
