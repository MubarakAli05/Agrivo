"""Train-only structured IDs plus lossless numeric/provenance side channels."""

from bisect import bisect_left
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

from agri.data_schema import SOIL_FEATURES


CATEGORIES = ("crop", "region", "source", "source_version", "depth")


def _json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _finite(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate structured tokenizer key: {key}")
        result[key] = value
    return result


class StructuredSoilTokenizer:
    """Fixed ordered field IDs; raw numbers are retained, never reconstructed from bins."""

    pad_token_id = 0

    def __init__(self, n_bins: int = 16) -> None:
        if type(n_bins) is not int or not 1 <= n_bins <= 256:
            raise ValueError("n_bins must be an integer in [1, 256]")
        self.n_bins = n_bins
        self._state: dict[str, Any] | None = None
        self._vocab: dict[str, int] = {}

    @staticmethod
    def _validate(row: dict[str, Any]) -> None:
        if not isinstance(row, dict):
            raise ValueError("Expected a structured soil record")
        if row.get("units") != {name: spec["unit"] for name, spec in SOIL_FEATURES.items()}:
            raise ValueError("Canonical units required; implicit unit conversion is forbidden")
        uncertainty = row.get("uncertainty")
        if not isinstance(uncertainty, dict) or set(uncertainty) != set(SOIL_FEATURES):
            raise ValueError("Explicit uncertainty fields required")
        for name, spec in SOIL_FEATURES.items():
            if name not in row or type(row.get(name + "_missing")) is not bool:
                raise ValueError(f"{name}: explicit value and missingness required")
            value = row[name]
            if row[name + "_missing"] != (value is None):
                raise ValueError(f"{name}: inconsistent missingness")
            if value is not None and (not _finite(value) or value < spec["minimum"]
                                      or (spec["maximum"] is not None and value > spec["maximum"])):
                raise ValueError(f"{name}: invalid numeric value")
            interval = uncertainty[name]
            if interval is not None:
                if (not isinstance(interval, dict) or set(interval) != {"lower", "upper", "method"}
                        or value is None or not _finite(interval["lower"]) or not _finite(interval["upper"])
                        or not interval["lower"] <= value <= interval["upper"]
                        or not isinstance(interval["method"], str) or not interval["method"].strip()):
                    raise ValueError(f"{name}: invalid uncertainty interval")
        depth = row.get("depth")
        if (not isinstance(depth, dict) or set(depth) != {"top", "bottom", "unit"}
                or depth["unit"] != "cm" or not _finite(depth["top"]) or not _finite(depth["bottom"])
                or not 0 <= depth["top"] < depth["bottom"]):
            raise ValueError("Expected a finite increasing depth interval in cm")
        for name in CATEGORIES[:-1]:
            if name not in row or (row[name] is not None and
                                  (not isinstance(row[name], str) or not row[name].strip())):
                raise ValueError(f"{name}: expected text or explicit null")

    @staticmethod
    def _category(row: dict[str, Any], name: str) -> str | None:
        if name == "depth":
            depth = row["depth"]
            return _json(depth).decode("utf-8").strip()
        return row[name]

    def fit(self, rows: list[dict[str, Any]]) -> "StructuredSoilTokenizer":
        if not rows:
            raise ValueError("Training rows must not be empty")
        for row in rows:
            self._validate(row)
            if row.get("split") != "train":
                raise ValueError("Only explicitly training-split rows may fit structured tokens")
        features = {}
        for name in SOIL_FEATURES:
            values = sorted(row[name] for row in rows if row[name] is not None)
            cuts = sorted({values[(i * len(values) + self.n_bins - 1) // self.n_bins - 1]
                           for i in range(1, self.n_bins)} - {values[-1]}) if values else []
            features[name] = {"cuts": cuts, "minimum": values[0] if values else None,
                              "maximum": values[-1] if values else None, "count": len(values)}
        categories = {name: sorted({self._category(row, name) for row in rows
                                    if self._category(row, name) is not None}) for name in CATEGORIES}
        self._state = {"version": 1, "n_bins": self.n_bins, "features": features, "categories": categories,
                       "units": {name: spec["unit"] for name, spec in SOIL_FEATURES.items()}}
        self._make_vocab()
        return self

    def _make_vocab(self) -> None:
        assert self._state is not None
        names = ["<pad>"]
        for name in SOIL_FEATURES:
            names.extend([f"{name}:missing", f"{name}:uncalibrated"])
            names.extend(f"{name}:bin:{index}" for index in range(self.n_bins))
            names.extend([f"{name}:unit", f"{name}:uncertainty:missing", f"{name}:uncertainty:present"])
        for name in CATEGORIES:
            names.extend([f"{name}:missing", f"{name}:unknown"])
            names.extend(f"{name}:category:{index}" for index in range(len(self._state["categories"][name])))
        self._vocab = {name: index for index, name in enumerate(names)}

    @property
    def vocab_size(self) -> int:
        if self._state is None:
            raise RuntimeError("Fit or load the structured tokenizer first")
        return len(self._vocab)

    def encode(self, row: dict[str, Any]) -> dict[str, Any]:
        if self._state is None:
            raise RuntimeError("Fit or load the structured tokenizer first")
        self._validate(row)
        ids = []
        numeric = {}
        for name in SOIL_FEATURES:
            value, stats = row[name], self._state["features"][name]
            label = "missing" if value is None else (
                "uncalibrated" if not stats["count"] else f"bin:{bisect_left(stats['cuts'], value)}")
            token = self._vocab[f"{name}:{label}"]
            interval = row["uncertainty"][name]
            ids.extend([token, self._vocab[f"{name}:unit"],
                        self._vocab[f"{name}:uncertainty:{'missing' if interval is None else 'present'}"]])
            numeric[name] = {"token_id": token, "value": value, "missing": value is None,
                             "unit": row["units"][name], "uncertainty": deepcopy(interval),
                             "calibrated": bool(stats["count"]),
                             "outside_training_range": None if value is None or not stats["count"] else
                             not stats["minimum"] <= value <= stats["maximum"]}
        categories = {}
        for name in CATEGORIES:
            value = self._category(row, name)
            known = self._state["categories"][name]
            label = "missing" if value is None else (f"category:{known.index(value)}" if value in known else "unknown")
            token = self._vocab[f"{name}:{label}"]
            ids.append(token)
            categories[name] = {"token_id": token, "value": deepcopy(row[name]), "known": value in known}
        return {"token_ids": ids, "numeric": numeric, "categories": categories}

    def artifacts(self) -> dict[str, bytes]:
        if self._state is None:
            raise RuntimeError("Cannot serialize an unfitted structured tokenizer")
        return {"structured.json": _json(self._state)}

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(self.artifacts()["structured.json"])

    @classmethod
    def load(cls, path: str | Path) -> "StructuredSoilTokenizer":
        state = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique)
        try:
            if (not isinstance(state, dict) or set(state) != {"version", "n_bins", "features", "categories", "units"}
                    or type(state["version"]) is not int or state["version"] != 1):
                raise ValueError("Invalid structured tokenizer format")
            tokenizer = cls(state["n_bins"])
            if (set(state["features"]) != set(SOIL_FEATURES) or set(state["categories"]) != set(CATEGORIES)
                    or state["units"] != {name: spec["unit"] for name, spec in SOIL_FEATURES.items()}):
                raise ValueError("Unexpected fields or units")
            for name, stats in state["features"].items():
                if not isinstance(stats, dict) or set(stats) != {"count", "cuts", "minimum", "maximum"}:
                    raise ValueError("Invalid numeric calibration")
                cuts, count = stats["cuts"], stats["count"]
                if (type(count) is not int or count < 0 or not isinstance(cuts, list)
                        or len(cuts) >= tokenizer.n_bins or any(not _finite(cut) for cut in cuts)
                        or any(a >= b for a, b in zip(cuts, cuts[1:]))):
                    raise ValueError("Invalid quantile boundaries")
                lo, hi = stats["minimum"], stats["maximum"]
                spec = SOIL_FEATURES[name]
                if count == 0:
                    if cuts or lo is not None or hi is not None:
                        raise ValueError("Uncalibrated features must not have invented statistics")
                elif (not _finite(lo) or not _finite(hi) or not spec["minimum"] <= lo <= hi
                      or (spec["maximum"] is not None and hi > spec["maximum"])
                      or any(not lo <= cut < hi for cut in cuts)):
                    raise ValueError("Invalid observed range")
            for values in state["categories"].values():
                if (not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values)
                        or values != sorted(set(values))):
                    raise ValueError("Invalid category vocabulary")
        except (KeyError, TypeError, OverflowError, AttributeError) as exc:
            raise ValueError("Invalid structured tokenizer data") from exc
        tokenizer._state = state
        tokenizer._make_vocab()
        return tokenizer
