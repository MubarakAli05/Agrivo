"""Phase 10 shared canonical soil primitives used lazily by Phases 11 and 13."""

from copy import deepcopy
import math
from typing import Any

from agri.data_schema import SOIL_FEATURES
from sources._common import exact, number, text

# Physical unit conversions only; no inferred units, area/mass or OM-to-OC conversion.
UNIT_FACTORS = {
    "ph": {"1": 1.0, "pH x 10": 0.1},
    "organic_carbon": {"g/kg": 1.0, "dg/kg": 0.1, "%": 10.0},
    "nitrogen": {"g/kg": 1.0, "cg/kg": 0.01, "mg/kg": 0.001, "%": 10.0},
    "phosphorus": {"mg/kg": 1.0, "g/kg": 1000.0},
    "potassium": {"mg/kg": 1.0, "g/kg": 1000.0},
    "bulk_density": {"kg/dm3": 1.0, "g/cm3": 1.0, "cg/cm3": 0.01, "kg/m3": 0.001},
    "sand": {"%": 1.0, "g/kg": 0.1}, "silt": {"%": 1.0, "g/kg": 0.1},
    "clay": {"%": 1.0, "g/kg": 0.1},
    "cec": {"cmol(+)/kg": 1.0, "mmol(c)/kg": 0.1, "cmol(c)/kg": 1.0},
    "moisture": {"% volume": 1.0, "m3/m3": 100.0},
    "ec": {"dS/m": 1.0, "mS/cm": 1.0, "uS/cm": 0.001},
}


def _empty_soil() -> dict[str, Any]:
    result: dict[str, Any] = {
        "units": {name: spec["unit"] for name, spec in SOIL_FEATURES.items()},
        "uncertainty": {name: None for name in SOIL_FEATURES},
        "uncertainty_missing": {name: True for name in SOIL_FEATURES},
    }
    for name in SOIL_FEATURES:
        result[name] = None
        result[name + "_missing"] = True
    return result


def _depth(raw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    exact(raw, {"top", "bottom", "unit"}, "depth")
    unit = text(raw["unit"], "depth.unit")
    if unit not in ("cm", "m", "mm"):
        raise ValueError("Depth requires explicit cm, m or mm units")
    factor = {"cm": 1.0, "m": 100.0, "mm": 0.1}[unit]
    top = number(number(raw["top"], "depth.top", 0) * factor, "depth.top_cm", 0)
    bottom = number(number(raw["bottom"], "depth.bottom", 0) * factor, "depth.bottom_cm", 0)
    if top >= bottom:
        raise ValueError("Depth must be a nonempty increasing interval")
    return ({"top": top, "bottom": bottom, "unit": "cm"},
            {"operation": "depth_conversion", "input_unit": unit, "output_unit": "cm", "factor": factor})


def _coordinates(latitude: Any, longitude: Any, crs: Any, unit: Any) -> tuple[float | None, float | None]:
    missing = latitude is None and longitude is None
    if not (missing and crs is None and unit is None) and (crs != "EPSG:4326" or unit != "degree"):
        raise ValueError("Coordinates require explicit EPSG:4326 CRS and degree units; no CRS/unit inference or reprojection")
    if missing:
        return None, None
    return number(latitude, "latitude", -90, 90), number(longitude, "longitude", -180, 180)


def _measurement(record: dict[str, Any], feature: str, raw: Any,
                 conversions: dict[str, dict[str, float]] = UNIT_FACTORS) -> dict[str, Any]:
    exact(raw, {"value", "unit", "uncertainty"}, feature)
    unit = text(raw["unit"], feature + ".unit")
    if unit not in conversions[feature]:
        raise ValueError(f"{feature}: unsupported explicit unit {unit}")
    factor = conversions[feature][unit]
    spec = SOIL_FEATURES[feature]

    def converted(value: Any) -> float:
        return number(number(value, feature) * factor, feature, spec["minimum"], spec["maximum"])

    value = None if raw["value"] is None else converted(raw["value"])
    record[feature] = value
    record[feature + "_missing"] = value is None
    interval = raw["uncertainty"]
    if interval is not None:
        exact(interval, {"lower", "upper", "method"}, feature + ".uncertainty")
        text(interval["method"], "uncertainty.method")
        lower, upper = converted(interval["lower"]), converted(interval["upper"])
        if value is None or not lower <= value <= upper:
            raise ValueError("Uncertainty interval must bracket a present value")
        record["uncertainty"][feature] = {"lower": lower, "upper": upper, "method": interval["method"]}
        record["uncertainty_missing"][feature] = False
    return {"operation": "unit_conversion", "feature": feature, "input_unit": unit,
            "output_unit": spec["unit"], "factor": factor, "applies_to": "value_and_uncertainty"}


def _texture(record: dict[str, Any]) -> None:
    fractions = [record[name] for name in ("sand", "silt", "clay")]
    if all(value is not None for value in fractions) and not math.isclose(sum(fractions), 100, abs_tol=1):
        raise ValueError("Complete texture fractions must sum to 100 percent (rounding tolerance 1 point)")


def _nonoverlap(intervals: dict[str, list[tuple[float, float]]], key: str,
                depth: dict[str, Any]) -> None:
    top, bottom = depth["top"], depth["bottom"]
    previous = intervals.setdefault(key, [])
    if any(top < end and start < bottom for start, end in previous):
        raise ValueError("Duplicate or overlapping layers for one profile/location")
    previous.append((top, bottom))


def _soil_schema() -> dict[str, Any]:
    return {
        "soil_features": deepcopy(SOIL_FEATURES), "unit_conversion_factors": deepcopy(UNIT_FACTORS),
        "depth_schema": {"required_fields": ["top", "bottom", "unit"], "additional_fields": False,
                         "top": "finite JSON number >=0", "bottom": "finite JSON number >top",
                         "unit_factors_to_cm": {"cm": 1.0, "m": 100.0, "mm": 0.1}},
        "measurement_schema": {"required_fields": ["value", "unit", "uncertainty"],
                               "additional_fields": False, "value": "finite JSON number or null",
                               "unit": "explicit supported unit, even for missing values",
                               "uncertainty": "null or {lower: number, upper: number, method: nonempty text}; interval must bracket value"},
        "missingness": "Absent properties and null values remain null with <feature>_missing=true. Every feature has uncertainty and uncertainty_missing entries; no missing-to-zero conversion.",
        "texture_policy": "Complete sand/silt/clay percentages must sum to 100 with 1 percentage-point absolute tolerance; never renormalized.",
    }
