"""Phase 12: explicitly mapped data.gov.in crop-statistics snapshot exports."""

import csv
import io
import re
from typing import Any

from sources._common import exact, identifier, number, rows, text

_COLUMNS = ["record_id", "resource_id", "state", "district", "crop", "year", "season", "metric", "value", "unit"]
_CONVERSIONS = {
    "area": {"ha": 1.0, "hectare": 1.0, "acre": 0.40468564224},
    "production": {"tonne": 1.0, "kg": 0.001, "quintal": 0.1},
    "yield": {"tonne/ha": 1.0, "kg/ha": 0.001, "quintal/ha": 0.1},
}
_CANONICAL = {"area": "ha", "production": "tonne", "yield": "tonne/ha"}
_DECIMAL = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?\Z")


def _schema() -> dict[str, Any]:
    return {
        "formats": ["json", "csv"],
        "records_type": "JSON array of crop-statistics objects, or CSV text inside the JSON envelope records field",
        "required_columns": list(_COLUMNS), "additional_columns": False,
        "column_types": {"record_id": "unique identifier", "resource_id": "identifier equal to envelope dataset_id",
                         "state": "nonempty text", "district": "nonempty text or null (CSV empty)",
                         "crop": "nonempty text", "year": "JSON integer in [1,9999]; CSV digits only",
                         "season": "nonempty text", "metric": list(_CONVERSIONS),
                         "value": "finite nonnegative JSON number or null; CSV strict decimal/scientific text or empty",
                         "unit": "explicit member of unit_conversion_factors for the selected metric"},
        "csv_policy": "RFC-style CSV header exactly required_columns in any order, unique headers, equal field counts; no trimming/coercing labels, no NA/null/Infinity/bool tokens; empty value/district means missing",
        "unit_conversion_factors": _CONVERSIONS, "canonical_units": _CANONICAL,
        "unit_definitions": "1 hectare=10000 m2; 1 international acre=4046.8564224 m2; 1 tonne=1000 kg; 1 quintal=100 kg. No lakh/thousand multipliers, fertilizer measures or crop-derived yield inference.",
        "canonical_output": ["record_id", "resource_id", "state", "district", "district_missing", "crop",
                             "year", "season", "metric", "value", "value_missing", "unit", "raw", "provenance"],
        "limitations": ["The portal catalog is not a reviewed dataset; existing source/license/inspection/catalog gates remain active.",
                        "Only explicitly mapped area/production/yield resource exports are accepted, not arbitrary API JSON/columns or all government datasets.",
                        "CSV is embedded in a metadata envelope so version, resource ID, units, citations and rights cannot be omitted.",
                        "No API keys, HTTP requests, live endpoint availability checks or derived yield calculation. Real-data quality remains UNVERIFIED."],
    }


def _csv_rows(value: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(value, str):
        raise ValueError("CSV records must be CSV text in the JSON envelope")
    try:
        reader = csv.reader(io.StringIO(value, newline=""), strict=True)
        header = next(reader)
        if len(header) != len(_COLUMNS) or set(header) != set(_COLUMNS):
            raise ValueError("CSV header must contain exactly the required unique columns")
        original = []
        parsed = []
        for values in reader:
            if len(values) != len(header):
                raise ValueError("CSV row has incorrect field count")
            raw = dict(zip(header, values))
            row: dict[str, Any] = dict(raw)
            if not re.fullmatch(r"[1-9]\d{0,3}", row["year"]):
                raise ValueError("CSV year must contain integer digits only")
            row["year"] = int(row["year"])
            if row["value"] == "":
                row["value"] = None
            elif not _DECIMAL.fullmatch(row["value"]):
                raise ValueError("CSV value requires explicit decimal numeric format")
            else:
                row["value"] = number(float(row["value"]), "CSV value", 0)
            if row["district"] == "":
                row["district"] = None
            original.append(raw)
            parsed.append(row)
    except (csv.Error, StopIteration) as exc:
        raise ValueError("Invalid or empty CSV snapshot") from exc
    if not parsed:
        raise ValueError("CSV snapshot contains no records")
    return parsed, original


def _normalize(snapshot: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if snapshot["format"] == "csv":
        parsed, originals = _csv_rows(snapshot["records"])
    else:
        parsed = rows(snapshot["records"])
        originals = parsed
    output, history = [], []
    observations = set()
    for row, raw in zip(parsed, originals):
        exact(row, set(_COLUMNS), "data.gov.in resource row")
        for key in ("record_id", "resource_id"):
            identifier(row[key], key)
        if row["resource_id"] != snapshot["dataset_id"]:
            raise ValueError("resource_id must match the reviewed snapshot dataset_id")
        for key in ("state", "crop", "season"):
            text(row[key], key)
        if row["district"] is not None:
            text(row["district"], "district")
        if type(row["year"]) is not int or not 1 <= row["year"] <= 9999:
            raise ValueError("year must be an integer in [1,9999]")
        metric = text(row["metric"], "metric")
        if metric not in _CONVERSIONS:
            raise ValueError("Unsupported data.gov.in metric")
        unit = text(row["unit"], "unit")
        if unit not in _CONVERSIONS[metric]:
            raise ValueError("Unsupported or ambiguous data.gov.in unit")
        observation = tuple(row[key] for key in ("resource_id", "state", "district", "crop", "year", "season", "metric"))
        if observation in observations:
            raise ValueError("Duplicate or ambiguous resource observation")
        observations.add(observation)
        factor = _CONVERSIONS[metric][unit]
        value = None if row["value"] is None else number(number(row["value"], "value", 0) * factor, "converted value", 0)
        output.append({**row, "value": value, "value_missing": value is None,
                       "district_missing": row["district"] is None, "unit": _CANONICAL[metric], "raw": raw})
        history.append({"record_id": row["record_id"], "operation": "normalize_crop_statistic",
                        "metric": metric, "input_unit": unit, "output_unit": _CANONICAL[metric],
                        "factor": factor, "numeric_parser": "strict_csv_decimal" if snapshot["format"] == "csv" else "json_number"})
    return output, history
