"""Phase 11: offline SoilGrids point/layer export, not a REST dependency."""

from typing import Any

from sources._common import exact, identifier, rows
from sources._soil import (
    _coordinates, _depth, _empty_soil, _measurement, _nonoverlap, _soil_schema, _texture,
)

_PROPERTIES = {"phh2o": "ph", "soc": "organic_carbon", "nitrogen": "nitrogen",
               "bdod": "bulk_density", "sand": "sand", "silt": "silt", "clay": "clay", "cec": "cec"}
_NATIVE_UNITS = {"ph": {"pH x 10": 0.1}, "organic_carbon": {"dg/kg": 0.1},
                 "nitrogen": {"cg/kg": 0.01}, "bulk_density": {"cg/cm3": 0.01},
                 "sand": {"g/kg": 0.1}, "silt": {"g/kg": 0.1}, "clay": {"g/kg": 0.1},
                 "cec": {"mmol(c)/kg": 0.1}}
_LAYERS = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 100), (100, 200)]
_FIELDS = {"record_id", "location_id", "latitude", "longitude", "coordinate_crs", "coordinate_unit", "depth", "statistic", "properties"}


def _schema() -> dict[str, Any]:
    return {**_soil_schema(), "formats": ["json"], "records_type": "nonempty array of point/layer objects",
            "required_columns": sorted(_FIELDS), "additional_columns": False,
            "column_types": {"record_id": "unique identifier", "location_id": "stable point identifier",
                             "latitude": "JSON number [-90,90]", "longitude": "JSON number [-180,180]",
                             "coordinate_crs": "EPSG:4326", "coordinate_unit": "degree",
                             "depth": "depth_schema; converted interval must equal one standard layer",
                             "statistic": ["mean", "Q0.5"], "properties": "nonempty subset of property_mapping; each value is measurement_schema"},
            "property_mapping": dict(_PROPERTIES), "standard_layers_cm": [list(x) for x in _LAYERS],
            "unit_conversion_factors": {k: dict(v) for k, v in _NATIVE_UNITS.items()},
            "unit_evidence": "https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_01.html",
            "canonical_output": "Canonical soil features/masks/units/uncertainty plus record_id, location_id, latitude, longitude, coordinate_crs, coordinate_unit, depth, statistic, value_origin, raw and provenance",
            "limitations": ["Only explicitly unit-labelled point/layer exports; no raster sampling or WebDAV/WCS/REST requests.",
                            "SoilGrids values are modeled estimates, never local laboratory measurements.",
                            "Source property names, statistic and raw numeric scale are preserved. One statistic per point/layer; no aggregation or uncertainty inference.",
                            "All real-data quality and upstream acquisition metadata remain UNVERIFIED."]}


def _normalize(snapshot: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if snapshot["format"] != "json":
        raise ValueError("SoilGrids requires JSON point/layer exports")
    output = []
    history = []
    locations = {}
    intervals: dict[str, list[tuple[float, float]]] = {}
    coordinate_ids = {}
    for row in rows(snapshot["records"]):
        exact(row, _FIELDS, "SoilGrids row")
        identifier(row["record_id"], "record_id")
        location = identifier(row["location_id"], "location_id")
        if row["latitude"] is None or row["longitude"] is None:
            raise ValueError("SoilGrids point export requires explicit coordinates")
        coordinates = _coordinates(row["latitude"], row["longitude"], row["coordinate_crs"], row["coordinate_unit"])
        if location in locations and locations[location] != coordinates:
            raise ValueError("One location_id has conflicting coordinates")
        if coordinates in coordinate_ids and coordinate_ids[coordinates] != location:
            raise ValueError("Same coordinates have ambiguous location identities")
        locations[location] = coordinates
        coordinate_ids[coordinates] = location
        depth, depth_step = _depth(row["depth"])
        if (depth["top"], depth["bottom"]) not in _LAYERS:
            raise ValueError("Unsupported or ambiguous SoilGrids layer")
        _nonoverlap(intervals, location, depth)
        if row["statistic"] not in ("mean", "Q0.5"):
            raise ValueError("SoilGrids statistic must be explicit mean or Q0.5")
        properties = row["properties"]
        if not isinstance(properties, dict) or not properties or not set(properties) <= set(_PROPERTIES):
            raise ValueError("Unsupported SoilGrids properties")
        record = {**_empty_soil(), "record_id": row["record_id"], "location_id": location,
                  "latitude": coordinates[0], "longitude": coordinates[1], "depth": depth,
                  "coordinate_crs": row["coordinate_crs"], "coordinate_unit": row["coordinate_unit"],
                  "statistic": row["statistic"], "value_origin": "SYNTHETIC" if snapshot["data_origin"] == "synthetic_fixture" else "MODELED_ESTIMATE",
                  "raw": row}
        steps = [depth_step]
        for key, raw in properties.items():
            steps.append(_measurement(record, _PROPERTIES[key], raw, _NATIVE_UNITS))
        _texture(record)
        output.append(record)
        history.append({"record_id": row["record_id"], "operation": "normalize_soilgrids", "steps": steps})
    return output, history
