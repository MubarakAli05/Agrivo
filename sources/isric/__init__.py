"""Phase 11: one explicitly reviewed ISRIC resource, never blanket catalog ingestion."""

from typing import Any

from sources._common import exact, identifier, rows, text
from sources._soil import (
    UNIT_FACTORS, _coordinates, _depth, _empty_soil, _measurement, _nonoverlap, _soil_schema, _texture,
)

_FIELDS = {"record_id", "profile_id", "latitude", "longitude", "coordinate_crs", "coordinate_unit", "depth", "measurement_method", "properties"}


def _schema() -> dict[str, Any]:
    return {**_soil_schema(), "formats": ["json"], "records_type": "nonempty array of profile/layer objects",
            "required_columns": sorted(_FIELDS), "additional_columns": False,
            "column_types": {"record_id": "unique identifier", "profile_id": "stable reviewed profile ID",
                             "latitude": "JSON number [-90,90] or null with longitude null",
                             "longitude": "JSON number [-180,180] or null with latitude null",
                             "coordinate_crs": "EPSG:4326; null allowed only when both coordinates and coordinate_unit are null",
                             "coordinate_unit": "degree; null allowed only when both coordinates and coordinate_crs are null",
                             "depth": "depth_schema", "measurement_method": "explicit nonempty method text",
                             "properties": "nonempty subset of canonical soil_features, each value measurement_schema"},
            "canonical_output": "Canonical soil features/masks/units/uncertainty plus record_id, location_id (profile_id), latitude, longitude, coordinate_crs, coordinate_unit, depth, measurement_method, value_origin, raw and provenance",
            "limitations": ["ISRIC Explore is a catalog: the existing catalog gate remains blocked. A reviewer must register an exact resource URL, dataset type, version, access mode and dataset-specific license before use.",
                            "No SoilGrids license inheritance. This accepts explicit profile exports, not arbitrary catalog entries, archives or rasters.",
                            "Measurement methods and raw values are preserved without claiming verified laboratory observations.",
                            "No OM-to-carbon, area-to-mass, depth interpolation or missing-unit inference."]}


def _normalize(snapshot: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if snapshot["format"] != "json":
        raise ValueError("ISRIC profile export requires JSON")
    output, history = [], []
    profiles = {}
    intervals: dict[str, list[tuple[float, float]]] = {}
    for row in rows(snapshot["records"]):
        exact(row, _FIELDS, "ISRIC profile row")
        identifier(row["record_id"], "record_id")
        profile = identifier(row["profile_id"], "profile_id")
        coordinates = _coordinates(row["latitude"], row["longitude"], row["coordinate_crs"], row["coordinate_unit"])
        if profile in profiles and profiles[profile] != coordinates:
            raise ValueError("Profile has conflicting coordinates")
        profiles[profile] = coordinates
        text(row["measurement_method"], "measurement_method")
        depth, step = _depth(row["depth"])
        _nonoverlap(intervals, profile, depth)
        properties = row["properties"]
        if not isinstance(properties, dict) or not properties or not set(properties) <= set(UNIT_FACTORS):
            raise ValueError("ISRIC properties must use supported explicit canonical names")
        record = {**_empty_soil(), "record_id": row["record_id"], "location_id": profile,
                  "latitude": coordinates[0], "longitude": coordinates[1], "depth": depth,
                  "coordinate_crs": row["coordinate_crs"], "coordinate_unit": row["coordinate_unit"],
                  "measurement_method": row["measurement_method"],
                  "value_origin": "SYNTHETIC" if snapshot["data_origin"] == "synthetic_fixture" else "SOURCE_REPORTED_UNVERIFIED",
                  "raw": row}
        steps = [step]
        for feature, raw in properties.items():
            steps.append(_measurement(record, feature, raw))
        _texture(record)
        output.append(record)
        history.append({"record_id": row["record_id"], "operation": "normalize_isric_profile", "steps": steps})
    return output, history
