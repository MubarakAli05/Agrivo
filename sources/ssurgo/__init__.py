"""Phase 13: local SSURGO mapunit/component/chorizon representative-value joins."""

from typing import Any

from sources._common import exact, identifier, number, rows, text, unique_ids
from sources._soil import _depth, _empty_soil, _measurement, _nonoverlap, _soil_schema, _texture

_MAPUNIT = {"mukey", "muname", "areasymbol"}
_COMPONENT = {"cokey", "mukey", "compname", "comppct_r"}
_PROPERTY_MAP = {"ph1to1h2o_r": "ph", "dbthirdbar_r": "bulk_density",
                 "sandtotal_r": "sand", "silttotal_r": "silt", "claytotal_r": "clay",
                 "cec7_r": "cec", "ec_r": "ec"}
_UNITS = {"ph1to1h2o_r": "1", "dbthirdbar_r": "g/cm3", "sandtotal_r": "%",
          "silttotal_r": "%", "claytotal_r": "%", "cec7_r": "cmol(+)/kg", "ec_r": "dS/m", "om_r": "%"}
_HORIZON = {"chkey", "cokey", "hzdept_r", "hzdepb_r", "depth_unit", "units", "om_r"} | set(_PROPERTY_MAP)


def _schema() -> dict[str, Any]:
    return {
        **_soil_schema(), "formats": ["json"],
        "records_type": "object containing exactly mapunit, component, chorizon nonempty arrays",
        "tables": {
            "mapunit": {"required_columns": sorted(_MAPUNIT), "additional_columns": False,
                        "types": {"mukey": "unique identifier", "muname": "nonempty text", "areasymbol": "identifier"}},
            "component": {"required_columns": sorted(_COMPONENT), "additional_columns": False,
                          "types": {"cokey": "unique identifier", "mukey": "existing mapunit key",
                                    "compname": "nonempty text", "comppct_r": "JSON number [0,100] or null; sum of known percentages per mapunit <=100"}},
            "chorizon": {"required_columns": sorted(_HORIZON), "additional_columns": False,
                         "types": {"chkey": "unique identifier (becomes record_id)", "cokey": "existing component key",
                                   "hzdept_r": "finite JSON number >=0", "hzdepb_r": "finite JSON number > hzdept_r",
                                   "depth_unit": "explicit cm, m or mm",
                                   "units": "exact property_units object; every property needs explicit units, including nulls",
                                   **{key: "finite nonnegative JSON number or null" for key in _UNITS}}},
        },
        "property_mapping": dict(_PROPERTY_MAP), "property_units": dict(_UNITS),
        "unit_conversion_factors": {"ph": {"1": 1.0}, "bulk_density": {"g/cm3": 1.0},
                                    "sand": {"%": 1.0}, "silt": {"%": 1.0}, "clay": {"%": 1.0},
                                    "cec": {"cmol(+)/kg": 1.0}, "ec": {"dS/m": 1.0}},
        "canonical_output": "One canonical soil record per horizon: record_id=chkey, location_id=cokey, mukey, cokey, areasymbol, mapunit_name, component_name, component_percent/missing, null latitude/longitude/coordinate_crs/coordinate_unit, depth, organic_matter/organic_matter_missing/organic_matter_unit, value_origin, raw joined rows and provenance",
        "limitations": ["Only explicit JSON exports of mapunit/component/chorizon representative fields; not native pipe files, spatial geometry, survey ZIPs or remote Soil Data Access.",
                        "No component-weighted averaging or horizon aggregation; representative values are not observations at a point.",
                        "Organic matter is preserved separately and never inferred as organic carbon. Unrepresented nutrients/uncertainty remain missing.",
                        "Survey release/version, dataset-specific rights and quality require independent review. Federal hosting is not inferred permission."],
    }


def _normalize(snapshot: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if snapshot["format"] != "json":
        raise ValueError("SSURGO requires JSON relational export")
    tables = exact(snapshot["records"], {"mapunit", "component", "chorizon"}, "SSURGO tables")
    mapunits, components = {}, {}
    totals: dict[str, float] = {}
    for table, columns, primary in (("mapunit", _MAPUNIT, "mukey"), ("component", _COMPONENT, "cokey"),
                                     ("chorizon", _HORIZON, "chkey")):
        values = rows(tables[table], table)
        for row in values:
            exact(row, columns, table)
        unique_ids(values, primary)
    for row in tables["mapunit"]:
        identifier(row["areasymbol"], "areasymbol")
        text(row["muname"], "muname")
        mapunits[row["mukey"]] = row
    for row in tables["component"]:
        identifier(row["mukey"], "mukey")
        if row["mukey"] not in mapunits:
            raise ValueError("Orphan component: mukey does not exist")
        text(row["compname"], "compname")
        if row["comppct_r"] is not None:
            amount = number(row["comppct_r"], "comppct_r", 0, 100)
            totals[row["mukey"]] = totals.get(row["mukey"], 0) + amount
            if totals[row["mukey"]] > 100:
                raise ValueError("Component percentages exceed 100 for a mapunit")
        components[row["cokey"]] = row
    output, history = [], []
    intervals: dict[str, list[tuple[float, float]]] = {}
    for row in tables["chorizon"]:
        identifier(row["cokey"], "cokey")
        if row["cokey"] not in components:
            raise ValueError("Orphan horizon: cokey does not exist")
        component = components[row["cokey"]]
        mapunit = mapunits[component["mukey"]]
        units = exact(row["units"], set(_UNITS), "SSURGO units")
        if units != _UNITS:
            raise ValueError("SSURGO representative property units must match the explicit export contract")
        depth, step = _depth({"top": row["hzdept_r"], "bottom": row["hzdepb_r"], "unit": row["depth_unit"]})
        _nonoverlap(intervals, row["cokey"], depth)
        organic_matter = None if row["om_r"] is None else number(row["om_r"], "om_r", 0, 100)
        record = {**_empty_soil(), "record_id": row["chkey"], "location_id": row["cokey"],
                  "mukey": component["mukey"], "cokey": row["cokey"], "areasymbol": mapunit["areasymbol"],
                  "mapunit_name": mapunit["muname"], "component_name": component["compname"],
                  "component_percent": component["comppct_r"], "component_percent_missing": component["comppct_r"] is None,
                  "latitude": None, "longitude": None, "coordinate_crs": None, "coordinate_unit": None, "depth": depth,
                  "organic_matter": organic_matter, "organic_matter_missing": organic_matter is None,
                  "organic_matter_unit": "%", "value_origin": "SYNTHETIC" if snapshot["data_origin"] == "synthetic_fixture" else "SURVEY_REPRESENTATIVE",
                  "raw": {"mapunit": mapunit, "component": component, "chorizon": row}}
        steps = [step]
        for column, feature in _PROPERTY_MAP.items():
            steps.append(_measurement(record, feature, {"value": row[column], "unit": units[column], "uncertainty": None}))
        _texture(record)
        output.append(record)
        history.append({"record_id": row["chkey"], "operation": "join_ssurgo_horizon_component_mapunit",
                        "keys": {"chkey": row["chkey"], "cokey": row["cokey"], "mukey": component["mukey"]},
                        "steps": steps})
    return output, history
