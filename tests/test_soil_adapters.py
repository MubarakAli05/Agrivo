"""Explicitly synthetic SoilGrids/ISRIC exports; never real source data."""

from copy import deepcopy
import unittest

from agri.source_registry import load_registries
from sources.ingestion import adapter_schema, inspect_adapter
from tests.test_plant_adapters import SnapshotFixtureMixin


def measurement(value, unit, uncertainty=None):
    return {"value": value, "unit": unit, "uncertainty": uncertainty}


def soilgrids_row():
    return {"record_id": "invented-layer-1", "location_id": "invented-point", "latitude": 0, "longitude": 0,
            "coordinate_crs": "EPSG:4326", "coordinate_unit": "degree",
            "depth": {"top": 0, "bottom": 0.05, "unit": "m"}, "statistic": "mean",
            "properties": {"phh2o": measurement(65, "pH x 10", {"lower": 60, "upper": 70, "method": "invented_interval"}),
                           "soc": measurement(120, "dg/kg"), "nitrogen": measurement(250, "cg/kg"),
                           "bdod": measurement(130, "cg/cm3"), "sand": measurement(400, "g/kg"),
                           "silt": measurement(350, "g/kg"), "clay": measurement(250, "g/kg"),
                           "cec": measurement(125, "mmol(c)/kg")}}


def isric_row():
    return {"record_id": "invented-profile-layer", "profile_id": "invented-profile", "latitude": None, "longitude": None,
            "coordinate_crs": None, "coordinate_unit": None,
            "depth": {"top": 0, "bottom": 150, "unit": "mm"}, "measurement_method": "invented laboratory-like fixture, not a measurement",
            "properties": {"ph": measurement(None, "1"), "nitrogen": measurement(1000, "mg/kg"),
                           "organic_carbon": measurement(1.5, "%"), "bulk_density": measurement(1200, "kg/m3"),
                           "ec": measurement(250, "uS/cm"), "moisture": measurement(0.2, "m3/m3")}}


class SoilAdapterTests(SnapshotFixtureMixin, unittest.TestCase):
    def test_gates_block_both_sources(self):
        for source in ("soilgrids", "isric"):
            self.assert_blocked_before_snapshot(source)
            self.assertEqual(inspect_adapter(self.root, source)["status"], "YELLOW")

    def test_soilgrids_native_scaling_depth_uncertainty_and_missingness(self):
        self.approve("soilgrids")
        original = soilgrids_row()
        result = self.ingest(self.envelope("soilgrids", [original]))
        row = self.output(result)[0]
        for key, expected in {"ph": 6.5, "organic_carbon": 12, "nitrogen": 2.5, "bulk_density": 1.3,
                              "sand": 40, "silt": 35, "clay": 25, "cec": 12.5}.items():
            self.assertAlmostEqual(row[key], expected)
            self.assertFalse(row[key + "_missing"])
        self.assertEqual(row["depth"], {"top": 0, "bottom": 5, "unit": "cm"})
        self.assertEqual(row["uncertainty"]["ph"], {"lower": 6, "upper": 7, "method": "invented_interval"})
        self.assertFalse(row["uncertainty_missing"]["ph"])
        for key in ("phosphorus", "potassium", "moisture", "ec"):
            self.assertIsNone(row[key])
            self.assertTrue(row[key + "_missing"])
            self.assertTrue(row["uncertainty_missing"][key])
        self.assertEqual(row["raw"], original)
        self.assertEqual(row["value_origin"], "SYNTHETIC")
        self.assertTrue(row["provenance"]["transformation_history"])
        self.assertEqual(result["status"], "YELLOW")

    def test_isric_explicit_units_and_missingness(self):
        self.approve("isric")
        row = self.output(self.ingest(self.envelope("isric", [isric_row()])))[0]
        self.assertEqual(row["nitrogen"], 1)
        self.assertEqual(row["organic_carbon"], 15)
        self.assertEqual(row["bulk_density"], 1.2)
        self.assertEqual(row["ec"], 0.25)
        self.assertEqual(row["moisture"], 20)
        self.assertEqual(row["depth"]["bottom"], 15)
        self.assertIsNone(row["ph"])
        self.assertTrue(row["ph_missing"])
        self.assertIsNone(row["latitude"])
        self.assertEqual(row["raw"]["measurement_method"], isric_row()["measurement_method"])

    def test_catalog_license_cannot_inherit_soilgrids(self):
        self.approve("isric")
        sources, licenses = load_registries(self.root)
        sources["sources"]["isric"]["type"] = "catalog"
        self.write_registry(sources, licenses)
        result = self.ingest(self.envelope("isric", [isric_row()]))
        self.assertTrue(result["blocked"])
        self.assertTrue(any("Catalog only" in x for x in result["reasons"]))
        self.assertFalse((self.root / "data" / "releases").exists())

    def test_invalid_json_numbers_and_units(self):
        for source, factory, prop in (("soilgrids", soilgrids_row, "soc"), ("isric", isric_row, "nitrogen")):
            self.approve(source)
            for value in (True, "12", float("inf"), float("nan"), -1, 10 ** 400):
                row = factory()
                row["properties"][prop]["value"] = value
                with self.subTest(source=source, value=str(value)[:30]):
                    self.assert_invalid(self.envelope(source, [row]))
            for unit in (None, "", "unknown", "kg/ha"):
                row = factory()
                row["properties"][prop]["unit"] = unit
                self.assert_invalid(self.envelope(source, [row]))
            row = factory()
            del row["properties"][prop]["unit"]
            self.assert_invalid(self.envelope(source, [row]))

    def test_ambiguous_layers_coordinates_and_statistic_rejected(self):
        self.approve("soilgrids")
        for key, value in (("depth", {"top": 0, "bottom": 10, "unit": "cm"}),
                           ("latitude", 91), ("longitude", None), ("coordinate_crs", "EPSG:3857"),
                           ("coordinate_unit", None), ("statistic", "unknown")):
            row = soilgrids_row()
            row[key] = value
            self.assert_invalid(self.envelope("soilgrids", [row]))
        self.approve("isric")
        for key, value in (("depth", {"top": 15, "bottom": 10, "unit": "cm"}),
                           ("depth", {"top": 0, "bottom": 10, "unit": ""}),
                           ("depth", {"top": False, "bottom": 10, "unit": "cm"}), ("longitude", 10)):
            row = isric_row()
            row[key] = value
            self.assert_invalid(self.envelope("isric", [row]))

    def test_duplicate_ids_and_overlapping_profiles(self):
        for source, factory in (("soilgrids", soilgrids_row), ("isric", isric_row)):
            self.approve(source)
            first = factory()
            second = deepcopy(first)
            second["record_id"] = "different-id"
            self.assert_invalid(self.envelope(source, [first, second]))
            second["depth"] = {"top": 5 if source == "soilgrids" else 15, "bottom": 15 if source == "soilgrids" else 30, "unit": "cm"}
            second["record_id"] = first["record_id"]
            self.assert_invalid(self.envelope(source, [first, second]))

    def test_unknown_properties_invalid_uncertainty_and_texture(self):
        self.approve("soilgrids")
        row = soilgrids_row()
        row["properties"]["phh2o"]["uncertainty"]["lower"] = 68
        self.assert_invalid(self.envelope("soilgrids", [row]))
        row = soilgrids_row()
        row["properties"]["phh2o"]["value"] = None
        self.assert_invalid(self.envelope("soilgrids", [row]))
        row = soilgrids_row()
        row["properties"]["sand"]["value"] = 700
        self.assert_invalid(self.envelope("soilgrids", [row]))
        row = soilgrids_row()
        row["properties"]["invented_unknown_property"] = measurement(1, "g/kg")
        self.assert_invalid(self.envelope("soilgrids", [row]))

    def test_schema_is_exported_and_defensive(self):
        schema = adapter_schema("soilgrids")
        self.assertEqual(schema["property_mapping"]["phh2o"], "ph")
        schema["property_mapping"].clear()
        self.assertIn("phh2o", adapter_schema("soilgrids")["property_mapping"])
        self.assertIn("measurement_schema", adapter_schema("isric"))


if __name__ == "__main__":
    unittest.main()
