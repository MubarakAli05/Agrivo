"""Tiny invented relational survey fixtures; no real SSURGO rows or downloads."""

from copy import deepcopy
import unittest

from sources.ingestion import adapter_schema, inspect_adapter
from tests.test_plant_adapters import SnapshotFixtureMixin


def ssurgo_tables():
    return {
        "mapunit": [{"mukey": "invented-mu", "muname": "Invented map unit", "areasymbol": "TEST00"}],
        "component": [{"cokey": "invented-component", "mukey": "invented-mu", "compname": "Invented component", "comppct_r": 70}],
        "chorizon": [{"chkey": "invented-horizon", "cokey": "invented-component", "hzdept_r": 0, "hzdepb_r": 0.15,
                      "depth_unit": "m", "ph1to1h2o_r": 6.8, "dbthirdbar_r": 1.25,
                      "sandtotal_r": 40, "silttotal_r": 35, "claytotal_r": 25, "cec7_r": 12, "ec_r": None, "om_r": 3,
                      "units": {"ph1to1h2o_r": "1", "dbthirdbar_r": "g/cm3", "sandtotal_r": "%", "silttotal_r": "%",
                                "claytotal_r": "%", "cec7_r": "cmol(+)/kg", "ec_r": "dS/m", "om_r": "%"}}],
    }


class SsurgoAdapterTests(SnapshotFixtureMixin, unittest.TestCase):
    def test_default_gate_and_offline_inspection(self):
        self.assert_blocked_before_snapshot("ssurgo")
        report = inspect_adapter(self.root, "ssurgo")
        self.assertEqual(report["status"], "YELLOW")
        self.assertIn("Dataset license unresolved", report["ingestion"]["research_training"]["blockers"])

    def test_relational_join_provenance_depth_and_no_organic_carbon_inference(self):
        self.approve("ssurgo")
        tables = ssurgo_tables()
        result = self.ingest(self.envelope("ssurgo", tables))
        row = self.output(result)[0]
        self.assertEqual(row["ph"], 6.8)
        self.assertEqual(row["bulk_density"], 1.25)
        self.assertEqual(row["units"]["bulk_density"], "kg/dm3")
        self.assertEqual(row["depth"], {"top": 0, "bottom": 15, "unit": "cm"})
        self.assertEqual(row["mukey"], "invented-mu")
        self.assertEqual(row["component_percent"], 70)
        self.assertEqual(row["organic_matter"], 3)
        for feature in ("organic_carbon", "nitrogen", "phosphorus", "potassium", "ec"):
            self.assertIsNone(row[feature])
            self.assertTrue(row[feature + "_missing"])
        self.assertIsNone(row["latitude"])
        self.assertTrue(all(row["uncertainty_missing"].values()))
        self.assertEqual(row["raw"], {key: value[0] for key, value in tables.items()})
        self.assertEqual(row["value_origin"], "SYNTHETIC")
        self.assertEqual(row["provenance"]["real_data_quality"], "UNVERIFIED")
        self.assertEqual(result["status"], "YELLOW")

    def test_distinct_horizons_not_averaged(self):
        self.approve("ssurgo")
        tables = ssurgo_tables()
        other = {**tables["chorizon"][0], "chkey": "second-horizon", "hzdept_r": 0.15, "hzdepb_r": 0.3, "ph1to1h2o_r": 7.2}
        tables["chorizon"].append(other)
        rows = self.output(self.ingest(self.envelope("ssurgo", tables)))
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["ph"] for row in rows], [6.8, 7.2])
        self.assertEqual(rows[1]["depth"]["top"], 15)
        for row in rows:
            record_steps = [step for step in row["provenance"]["transformation_history"] if "record_id" in step]
            self.assertEqual([step["record_id"] for step in record_steps], [row["record_id"]])

    def test_duplicate_keys_and_orphan_references(self):
        self.approve("ssurgo")
        for table in ("mapunit", "component", "chorizon"):
            tables = ssurgo_tables()
            tables[table].append(deepcopy(tables[table][0]))
            self.assert_invalid(self.envelope("ssurgo", tables))
        for table, key in (("component", "mukey"), ("chorizon", "cokey")):
            tables = ssurgo_tables()
            tables[table][0][key] = "missing-parent"
            self.assert_invalid(self.envelope("ssurgo", tables))

    def test_overlapping_horizons_and_unknown_units_rejected(self):
        self.approve("ssurgo")
        tables = ssurgo_tables()
        tables["chorizon"].append({**tables["chorizon"][0], "chkey": "another-horizon"})
        self.assert_invalid(self.envelope("ssurgo", tables))
        for key, value in (("depth_unit", ""), ("hzdept_r", True), ("hzdepb_r", "15"), ("hzdepb_r", 0)):
            tables = ssurgo_tables()
            tables["chorizon"][0][key] = value
            self.assert_invalid(self.envelope("ssurgo", tables))
        for unit in ("", "unknown", "%", None):
            tables = ssurgo_tables()
            tables["chorizon"][0]["units"]["ph1to1h2o_r"] = unit
            self.assert_invalid(self.envelope("ssurgo", tables))

    def test_numeric_string_bool_nonfinite_and_bad_percent_rejected(self):
        self.approve("ssurgo")
        for value in (True, "6.5", float("inf"), float("nan"), -1, 15):
            tables = ssurgo_tables()
            tables["chorizon"][0]["ph1to1h2o_r"] = value
            self.assert_invalid(self.envelope("ssurgo", tables))
        tables = ssurgo_tables()
        tables["component"].append({**tables["component"][0], "cokey": "other-component", "comppct_r": 50})
        self.assert_invalid(self.envelope("ssurgo", tables))

    def test_missing_fields_citations_and_versions_rejected(self):
        self.approve("ssurgo")
        for table, key in (("mapunit", "areasymbol"), ("component", "comppct_r"), ("chorizon", "units")):
            tables = ssurgo_tables()
            del tables[table][0][key]
            self.assert_invalid(self.envelope("ssurgo", tables))
        for key, value in (("citations", []), ("source_version", "")):
            payload = self.envelope("ssurgo", ssurgo_tables())
            payload[key] = value
            self.assert_invalid(payload)

    def test_null_representative_and_component_values_preserved(self):
        self.approve("ssurgo")
        tables = ssurgo_tables()
        tables["component"][0]["comppct_r"] = None
        tables["chorizon"][0]["ph1to1h2o_r"] = None
        row = self.output(self.ingest(self.envelope("ssurgo", tables)))[0]
        self.assertIsNone(row["component_percent"])
        self.assertTrue(row["component_percent_missing"])
        self.assertIsNone(row["ph"])
        self.assertTrue(row["ph_missing"])

    def test_schema_exports_exact_relational_contract(self):
        schema = adapter_schema("ssurgo")
        self.assertEqual(set(schema["tables"]), {"mapunit", "component", "chorizon"})
        self.assertNotIn("om_r", schema["property_mapping"])
        self.assertEqual(schema["property_units"]["om_r"], "%")


if __name__ == "__main__":
    unittest.main()
