"""Invented, approval-isolated crop statistics; no real portal data/API keys."""

import csv
import io
import unittest

from agri.source_registry import load_registries
from sources.ingestion import adapter_schema, inspect_adapter
from tests.test_plant_adapters import SnapshotFixtureMixin


def data_gov_row():
    return {"record_id": "invented-statistic-1", "resource_id": "invented-resource",
            "state": "Invented State", "district": "Invented District", "crop": "invented crop",
            "year": 2026, "season": "invented season", "metric": "production", "value": 35, "unit": "quintal"}


def as_csv(row):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)
    return stream.getvalue()


class DataGovAdapterTests(SnapshotFixtureMixin, unittest.TestCase):
    def test_default_blockers_and_read_only_inspection(self):
        self.assert_blocked_before_snapshot("data_gov")
        report = inspect_adapter(self.root, "data_gov")
        self.assertEqual(report["status"], "YELLOW")
        self.assertIn("Dataset license unresolved", report["ingestion"]["research_training"]["blockers"])
        self.assertTrue(any("Catalog only" in x for x in report["ingestion"]["research_training"]["blockers"]))

    def test_explicit_json_unit_conversions(self):
        self.approve("data_gov")
        for metric, unit, value, expected in (("production", "quintal", 35, 3.5), ("production", "kg", 1200, 1.2),
                                             ("area", "acre", 10, 4.0468564224), ("area", "hectare", 2, 2),
                                             ("yield", "kg/ha", 1500, 1.5), ("yield", "quintal/ha", 10, 1)):
            with self.subTest(metric=metric, unit=unit):
                raw = {**data_gov_row(), "metric": metric, "unit": unit, "value": value}
                report = self.ingest(self.envelope("data_gov", [raw]))
                row = self.output(report)[0]
                self.assertAlmostEqual(row["value"], expected)
                self.assertEqual(row["raw"], raw)
                self.assertFalse(row["value_missing"])
                self.assertEqual(row["provenance"]["source_version"], "invented-v1")
                self.assertEqual(report["status"], "YELLOW")

    def test_csv_explicit_decimal_parser_and_raw_fields(self):
        self.approve("data_gov")
        raw = {**data_gov_row(), "value": "1.25e2", "district": "District, invented"}
        report = self.ingest(self.envelope("data_gov", as_csv(raw), format="csv"))
        row = self.output(report)[0]
        self.assertEqual(row["value"], 12.5)
        self.assertEqual(row["raw"]["value"], "1.25e2")
        self.assertEqual(row["district"], "District, invented")
        self.assertEqual(row["provenance"]["transformation_history"][-1]["numeric_parser"], "strict_csv_decimal")

    def test_null_and_csv_empty_values_remain_missing_not_zero(self):
        self.approve("data_gov")
        for format, value in (("json", None), ("csv", "")):
            raw = {**data_gov_row(), "value": value, "district": None if format == "json" else ""}
            report = self.ingest(self.envelope("data_gov", [raw] if format == "json" else as_csv(raw), format=format))
            row = self.output(report)[0]
            self.assertIsNone(row["value"])
            self.assertTrue(row["value_missing"])
            self.assertIsNone(row["district"])
            self.assertTrue(row["district_missing"])

    def test_rejects_wrong_resources_units_and_json_numeric_strings(self):
        self.approve("data_gov")
        for key, value in (("resource_id", "another-resource"), ("unit", ""), ("unit", "thousand tonnes"),
                           ("metric", "fertilizer"), ("value", "35"), ("value", True), ("value", float("inf")),
                           ("value", -1), ("year", "2026"), ("year", True), ("year", 2026.5)):
            with self.subTest(key=key, value=value):
                self.assert_invalid(self.envelope("data_gov", [{**data_gov_row(), key: value}]))
        row = data_gov_row()
        del row["unit"]
        self.assert_invalid(self.envelope("data_gov", [row]))

    def test_rejects_malformed_csv(self):
        self.approve("data_gov")
        for value in ("NaN", "Infinity", "true", "null", "NA", "1,000", "1e9999", " 3", "03"):
            with self.subTest(value=value):
                self.assert_invalid(self.envelope("data_gov", as_csv({**data_gov_row(), "value": value}), format="csv"))
        good = as_csv(data_gov_row())
        for text in (good.replace("record_id,", "year,"), good + "broken,row\n", "", good.splitlines()[0] + "\n"):
            self.assert_invalid(self.envelope("data_gov", text, format="csv"))

    def test_duplicate_ids_and_ambiguous_observations(self):
        self.approve("data_gov")
        first = data_gov_row()
        self.assert_invalid(self.envelope("data_gov", [first, {**first, "record_id": "other"}]))
        self.assert_invalid(self.envelope("data_gov", [first, {**first, "year": 2025}]))

    def test_catalog_gate_remains_after_license_approval(self):
        self.approve("data_gov")
        sources, licenses = load_registries(self.root)
        sources["sources"]["data_gov"]["type"] = "catalog"
        self.write_registry(sources, licenses)
        report = self.ingest(self.envelope("data_gov", [data_gov_row()]))
        self.assertTrue(report["blocked"])
        self.assertFalse((self.root / "data" / "releases").exists())

    def test_schema_has_csv_contract(self):
        schema = adapter_schema("data_gov")
        self.assertEqual(schema["formats"], ["json", "csv"])
        self.assertIn("resource_id", schema["required_columns"])
        self.assertIn("csv_policy", schema)


if __name__ == "__main__":
    unittest.main()
