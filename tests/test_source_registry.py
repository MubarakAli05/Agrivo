from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from agri.cli import main
from agri.source_catalog import default_registries
from agri.source_registry import (
    LICENSE_PATH, SOURCE_PATH, PURPOSES, initialize_registries, inspect_sources,
    load_registries, require_ingestion_approval, validate_registries,
)
from agri.workspace import setup, status


class SourceRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        initialize_registries(self.root)

    def write(self, sources, licenses):
        (self.root / SOURCE_PATH).write_text(yaml.safe_dump(sources), encoding="utf-8")
        (self.root / LICENSE_PATH).write_text(json.dumps(licenses), encoding="utf-8")

    def approve(self, source_id="soilgrids", purpose="research_training"):
        sources, licenses = load_registries(self.root)
        sources["sources"][source_id]["enabled"] = True
        licenses["licenses"][source_id]["approval"] = {
            "status": "approved", "intended_use": purpose,
            "reviewer": "unit-test-only", "reviewed_on": "2026-10-07",
        }
        self.write(sources, licenses)
        return sources, licenses

    def test_all_ten_requested_sources_present(self):
        sources, licenses = load_registries(self.root)
        self.assertEqual(set(sources["sources"]), {
            "soilgrids", "isric", "ssurgo", "data_gov", "plantvillage", "plantdoc",
            "plant_pathology_2020", "plant_pathology_2021", "ai_challenger", "kaggle_agriculture",
        })
        self.assertEqual(set(sources["sources"]), set(licenses["licenses"]))
        self.assertTrue(all(s["source_version"] is None for s in sources["sources"].values()))
        self.assertTrue(all(not s["enabled"] for s in sources["sources"].values()))

    def test_default_licenses_and_access_are_evidence_based(self):
        sources, licenses = load_registries(self.root)
        terms = licenses["licenses"]
        self.assertEqual(terms["soilgrids"]["license"], "CC-BY-4.0")
        self.assertEqual(terms["plantvillage"]["license"], "CC-BY-SA-3.0")
        self.assertEqual(terms["plantdoc"]["license"], "CC-BY-4.0")
        for key in ("ssurgo", "isric", "data_gov"):
            self.assertIsNone(terms[key]["license"])
            self.assertEqual(terms[key]["commercial_use"], "unknown")
        access = {a["mode"]: a["status"] for a in sources["sources"]["soilgrids"]["access_modes"]}
        self.assertEqual(access["rest"], "paused")
        self.assertEqual(access["webdav"], "documented")
        self.assertEqual(access["wcs"], "documented")

    def test_every_default_ingestion_is_blocked_for_every_purpose(self):
        sources, _ = load_registries(self.root)
        for source_id in sources["sources"]:
            for purpose in PURPOSES:
                with self.subTest(source=source_id, purpose=purpose):
                    with self.assertRaises(PermissionError):
                        require_ingestion_approval(self.root, source_id, purpose)

    def test_enabling_source_is_not_approval(self):
        sources, licenses = load_registries(self.root)
        sources["sources"]["soilgrids"]["enabled"] = True
        self.write(sources, licenses)
        with self.assertRaisesRegex(PermissionError, "Intended use not approved"):
            require_ingestion_approval(self.root, "soilgrids", "research_training")

    def test_approval_is_specific_to_intended_use(self):
        self.approve()
        self.assertEqual(require_ingestion_approval(self.root, "soilgrids", "research_training")["name"],
                         "ISRIC SoilGrids")
        with self.assertRaises(PermissionError):
            require_ingestion_approval(self.root, "soilgrids", "commercial_training")
        report = inspect_sources(self.root, "soilgrids")
        self.assertEqual(report["status"], "GREEN")
        self.assertEqual(report["sources"]["soilgrids"]["license_gate"], "GREEN")

    def test_approved_but_unavailable_or_paused_source_is_blocked(self):
        sources, licenses = self.approve()
        original = deepcopy(sources)
        for status_value in ("unavailable", "restricted"):
            sources["sources"]["soilgrids"]["inspection"]["status"] = status_value
            self.write(sources, licenses)
            with self.assertRaises(PermissionError):
                require_ingestion_approval(self.root, "soilgrids", "research_training")
        sources = original
        for access in sources["sources"]["soilgrids"]["access_modes"]:
            access["status"] = "paused"
        self.write(sources, licenses)
        with self.assertRaisesRegex(PermissionError, "No documented data acquisition mode"):
            require_ingestion_approval(self.root, "soilgrids", "research_training")

    def test_wms_only_is_not_data_acquisition(self):
        sources, licenses = self.approve()
        sources["sources"]["soilgrids"]["access_modes"] = [
            a for a in sources["sources"]["soilgrids"]["access_modes"] if a["mode"] == "wms"]
        self.write(sources, licenses)
        with self.assertRaises(PermissionError):
            require_ingestion_approval(self.root, "soilgrids", "research_training")

    def test_catalog_cannot_be_approved_as_dataset(self):
        sources, licenses = self.approve()
        sources["sources"]["soilgrids"]["type"] = "catalog"
        self.write(sources, licenses)
        with self.assertRaisesRegex(PermissionError, "Catalog only"):
            require_ingestion_approval(self.root, "soilgrids", "research_training")

    def test_unknown_or_prohibited_commercial_and_redistribution_rights_block(self):
        for purpose, field in (("commercial_training", "commercial_use"), ("redistribution", "redistribution")):
            for permission in ("unknown", "prohibited"):
                sources, licenses = self.approve(purpose=purpose)
                licenses["licenses"]["soilgrids"][field] = permission
                self.write(sources, licenses)
                with self.assertRaises(PermissionError):
                    require_ingestion_approval(self.root, "soilgrids", purpose)

    def test_unknown_source_and_purpose_fail(self):
        with self.assertRaises(ValueError):
            inspect_sources(self.root, "nonexistent")
        with self.assertRaises(ValueError):
            require_ingestion_approval(self.root, "nonexistent", "research_training")
        with self.assertRaises(ValueError):
            require_ingestion_approval(self.root, "soilgrids", "anything")

    def test_inspection_is_offline_read_only_and_fingerprinted(self):
        paths = [self.root / SOURCE_PATH, self.root / LICENSE_PATH]
        before = [p.read_bytes() for p in paths]
        with patch("socket.create_connection", side_effect=AssertionError("Network access forbidden")):
            report = inspect_sources(self.root)
        self.assertEqual(report["source_count"], 10)
        self.assertEqual(report["declared_license_count"], 3)
        self.assertEqual(report["unresolved_license_count"], 7)
        self.assertEqual(report["status"], "YELLOW")
        self.assertEqual(len(report["registry_sha256"]), 64)
        self.assertEqual([p.read_bytes() for p in paths], before)
        self.assertFalse((self.root / "data" / "raw").exists())
        self.assertEqual(report["sources"]["data_gov"]["license_gate"], "RED")

    def test_setup_preserves_user_registry_edits(self):
        sources, licenses = load_registries(self.root)
        sources["sources"]["soilgrids"]["priority"] = 2
        self.write(sources, licenses)
        paths = [self.root / SOURCE_PATH, self.root / LICENSE_PATH]
        before = [p.read_bytes() for p in paths]
        report = setup(self.root)
        self.assertEqual([p.read_bytes() for p in paths], before)
        self.assertEqual(report["phase"], 2)
        self.assertEqual(report["phase1_status"], "GREEN")
        self.assertEqual(report["phase2_status"], "YELLOW")

    def test_invalid_registry_is_not_replaced(self):
        path = self.root / SOURCE_PATH
        path.write_text("sources: []", encoding="utf-8")
        with self.assertRaises(ValueError):
            initialize_registries(self.root)
        self.assertEqual(path.read_text(encoding="utf-8"), "sources: []")

    def test_missing_license_and_mismatched_pair_fail_closed(self):
        (self.root / LICENSE_PATH).unlink()
        with self.assertRaises(OSError):
            load_registries(self.root)
        with self.assertRaises(ValueError):
            status(self.root)
        sources, licenses = default_registries()
        del licenses["licenses"]["soilgrids"]
        with self.assertRaises(ValueError):
            validate_registries(sources, licenses)

    def test_duplicate_keys_rejected_in_yaml_and_json(self):
        source_path = self.root / SOURCE_PATH
        source_path.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unique strings"):
            load_registries(self.root)
        self.write(*default_registries())
        (self.root / LICENSE_PATH).write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unique strings"):
            load_registries(self.root)

    def test_unsafe_yaml_tags_rejected(self):
        (self.root / SOURCE_PATH).write_text("!!python/object:builtins.object {}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid registry YAML"):
            load_registries(self.root)

    def test_malformed_fields_rejected(self):
        source_changes = [
            ("enabled", "false"), ("priority", True), ("priority", 6),
            ("url", "http://example.org"), ("url", "https://user:password@example.org"),
            ("source_version", 1), ("type", "unknown"), ("access_modes", {}),
        ]
        for key, value in source_changes:
            with self.subTest(key=key, value=value):
                sources, licenses = default_registries()
                sources["sources"]["soilgrids"][key] = value
                with self.assertRaises(ValueError):
                    validate_registries(sources, licenses)
        for key, value in [("license_url", None), ("evidence_urls", []), ("scope", "unknown")]:
            sources, licenses = default_registries()
            licenses["licenses"]["soilgrids"][key] = value
            with self.assertRaises(ValueError):
                validate_registries(sources, licenses)
        sources, licenses = default_registries()
        sources["schema_version"] = True
        with self.assertRaises(ValueError):
            validate_registries(sources, licenses)

    def test_unresolved_license_cannot_claim_rights_or_approval(self):
        for mutation in ("commercial_use", "approval"):
            sources, licenses = default_registries()
            entry = licenses["licenses"]["ssurgo"]
            if mutation == "approval":
                entry["approval"] = {"status": "approved", "intended_use": "research_training",
                                     "reviewer": "test", "reviewed_on": "2026-10-07"}
            else:
                entry[mutation] = "allowed_with_conditions"
            with self.assertRaises(ValueError):
                validate_registries(sources, licenses)

    def test_approval_requires_complete_dated_review(self):
        sources, licenses = self.approve()
        for key, value in [("reviewer", ""), ("reviewed_on", "2026-99-99"), ("intended_use", None)]:
            modified = deepcopy(licenses)
            modified["licenses"]["soilgrids"]["approval"][key] = value
            with self.assertRaises(ValueError):
                validate_registries(sources, modified)

    def test_cli_inspection_filter_and_failure(self):
        for source_id, code in [("soilgrids", 0), ("missing", 1)]:
            output = io.StringIO()
            with redirect_stdout(output):
                result = main(["inspect-sources", "--root", str(self.root), "--source", source_id])
            self.assertEqual(result, code)
            report = json.loads(output.getvalue())
            if code == 0:
                self.assertEqual(set(report["sources"]), {"soilgrids"})
            else:
                self.assertEqual(report["status"], "RED")


if __name__ == "__main__":
    unittest.main()
