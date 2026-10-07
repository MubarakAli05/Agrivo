"""Invented metadata only; approval registries are isolated beneath the project."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

import yaml

from agri.source_registry import LICENSE_PATH, SOURCE_PATH, initialize_registries, load_registries
from sources.ingestion import adapter_schema, ingest_snapshot, inspect_adapter


class SnapshotFixtureMixin:
    """Shared Phase 10 test helpers, without tests inherited by later phases."""

    def setUp(self):
        self.workspace = Path(__file__).resolve().parents[1]
        self.base = self.workspace / ".adapter-test-fixtures"
        self.root = self.base / uuid4().hex
        self.root.mkdir(parents=True)
        self.addCleanup(self.cleanup_fixture)
        initialize_registries(self.root)
        self.snapshot_path = self.root / "invented-snapshot.json"

    def cleanup_fixture(self):
        shutil.rmtree(self.root)
        try:
            self.base.rmdir()
        except OSError:
            pass

    def approve(self, source_id):
        sources, licenses = load_registries(self.root)
        source = sources["sources"][source_id]
        source.update(enabled=True, source_version="invented-v1",
                      url=f"https://example.invalid/invented/{source_id}",
                      type="plant_vision" if source_id in ("plantvillage", "plantdoc") else "soil_survey")
        source["inspection"].update(status="reviewed", notes="Invented test metadata, not source approval")
        source["access_modes"] = [{"mode": "download", "url": source["url"], "status": "documented", "evidence_url": source["url"]}]
        licenses["licenses"][source_id].update(
            source_url=source["url"], license="TEST-ONLY-NOT-A-REAL-LICENSE", status="declared", scope="dataset",
            license_url="https://example.invalid/invented-license", evidence_urls=[source["url"]],
            attribution_requirement="Invented fixture authors", commercial_use="unknown", redistribution="unknown",
            approval={"status": "approved", "intended_use": "research_training", "reviewer": "synthetic-test-only", "reviewed_on": "2026-10-07"})
        self.write_registry(sources, licenses)

    def write_registry(self, sources, licenses):
        (self.root / SOURCE_PATH).write_text(yaml.safe_dump(sources), encoding="utf-8")
        (self.root / LICENSE_PATH).write_text(json.dumps(licenses), encoding="utf-8")

    def envelope(self, source_id, records, format="json"):
        return {"schema_version": 1, "source_id": source_id, "dataset_id": "invented-resource",
                "source_version": "invented-v1", "source_url": f"https://example.invalid/invented/{source_id}",
                "retrieved_at": "2026-10-07T10:00:00Z", "citations": ["Invented local parser fixture; not an upstream dataset"],
                "license": {"name": "TEST-ONLY-NOT-A-REAL-LICENSE", "url": "https://example.invalid/invented-license",
                            "attribution": "Invented fixture authors", "upstream_rights": "Synthetic test metadata only"},
                "data_origin": "synthetic_fixture", "format": format, "records": records}

    def write_snapshot(self, payload):
        self.snapshot_path.write_text(json.dumps(payload), encoding="utf-8")
        return self.snapshot_path

    def ingest(self, payload):
        path = self.write_snapshot(payload)
        return ingest_snapshot(self.root, payload["source_id"], path)

    def output(self, report):
        return [json.loads(line) for line in Path(report["records"]).read_text(encoding="utf-8").splitlines()]

    def assert_invalid(self, payload):
        with self.assertRaises(ValueError):
            self.ingest(payload)
        self.assertFalse((self.root / "data" / "releases").exists())

    def assert_blocked_before_snapshot(self, source_id):
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        nonexistent = self.root / "not-present.json"
        original = Path.read_bytes

        def guarded(path):
            if path == nonexistent:
                raise AssertionError("Blocked snapshot was read")
            return original(path)

        with patch.object(Path, "read_bytes", guarded), patch("socket.create_connection", side_effect=AssertionError("network forbidden")):
            report = ingest_snapshot(self.root, source_id, nonexistent)
        self.assertTrue(report["blocked"])
        self.assertEqual(report["status"], "YELLOW")
        self.assertIn("Source disabled", report["reasons"])
        self.assertEqual(before, {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()})


def plant_row(source_id="plantvillage"):
    base = {"record_id": "invented-image-1", "image_path": "invented/leaf.png", "image_sha256": "a" * 64,
            "crop": "invented crop", "disease": "invented label", "split": "train",
            "image_source_url": "https://example.invalid/invented-image", "image_license": "synthetic-only",
            "rights_notes": "No actual image supplied"}
    return {**base, **({"leaf_id": "leaf-1", "variant": "color"} if source_id == "plantvillage" else {
        "original_image_id": "original-1", "annotation_id": "annotation-1"})}


class PlantAdapterTests(SnapshotFixtureMixin, unittest.TestCase):
    def test_both_sources_blocked_before_snapshot_access(self):
        for source_id in ("plantvillage", "plantdoc"):
            self.assert_blocked_before_snapshot(source_id)

    def test_inspect_offline_read_only_and_yellow(self):
        for source_id in ("plantvillage", "plantdoc"):
            report = inspect_adapter(self.root, source_id)
            self.assertEqual(report["status"], "YELLOW")
            self.assertEqual(report["adapter_status"], "READY")
            self.assertEqual(report["real_data_quality"], "UNVERIFIED")
            self.assertFalse(report["ingestion"]["research_training"]["allowed"])
            self.assertFalse((self.root / "data" / "releases").exists())

    def test_metadata_provenance_and_immutable_manifest(self):
        for source_id in ("plantvillage", "plantdoc"):
            with self.subTest(source=source_id):
                self.approve(source_id)
                payload = self.envelope(source_id, [plant_row(source_id)])
                report = self.ingest(payload)
                self.assertTrue(report["ingested"])
                self.assertEqual(report["status"], "YELLOW")
                row = self.output(report)[0]
                self.assertEqual(row["raw"], payload["records"][0])
                self.assertEqual(row["image_license"], "synthetic-only")
                self.assertEqual(row["provenance"]["data_origin"], "synthetic_fixture")
                self.assertEqual(row["provenance"]["real_data_quality"], "UNVERIFIED")
                manifest_path = Path(report["manifest"])
                original = manifest_path.read_bytes()
                manifest = json.loads(original)
                self.assertEqual(manifest["record_count"], 1)
                self.assertEqual(manifest["snapshot_sha256"], hashlib.sha256(self.snapshot_path.read_bytes()).hexdigest())
                self.assertEqual(report["release_id"], hashlib.sha256(original).hexdigest())
                self.assertEqual((manifest_path.parent / "snapshot.json").read_bytes(), self.snapshot_path.read_bytes())
                repeated = self.ingest(payload)
                self.assertTrue(repeated["reused"])
                self.assertEqual(manifest_path.read_bytes(), original)
                Path(report["records"]).write_text("tampered", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    self.ingest(payload)
                self.assertEqual(Path(report["records"]).read_text(encoding="utf-8"), "tampered")

    def test_groups_shared_across_variants_and_stable_across_versions(self):
        self.approve("plantvillage")
        first = plant_row()
        second = {**first, "record_id": "image-2", "variant": "segmented", "image_path": "invented/seg.png", "image_sha256": "b" * 64}
        rows = self.output(self.ingest(self.envelope("plantvillage", [first, second])))
        self.assertEqual(rows[0]["group_id"], rows[1]["group_id"])
        sources, licenses = load_registries(self.root)
        sources["sources"]["plantvillage"]["source_version"] = "invented-v2"
        self.write_registry(sources, licenses)
        payload = self.envelope("plantvillage", [first])
        payload["source_version"] = "invented-v2"
        self.assertEqual(rows[0]["group_id"], self.output(self.ingest(payload))[0]["group_id"])

    def test_leaf_and_image_leakage_rejected(self):
        for source_id in ("plantvillage", "plantdoc"):
            self.approve(source_id)
            first = plant_row(source_id)
            for shared in ("group", "checksum", "path", "url"):
                with self.subTest(source=source_id, identity=shared):
                    second = {**first, "record_id": "image-2", "split": "test", "image_sha256": "b" * 64,
                              "image_path": "invented/other.png", "image_source_url": "https://example.invalid/other"}
                    group_key = "leaf_id" if source_id == "plantvillage" else "original_image_id"
                    second[group_key] = "other-group"
                    if source_id == "plantdoc":
                        second["annotation_id"] = "annotation-2"
                    key = {"group": group_key, "checksum": "image_sha256", "path": "image_path", "url": "image_source_url"}[shared]
                    second[key] = first[key]
                    self.assert_invalid(self.envelope(source_id, [first, second]))

    def test_duplicate_ids_rejected(self):
        self.approve("plantvillage")
        self.assert_invalid(self.envelope("plantvillage", [plant_row(), plant_row()]))

    def test_unsafe_image_paths_rejected(self):
        self.approve("plantvillage")
        for path in ("../leaf.png", "..\\leaf.png", "C:\\leaf.png", "C:leaf.png", "/leaf.png", "\\\\server\\leaf.png", "a//b", "a/./b", "a:b"):
            with self.subTest(path=path):
                self.assert_invalid(self.envelope("plantvillage", [{**plant_row(), "image_path": path}]))

    def test_snapshot_path_traversal_rejected_after_approval(self):
        self.approve("plantvillage")
        with self.assertRaisesRegex(ValueError, "traversal"):
            ingest_snapshot(self.root, "plantvillage", self.root / ".." / "anything.json")
        self.assertFalse((self.root / "data" / "releases").exists())

    def test_schema_and_required_metadata_are_strict(self):
        self.approve("plantvillage")
        original = self.envelope("plantvillage", [plant_row()])
        for key in original:
            with self.subTest(missing=key):
                payload = deepcopy(original)
                del payload[key]
                self.assert_invalid(payload) if key != "source_id" else self._invalid_source_envelope(payload)
        mutations = {"schema_version": True, "source_version": "latest", "citations": [], "retrieved_at": "2026-10-07",
                     "source_url": "https://example.invalid/unapproved", "data_origin": "real_verified", "format": "csv"}
        for key, value in mutations.items():
            with self.subTest(field=key):
                self.assert_invalid({**original, key: value})
        self.assert_invalid({**original, "extra": 1})
        for key in plant_row():
            payload = deepcopy(original)
            del payload["records"][0][key]
            self.assert_invalid(payload)
        self.assertEqual(adapter_schema("plantvillage")["phase"], 10)

    def _invalid_source_envelope(self, payload):
        with self.assertRaises(ValueError):
            ingest_snapshot(self.root, "plantvillage", self.write_snapshot(payload))

    def test_wrong_license_version_and_purpose(self):
        self.approve("plantvillage")
        payload = self.envelope("plantvillage", [plant_row()])
        payload["license"]["name"] = "unapproved"
        self.assert_invalid(payload)
        payload = self.envelope("plantvillage", [plant_row()])
        payload["source_version"] = "unapproved-v2"
        self.assert_invalid(payload)
        report = ingest_snapshot(self.root, "plantvillage", self.root / "absent.json", "redistribution")
        self.assertTrue(report["blocked"])
        self.assertTrue(any("Intended use" in x for x in report["reasons"]))
        with self.assertRaises(ValueError):
            ingest_snapshot(self.root, "plantvillage", self.root / "absent.json", "anything")

    def test_duplicate_json_keys_and_nonfinite_constants(self):
        self.approve("plantvillage")
        for invalid in ('{"source_id":"a","source_id":"b"}', '{"bad":NaN}', '{"bad":Infinity}'):
            self.snapshot_path.write_text(invalid, encoding="utf-8")
            with self.assertRaises(ValueError):
                ingest_snapshot(self.root, "plantvillage", self.snapshot_path)
        self.assertFalse((self.root / "data" / "releases").exists())

    def test_missing_registry_pin_readiness_and_ingestion(self):
        self.approve("plantvillage")
        sources, licenses = load_registries(self.root)
        sources["sources"]["plantvillage"]["source_version"] = None
        self.write_registry(sources, licenses)
        self.assertFalse(inspect_adapter(self.root, "plantvillage")["ingestion"]["research_training"]["allowed"])
        self.assert_invalid(self.envelope("plantvillage", [plant_row()]))

    def test_gate_is_called_before_snapshot_stat(self):
        with patch("agri.source_registry.require_ingestion_approval", side_effect=PermissionError("pending review")) as gate, patch.object(Path, "is_file", side_effect=AssertionError("snapshot stat before gate")):
            report = ingest_snapshot(self.root, "plantvillage", self.snapshot_path)
        gate.assert_called_once_with(self.root, "plantvillage", "research_training")
        self.assertTrue(report["blocked"])

    def test_approval_revoked_before_publication_returns_yellow_without_writes(self):
        self.approve("plantvillage")
        sources, _ = load_registries(self.root)
        payload = self.envelope("plantvillage", [plant_row()])
        self.write_snapshot(payload)
        with patch("agri.source_registry.require_ingestion_approval", side_effect=[
                sources["sources"]["plantvillage"], PermissionError("Approval revoked during parsing")]):
            report = ingest_snapshot(self.root, "plantvillage", self.snapshot_path)
        self.assertTrue(report["blocked"])
        self.assertTrue(report["snapshot_read"])
        self.assertEqual(report["status"], "YELLOW")
        self.assertFalse((self.root / "data" / "releases").exists())

    def test_snapshot_symlinks_and_alternate_stream_paths_rejected(self):
        self.approve("plantvillage")
        path = self.write_snapshot(self.envelope("plantvillage", [plant_row()]))
        original = Path.is_symlink

        def linked(candidate):
            return candidate == path or original(candidate)

        with patch.object(Path, "is_symlink", linked):
            with self.assertRaisesRegex(ValueError, "Symlinks"):
                ingest_snapshot(self.root, "plantvillage", path)
        with self.assertRaisesRegex(ValueError, "alternate data streams"):
            ingest_snapshot(self.root, "plantvillage", self.root / "file.json:stream.json")
        self.assertFalse((self.root / "data" / "releases").exists())

    def test_fully_approved_readiness_still_does_not_certify_real_data(self):
        self.approve("plantvillage")
        report = inspect_adapter(self.root, "plantvillage")
        self.assertTrue(report["ingestion"]["research_training"]["allowed"])
        self.assertEqual(report["status"], "YELLOW")
        self.assertEqual(report["real_data_quality"], "UNVERIFIED")
        self.assertFalse((self.root / "data" / "releases").exists())

    def test_default_real_registry_unchanged_and_no_external_quality_claim(self):
        paths = [self.workspace / SOURCE_PATH, self.workspace / LICENSE_PATH]
        before = [p.read_bytes() for p in paths]
        report = inspect_adapter(self.workspace, "plantvillage")
        self.assertEqual(report["status"], "YELLOW")
        self.assertEqual(before, [p.read_bytes() for p in paths])


if __name__ == "__main__":
    unittest.main()
