from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from agri.cli import main
from agri.config import default_config
from agri.data_schema import SOIL_FEATURES, SOURCE_ID, SPLITS, validate_soil_record
from agri.synthetic_data import (
    FILE_NAMES, _check_consistency, build_fixture, generate_synthetic, release_path, validate_data,
)
from agri.workspace import setup, status


class FixtureContentTests(unittest.TestCase):
    def test_counts_and_explicit_synthetic_provenance(self):
        soil, plants, qa = build_fixture(42)
        self.assertEqual((len(soil), len(plants), len(qa)), (24, 6, 108))
        for record in soil + plants:
            self.assertIs(record["synthetic"], True)
            self.assertEqual(record["source"], SOURCE_ID)
        for record in soil:
            validate_soil_record(record)
            self.assertIsNone(record["latitude"])
            self.assertIsNone(record["longitude"])
            self.assertEqual(record["value_origin"], "SYNTHETIC")
        for record in plants:
            self.assertIsNone(record["image_path"])
            self.assertIsNone(record["image_checksum"])
            self.assertIsNone(record["plant_id"])
            self.assertTrue(record["label"].startswith("SYNTHETIC_"))

    def test_deterministic_seed_without_global_rng_mutation(self):
        state = random.getstate()
        self.assertEqual(build_fixture(42), build_fixture(42))
        self.assertNotEqual(build_fixture(42), build_fixture(7))
        self.assertEqual(random.getstate(), state)
        for seed in (True, -1, 1.5):
            with self.assertRaises(ValueError):
                build_fixture(seed)

    def test_splits_are_nonempty_and_group_disjoint(self):
        soil, plants, qa = build_fixture(42)
        _check_consistency(soil, plants, qa)
        groups = {split: {r.get("location_id", r.get("group_id")) for r in soil + plants
                          if r["split"] == split} for split in SPLITS}
        self.assertEqual([len(groups[split]) for split in SPLITS], [12, 3, 3])
        self.assertFalse(groups["train"] & groups["validation"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["validation"] & groups["test"])
        self.assertEqual([sum(q["split"] == split for q in qa) for split in SPLITS], [72, 18, 18])

    def test_zero_and_missingness_are_distinct(self):
        soil, _, _ = build_fixture(42)
        self.assertEqual(soil[0]["nitrogen"], 0)
        self.assertIs(soil[0]["nitrogen_missing"], False)
        for record in soil:
            self.assertIsNone(record["phosphorus"])
            self.assertIs(record["phosphorus_missing"], True)
            for feature in SOIL_FEATURES:
                self.assertEqual(record[feature + "_missing"], record[feature] is None)

    def test_qa_context_citations_and_unknown_answers(self):
        soil, plants, qa = build_fixture(42)
        _check_consistency(soil, plants, qa)
        self.assertEqual({q["kind"] for q in qa}, {
            "property", "unit", "missing", "uncertainty", "source", "comparison",
            "definition", "location_unknown", "dataset", "image_unknown",
        })
        for example in qa:
            self.assertEqual(example["quality"], "synthetic_verified")
            self.assertEqual(example["source_ids"], [SOURCE_ID])
            context = json.loads(example["context"])
            self.assertTrue(set(example["evidence_ids"]) <= {r["record_id"] for r in context["records"]})
            self.assertEqual(example["unknown"], example["answer"].startswith("UNKNOWN:"))
            if example["kind"] in ("missing", "location_unknown", "image_unknown"):
                self.assertTrue(example["unknown"])

    def test_schema_rejects_bad_missingness_numbers_units_and_origin(self):
        original = build_fixture(42)[0][0]
        changes = [("ph", float("nan")), ("ph", float("inf")), ("ph", True), ("ph", 20),
                   ("ph_missing", True), ("nitrogen_missing", 0), ("synthetic", False),
                   ("source", "soilgrids"), ("latitude", 12.0), ("timestamp", "2026-01-01"),
                   ("sand", 100), ("depth", {"top": 5, "bottom": 0, "unit": "cm"})]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                record = deepcopy(original)
                record[key] = value
                with self.assertRaises(ValueError):
                    validate_soil_record(record)
        record = deepcopy(original)
        record["units"]["nitrogen"] = "kg/ha"
        with self.assertRaises(ValueError):
            validate_soil_record(record)
        record = deepcopy(original)
        record["uncertainty"]["ph"]["method"] = "calibrated"
        with self.assertRaises(ValueError):
            validate_soil_record(record)

    def test_group_leakage_and_altered_context_are_rejected(self):
        soil, plants, qa = build_fixture(42)
        soil[1]["split"] = "test" if soil[0]["split"] != "test" else "train"
        with self.assertRaisesRegex(ValueError, "Group leakage"):
            _check_consistency(soil, plants, qa)
        soil, plants, qa = build_fixture(42)
        context = json.loads(qa[0]["context"])
        context["records"][0]["ph"] = 14
        qa[0]["context"] = json.dumps(context)
        with self.assertRaisesRegex(ValueError, "context is altered"):
            _check_consistency(soil, plants, qa)

    def test_duplicate_ids_and_missing_citations_are_rejected(self):
        soil, plants, qa = build_fixture(42)
        with self.assertRaisesRegex(ValueError, "Duplicate record"):
            _check_consistency(soil + [soil[0]], plants, qa)
        with self.assertRaisesRegex(ValueError, "Duplicate QA"):
            _check_consistency(soil, plants, qa + [qa[0]])
        qa[0]["evidence_ids"] = ["unavailable"]
        with self.assertRaisesRegex(ValueError, "citation does not resolve"):
            _check_consistency(soil, plants, qa)


class FixtureReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        setup(self.root)

    def test_publish_validate_and_status(self):
        report = generate_synthetic(self.root)
        self.assertEqual(report["status"], "GREEN")
        self.assertEqual(report["counts"]["images"], 0)
        self.assertEqual(report["counts"]["qa_examples"], 108)
        self.assertLess(report["storage_bytes"], 1_000_000)
        self.assertEqual(report["duplicate_records"], 0)
        self.assertEqual(report["cross_split_group_leakage"], 0)
        self.assertEqual(validate_data(self.root), report)
        state = status(self.root)
        self.assertEqual(state["phase"], 3)
        self.assertEqual(state["phase3_status"], "GREEN")
        self.assertEqual(state["phase2_status"], "YELLOW")
        self.assertEqual(state["status"], "YELLOW")
        self.assertIn("Not run", report["model_hallucination_test"])

    def test_manifest_checksums_metadata_and_no_real_observation_time(self):
        import hashlib
        generate_synthetic(self.root)
        release = release_path(self.root)
        manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
        self.assertIsNone(manifest["original_url"])
        self.assertIsNone(manifest["downloaded_at"])
        self.assertIsNone(manifest["license"])
        self.assertTrue(manifest["created_at"])
        self.assertEqual(set(manifest["files"]), set(FILE_NAMES))
        for name, metadata in manifest["files"].items():
            content = release.joinpath(*name.split("/")).read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), metadata["sha256"])
            self.assertEqual(len(content), metadata["size_bytes"])

    def test_repeat_generation_is_read_only_and_preserves_raw_and_registries(self):
        raw = self.root / "data" / "raw" / "existing.bin"
        raw.write_bytes(b"untouched")
        source = self.root / "data" / "source_registry.yaml"
        before_source = source.read_bytes()
        generate_synthetic(self.root)
        release = release_path(self.root)
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in release.rglob("*") if p.is_file()}
        generate_synthetic(self.root)
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})
        self.assertEqual(source.read_bytes(), before_source)
        self.assertEqual(raw.read_bytes(), b"untouched")

    def test_no_network_and_no_external_approval_changes(self):
        source = self.root / "licenses" / "registry.json"
        before = source.read_bytes()
        with patch("socket.create_connection", side_effect=AssertionError("No network allowed")):
            generate_synthetic(self.root)
        self.assertEqual(source.read_bytes(), before)

    def test_corruption_is_rejected_and_never_overwritten(self):
        generate_synthetic(self.root)
        path = release_path(self.root) / "soil.jsonl"
        path.write_bytes(b"corrupted")
        for operation in (validate_data, generate_synthetic, status):
            with self.subTest(operation=operation.__name__):
                with self.assertRaisesRegex(ValueError, "corrupt"):
                    operation(self.root)
        self.assertEqual(path.read_bytes(), b"corrupted")

    def test_manifest_tampering_and_extra_files_are_rejected(self):
        generate_synthetic(self.root)
        release = release_path(self.root)
        path = release / "manifest.json"
        before = path.read_bytes()
        manifest = json.loads(before)
        manifest["counts"]["images"] = 6
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Manifest differs"):
            validate_data(self.root)
        path.write_bytes(before)
        (release / "unexpected.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inventory"):
            validate_data(self.root)

    def test_missing_files_fail_without_recreation(self):
        generate_synthetic(self.root)
        path = release_path(self.root) / "qa" / "test.jsonl"
        path.unlink()
        with self.assertRaises(ValueError):
            generate_synthetic(self.root)
        self.assertFalse(path.exists())

    def test_manifest_rejects_boolean_counts_and_naive_timestamps(self):
        generate_synthetic(self.root)
        path = release_path(self.root) / "manifest.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        for changes in ({"counts": {**original["counts"], "images": False}},
                        {"created_at": "2026-01-01T00:00:00"}):
            with self.subTest(changes=changes):
                path.write_text(json.dumps({**original, **changes}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    validate_data(self.root)

    def test_generation_requires_valid_config_and_registry(self):
        for relative in ("configs/agri-mini.json", "licenses/registry.json"):
            path = self.root.joinpath(*relative.split("/"))
            original = path.read_bytes()
            path.unlink()
            with self.assertRaises((OSError, ValueError)):
                generate_synthetic(self.root)
            path.write_bytes(original)
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                generate_synthetic(self.root)
            path.write_bytes(original)
        self.assertFalse(release_path(self.root).exists())

    def test_setup_and_status_do_not_generate_or_change_data(self):
        self.assertEqual(status(self.root)["phase"], 2)
        self.assertFalse(release_path(self.root).exists())
        generate_synthetic(self.root)
        release = release_path(self.root)
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in release.rglob("*") if p.is_file()}
        setup(self.root)
        status(self.root)
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})

    def test_new_seed_creates_new_release_preserving_old(self):
        generate_synthetic(self.root)
        old = release_path(self.root)
        config = default_config()
        config["seed"] = 7
        (self.root / "configs" / "agri-mini.json").write_text(json.dumps(config), encoding="utf-8")
        report = generate_synthetic(self.root)
        self.assertNotEqual(Path(report["release_path"]), old)
        self.assertTrue(old.exists())
        self.assertEqual(report["seed"], 7)

    def test_failed_publish_cleans_staging_without_exposing_partial_release(self):
        with patch.object(Path, "rename", side_effect=OSError("Simulated publish failure")):
            with self.assertRaises(OSError):
                generate_synthetic(self.root)
        self.assertFalse(release_path(self.root).exists())
        self.assertFalse(list((self.root / "data" / "releases").glob(".synthetic-staging-*")))

    def test_cli_generation_validation_and_missing_data_error(self):
        for command, expected in [("validate-data", 1), ("generate-synthetic", 0), ("validate-data", 0)]:
            output = io.StringIO()
            with redirect_stdout(output):
                code = main([command, "--root", str(self.root)])
            self.assertEqual(code, expected)
            report = json.loads(output.getvalue())
            self.assertEqual(report["phase"], 3)
            self.assertEqual(report["status"], "RED" if expected else "GREEN")


if __name__ == "__main__":
    unittest.main()
