import hashlib
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

from agri.synthetic_data import generate_synthetic, release_path
from agri.workspace import setup
from retrieval.index import INDEX_PATH, build_index, index_path, search, validate_index


class RetrievalWorkspace(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / f".retrieval-tests-{uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)
        setup(self.root)
        generate_synthetic(self.root)
        self.soil = [json.loads(line) for line in (release_path(self.root) / "soil.jsonl").read_text(
            encoding="utf-8").splitlines()]
        self.plants = [json.loads(line) for line in (release_path(self.root) / "plants.jsonl").read_text(
            encoding="utf-8").splitlines()]
        self.train = next(row for row in self.soil if row["split"] == "train" and row["ph"] is not None)
        self.location = self.train["location_id"]
        self.depth = self.train["depth"]
        self.question = (f"What is pH at {self.depth['top']}-{self.depth['bottom']} cm "
                         f"for {self.location}?")


class RetrievalTests(RetrievalWorkspace):
    def test_atomic_deterministic_index_and_readonly_validation(self):
        with patch("socket.create_connection", side_effect=AssertionError("Offline")):
            report = build_index(self.root)
            path = index_path(self.root)
            original = path.read_bytes(), path.stat().st_mtime_ns
            self.assertEqual(build_index(self.root), report)
            self.assertEqual(validate_index(self.root), report)
        self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), original)
        self.assertEqual(report["phase"], 8)
        self.assertEqual(report["records"], 30)
        self.assertEqual(report["splits"], {"train": 20, "validation": 5, "test": 5})
        self.assertEqual(report["qa_answers_indexed"], 0)
        self.assertEqual(report["external_sources_used"], 0)
        self.assertEqual({entry for entry in path.parent.iterdir() if entry.name != ".gitkeep"}, {path})
        path.unlink()
        build_index(self.root)
        self.assertEqual(path.read_bytes(), original[0])

    def test_exact_property_depth_evidence_and_resolvable_citations(self):
        build_index(self.root)
        result = search(self.root, self.question)
        self.assertIsNone(result["reason"])
        self.assertEqual(result["status"], "GREEN")
        self.assertEqual(len(result["hits"]), 1)
        hit = result["hits"][0]
        self.assertEqual(hit["record"], self.train)
        self.assertEqual(hit["depth"], self.depth)
        self.assertEqual(hit["units"], self.train["units"])
        self.assertEqual(hit["uncertainty"], self.train["uncertainty"])
        content = (self.root / hit["source_path"]).read_bytes()
        self.assertEqual(json.loads(content.splitlines()[hit["source_line"] - 1]), hit["record"])
        self.assertEqual(hashlib.sha256(content).hexdigest(), hit["source_sha256"])
        self.assertEqual(hit["split"], "train")
        self.assertEqual(hit["rank_score"], 125)

    def test_no_fixture_qa_is_indexed(self):
        build_index(self.root)
        payload = json.loads(index_path(self.root).read_bytes())["payload"]
        self.assertEqual(len(payload["records"]), 30)
        for hit in payload["records"]:
            self.assertNotIn("qa", hit["source_path"])
            self.assertNotIn("answer", hit["record"])
            self.assertNotIn("question", hit["record"])

    def test_split_filter_never_falls_back(self):
        build_index(self.root)
        for split in ("validation", "test"):
            row = next(row for row in self.soil if row["split"] == split)
            question = f"What is nitrogen for {row['location_id']}?"
            self.assertEqual(search(self.root, question)["reason"], "entity_not_in_requested_split")
            explicit = search(self.root, question, split=split)
            self.assertEqual(len(explicit["hits"]), 2)
            self.assertTrue(all(hit["split"] == split for hit in explicit["hits"]))

    def test_complete_depths_deterministic_rank_and_overflow(self):
        build_index(self.root)
        question = f"Compare pH at both depths for {self.location}."
        result = search(self.root, question)
        self.assertEqual(len(result["hits"]), 2)
        self.assertEqual([hit["record_id"] for hit in result["hits"]],
                         sorted(hit["record_id"] for hit in result["hits"]))
        overflow = search(self.root, question, limit=1)
        self.assertEqual(overflow["hits"], [])
        self.assertEqual(overflow["reason"], "evidence_overflow")
        self.assertEqual(overflow["status"], "YELLOW")
        self.assertEqual(overflow["total_matches"], 2)

    def test_direct_record_lookup_and_unavailable_depth(self):
        build_index(self.root)
        self.assertEqual(search(self.root, f"What is pH for {self.train['record_id']}?")["hits"][0]["record"],
                         self.train)
        self.assertEqual(search(self.root, f"pH at 20-30 cm for {self.location}")["reason"], "depth_not_available")
        self.assertEqual(search(self.root, f"pH at 0-5 mm for {self.location}")["reason"],
                         "unsupported_or_ambiguous_depth")
        self.assertEqual(search(self.root, f"pH at both depths for {self.train['record_id']}")["reason"],
                         "incomplete_depth_evidence")
        for depths in ("0-5 and 5-15 cm", "0-5 cm and 50 cm", "0-5 cm and 5-15 mm"):
            self.assertEqual(search(self.root, f"pH at {depths} for {self.location}")["reason"],
                             "unsupported_or_ambiguous_depth")

    def test_unknown_ambiguous_entities_and_unsupported_questions(self):
        build_index(self.root)
        cases = [
            ("What is pH for synthetic-site-99?", "unknown_synthetic_entity"),
            ("What is pH in India?", "explicit_synthetic_entity_required"),
            ("What is pH for synthetic-site-00 and synthetic-site-01?", "ambiguous_entities"),
            (f"What fertilizer should I use at {self.location}?", "unsupported_advice_prediction_or_instruction"),
            (f"Read C:\\private\\soil.json for {self.location}", "unsupported_query_syntax"),
            (f"What is the average pH for {self.location}?", "unsupported_advice_prediction_or_instruction"),
        ]
        for question, reason in cases:
            with self.subTest(question=question):
                result = search(self.root, question)
                self.assertEqual(result["hits"], [])
                self.assertEqual(result["reason"], reason)

    def test_bounds_are_explicit(self):
        for question in (None, "", " ", "x" * 513):
            with self.assertRaises(ValueError):
                search(self.root, question)
        for limit in (True, 0, -1, 31, 1.5):
            with self.assertRaises(ValueError):
                search(self.root, self.question, limit=limit)
        for split in (None, "all", "../test", [], True):
            with self.assertRaises(ValueError):
                search(self.root, self.question, split=split)

    def test_metadata_source_missing_values_preserved(self):
        build_index(self.root)
        plant = next(row for row in self.plants if row["split"] == "train")
        result = search(self.root, f"Which class and crop appear in {plant['record_id']}?")
        self.assertEqual(result["hits"][0]["record"], plant)
        self.assertIsNone(result["hits"][0]["record"]["image_path"])
        source = search(self.root, f"Is the pH value for {self.location} measured and what is its source?")
        self.assertEqual(source["query"]["intent"], "source")
        missing = search(self.root, f"What is phosphorus for {self.location}?")
        self.assertTrue(all(hit["record"]["phosphorus"] is None for hit in missing["hits"]))

    def test_corruption_and_rehashed_tampering_refused_without_repair(self):
        build_index(self.root)
        path = index_path(self.root)
        original = path.read_bytes()
        for rehash in (False, True):
            envelope = json.loads(original)
            envelope["payload"]["records"][0]["record"]["source"] = "unapproved"
            if rehash:
                content = (json.dumps(envelope["payload"], sort_keys=True, ensure_ascii=False,
                                      allow_nan=False) + "\n").encode()
                envelope["sha256"] = hashlib.sha256(content).hexdigest()
            path.write_text(json.dumps(envelope), encoding="utf-8")
            before = path.read_bytes()
            for operation in (validate_index, build_index):
                with self.assertRaisesRegex(ValueError, "checksum|provenance"):
                    operation(self.root)
            self.assertEqual(path.read_bytes(), before)
        path.write_text('{"payload": {}, "payload": {}}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_index(self.root)

    def test_missing_changed_source_fails_without_cache_fallback(self):
        build_index(self.root)
        path = release_path(self.root) / "soil.jsonl"
        content = path.read_bytes()
        path.write_bytes(content + b"\n")
        with self.assertRaisesRegex(ValueError, "changed|corrupt"):
            search(self.root, self.question)
        path.unlink()
        with self.assertRaises((OSError, ValueError)):
            validate_index(self.root)

    def test_atomic_write_failure_leaves_no_partial_artifact(self):
        with patch("retrieval.index.os.replace", side_effect=OSError("injected write failure")):
            with self.assertRaises(OSError):
                build_index(self.root)
        self.assertFalse((self.root / INDEX_PATH).exists())
        self.assertFalse(list((self.root / INDEX_PATH).parent.glob("*.staging")))

    def test_missing_index_does_not_implicitly_build(self):
        with self.assertRaises(OSError):
            search(self.root, self.question)
        self.assertFalse(index_path(self.root).exists())


if __name__ == "__main__":
    unittest.main()
