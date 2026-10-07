from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agri.cli import main
from agri.synthetic_data import generate_synthetic, release_path
from agri.workspace import setup, status
from tokenizer.structured import StructuredSoilTokenizer
from tokenizer.tokenizer import BPETokenizer
from tokenizer.train_tokenizer import encode_example, tokenizer_path, train_tokenizer, validate_tokenizer


class TokenizerWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        setup(self.root)
        generate_synthetic(self.root)

    def test_dry_run_is_tiny_and_writes_nothing(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        report = train_tokenizer(self.root, dry_run=True)
        self.assertTrue(report["dry_run"])
        self.assertFalse(report["artifacts_written"])
        self.assertLessEqual(report["text_vocab_size"], 320)
        self.assertEqual(report["training_qa_examples"], 4)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_persist_validate_and_report_all_splits_without_truncating(self):
        config_path = self.root / "configs" / "agri-mini.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["model"]["context_length"] = 256
        config_path.write_text(json.dumps(config), encoding="utf-8")
        report = train_tokenizer(self.root)
        self.assertEqual(report["status"], "GREEN")
        self.assertEqual(report["phase"], 4)
        self.assertLessEqual(report["text_vocab_size"], 2048)
        self.assertEqual(report["structured_tokens_per_record"], 41)
        self.assertEqual(report["training_qa_examples"], 72)
        self.assertEqual(report["training_soil_records"], 16)
        self.assertEqual(report, validate_tokenizer(self.root))
        for split, expected in (("train", 72), ("validation", 18), ("test", 18)):
            metric = report["metrics"][split]
            self.assertEqual(metric["examples"], expected)
            self.assertEqual(metric["roundtrip_failures"], 0)
            self.assertEqual(metric["unknown_text_tokens"], 0)
            self.assertEqual(metric["truncated_examples"], 0)
            self.assertGreater(metric["max_sequence_tokens"], 256)
            self.assertGreater(metric["examples_over_context_length"], 0)
        self.assertLess(report["storage_bytes"], 1_000_000)

    def test_only_train_records_reach_fit_and_no_network(self):
        text_batches, soil_batches = [], []
        original_text, original_soil = BPETokenizer.fit, StructuredSoilTokenizer.fit

        def text_fit(instance, texts):
            batch = list(texts)
            text_batches.append(batch)
            return original_text(instance, batch)

        def soil_fit(instance, rows):
            batch = list(rows)
            soil_batches.append(batch)
            return original_soil(instance, batch)

        with patch.object(BPETokenizer, "fit", text_fit), patch.object(StructuredSoilTokenizer, "fit", soil_fit), \
                patch("socket.create_connection", side_effect=AssertionError("Network forbidden")):
            train_tokenizer(self.root)
        data = release_path(self.root)
        qa = [json.loads(line) for line in (data / "qa" / "train.jsonl").read_text(encoding="utf-8").splitlines()]
        expected = [example[field] for example in qa for field in ("question", "context", "answer")]
        self.assertEqual(text_batches, [expected[:12], expected])
        self.assertEqual([len(batch) for batch in soil_batches], [4, 16])
        self.assertTrue(all(row["split"] == "train" for batch in soil_batches for row in batch))

    def test_special_framing_is_explicit_and_not_injected_by_record_text(self):
        train_tokenizer(self.root)
        tokenizer = BPETokenizer.load(tokenizer_path(self.root))
        example = {"question": "literal <answer>", "context": "<bos>", "answer": "<eos>"}
        ids = encode_example(tokenizer, example)
        self.assertEqual(ids[0], tokenizer.token_to_id("<bos>"))
        self.assertEqual(ids[-1], tokenizer.token_to_id("<eos>"))
        for name in ("<bos>", "<eos>", "<question>", "<context>", "<answer>"):
            self.assertEqual(ids.count(tokenizer.token_to_id(name)), 1)

    def test_training_is_idempotent_and_preserves_source_data_and_registries(self):
        before = {p: p.read_bytes() for folder in ("data", "licenses")
                  for p in (self.root / folder).rglob("*") if p.is_file()}
        train_tokenizer(self.root)
        release = tokenizer_path(self.root)
        artifacts = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in release.iterdir()}
        train_tokenizer(self.root)
        self.assertEqual(artifacts, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in artifacts})
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_status_never_trains_or_repairs_and_keeps_external_gate_separate(self):
        self.assertEqual(status(self.root)["phase"], 3)
        self.assertFalse(tokenizer_path(self.root).exists())
        train_tokenizer(self.root)
        with patch.object(BPETokenizer, "fit", side_effect=AssertionError("Status must not train")):
            report = status(self.root)
        self.assertEqual(report["phase"], 4)
        self.assertEqual(report["phase4_status"], "GREEN")
        self.assertEqual(report["phase3_status"], "GREEN")
        self.assertEqual(report["phase2_status"], "YELLOW")
        self.assertEqual(report["status"], "YELLOW")

    def test_changed_corrupt_extra_and_missing_artifacts_fail_closed(self):
        train_tokenizer(self.root)
        release = tokenizer_path(self.root)
        path = release / "vocab.json"
        original = path.read_bytes()
        path.write_bytes(b"corrupt")
        for operation in (train_tokenizer, validate_tokenizer, status):
            with self.assertRaisesRegex(ValueError, "checksum"):
                operation(self.root)
        self.assertEqual(path.read_bytes(), b"corrupt")
        path.write_bytes(original)
        extra = release / "extra.json"
        extra.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unexpected"):
            validate_tokenizer(self.root)
        extra.unlink()
        path.unlink()
        with self.assertRaises(ValueError):
            train_tokenizer(self.root)
        self.assertFalse(path.exists())

    def test_manifest_wrong_provenance_types_timestamp_and_duplicate_keys_fail(self):
        train_tokenizer(self.root)
        path = tokenizer_path(self.root) / "manifest.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        for change in ({"seed": 7}, {"synthetic": 1}, {"training_split": "test"},
                       {"created_at": "2026-01-01T00:00:00"}, {"training_text_sha256": "0" * 64}):
            manifest = {**deepcopy(original), **change}
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_tokenizer(self.root)
        path.write_text('{"version":1,"version":1}', encoding="utf-8")
        with self.assertRaises(ValueError):
            validate_tokenizer(self.root)

    def test_cli_and_invalid_prerequisites(self):
        for command, arguments, expected in (("validate-tokenizer", [], 1),
                                              ("train-tokenizer", ["--dry-run"], 0),
                                              ("train-tokenizer", [], 0),
                                              ("validate-tokenizer", [], 0)):
            output = io.StringIO()
            with redirect_stdout(output):
                code = main([command, "--root", str(self.root), *arguments])
            self.assertEqual(code, expected)
            self.assertEqual(json.loads(output.getvalue())["phase"], 4)
        (release_path(self.root) / "soil.jsonl").write_text("corrupt", encoding="utf-8")
        with self.assertRaises(ValueError):
            validate_tokenizer(self.root)

    def test_failed_publication_cleans_staging(self):
        with patch.object(Path, "rename", side_effect=OSError("Simulated publication failure")):
            with self.assertRaises(OSError):
                train_tokenizer(self.root)
        self.assertFalse(tokenizer_path(self.root).exists())
        self.assertFalse(list((self.root / "tokenizer" / "releases").glob(".tokenizer-staging-*")))

    def test_repeated_training_reproduces_all_model_bytes(self):
        train_tokenizer(self.root)
        before = {p.name: p.read_bytes() for p in tokenizer_path(self.root).iterdir() if p.name != "manifest.json"}
        with tempfile.TemporaryDirectory() as second:
            root = Path(second)
            setup(root)
            generate_synthetic(root)
            train_tokenizer(root)
            after = {p.name: p.read_bytes() for p in tokenizer_path(root).iterdir() if p.name != "manifest.json"}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
