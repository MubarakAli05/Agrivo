from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from agri.cli import main
from agri.config import default_config
from agri.model_check import check_model, model_report_path, validate_model
from agri.synthetic_data import generate_synthetic
from agri.workspace import setup, status
from tokenizer.tokenizer import BPETokenizer
from tokenizer.train_tokenizer import train_tokenizer


class ModelWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.base_temp.cleanup)
        cls.base = Path(cls.base_temp.name)
        setup(cls.base)
        generate_synthetic(cls.base)
        train_tokenizer(cls.base)
        config_path = cls.base / "configs" / "agri-mini.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["model"].update(layers=1, hidden_size=16, attention_heads=2, feed_forward_size=32, dropout=0)
        config_path.write_text(json.dumps(config), encoding="utf-8")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "workspace"
        shutil.copytree(self.base, self.root)

    def test_default_context_is_explicitly_increased(self):
        self.assertEqual(default_config()["model"]["context_length"], 1536)

    def test_dry_run_reports_estimates_and_never_writes(self):
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        report = check_model(self.root, dry_run=True)
        self.assertTrue(report["dry_run"])
        self.assertFalse(report["artifacts_written"])
        self.assertEqual(report["tiny_forward_backward"], "passed")
        self.assertEqual(report["resource_estimates"]["optimizer_steps"], 0)
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        self.assertEqual(set(before), {path for path in self.root.rglob("*") if path.is_file()})

    def test_persisted_check_and_read_only_status_without_model_or_fit(self):
        with patch.object(BPETokenizer, "fit", side_effect=AssertionError("Do not refit")), \
                patch("socket.create_connection", side_effect=AssertionError("Offline only")), \
                patch("torch.optim.AdamW.step", side_effect=AssertionError("No training")):
            report = check_model(self.root)
        self.assertEqual(report["status"], "GREEN")
        self.assertEqual(report["checks"]["optimizer_steps"], 0)
        self.assertEqual(report["provenance"]["architecture"]["vocab_size"], 760)
        self.assertEqual(report["provenance"]["architecture"]["structured_vocab_size"], 272)
        self.assertEqual(report["checks"]["forward_positions"], 1401)
        self.assertLess(report["storage_bytes"], 20_480)
        path = model_report_path(self.root)
        before = path.read_bytes(), path.stat().st_mtime_ns
        with patch("models.transformer.model.AgriTransformer", side_effect=AssertionError("No initialization")), \
                patch.object(BPETokenizer, "fit", side_effect=AssertionError("No fitting")):
            self.assertEqual(validate_model(self.root), report)
            ready = status(self.root)
        self.assertEqual(ready["phase"], 5)
        self.assertEqual(ready["phase5_status"], "GREEN")
        self.assertEqual(ready["phase4_status"], "GREEN")
        self.assertEqual(ready["phase2_status"], "YELLOW")
        self.assertEqual(ready["status"], "YELLOW")
        self.assertIsNone(ready["metrics"]["model_quality"])
        self.assertEqual(before, (path.read_bytes(), path.stat().st_mtime_ns))
        self.assertFalse(list(self.root.rglob("*.pt")))

    def test_rng_and_thread_count_restored(self):
        before = torch.get_rng_state().clone()
        threads = torch.get_num_threads()
        check_model(self.root, dry_run=True)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(torch.get_num_threads(), threads)

    def test_context_overflow_and_resource_guard_before_initialization(self):
        path = self.root / "configs" / "agri-mini.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        for value, message in ((256, "no truncation"), (100000, "limits")):
            config["model"]["context_length"] = value
            path.write_text(json.dumps(config), encoding="utf-8")
            with patch("models.transformer.model.AgriTransformer", side_effect=AssertionError("No allocation")), \
                    self.assertRaisesRegex(ValueError, message):
                check_model(self.root)
        self.assertFalse(model_report_path(self.root).exists())

    def test_corrupt_and_stale_reports_fail_without_repair(self):
        check_model(self.root)
        path = model_report_path(self.root)
        original = path.read_bytes()
        envelope = json.loads(original)
        envelope["report"]["checks"]["optimizer_steps"] = 1
        path.write_text(json.dumps(envelope), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_model(self.root)
        path.write_bytes(original)
        config_path = self.root / "configs" / "agri-mini.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["model"]["dropout"] = 0.2
        config_path.write_text(json.dumps(config), encoding="utf-8")
        for operation in (validate_model, status):
            with self.assertRaisesRegex(ValueError, "provenance"):
                operation(self.root)
        self.assertEqual(path.read_bytes(), original)

    def test_boolean_zero_result_and_duplicate_keys_rejected(self):
        check_model(self.root)
        path = model_report_path(self.root)
        envelope = json.loads(path.read_bytes())
        envelope["report"]["checks"]["optimizer_steps"] = False
        content = (json.dumps(envelope["report"], sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()
        envelope["sha256"] = hashlib.sha256(content).hexdigest()
        path.write_text(json.dumps(envelope), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "without training"):
            validate_model(self.root)
        path.write_text('{"report": {}, "report": {}}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_model(self.root)

    def test_cli_missing_check_preflight_and_validation(self):
        for command, extra, expected in (("validate-model", [], 1), ("check-model", ["--dry-run"], 0),
                                         ("check-model", [], 0), ("validate-model", [], 0)):
            output = io.StringIO()
            with redirect_stdout(output):
                code = main([command, "--root", str(self.root), *extra])
            self.assertEqual(code, expected)
            self.assertEqual(json.loads(output.getvalue())["phase"], 5)


if __name__ == "__main__":
    unittest.main()
