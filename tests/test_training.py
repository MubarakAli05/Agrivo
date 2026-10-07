import contextlib
import io
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4

import torch

from agri.config import load_config
from agri.synthetic_data import release_path
from models.transformer.model import AgriTransformer, TransformerConfig
from tokenizer.tokenizer import BPETokenizer
from tokenizer.structured import StructuredSoilTokenizer
from training.trainer import (checkpoint_path, cpu_session, load_checkpoint, train_model,
                              training_plan, training_report_path, validate_training, _publish)


PROJECT = Path(__file__).resolve().parents[1]


def copy_inputs(destination):
    destination.mkdir(parents=True)
    for name in ("configs", "data/releases", "tokenizer/releases"):
        shutil.copytree(PROJECT.joinpath(*name.split("/")), destination.joinpath(*name.split("/")))
    path = destination / "configs" / "agri-mini.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["model"].update(layers=1, hidden_size=16, attention_heads=2, feed_forward_size=32, dropout=0)
    config["training"].update(max_steps=2, checkpoint_every_steps=1, batch_size=1)
    path.write_text(json.dumps(config), encoding="utf-8")


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = PROJECT / "tests" / f".training-fixture-{uuid4().hex}"
        cls.addClassCleanup(shutil.rmtree, cls.base)
        copy_inputs(cls.base)
        with patch.object(BPETokenizer, "fit", side_effect=AssertionError("Never refit")), \
                patch.object(StructuredSoilTokenizer, "fit", side_effect=AssertionError("Never refit")), \
                contextlib.redirect_stderr(io.StringIO()):
            cls.report = train_model(cls.base)

    def setUp(self):
        self.root = PROJECT / "tests" / f".training-case-{uuid4().hex}"
        shutil.copytree(self.base, self.root)
        self.addCleanup(shutil.rmtree, self.root)

    def test_real_causal_optimization_and_weighted_validation(self):
        metrics = self.report["metrics"]
        self.assertEqual(metrics["optimizer_steps"], 2)
        self.assertEqual(len(metrics["train_step_losses"]), 2)
        self.assertGreater(metrics["embedding_update_l2"], 0)
        self.assertTrue(metrics["train_probe_loss_decreased"])
        self.assertEqual(metrics["validation_after"]["examples"], 18)
        self.assertGreater(metrics["validation_after"]["supervised_tokens"], 18)
        self.assertEqual(metrics["test_examples_evaluated"], 0)
        self.assertFalse(self.report["resume_supported"])
        with cpu_session(42):
            model, _ = load_checkpoint(self.root)
            torch.random.default_generator.manual_seed(42)
            initial = AgriTransformer(TransformerConfig(**self.report["provenance"]["architecture"]))
            self.assertFalse(torch.equal(model.text_embeddings.weight, initial.text_embeddings.weight))
            self.assertEqual(sum(p.numel() for p in model.parameters()), self.report["resource_estimates"]["parameters"])

    def test_no_optimizer_split_leakage_and_no_tokenizer_refitting(self):
        data = release_path(self.root)
        by_split = {split: [json.loads(line) for line in (data / "qa" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
                    for split in ("train", "validation", "test")}
        trained = set(self.report["metrics"]["training_example_ids"])
        self.assertTrue(trained <= {row["example_id"] for row in by_split["train"]})
        self.assertFalse(trained & {row["example_id"] for split in ("validation", "test") for row in by_split[split]})
        self.assertFalse({row["group_id"] for row in by_split["train"]} & {row["group_id"] for row in by_split["validation"]})
        with patch.object(BPETokenizer, "fit", side_effect=AssertionError("No fitting")), \
                patch("models.transformer.model.AgriTransformer", side_effect=AssertionError("No allocation")):
            self.assertEqual(validate_training(self.root), self.report)
            self.assertEqual(training_plan(self.root)["resource_estimates"]["optimizer_steps"], 2)

    def test_dry_run_tiny_side_effect_free_and_restores_global_state(self):
        before = {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        rng, threads, python_rng = torch.get_rng_state().clone(), torch.get_num_threads(), random.getstate()
        with contextlib.redirect_stderr(io.StringIO()):
            result = train_model(self.root, dry_run=True)
        self.assertTrue(result["dry_run"])
        self.assertFalse(result["artifacts_written"])
        self.assertLess(result["tiny_parameters"], self.report["resource_estimates"]["parameters"])
        self.assertLess(result["metrics"]["smoke_loss_after"], result["metrics"]["smoke_loss_before"])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(threads, torch.get_num_threads())
        self.assertEqual(python_rng, random.getstate())
        self.assertEqual(before, {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob("*") if path.is_file()})

    def test_status_import_and_validation_never_import_torch(self):
        script = '''
import importlib.abc
import sys
from pathlib import Path
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("Status must not import torch")
sys.meta_path.insert(0, NoTorch())
from training.trainer import training_report_path, validate_training
root = Path(sys.argv[1])
assert training_report_path(root).exists()
assert validate_training(root)["complete"]
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
'''
        path = training_report_path(self.root)
        before = path.read_bytes(), path.stat().st_mtime_ns
        result = subprocess.run([sys.executable, "-B", "-c", script, str(self.root)], cwd=PROJECT,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(before, (path.read_bytes(), path.stat().st_mtime_ns))
        self.assertEqual(path.name, "latest.json")

    def test_report_contract_and_truthful_resource_usage(self):
        required = {"phase", "status", "model", "training", "metrics", "source_status",
                    "next_required_step", "blockers", "resource_usage"}
        with contextlib.redirect_stderr(io.StringIO()):
            smoke = train_model(self.root, dry_run=True)
        plan = training_plan(self.root)
        for report in (plan, smoke, self.report, validate_training(self.root)):
            self.assertTrue(required <= report.keys())
            self.assertIsNone(report["resource_usage"]["peak_ram_bytes"])
        self.assertEqual(plan["resource_usage"]["storage_bytes_written"], 0)
        self.assertEqual(smoke["resource_usage"]["storage_bytes_written"], 0)
        self.assertEqual(self.report["resource_usage"]["checkpoint_weights_bytes"], checkpoint_path(self.root).stat().st_size)
        self.assertGreater(self.report["resource_usage"]["wall_seconds"], 0)

    def test_completed_run_is_read_only_idempotent(self):
        path = checkpoint_path(self.root)
        before = path.read_bytes(), path.stat().st_mtime_ns
        with patch("torch.optim.AdamW.step", side_effect=AssertionError("No additional updates")), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(train_model(self.root), self.report)
        self.assertEqual(before, (path.read_bytes(), path.stat().st_mtime_ns))

    def test_guard_fails_before_model_allocation_even_if_large_flag_set(self):
        path = self.root / "configs" / "agri-mini.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        for section, key, value in (("training", "max_steps", 21), ("training", "epochs", 2),
                                     ("training", "batch_size", 3), ("training", "allow_large_runs", True),
                                     ("model", "hidden_size", 512)):
            config = json.loads(json.dumps(original))
            config[section][key] = value
            path.write_text(json.dumps(config), encoding="utf-8")
            with patch("models.transformer.model.AgriTransformer", side_effect=AssertionError("No allocation")), self.assertRaises(ValueError):
                training_plan(self.root)

    def test_bad_checkpoint_hash_rejected(self):
        path = checkpoint_path(self.root)
        path.write_bytes(path.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_training(self.root)

    def test_bad_tokenizer_hash_and_config_rejected(self):
        from tokenizer.train_tokenizer import tokenizer_path
        path = tokenizer_path(self.root) / "merges.txt"
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_training(self.root)
        path.write_bytes(original)
        path = self.root / "configs" / "agri-mini.json"
        config = load_config(path)
        config["model"]["dropout"] = 0.2
        path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_training(self.root)

    def test_publishing_failure_preserves_previous_checkpoint(self):
        class EmptyModel:
            def state_dict(self):
                return {}
        latest = checkpoint_path(self.root)
        report = {"metrics": {"optimizer_steps": 3}}
        with patch("torch.save", side_effect=OSError("simulated disk failure")), self.assertRaises(OSError):
            _publish(self.root, EmptyModel(), report)
        self.assertEqual(checkpoint_path(self.root), latest)
        self.assertFalse(list(latest.parent.parent.glob(".publishing-*")))
        self.assertFalse(list(latest.parent.parent.glob(".latest-*")))
        self.assertEqual(validate_training(self.root), self.report)

    def test_incomplete_checkpoint_is_not_silently_restarted(self):
        import hashlib
        latest = checkpoint_path(self.root).parent.parent / "latest.json"
        manifest = latest.parent / "step-000001" / "manifest.json"
        latest.write_text(json.dumps({"directory": "step-000001", "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}), encoding="utf-8")
        self.assertFalse(validate_training(self.root)["complete"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(ValueError, "resume is unsupported"):
            train_model(self.root)
        with cpu_session(42), self.assertRaisesRegex(ValueError, "incomplete"):
            load_checkpoint(self.root)


if __name__ == "__main__":
    unittest.main()
