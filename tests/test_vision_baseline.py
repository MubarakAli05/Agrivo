"""Offline phase-14 mechanics tests; generated arrays are not plant evidence."""

import hashlib
import json
import math
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

import torch
import yaml

from models.vision_baseline import workflow
from models.vision_baseline.model import (
    CLASSES, PARAMETER_COUNT, SmallVisionCNN, classification_loss, isolated_cpu,
    run_synthetic_smoke, synthetic_patterns,
)


PROJECT = Path(__file__).resolve().parents[1]


def snapshot(root):
    return {str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


class VisionBaselineTests(unittest.TestCase):
    def setUp(self):
        self.root = PROJECT / "checkpoints" / f"vision-tests-{uuid4().hex}"
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        for relative in (Path("configs") / "agri-mini.json", Path("data") / "source_registry.yaml",
                         Path("licenses") / "registry.json"):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT / relative, target)
        # Every test obtains the resource plan before any allocation/training.
        self.plan = workflow.vision_plan(self.root)

    def test_plan_is_read_only_bounded_and_honest(self):
        before = snapshot(self.root)
        with patch("models.vision_baseline.model.SmallVisionCNN", side_effect=AssertionError("No model allocation")):
            plan = workflow.vision_plan(self.root)
        self.assertEqual(before, snapshot(self.root))
        self.assertEqual(plan["status"], "YELLOW")
        self.assertEqual(plan["resource_estimates"]["optimizer_steps"], 2)
        self.assertEqual(plan["resource_estimates"]["threads"], 1)
        self.assertEqual(plan["resource_estimates"]["gpu_bytes"], 0)
        self.assertEqual(plan["classes"], list(CLASSES))
        self.assertTrue(all(label.startswith("SYNTHETIC_PATTERN_") for label in CLASSES))
        self.assertIsNone(plan["quality"]["plant_disease_accuracy"])
        self.assertFalse(plan["agronomic_model"])

    def test_synthetic_forward_backward(self):
        with isolated_cpu(7):
            model = SmallVisionCNN()
            images, labels = synthetic_patterns(7)
            logits = model(images)
            self.assertEqual(tuple(logits.shape), (6, 3))
            self.assertEqual(sum(p.numel() for p in model.parameters()), PARAMETER_COUNT)
            loss = classification_loss(logits, labels)
            loss.backward()
            self.assertTrue(math.isfinite(loss.item()))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_deterministic_two_steps_finite_loss_changed_weights(self):
        first, first_state = run_synthetic_smoke(17)
        second, second_state = run_synthetic_smoke(17)
        self.assertEqual(first, second)
        self.assertTrue(all(torch.equal(first_state[key], second_state[key]) for key in first_state))
        self.assertEqual(first["optimizer_steps"], 2)
        self.assertTrue(all(math.isfinite(loss) for loss in first["training_batch_losses"]))
        self.assertNotEqual(first["initial_state_sha256"], first["final_state_sha256"])
        self.assertTrue(first["weights_changed"])
        self.assertTrue(first["finite_gradients"])

    def test_generated_patterns_local_rng_only(self):
        state = torch.get_rng_state().clone()
        a, labels_a = synthetic_patterns(17)
        b, labels_b = synthetic_patterns(17)
        c, _ = synthetic_patterns(18)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(labels_a, labels_b))
        self.assertFalse(torch.equal(a, c))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_rng_and_threads_restore_success_and_failure(self):
        before = torch.get_rng_state().clone()
        threads = torch.get_num_threads()
        run_synthetic_smoke(19)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(threads, torch.get_num_threads())
        with patch("torch.optim.SGD.step", side_effect=RuntimeError("injected failure")):
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                run_synthetic_smoke(19)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(threads, torch.get_num_threads())

    def test_invalid_images_and_parameters(self):
        with isolated_cpu(7):
            model = SmallVisionCNN()
            invalid = [None, torch.zeros(3, 16, 16), torch.zeros(0, 3, 16, 16),
                       torch.zeros(9, 3, 16, 16), torch.zeros(1, 1, 16, 16),
                       torch.zeros(1, 3, 17, 16), torch.zeros(1, 3, 16, 16, dtype=torch.float64),
                       torch.zeros(1, 3, 16, 16, dtype=torch.uint8),
                       torch.full((1, 3, 16, 16), float("nan")),
                       torch.full((1, 3, 16, 16), float("inf")),
                       torch.full((1, 3, 16, 16), -0.1), torch.full((1, 3, 16, 16), 1.1),
                       torch.empty(1, 3, 16, 16, device="meta"),
                       torch.zeros(1, 3, 16, 16).to_sparse()]
            for images in invalid:
                with self.subTest(images=str(type(images))):
                    with self.assertRaises(ValueError):
                        model(images)
            images, _ = synthetic_patterns()
            model.double()
            with self.assertRaisesRegex(ValueError, "parameters"):
                model(images)
            model.float()
            with torch.no_grad():
                next(model.parameters()).fill_(float("nan"))
            with self.assertRaisesRegex(ValueError, "finite"):
                model(images)

    def test_invalid_loss_inputs(self):
        logits = torch.zeros(2, 3)
        targets = torch.tensor([0, 1])
        for bad in (None, torch.tensor([0]), torch.tensor([[0, 1]]), torch.tensor([0., 1.]),
                    torch.tensor([-1, 0]), torch.tensor([0, 3]), torch.empty(2, device="meta", dtype=torch.int64)):
            with self.assertRaises(ValueError):
                classification_loss(logits, bad)
        for bad in (None, torch.zeros(2, 4), torch.zeros(2, 3, dtype=torch.float64),
                    torch.full((2, 3), float("nan")), torch.empty(2, 3, device="meta")):
            with self.assertRaises(ValueError):
                classification_loss(bad, targets)

    def test_strict_budget_device_and_seed(self):
        for device in ("cuda", "meta", "cpu:0", None):
            with self.assertRaisesRegex(ValueError, "cpu"):
                SmallVisionCNN(device)
            with self.assertRaisesRegex(ValueError, "cpu"):
                run_synthetic_smoke(device=device)
        for steps in (0, 3, True, 1.0):
            with self.assertRaisesRegex(ValueError, "budget"):
                run_synthetic_smoke(steps=steps)
        for seed in (-1, 2**32, True, 0.5):
            with self.assertRaisesRegex(ValueError, "seed"):
                run_synthetic_smoke(seed)
            with self.assertRaisesRegex(ValueError, "seed"):
                synthetic_patterns(seed)

    def test_dry_run_no_writes_or_training(self):
        before = snapshot(self.root)
        directories = {p for p in self.root.rglob("*") if p.is_dir()}
        with patch("models.vision_baseline.model.run_synthetic_smoke", side_effect=AssertionError("No training")), \
                patch("socket.create_connection", side_effect=AssertionError("Offline")):
            report = workflow.check_vision(self.root, dry_run=True)
        self.assertEqual(report["synthetic_model_mechanics"], "NOT_RUN")
        self.assertEqual(report["checks"]["optimizer_steps"], 0)
        self.assertTrue(report["dry_run"])
        self.assertFalse(report["artifacts_written"])
        self.assertEqual(before, snapshot(self.root))
        self.assertEqual(directories, {p for p in self.root.rglob("*") if p.is_dir()})
        with self.assertRaises(ValueError):
            workflow.check_vision(self.root, dry_run=1)

    def test_missing_report_status_does_not_fit_or_write(self):
        before = snapshot(self.root)
        with patch("models.vision_baseline.model.SmallVisionCNN", side_effect=AssertionError("No model")), \
                patch("torch.optim.SGD.step", side_effect=AssertionError("No training")):
            report = workflow.validate_vision(self.root)
        self.assertEqual(report["status"], "YELLOW")
        self.assertEqual(report["synthetic_model_mechanics"], "NOT_RUN")
        self.assertEqual(before, snapshot(self.root))

    def test_pending_sources_remain_blocked_and_no_image_reads(self):
        source = self.plan["provenance"]["source_approval_snapshot"]["sources"]["plantvillage"]
        self.assertFalse(source["enabled"])
        self.assertEqual(source["approval"]["status"], "pending")
        self.assertFalse(source["research_training"]["allowed"])
        self.assertTrue(source["research_training"]["blockers"])
        with patch("socket.create_connection", side_effect=AssertionError("Offline")):
            report = workflow.check_vision(self.root)
        self.assertEqual(report["status"], "YELLOW")
        self.assertEqual(report["synthetic_model_mechanics"], "GREEN")
        self.assertEqual(report["real_image_integration"], "BLOCKED")
        self.assertIsNone(report["quality"]["plant_disease_confidence"])

    def test_even_approved_sources_do_not_enable_diagnosis(self):
        source_path = self.root / "data" / "source_registry.yaml"
        sources = yaml.safe_load(source_path.read_text())
        sources["sources"]["plantvillage"]["enabled"] = True
        source_path.write_text(yaml.safe_dump(sources), encoding="utf-8")
        license_path = self.root / "licenses" / "registry.json"
        licenses = json.loads(license_path.read_text())
        licenses["licenses"]["plantvillage"]["approval"] = {
            "status": "approved", "intended_use": "research_training", "reviewer": "synthetic test reviewer",
            "reviewed_on": "2026-10-07"}
        license_path.write_text(json.dumps(licenses), encoding="utf-8")
        plan = workflow.vision_plan(self.root)
        self.assertTrue(plan["provenance"]["source_approval_snapshot"]["sources"]["plantvillage"]["research_training"]["allowed"])
        self.assertEqual(plan["real_image_integration"], "BLOCKED")
        with patch("torch.load", side_effect=AssertionError("No checkpoint load")):
            for image in (None, "https://invalid.example/image.jpg", self.root / "missing.jpg", object()):
                result = workflow.diagnose_image(image, root=self.root)
                self.assertEqual(result["status"], "UNKNOWN")
                self.assertIsNone(result["confidence"])
                self.assertIsNone(result["diagnosis"])
                self.assertEqual(result["evidence"], [])

    def test_missing_source_registry_is_unknown_not_approved(self):
        (self.root / "licenses" / "registry.json").unlink()
        plan = workflow.vision_plan(self.root)
        self.assertEqual(plan["provenance"]["source_approval_snapshot"]["status"], "UNKNOWN")
        self.assertEqual(plan["real_image_integration"], "BLOCKED")

    def test_persisted_artifacts_and_read_only_validation(self):
        original_files = snapshot(self.root)
        with patch.object(workflow, "vision_plan", wraps=workflow.vision_plan) as planner, \
                patch("models.vision_baseline.model.run_synthetic_smoke", wraps=run_synthetic_smoke) as smoke:
            report = workflow.check_vision(self.root)
        self.assertGreaterEqual(planner.call_count, 1)
        self.assertEqual(smoke.call_count, 1)
        self.assertEqual(report["status"], "YELLOW")
        report_path = workflow.vision_report_path(self.root)
        envelope = json.loads(report_path.read_bytes())
        self.assertEqual(envelope["sha256"], hashlib.sha256(workflow._bytes(report)).hexdigest())
        checkpoint = self.root / workflow.ARTIFACT_DIR / report["checkpoint"]["name"]
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), report["checkpoint"]["sha256"])
        stored = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.assertFalse(stored["agronomic_model"])
        self.assertEqual(stored["classes"], list(CLASSES))
        self.assertEqual(stored["provenance"], report["provenance"])
        with isolated_cpu(1):
            model = SmallVisionCNN()
            model.load_state_dict(stored["state_dict"], strict=True)
        before = snapshot(self.root)
        with patch("models.vision_baseline.model.SmallVisionCNN", side_effect=AssertionError("No model")), \
                patch("models.vision_baseline.model.run_synthetic_smoke", side_effect=AssertionError("No smoke")), \
                patch("torch.load", side_effect=AssertionError("No unpickling")), \
                patch("torch.optim.SGD.step", side_effect=AssertionError("No fitting")):
            self.assertEqual(workflow.validate_vision(self.root), report)
        self.assertEqual(before, snapshot(self.root))
        for name, entry in original_files.items():
            self.assertEqual(before[name], entry)
        self.assertFalse(list(self.root.rglob("*.part")))

    def test_plan_precedes_training_and_bad_config_allocates_nothing(self):
        with patch.object(workflow, "vision_plan", side_effect=ValueError("plan refused")), \
                patch("models.vision_baseline.model.run_synthetic_smoke", side_effect=AssertionError("No training")):
            with self.assertRaisesRegex(ValueError, "plan refused"):
                workflow.check_vision(self.root)
        config_path = self.root / "configs" / "agri-mini.json"
        original = json.loads(config_path.read_bytes())
        for field, value in (("device", "cuda"), ("offline", False), ("seed", True), ("seed", -1)):
            config_path.write_text(json.dumps({**original, field: value}), encoding="utf-8")
            with patch("models.vision_baseline.model.SmallVisionCNN", side_effect=AssertionError("No allocation")):
                with self.assertRaises(ValueError):
                    workflow.check_vision(self.root)
        self.assertFalse(workflow.vision_report_path(self.root).exists())

    def test_report_and_checkpoint_tampering(self):
        report = workflow.check_vision(self.root)
        path = workflow.vision_report_path(self.root)
        original = path.read_bytes()
        envelope = json.loads(original)
        envelope["report"]["checks"]["optimizer_steps"] = 100
        path.write_text(json.dumps(envelope), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            workflow.validate_vision(self.root)
        path.write_bytes(original)
        checkpoint = self.root / workflow.ARTIFACT_DIR / report["checkpoint"]["name"]
        blob = bytearray(checkpoint.read_bytes())
        blob[-1] ^= 1
        checkpoint.write_bytes(blob)
        with self.assertRaisesRegex(ValueError, "checksum"):
            workflow.validate_vision(self.root)
        checkpoint.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            workflow.validate_vision(self.root)

    def test_stale_config_source_and_code_provenance(self):
        workflow.check_vision(self.root)
        path = workflow.vision_report_path(self.root)
        original_report = path.read_bytes()
        for changed_path in (self.root / "configs" / "agri-mini.json", self.root / "licenses" / "registry.json",
                             self.root / "data" / "source_registry.yaml"):
            original = changed_path.read_bytes()
            changed_path.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "provenance"):
                workflow.validate_vision(self.root)
            changed_path.write_bytes(original)
        envelope = json.loads(original_report)
        code = envelope["report"]["provenance"]["code_sha256"]
        code[next(iter(code))] = "0" * 64
        envelope["sha256"] = hashlib.sha256(workflow._bytes(envelope["report"])).hexdigest()
        path.write_bytes(workflow._bytes(envelope))
        with self.assertRaisesRegex(ValueError, "provenance"):
            workflow.validate_vision(self.root)

    def test_invalid_resigned_report_semantics_and_json(self):
        workflow.check_vision(self.root)
        path = workflow.vision_report_path(self.root)
        original = path.read_bytes()
        for section, key, value in (("checks", "optimizer_steps", True), ("checks", "weights_changed", False),
                                    ("checks", "training_batch_losses", [1.0, True, 0.1]),
                                    ("checkpoint", "name", "..\\outside.pt")):
            envelope = json.loads(original)
            envelope["report"][section][key] = value
            envelope["sha256"] = hashlib.sha256(workflow._bytes(envelope["report"])).hexdigest()
            path.write_bytes(workflow._bytes(envelope))
            with self.assertRaises(ValueError):
                workflow.validate_vision(self.root)
        for text in ('{"report": {}, "report": {}}', '{"report": NaN}', '[]', '{}', '{'):
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                workflow.validate_vision(self.root)

    def test_failed_atomic_publication_preserves_previous_report(self):
        report = workflow.check_vision(self.root)
        before = workflow.vision_report_path(self.root).read_bytes()
        with patch("models.vision_baseline.workflow.os.replace", side_effect=OSError("publish failed")):
            with self.assertRaisesRegex(OSError, "publish failed"):
                workflow.check_vision(self.root)
        self.assertEqual(workflow.vision_report_path(self.root).read_bytes(), before)
        self.assertEqual(workflow.validate_vision(self.root), report)
        self.assertFalse(list((self.root / workflow.ARTIFACT_DIR).glob("*.part")))


if __name__ == "__main__":
    unittest.main()
