from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.demo import DEMO_QUESTIONS
from evaluation.integration import KNOWN_QUESTION, _canonical, integration_report_path, run_integration, validate_integration


class IntegrationReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        unknown = {"answer": "UNKNOWN: no approved evidence", "data_used": [], "sources": [],
                   "model_version": None, "confidence": None, "unknown": True, "status": "YELLOW",
                   "model_used": False, "answer_origin": "abstention"}
        known = {**unknown, "answer": "Synthetic fixture only", "data_used": ["fixture"],
                 "sources": ["agrimini_synthetic_fixture"], "unknown": False, "model_used": True,
                 "status": "GREEN", "answer_origin": "deterministic_evidence_renderer",
                 "neural_draft": {"used_as_factual_answer": False}}
        self.report = {"phase": 16, "status": "YELLOW", "integration_mechanics": "GREEN",
                       "model_execution": "GREEN", "training": {"status": "GREEN"},
                       "metrics": {"response_contracts_passed": 21, "original_demo_questions": 20,
                                   "unknown_responses": 20, "model_executed_responses": 1,
                                   "agricultural_accuracy": None},
                       "fingerprints": {"test": "digest"},
                       "cases": [{"question": question, "response": unknown} for question in DEMO_QUESTIONS]
                                + [{"question": KNOWN_QUESTION, "response": known}]}
        self.path = integration_report_path(self.root)
        self.path.parent.mkdir(parents=True)

    def write(self):
        self.path.write_text(json.dumps({"report": self.report,
            "sha256": hashlib.sha256(_canonical(self.report)).hexdigest()}), encoding="utf-8")

    def test_validation_is_read_only(self):
        self.write()
        before = self.path.read_bytes()
        with patch("evaluation.integration._fingerprints", return_value={"test": "digest"}):
            self.assertEqual(validate_integration(self.root), self.report)
        self.assertEqual(self.path.read_bytes(), before)

    def test_corrupt_checksum_rejected(self):
        self.write()
        self.path.write_text(self.path.read_text().replace("UNKNOWN:", "CHANGED:"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_integration(self.root)

    def test_changed_dependencies_rejected(self):
        self.write()
        with patch("evaluation.integration._fingerprints", return_value={"test": "changed"}):
            with self.assertRaisesRegex(ValueError, "stale"):
                validate_integration(self.root)

    def test_missing_demo_case_rejected(self):
        self.report["cases"].pop(0)
        self.write()
        with patch("evaluation.integration._fingerprints", return_value={"test": "digest"}):
            with self.assertRaisesRegex(ValueError, "question set"):
                validate_integration(self.root)

    def make_partial(self):
        self.report.update(integration_mechanics="PARTIAL", model_execution="BLOCKED",
                           training={"status": "YELLOW", "reason": "training_checkpoint_absent"})
        self.report["cases"][-1]["response"].update(
            answer="UNKNOWN: checkpoint missing or unavailable.", unknown=True, model_used=False,
            status="YELLOW", answer_origin="abstention", neural_draft=None,
            reason="checkpoint_missing_or_unavailable")
        self.report["metrics"].update(unknown_responses=21, model_executed_responses=0)

    def test_deferred_training_partial_report_is_valid(self):
        self.make_partial()
        self.write()
        with patch("evaluation.integration._fingerprints", return_value={"test": "digest"}):
            self.assertEqual(validate_integration(self.root)["model_execution"], "BLOCKED")

    def test_partial_report_cannot_claim_model_execution(self):
        self.make_partial()
        self.report["cases"][0]["response"] = {**self.report["cases"][0]["response"], "model_used": True}
        self.write()
        with patch("evaluation.integration._fingerprints", return_value={"test": "digest"}):
            with self.assertRaisesRegex(ValueError, "abstention"):
                validate_integration(self.root)

    def test_unverified_draft_cannot_be_factual(self):
        self.report["cases"][-1]["response"]["neural_draft"]["used_as_factual_answer"] = True
        self.write()
        with patch("evaluation.integration._fingerprints", return_value={"test": "digest"}):
            with self.assertRaisesRegex(ValueError, "draft"):
                validate_integration(self.root)

    def test_incorrect_metrics_rejected(self):
        self.report["metrics"]["model_executed_responses"] = 21
        self.write()
        with patch("evaluation.integration._fingerprints", return_value={"test": "digest"}):
            with self.assertRaisesRegex(ValueError, "metrics"):
                validate_integration(self.root)

    def run_with_mock_dependencies(self, training_error=None):
        with ExitStack() as stack:
            for target, value in (
                ("evaluation.integration._fingerprints", {"test": "digest"}),
                ("training.trainer.training_report_path", self.root / "training" / "latest.json"),
                ("agri.model_check.validate_model", {"status": "GREEN"}),
                ("retrieval.index.validate_index", {"status": "GREEN"}),
                ("models.vision_baseline.workflow.validate_vision", {"synthetic_model_mechanics": "GREEN"}),
                ("agri.source_registry.inspect_sources", {"status": "YELLOW"}),
            ):
                stack.enter_context(patch(target, return_value=value))
            training = stack.enter_context(patch("training.trainer.validate_training",
                return_value={"status": "GREEN"}, side_effect=training_error))
            answer = stack.enter_context(patch("retrieval.qa.answer_query",
                side_effect=[case["response"] for case in self.report["cases"]]))
            result = run_integration(self.root)
            self.assertEqual(answer.call_count, 21)
            self.assertEqual(training.call_count, int((self.root / "training" / "latest.json").exists()))
            return result

    def test_runner_persists_partial_report_without_training(self):
        self.make_partial()
        result = self.run_with_mock_dependencies()
        self.assertEqual(result["integration_mechanics"], "PARTIAL")
        self.assertEqual(result["metrics"]["model_executed_responses"], 0)
        with patch("evaluation.integration._fingerprints", return_value={"test": "digest"}):
            self.assertEqual(validate_integration(self.root), result)

    def test_runner_requires_checkpoint_execution_when_trained(self):
        training = self.root / "training" / "latest.json"
        training.parent.mkdir()
        training.write_text("mock checkpoint metadata")
        result = self.run_with_mock_dependencies()
        self.assertEqual(result["integration_mechanics"], "GREEN")
        self.assertEqual(result["metrics"]["model_executed_responses"], 1)

    def test_corrupt_existing_training_is_not_treated_as_deferred(self):
        training = self.root / "training" / "latest.json"
        training.parent.mkdir()
        training.write_text("corrupt")
        with self.assertRaisesRegex(ValueError, "corrupt"):
            self.run_with_mock_dependencies(training_error=ValueError("corrupt training"))
        self.assertFalse(self.path.exists())

    def test_orphan_training_artifacts_are_not_treated_as_deferred(self):
        training = self.root / "training"
        training.mkdir()
        (training / "weights.pt").write_bytes(b"incomplete")
        with self.assertRaisesRegex(ValueError, "Incomplete training"):
            self.run_with_mock_dependencies()
        self.assertFalse(self.path.exists())

    def test_missing_model_execution_rejected(self):
        self.report["cases"][-1]["response"]["model_used"] = False
        self.write()
        with patch("evaluation.integration._fingerprints", return_value={"test": "digest"}):
            with self.assertRaisesRegex(ValueError, "checkpoint-backed"):
                validate_integration(self.root)


if __name__ == "__main__":
    unittest.main()
