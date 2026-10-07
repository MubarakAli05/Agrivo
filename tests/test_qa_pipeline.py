import json
import unittest
from unittest.mock import patch

from retrieval.index import build_index, index_path, search
from retrieval.qa import _pack_query, _render, answer_query
from tests.test_retrieval import RetrievalWorkspace


class QAPipelineTests(RetrievalWorkspace):
    def setUp(self):
        super().setUp()
        build_index(self.root)
        self.checkpoint = patch("retrieval.qa._validate_checkpoint", return_value={"status": "GREEN"}).start()
        self.pack = patch("retrieval.qa._pack_query", return_value={"input_ids": [1, 2, 3]}).start()
        self.generate = patch("retrieval.qa._generate", return_value={
            "text": "UNSUPPORTED_NEURAL_CLAIM", "status": "GREEN", "synthetic": True,
            "trained": True, "checkpoint": "mock-validated-checkpoint",
        }).start()
        self.addCleanup(patch.stopall)

    def test_evidence_to_empty_answer_prompt_then_model_then_safe_renderer(self):
        response = answer_query(self.root, self.question)
        self.assertFalse(response["unknown"])
        self.assertTrue(response["synthetic"])
        self.assertTrue(response["model_used"])
        self.assertEqual(response["answer_origin"], "deterministic_evidence_renderer")
        self.assertIn(str(self.train["ph"]), response["answer"])
        self.assertIn(self.train["record_id"], response["answer"])
        self.assertNotIn("UNSUPPORTED_NEURAL_CLAIM", json.dumps(response))
        self.assertIsNone(response["confidence"])
        self.assertEqual(response["neural_draft"]["status"], "withheld_unverified")
        self.checkpoint.assert_called_once_with(self.root)
        example = self.pack.call_args.args[1]
        self.assertEqual(example["answer"], "")
        self.assertEqual(example["split"], "train")
        self.assertEqual(json.loads(example["context"])["records"], [self.train])
        self.assertNotIn("answer", json.loads(example["context"]))
        self.assertEqual(example["evidence_ids"], [self.train["record_id"]])
        self.generate.assert_called_once_with(self.root, {"input_ids": [1, 2, 3]}, 32)
        self.assertEqual(response["evidence"][0]["record"], self.train)
        self.assertEqual(response["sources"][0]["source_sha256"], response["evidence"][0]["source_sha256"])

    def test_complete_two_depth_evidence_and_explicit_source_answer(self):
        response = answer_query(self.root, f"What is the source for {self.location}?")
        self.assertFalse(response["unknown"])
        self.assertEqual(len(response["evidence"]), 2)
        context = json.loads(self.pack.call_args.args[1]["context"])
        self.assertEqual(len(context["records"]), 2)
        self.assertIn("not measured", response["answer"])
        self.assertIn(self.train["source"], response["answer"])
        self.assertNotIn("UNSUPPORTED_NEURAL_CLAIM", response["answer"])

    def test_missing_property_and_uncertainty_abstain_before_model(self):
        for question, reason in ((f"What is phosphorus for {self.location}?", "missing_value"),
                                 (f"What nitrogen uncertainty is given for {self.location}?", "missing_uncertainty")):
            with self.subTest(question=question):
                response = answer_query(self.root, question)
                self.assertTrue(response["unknown"])
                self.assertEqual(response["reason"], reason)
                self.assertTrue(response["evidence"])
        self.checkpoint.assert_not_called()
        self.generate.assert_not_called()

    def test_zero_is_not_missing_and_units_are_preserved(self):
        row = next(row for row in self.soil if row["location_id"] == "synthetic-site-00")
        retrieved = search(self.root, "What is nitrogen for synthetic-site-00?", split=row["split"])
        answer, reason = _render(retrieved)
        self.assertIsNone(reason)
        self.assertIn("0.0", answer)
        self.assertIn(row["units"]["nitrogen"], answer)
        self.generate.assert_not_called()

    def test_unit_query_does_not_invent_a_missing_value(self):
        response = answer_query(self.root, f"Which phosphorus unit is used for {self.location}?")
        self.assertFalse(response["unknown"])
        self.assertIn(self.train["units"]["phosphorus"], response["answer"])
        self.assertTrue(all(hit["record"]["phosphorus"] is None for hit in response["evidence"]))

    def test_uncertainty_is_not_calibrated_confidence(self):
        question = self.question.replace("What is pH", "What pH uncertainty is given")
        response = answer_query(self.root, question)
        self.assertFalse(response["unknown"])
        self.assertIn(str(self.train["uncertainty"]["ph"]["lower"]), response["answer"])
        self.assertIn("not calibrated", response["answer"])
        self.assertIsNone(response["confidence"])

    def test_heldout_entity_never_reaches_model(self):
        for split in ("validation", "test"):
            row = next(row for row in self.soil if row["split"] == split)
            response = answer_query(self.root, f"What is pH for {row['location_id']}?")
            self.assertEqual(response["reason"], "entity_not_in_requested_split")
            self.assertEqual(response["sources"], [])
            self.assertEqual(response["evidence"], [])
        self.pack.assert_not_called()
        self.generate.assert_not_called()

    def test_unknown_location_no_real_coordinates_no_diagnosis_or_recommendation(self):
        plant = next(row for row in self.plants if row["split"] == "train")
        cases = [("What is pH for synthetic-site-99?", "unknown_synthetic_entity"),
                 (f"What real coordinates correspond to {self.location}?", "no_real_coordinates"),
                 (f"Can you diagnose the image for {plant['record_id']}?", "no_images_or_diagnostic_model"),
                 (f"Which fertilizer should I use for {self.location}?", "unsupported_advice_prediction_or_instruction"),
                 (f"What is the average pH for {self.location}?", "unsupported_advice_prediction_or_instruction")]
        for question, reason in cases:
            with self.subTest(question=question):
                response = answer_query(self.root, question)
                self.assertTrue(response["unknown"])
                self.assertFalse(response["model_used"])
                self.assertEqual(response["reason"], reason)
                self.assertTrue(response["answer"].startswith("UNKNOWN:"))
        self.generate.assert_not_called()

    def test_plant_metadata_remains_fictional(self):
        plant = next(row for row in self.plants if row["split"] == "train")
        response = answer_query(self.root, f"Which class and crop appear in {plant['record_id']}?")
        self.assertFalse(response["unknown"])
        self.assertIn(plant["label"], response["answer"])
        self.assertIn("fictional", response["answer"])
        self.assertEqual(json.loads(self.pack.call_args.args[1]["context"])["records"], [plant])

    def test_checkpoint_missing_abstains_with_complete_evidence(self):
        self.checkpoint.side_effect = FileNotFoundError("No checkpoint")
        response = answer_query(self.root, self.question)
        self.assertEqual(response["reason"], "checkpoint_missing_or_unavailable")
        self.assertEqual(response["status"], "YELLOW")
        self.assertIsNone(response["model_version"])
        self.assertTrue(response["unknown"])
        self.assertFalse(response["model_used"])
        self.assertEqual(response["evidence"][0]["record"], self.train)
        self.assertTrue(response["sources"])
        self.generate.assert_not_called()

    def test_context_overflow_never_truncates_or_calls_model(self):
        self.pack.side_effect = ValueError("Complete evidence exceeds context; no truncation allowed")
        response = answer_query(self.root, f"What is the source for {self.location}?")
        self.assertEqual(response["reason"], "evidence_context_overflow")
        self.assertTrue(response["unknown"])
        self.assertEqual(len(response["evidence"]), 2)
        self.assertEqual(len(json.loads(self.pack.call_args.args[1]["context"])["records"]), 2)
        self.generate.assert_not_called()

    def test_actual_pack_boundary_reserves_generation_without_truncation(self):
        with patch("retrieval.qa.validate_tokenizer"), patch("retrieval.qa.BPETokenizer.load"), \
                patch("retrieval.qa.StructuredSoilTokenizer.load"), \
                patch("retrieval.qa.load_config", return_value={"model": {"context_length": 16}}), \
                patch("retrieval.qa.tokenizer_path", return_value=self.root), \
                patch("retrieval.qa.pack_example", return_value={"input_ids": [1] * 15}) as pack:
            with self.assertRaisesRegex(ValueError, "no truncation"):
                _pack_query(self.root, {"answer": ""}, 3)
            pack.assert_called_once()
            self.assertEqual(_pack_query(self.root, {"answer": ""}, 2)["input_ids"], [1] * 15)

    def test_model_failure_abstains_and_does_not_use_fixture_answer(self):
        self.generate.side_effect = RuntimeError("inference failed")
        response = answer_query(self.root, self.question)
        self.assertTrue(response["unknown"])
        self.assertFalse(response["model_used"])
        self.assertEqual(response["reason"], "model_generation_failed")
        self.assertEqual(response["status"], "RED")
        self.assertEqual(response["answer_origin"], "abstention")

    def test_missing_and_corrupt_index_abstain_without_rebuilding(self):
        path = index_path(self.root)
        path.write_text("not-json", encoding="utf-8")
        corrupt = answer_query(self.root, self.question)
        self.assertTrue(corrupt["unknown"])
        self.assertEqual(corrupt["status"], "RED")
        self.assertEqual(corrupt["reason"], "source_or_index_invalid")
        self.assertEqual(path.read_text(encoding="utf-8"), "not-json")
        path.unlink()
        response = answer_query(self.root, self.question)
        self.assertEqual(response["reason"], "source_or_index_missing")
        self.assertEqual(response["status"], "YELLOW")
        self.assertIsNone(response["model_version"])
        self.assertFalse(path.exists())
        self.generate.assert_not_called()

    def test_every_answer_exposes_demo_fields_and_honest_status(self):
        supported = answer_query(self.root, self.question)
        unknown = answer_query(self.root, "What soil data exists in the United States?")
        self.checkpoint.side_effect = ValueError("checkpoint integrity mismatch")
        invalid = answer_query(self.root, self.question)
        for response, status in ((supported, "GREEN"), (unknown, "YELLOW"), (invalid, "RED")):
            with self.subTest(status=status):
                self.assertTrue({"answer", "data_used", "sources", "model_version", "confidence", "unknown", "status"}
                                <= response.keys())
                self.assertEqual(response["status"], status)
                self.assertIsNone(response["confidence"])
                self.assertTrue(response["data_used"])
        self.assertEqual(supported["model_version"], "AgriTransformer-v0")
        self.assertEqual(supported["model_checkpoint"], "mock-validated-checkpoint")
        self.assertIn("synthetic", supported["data_used"])
        self.assertIsNone(unknown["model_version"])
        self.assertIsNone(invalid["model_version"])
        self.assertEqual(invalid["reason"], "checkpoint_invalid")

    def test_failed_inference_status_cannot_be_reported_as_model_used(self):
        self.generate.return_value = {"status": "RED", "synthetic": True, "trained": True}
        response = answer_query(self.root, self.question)
        self.assertEqual(response["status"], "RED")
        self.assertTrue(response["unknown"])
        self.assertFalse(response["model_used"])
        self.assertIsNone(response["model_version"])
        self.assertEqual(response["reason"], "invalid_model_result")

    def test_api_input_bounds(self):
        for tokens in (True, 0, -1, 129, 1.2):
            with self.assertRaises(ValueError):
                answer_query(self.root, self.question, max_new_tokens=tokens)
        for question in (None, "", "x" * 513):
            with self.assertRaises(ValueError):
                answer_query(self.root, question)


if __name__ == "__main__":
    unittest.main()
