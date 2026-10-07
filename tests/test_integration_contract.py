import unittest

from evaluation.demo import DEMO_QUESTIONS, validate_response


class IntegrationContractTests(unittest.TestCase):
    def test_twenty_distinct_required_questions(self):
        self.assertEqual(len(DEMO_QUESTIONS), 20)
        self.assertEqual(len(set(DEMO_QUESTIONS)), 20)

    def test_explicit_unknown_with_no_evidence_is_valid(self):
        validate_response({
            "answer": "UNKNOWN: no approved image evidence is available.",
            "data_used": [], "sources": [], "model_version": None,
            "confidence": None, "unknown": True,
        })

    def test_known_answer_requires_sources_and_evidence(self):
        response = {
            "answer": "Synthetic example only.", "data_used": ["fixture-row"],
            "sources": ["agrimini_synthetic_fixture"], "model_version": "test",
            "confidence": None, "unknown": False,
        }
        validate_response(response)
        for field in ("sources", "data_used"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_response({**response, field: []})

    def test_missing_fields_and_invented_confidence_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_response({"answer": "UNKNOWN"})
        with self.assertRaises(ValueError):
            validate_response({
                "answer": "UNKNOWN", "data_used": [], "sources": [],
                "model_version": None, "confidence": 0.99, "unknown": True,
            })


if __name__ == "__main__":
    unittest.main()
