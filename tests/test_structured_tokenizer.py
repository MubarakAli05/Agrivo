from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from agri.synthetic_data import build_fixture
from tokenizer.structured import StructuredSoilTokenizer


class StructuredTokenizerTests(unittest.TestCase):
    def setUp(self):
        self.soil = build_fixture(42)[0]
        self.train = [row for row in self.soil if row["split"] == "train"]
        self.tokenizer = StructuredSoilTokenizer(n_bins=4).fit(self.train)

    def test_fixed_field_ids_and_exact_numeric_side_channels(self):
        for row in self.soil:
            encoded = self.tokenizer.encode(row)
            self.assertEqual(len(encoded["token_ids"]), 41)
            self.assertTrue(all(0 < token < self.tokenizer.vocab_size for token in encoded["token_ids"]))
            for feature, field in encoded["numeric"].items():
                self.assertEqual(field["value"], row[feature])
                self.assertEqual(field["unit"], row["units"][feature])
                self.assertEqual(field["uncertainty"], row["uncertainty"][feature])
            self.assertEqual(encoded["categories"]["depth"]["value"], row["depth"])
            self.assertEqual(encoded["categories"]["source"]["value"], row["source"])

    def test_all_missing_feature_has_distinct_uncalibrated_observation_token(self):
        row = deepcopy(self.train[0])
        missing = self.tokenizer.encode(row)["numeric"]["phosphorus"]
        row["phosphorus"], row["phosphorus_missing"] = 0.0, False
        present = self.tokenizer.encode(row)["numeric"]["phosphorus"]
        self.assertNotEqual(missing["token_id"], present["token_id"])
        self.assertIsNone(missing["value"])
        self.assertEqual(present["value"], 0)
        self.assertFalse(present["calibrated"])
        self.assertFalse(present["missing"])
        self.assertIsNone(present["outside_training_range"])
        self.assertEqual(json.loads(self.tokenizer.artifacts()["structured.json"])["features"]["phosphorus"],
                         {"count": 0, "cuts": [], "minimum": None, "maximum": None})

    def test_fit_rejects_held_out_rows_and_encoding_does_not_learn(self):
        before = self.tokenizer.artifacts()
        held_out = next(row for row in self.soil if row["split"] == "test")
        with self.assertRaisesRegex(ValueError, "training-split"):
            self.tokenizer.fit([held_out])
        self.assertEqual(before, self.tokenizer.artifacts())
        self.tokenizer.encode(held_out)
        self.assertEqual(before, self.tokenizer.artifacts())

    def test_unknown_category_and_missing_category_are_distinct(self):
        row = deepcopy(self.train[0])
        row["crop"] = "UNSEEN_CROP"
        unknown = self.tokenizer.encode(row)["categories"]["crop"]
        self.assertFalse(unknown["known"])
        self.assertEqual(unknown["value"], "UNSEEN_CROP")
        row["crop"] = None
        missing = self.tokenizer.encode(row)["categories"]["crop"]
        self.assertNotEqual(unknown["token_id"], missing["token_id"])

    def test_precise_unseen_depth_range_is_not_rounded_to_known_category(self):
        row = deepcopy(self.train[0])
        row["depth"]["bottom"] += 0.0000001
        depth = self.tokenizer.encode(row)["categories"]["depth"]
        self.assertFalse(depth["known"])
        self.assertEqual(depth["value"], row["depth"])

    def test_range_overflow_retains_value_and_reports_out_of_training_range(self):
        row = deepcopy(self.train[0])
        row["ph"] = 14
        row["ph_missing"] = False
        row["uncertainty"]["ph"] = {"lower": 13, "upper": 14, "method": "test_only"}
        field = self.tokenizer.encode(row)["numeric"]["ph"]
        self.assertEqual(field["value"], 14)
        self.assertTrue(field["outside_training_range"])
        self.assertEqual(field["uncertainty"], row["uncertainty"]["ph"])
        field["uncertainty"]["method"] = "mutated_output"
        self.assertEqual(row["uncertainty"]["ph"]["method"], "test_only")

    def test_exact_quantiles_and_constant_values(self):
        rows = [deepcopy(self.train[0]) for _ in range(4)]
        for index, row in enumerate(rows):
            row["nitrogen"] = index
            row["nitrogen_missing"] = False
            row["potassium"] = 50
            row["potassium_missing"] = False
        self.tokenizer.fit(rows)
        state = json.loads(self.tokenizer.artifacts()["structured.json"])
        self.assertEqual(state["features"]["nitrogen"]["cuts"], [0, 1, 2])
        self.assertEqual(state["features"]["potassium"]["cuts"], [])
        ids = [self.tokenizer.encode(row)["numeric"]["nitrogen"]["token_id"] for row in rows]
        self.assertEqual(len(set(ids)), 4)
        self.assertEqual(self.tokenizer.artifacts(), StructuredSoilTokenizer(4).fit(list(reversed(rows))).artifacts())

    def test_invalid_numbers_units_missingness_and_uncertainty(self):
        for value in (True, float("nan"), float("inf"), -1, "6.5", 10 ** 1000):
            row = deepcopy(self.train[0])
            row["nitrogen"] = value
            row["nitrogen_missing"] = False
            with self.subTest(value_type=type(value)), self.assertRaises(ValueError):
                self.tokenizer.encode(row)
        for mutate in (lambda r: r["units"].update(nitrogen="kg/ha"),
                       lambda r: r.update(ph_missing=0),
                       lambda r: r.update(ph_missing=not r["ph_missing"]),
                       lambda r: r["depth"].update(top=True),
                       lambda r: r["uncertainty"].update(ph={"lower": 8, "upper": 2, "method": "bad"})):
            row = deepcopy(self.train[0])
            mutate(row)
            with self.assertRaises(ValueError):
                self.tokenizer.encode(row)

    def test_serialization_roundtrip_and_malformed_state(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "structured.json"
            self.tokenizer.save(path)
            restored = StructuredSoilTokenizer.load(path)
            self.assertEqual(restored.artifacts(), self.tokenizer.artifacts())
            for row in self.soil:
                self.assertEqual(restored.encode(row), self.tokenizer.encode(row))
            original = json.loads(path.read_text(encoding="utf-8"))
            for mutate in (lambda s: s.update(version=True),
                           lambda s: s["features"]["phosphorus"].update(minimum=0),
                           lambda s: s["features"]["ph"].update(cuts=[float("nan")]),
                           lambda s: s["categories"].update(crop=["a", "a"]),
                           lambda s: s["units"].update(ph="mg/kg")):
                state = deepcopy(original)
                mutate(state)
                path.write_text(json.dumps(state), encoding="utf-8")
                with self.assertRaises(ValueError):
                    StructuredSoilTokenizer.load(path)
            path.write_text('{"version":1,"version":1}', encoding="utf-8")
            with self.assertRaises(ValueError):
                StructuredSoilTokenizer.load(path)

    def test_unfitted_and_invalid_training_fail(self):
        tokenizer = StructuredSoilTokenizer()
        with self.assertRaises(RuntimeError):
            tokenizer.encode(self.train[0])
        with self.assertRaises(RuntimeError):
            tokenizer.artifacts()
        with self.assertRaises(ValueError):
            tokenizer.fit([])
        for bins in (True, 0, -1, 1.5, 257):
            with self.assertRaises(ValueError):
                StructuredSoilTokenizer(bins)


if __name__ == "__main__":
    unittest.main()
