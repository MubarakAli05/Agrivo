import json
from pathlib import Path
import tempfile
import unittest

from soil_tokenizer import SoilTokenizer


class SoilTokenizerTests(unittest.TestCase):
    def test_nearest_rank_boundaries_and_extremes(self):
        tokenizer = SoilTokenizer(["ph"], n_bins=4).fit({"ph": x} for x in range(1, 9))
        self.assertEqual(
            tokenizer.transform({"ph": x} for x in [-100, 2, 3, 4, 5, 6, 7, 100]),
            [[2], [2], [3], [3], [4], [4], [5], [5]],
        )
        self.assertEqual(tokenizer.vocab_size, 6)

    def test_feature_specific_ids_and_fixed_order(self):
        tokenizer = SoilTokenizer(["ph", "nitrogen"], 2).fit(
            [{"ph": 5, "nitrogen": 10}, {"ph": 7, "nitrogen": 20}]
        )
        self.assertEqual(tokenizer.encode({"nitrogen": 20, "ph": 5, "label": "good"}), [2, 6])
        self.assertEqual(tokenizer.encode({}), [1, 4])
        self.assertEqual(tokenizer.vocab_size, 7)
        self.assertEqual(tokenizer.pad_token_id, 0)

    def test_missing_and_csv_numbers(self):
        tokenizer = SoilTokenizer(["ph"], 2).fit([{"ph": "5"}, {"ph": "7"}, {}])
        for value in [None, "", " ", float("nan"), "NaN"]:
            with self.subTest(value=value):
                self.assertEqual(tokenizer.encode({"ph": value}), [1])
        self.assertEqual(tokenizer.encode({"ph": " 7 "}), [3])

    def test_invalid_measurements(self):
        tokenizer = SoilTokenizer(["ph"]).fit([{"ph": 7}])
        for value in [True, False, "bad", [], {}, float("inf"), float("-inf")]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    tokenizer.encode({"ph": value})
                with self.assertRaises(ValueError):
                    SoilTokenizer(["ph"]).fit([{"ph": value}])

    def test_constant_and_single_bin(self):
        for bins in [1, 16]:
            tokenizer = SoilTokenizer(["ph"], bins).fit([{"ph": 7}] * 3)
            self.assertEqual(tokenizer.transform([{"ph": 1}, {"ph": 10}]), [[2], [2]])
        tokenizer = SoilTokenizer(["ph"], 1).fit([{"ph": 1}, {"ph": 10}])
        self.assertEqual(tokenizer.encode({"ph": 5}), [2])

    def test_duplicate_quantiles_are_collapsed(self):
        tokenizer = SoilTokenizer(["ph"], 16).fit([{"ph": 5}] * 8 + [{"ph": 7}] * 2)
        self.assertEqual(tokenizer.transform([{"ph": 5}, {"ph": 7}]), [[2], [3]])

    def test_unfitted_operations_fail(self):
        tokenizer = SoilTokenizer()
        for action in [lambda: tokenizer.encode({}), lambda: tokenizer.transform([]), lambda: tokenizer.save("unused.json")]:
            with self.assertRaises(RuntimeError):
                action()

    def test_empty_or_all_missing_training_fails(self):
        for rows in [[], [{}], [{"ph": None}], [{"ph": float("nan")}]]:
            with self.assertRaises(ValueError):
                SoilTokenizer(["ph"]).fit(rows)

    def test_invalid_configuration(self):
        for features in [[], ["ph", "ph"], [""], [" "], [1], "ph"]:
            with self.assertRaises(ValueError):
                SoilTokenizer(features)
        for bins in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                SoilTokenizer(n_bins=bins)

    def test_generator_fit_transform(self):
        tokenizer = SoilTokenizer(["ph"], 2)
        self.assertEqual(tokenizer.fit_transform({"ph": x} for x in [5, 7]), [[2], [3]])
        self.assertEqual(tokenizer.transform([]), [])

    def test_transform_does_not_fit_on_validation_data(self):
        tokenizer = SoilTokenizer(["ph"], 2).fit([{"ph": 5}, {"ph": 7}])
        before = tokenizer.encode({"ph": 6})
        tokenizer.transform([{"ph": 1000}, {"ph": -1000}])
        self.assertEqual(tokenizer.encode({"ph": 6}), before)

    def test_failed_refit_preserves_previous_state(self):
        tokenizer = SoilTokenizer(["ph", "nitrogen"], 2).fit([{"ph": 5, "nitrogen": 10}])
        before = tokenizer.encode({"ph": 5, "nitrogen": 10})
        with self.assertRaises(ValueError):
            tokenizer.fit([{"ph": 8}])
        self.assertEqual(tokenizer.encode({"ph": 5, "nitrogen": 10}), before)

    def test_round_trip(self):
        tokenizer = SoilTokenizer(["ph", "nitrogen"], 4).fit(
            [{"ph": 5, "nitrogen": 10}, {"ph": 7, "nitrogen": 20}]
        )
        rows = [{"ph": 6}, {"ph": 100, "nitrogen": 15}, {}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            restored = SoilTokenizer.load(path)
        self.assertEqual(restored.features, tokenizer.features)
        self.assertEqual(restored.vocab_size, tokenizer.vocab_size)
        self.assertEqual(restored.transform(rows), tokenizer.transform(rows))

    def test_invalid_saved_models(self):
        invalid = [
            [], {}, {"version": 2}, {"version": True},
            {"version": 1, "features": ["ph"], "n_bins": 3, "edges": {}},
        ]
        for cuts in [[2, 1], [1, 1], [float("nan")], [float("inf")], [True], ["1"], [1, 2, 3], "bad"]:
            invalid.append({"version": 1, "features": ["ph"], "n_bins": 3, "edges": {"ph": cuts}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            for payload in invalid:
                with self.subTest(payload=payload):
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        SoilTokenizer.load(path)


if __name__ == "__main__":
    unittest.main()
