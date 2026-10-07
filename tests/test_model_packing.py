from copy import deepcopy
import json
import unittest

import torch

from agri.data_schema import SOIL_FEATURES
from agri.synthetic_data import build_fixture
from models.transformer.packing import collate_examples, pack_example
from tokenizer.structured import StructuredSoilTokenizer
from tokenizer.tokenizer import BPETokenizer
from tokenizer.train_tokenizer import encode_example


class ModelPackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        soil, _, cls.qa = build_fixture(42)
        cls.text = BPETokenizer(2048).fit(row[field] for row in cls.qa if row["split"] == "train"
                                        for field in ("question", "context", "answer"))
        cls.structured = StructuredSoilTokenizer().fit([row for row in soil if row["split"] == "train"])

    def pack(self, example, length=1536):
        return pack_example(self.text, self.structured, example, length)

    def test_all_examples_fit_with_complete_text_and_declared_evidence(self):
        lengths = []
        for example in self.qa:
            packed = self.pack(example)
            lengths.append(len(packed["input_ids"]))
            for field, (start, end) in packed["text_spans"].items():
                self.assertEqual(self.text.decode(packed["input_ids"][start:end]), self.text.normalize(example[field]))
            self.assertEqual(packed["source_ids"], example["source_ids"])
            self.assertEqual(packed["evidence_ids"], example["evidence_ids"])
            self.assertEqual(packed["split"], example["split"])
            self.assertEqual(packed["group_id"], example["group_id"])
        self.assertEqual(max(lengths), 1411)

    def test_record_boundaries_and_disjoint_namespaces(self):
        packed = self.pack(self.qa[0])
        self.assertEqual(len(packed["input_ids"]), len(encode_example(self.text, self.qa[0])) + 86)
        self.assertEqual(len(packed["raw_structured"]), 2)
        self.assertEqual(sum(token != 0 for token in packed["structured_ids"]), 82)
        self.assertEqual(packed["input_ids"].count(self.text.token_to_id("<source>")), 2)
        self.assertEqual(packed["input_ids"].count(self.text.token_to_id("<sep>")), 2)
        for text_id, structured_id, label in zip(packed["input_ids"], packed["structured_ids"], packed["labels"]):
            self.assertNotEqual(bool(text_id), bool(structured_id))
            self.assertEqual(label, text_id if text_id else -100)

    def test_numeric_values_uncertainty_depth_and_zero_remain_distinct(self):
        example = deepcopy(self.qa[0])
        context = json.loads(example["context"])
        context["records"][0]["phosphorus"] = 0.0
        context["records"][0]["phosphorus_missing"] = False
        example["context"] = json.dumps(context)
        packed = self.pack(example)
        phosphorus = list(SOIL_FEATURES).index("phosphorus")
        positions = [i for i, feature in enumerate(packed["numeric_features"]) if feature == phosphorus]
        zero, missing = [packed["numeric_values"][i] for i in positions]
        self.assertEqual(zero, [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        self.assertEqual(missing, [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
        self.assertNotEqual(packed["structured_ids"][positions[0]], packed["structured_ids"][positions[1]])
        self.assertEqual(packed["raw_structured"][0]["numeric"]["phosphorus"]["value"], 0.0)
        raw = packed["raw_structured"][0]
        self.assertEqual(raw["numeric"]["ph"]["uncertainty"], context["records"][0]["uncertainty"]["ph"])
        self.assertEqual(raw["categories"]["depth"]["value"], context["records"][0]["depth"])
        self.assertEqual(packed["numeric_features"].count(12), 2)

    def test_packing_does_not_fit_or_mutate_inputs(self):
        original = deepcopy(self.qa[0])
        text_before, structured_before = self.text.artifacts(), self.structured.artifacts()
        first, second = self.pack(self.qa[0]), self.pack(self.qa[0])
        self.assertEqual(first, second)
        first["raw_structured"][0]["numeric"]["ph"]["value"] = -123
        self.assertEqual(self.qa[0], original)
        self.assertEqual(self.text.artifacts(), text_before)
        self.assertEqual(self.structured.artifacts(), structured_before)

    def test_overflow_rejected_never_truncated(self):
        with self.assertRaisesRegex(ValueError, "no truncation"):
            self.pack(self.qa[0], 256)
        for length in (True, 0, -1, 1.5):
            with self.subTest(length=length), self.assertRaises(ValueError):
                self.pack(self.qa[0], length)

    def test_group_split_source_and_evidence_mismatches_rejected(self):
        for change in ({"split": "invalid"}, {"group_id": "unrelated"}, {"source_ids": []},
                       {"evidence_ids": ["absent"]}, {"synthetic": False}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.pack({**self.qa[0], **change})
        example = deepcopy(self.qa[0])
        context = json.loads(example["context"])
        context["records"][0]["split"] = "test" if example["split"] != "test" else "train"
        example["context"] = json.dumps(context)
        with self.assertRaisesRegex(ValueError, "boundaries"):
            self.pack(example)

    def test_padding_masks_and_labels_and_split_isolation(self):
        examples = [self.pack(row) for row in self.qa if row["split"] == "train"]
        shortest, longest = min(examples, key=lambda row: len(row["input_ids"])), max(examples, key=lambda row: len(row["input_ids"]))
        batch = collate_examples([shortest, longest])
        length = len(shortest["input_ids"])
        self.assertEqual(batch["input_ids"].dtype, torch.long)
        self.assertEqual(batch["numeric_values"].shape, (2, len(longest["input_ids"]), 7))
        self.assertEqual(batch["attention_mask"].dtype, torch.bool)
        self.assertTrue((batch["labels"][0, length:] == -100).all().item())
        self.assertFalse(batch["attention_mask"][0, length:].any().item())
        self.assertTrue((batch["numeric_features"][0, length:] == -1).all().item())
        with self.assertRaisesRegex(ValueError, "one split"):
            collate_examples([shortest, self.pack(next(row for row in self.qa if row["split"] == "test"))])
        with self.assertRaises(ValueError):
            collate_examples([])

    def test_special_spellings_in_text_do_not_inject_boundaries(self):
        example = {**self.qa[0], "question": "literal <answer> and <source>"}
        packed = self.pack(example)
        self.assertEqual(packed["input_ids"].count(self.text.token_to_id("<answer>")), 1)
        self.assertEqual(packed["input_ids"].count(self.text.token_to_id("<source>")), 2)


if __name__ == "__main__":
    unittest.main()
