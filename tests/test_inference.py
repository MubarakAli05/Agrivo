import contextlib
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

import torch

from agri.synthetic_data import release_path
from models.transformer.inference import (_decode_bytes, _empty_channels, _generate,
                                          generate, generate_from_packed)
from models.transformer.model import AgriTransformer
from models.transformer.packing import pack_example
from tokenizer.structured import StructuredSoilTokenizer
from tokenizer.tokenizer import BPETokenizer, SPECIAL_TOKENS
from tokenizer.train_tokenizer import tokenizer_path
from training.trainer import checkpoint_path, cpu_session, load_checkpoint, train_model


PROJECT = Path(__file__).resolve().parents[1]


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = PROJECT / "tests" / f".inference-fixture-{uuid4().hex}"
        cls.base.mkdir()
        cls.addClassCleanup(shutil.rmtree, cls.base)
        for name in ("configs", "data/releases", "tokenizer/releases"):
            shutil.copytree(PROJECT.joinpath(*name.split("/")), cls.base.joinpath(*name.split("/")))
        path = cls.base / "configs" / "agri-mini.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["model"].update(layers=1, hidden_size=16, attention_heads=2, feed_forward_size=32, dropout=0)
        config["training"].update(max_steps=2, checkpoint_every_steps=2, batch_size=1)
        path.write_text(json.dumps(config), encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()):
            train_model(cls.base)

    def setUp(self):
        self.root = PROJECT / "tests" / f".inference-case-{uuid4().hex}"
        shutil.copytree(self.base, self.root)
        self.addCleanup(shutil.rmtree, self.root)
        self.text = BPETokenizer.load(tokenizer_path(self.root))

    def _packed(self, empty=True):
        example = json.loads((release_path(self.root) / "qa" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
        if empty:
            example["answer"] = ""
        structured = StructuredSoilTokenizer.load(tokenizer_path(self.root) / "structured.json")
        return pack_example(self.text, structured, example, 1536)

    def test_deterministic_bounded_finite_generation_restores_rng(self):
        before, threads = torch.get_rng_state().clone(), torch.get_num_threads()
        first = generate(self.root, "What is soil pH?", max_new_tokens=4)
        second = generate(self.root, "What is soil pH?", max_new_tokens=4)
        self.assertEqual(first, second)
        self.assertTrue({"phase", "status", "model", "training", "metrics", "source_status",
                         "next_required_step", "blockers", "resource_usage"} <= first.keys())
        self.assertIsNone(first["metrics"]["answer_quality"])
        self.assertEqual(first["resource_usage"]["storage_bytes_written"], 0)
        self.assertLessEqual(first["generated_tokens"], 4)
        self.assertEqual(first["generated_tokens"], len(first["generated_token_ids"]))
        self.assertTrue(all(0 <= value < self.text.vocab_size for value in first["generated_token_ids"]))
        self.assertFalse(first["answer_quality_validated"])
        self.assertTrue(first["synthetic"])
        self.assertTrue(first["trained"])
        self.assertIn("untrusted", " ".join(first["caveats"]))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(threads, torch.get_num_threads())

    def test_packed_removes_eos_preserves_numeric_context_and_never_mutates(self):
        packed = self._packed()
        original = deepcopy(packed)
        seen = []
        actual_forward = AgriTransformer.forward
        def capture(model, input_ids, **channels):
            seen.append((input_ids.tolist()[0], channels["numeric_values"].tolist()[0], channels["structured_ids"].tolist()[0]))
            return actual_forward(model, input_ids, **channels)
        with patch.object(AgriTransformer, "forward", capture):
            output = generate_from_packed(self.root, packed, 1)
        self.assertEqual(packed, original)
        self.assertEqual(seen[0][0], packed["input_ids"][:-1])
        self.assertEqual(seen[0][0][-1], self.text.token_to_id("<answer>"))
        self.assertEqual(seen[0][2], packed["structured_ids"][:-1])
        self.assertEqual(len(seen[0][1]), len(packed["numeric_values"]) - 1)
        self.assertEqual(output["prompt_tokens"], len(packed["input_ids"]) - 1)
        self.assertEqual(output["evidence_ids"], packed["evidence_ids"])

    def test_nonempty_answers_bad_eos_and_bad_channels_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty answer"):
            generate_from_packed(self.root, self._packed(empty=False), 1)
        packed = self._packed()
        packed["input_ids"][-1] = 12
        with self.assertRaisesRegex(ValueError, "terminal EOS"):
            generate_from_packed(self.root, packed, 1)
        packed = self._packed()
        packed["numeric_values"].pop()
        with self.assertRaisesRegex(ValueError, "identical lengths"):
            generate_from_packed(self.root, packed, 1)

    def test_arbitrary_byte_failures_preserve_valid_prefix_and_suffix(self):
        offset = len(SPECIAL_TOKENS)
        self.assertEqual(_decode_bytes(self.text, [offset + 65, offset + 0xE2]), ("A\ufffd", True))
        self.assertEqual(_decode_bytes(self.text, [offset + 65, offset + 0xFF, offset + 66]), ("A\ufffdB", True))
        self.assertEqual(_decode_bytes(self.text, [offset + 0xE2, offset + 0x82, offset + 0xAC]), ("\u20ac", False))
        with self.assertRaises(ValueError):
            _decode_bytes(self.text, [self.text.vocab_size])

    def test_control_ids_are_masked_and_invalid_utf8_is_reported(self):
        byte = len(SPECIAL_TOKENS) + 0xE2
        def fake(model, input_ids, **channels):
            logits = torch.zeros((*input_ids.shape, model.config.vocab_size))
            logits[..., 0] = 100
            logits[..., byte] = 50
            return logits
        with patch.object(AgriTransformer, "forward", fake):
            result = generate(self.root, "pH", 1)
        self.assertEqual(result["generated_token_ids"], [byte])
        self.assertTrue(result["invalid_utf8"])
        self.assertEqual(result["text"], "\ufffd")

    def test_eos_and_context_limits(self):
        eos = self.text.token_to_id("<eos>")
        def fake(model, input_ids, **channels):
            logits = torch.zeros((*input_ids.shape, model.config.vocab_size))
            logits[..., eos] = 10
            return logits
        with patch.object(AgriTransformer, "forward", fake):
            result = generate(self.root, "pH", 32)
        self.assertEqual(result["generated_token_ids"], [eos])
        self.assertEqual(result["stop_reason"], "eos")
        self.assertEqual(result["text"], "")
        prefix = _empty_channels([self.text.token_to_id("<answer>")] * 1536)
        result = _generate(self.root, self.text, prefix, 32)
        self.assertEqual(result["stop_reason"], "context_limit")
        self.assertEqual(result["generated_tokens"], 0)
        self.assertEqual(generate(self.root, "pH", 0)["generated_tokens"], 0)

    def test_bad_budgets_and_overflow(self):
        for value in (-1, True, 129, 1.5):
            with self.assertRaises(ValueError):
                generate(self.root, "pH", value)
        with self.assertRaisesRegex(ValueError, "budget"):
            generate(self.root, "a" * 30_000, 1)
        with self.assertRaisesRegex(ValueError, "truncation"):
            generate(self.root, "\u2603" * 1000, 1)

    def test_mismatched_actual_vocab_tensor_rejected_even_with_rehashed_files(self):
        path = checkpoint_path(self.root)
        data = torch.load(path, map_location="cpu", weights_only=True)
        data["model"]["text_embeddings.weight"] = data["model"]["text_embeddings.weight"][:-1]
        torch.save(data, path)
        manifest_path = path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["weights_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["weights_size_bytes"] = path.stat().st_size
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        pointer_path = path.parent.parent / "latest.json"
        pointer = json.loads(pointer_path.read_bytes())
        pointer["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
        with cpu_session(42), self.assertRaisesRegex(ValueError, "actual vocabulary"):
            load_checkpoint(self.root)

    def test_corrupt_hash_rejected_before_decode(self):
        path = checkpoint_path(self.root)
        path.write_bytes(path.read_bytes() + b"bad")
        with self.assertRaisesRegex(ValueError, "checksum"):
            generate(self.root, "pH", 1)


if __name__ == "__main__":
    unittest.main()
