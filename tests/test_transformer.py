"""Focused CPU tests for the manually assembled Phase 5 transformer."""

from dataclasses import replace
from io import BytesIO
from pathlib import Path
import subprocess
import sys
import unittest

import torch
from torch.nn import functional as F

from models.transformer.model import AgriTransformer, TransformerConfig, causal_lm_loss, _LayerNorm


class TransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, cls.previous_threads)

    def setUp(self):
        torch.manual_seed(71)
        self.config = TransformerConfig(
            vocab_size=19, structured_vocab_size=29, context_length=12,
            layers=2, hidden_size=16, attention_heads=2,
            feed_forward_size=32, dropout=0.0,
        )
        self.model = AgriTransformer(self.config).eval()

    def mixed(self):
        return {
            "input_ids": torch.tensor([[1, 0, 3, 0, 5, 0]]),
            "structured_ids": torch.tensor([[0, 22, 0, 23, 0, 0]]),
            "numeric_features": torch.tensor([[-1, 2, -1, 12, -1, -1]]),
            "numeric_values": torch.randn(1, 6, 7),
        }

    def test_package_imports_do_not_load_torch(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys; import models; import models.transformer; "
             "assert 'torch' not in sys.modules"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_config_defaults_and_validation(self):
        defaults = TransformerConfig(19, 29)
        self.assertEqual((defaults.context_length, defaults.layers, defaults.hidden_size,
                          defaults.attention_heads, defaults.feed_forward_size, defaults.dropout),
                         (1536, 4, 256, 4, 1024, 0.1))
        for name in ("vocab_size", "structured_vocab_size", "context_length", "layers",
                     "hidden_size", "attention_heads", "feed_forward_size"):
            for value in (0, -1, True, 1.5):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    replace(self.config, **{name: value})
        for value in (-0.1, 1.1, float("nan"), float("inf"), True, "0.1"):
            with self.subTest(dropout=value), self.assertRaises(ValueError):
                replace(self.config, dropout=value)
        with self.assertRaises(ValueError):
            replace(self.config, attention_heads=3)
        with self.assertRaises(TypeError):
            AgriTransformer({})

    def test_manual_layer_norm_uses_population_variance_and_affine(self):
        norm = _LayerNorm(4)
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([2., 3., 4., 5.]))
            norm.bias.copy_(torch.tensor([1., 2., 3., 4.]))
        values = torch.tensor([[1., 2., 3., 8.], [5., 5., 5., 5.]], requires_grad=True)
        reference = F.layer_norm(values, (4,), norm.weight, norm.bias, norm.eps)
        torch.testing.assert_close(norm(values), reference)
        norm(values).square().sum().backward()
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_logits_loss_and_all_parameter_gradients(self):
        inputs = self.mixed()
        inputs["numeric_values"].requires_grad_()
        logits = self.model(**inputs)
        self.assertEqual(logits.shape, (1, 6, 19))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(logits[:, -1].count_nonzero().item(), 0)
        loss = causal_lm_loss(logits, torch.tensor([[1, -100, 3, -100, 5, -100]]))
        loss.backward()
        for name, parameter in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(self.model.numeric_weight.grad[2].abs().sum().item(), 0)
        self.assertGreater(self.model.numeric_weight.grad[12].abs().sum().item(), 0)
        self.assertEqual(self.model.numeric_weight.grad[1].count_nonzero().item(), 0)
        self.assertGreater(inputs["numeric_values"].grad[:, 1].abs().sum().item(), 0)
        self.assertEqual(inputs["numeric_values"].grad[:, [0, 2, 4, 5]].count_nonzero().item(), 0)
        for embedding in (self.model.text_embeddings, self.model.structured_embeddings):
            self.assertEqual(embedding.weight.grad[0].count_nonzero().item(), 0)

    def test_causality_for_text_structured_values_and_feature_identity(self):
        inputs = self.mixed()
        original = self.model(**inputs)
        edits = [
            ("input_ids", (0, 4), 8),
            ("structured_ids", (0, 3), 28),
            ("numeric_features", (0, 3), 7),
            ("numeric_values", (0, 3, 0), 15.0),
        ]
        for name, index, value in edits:
            changed = {key: tensor.clone() for key, tensor in inputs.items()}
            changed[name][index] = value
            result = self.model(**changed)
            with self.subTest(channel=name):
                torch.testing.assert_close(original[:, :3], result[:, :3], rtol=0, atol=0)
                self.assertFalse(torch.equal(original[:, 3:5], result[:, 3:5]))
        numeric = inputs["numeric_values"].requires_grad_()
        self.model(**inputs)[:, :3].square().sum().backward()
        self.assertEqual(numeric.grad[:, 3:].count_nonzero().item(), 0)
        prefix = {name: tensor[:, :3] for name, tensor in inputs.items()}
        torch.testing.assert_close(original[:, :3], self.model(**prefix), rtol=1e-5, atol=1e-6)

    def test_padding_and_batch_equivalence_including_empty_prefixes(self):
        single = self.model(torch.tensor([[1, 2, 3]]))
        ids = torch.tensor([[1, 2, 3, 0, 0], [0, 0, 1, 2, 3],
                            [1, 0, 2, 0, 3], [0, 0, 0, 0, 0]])
        batched = self.model(ids)
        self.assertTrue(torch.isfinite(batched).all())
        for row, indices in ((0, [0, 1, 2]), (1, [2, 3, 4]), (2, [0, 2, 4])):
            torch.testing.assert_close(batched[row, indices], single[0], rtol=1e-5, atol=1e-6)
        self.assertEqual(batched[ids == 0].count_nonzero().item(), 0)
        batched.square().sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters()
                            if p.grad is not None))

    def test_explicit_mask_hides_text_and_structured_ids(self):
        mask = torch.tensor([[True, False, True, False]])
        expected = self.model(torch.tensor([[1, 3]]))
        text = torch.tensor([[1, 12, 3, 0]])
        structured = torch.tensor([[0, 0, 0, 22]])
        actual = self.model(text, mask, structured)
        torch.testing.assert_close(actual[:, [0, 2]], expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(actual[:, [1, 3]].count_nonzero().item(), 0)
        text[0, 1] = 18
        structured[0, 3] = 28
        torch.testing.assert_close(actual, self.model(text, mask, structured), rtol=0, atol=0)

    def test_independent_namespaces_and_default_structured_mask(self):
        text_result = self.model(torch.tensor([[2]]))
        structured_result = self.model(torch.tensor([[0]]), structured_ids=torch.tensor([[2]]))
        self.assertFalse(torch.equal(text_result, structured_result))
        self.assertNotEqual(self.model.text_embeddings.weight.data_ptr(),
                            self.model.structured_embeddings.weight.data_ptr())
        with torch.no_grad():
            self.model.text_embeddings.weight[2].add_(torch.arange(16))
        torch.testing.assert_close(structured_result,
                                   self.model(torch.tensor([[0]]), structured_ids=torch.tensor([[2]])),
                                   rtol=0, atol=0)
        # A structured ID beyond the text vocabulary is valid in its own table.
        self.assertTrue(torch.isfinite(self.model(torch.tensor([[0]]),
                                                 structured_ids=torch.tensor([[28]]))).all())

    def test_numeric_channels_affect_only_selected_features(self):
        inputs = self.mixed()
        original = self.model(**inputs)
        inputs["numeric_values"][0, 1] += 20
        changed = self.model(**inputs)
        torch.testing.assert_close(original[:, :1], changed[:, :1], rtol=0, atol=0)
        self.assertFalse(torch.equal(original[:, 1:5], changed[:, 1:5]))
        original = changed
        inputs["numeric_values"][:, [0, 2, 4, 5]] += 1000
        torch.testing.assert_close(original, self.model(**inputs), rtol=0, atol=0)
        inputs["numeric_values"] = inputs["numeric_values"].double()
        torch.testing.assert_close(original, self.model(**inputs), rtol=0, atol=0)

    def test_dropout_train_eval_and_seeded_initialization(self):
        config = replace(self.config, layers=1, dropout=0.5)
        torch.manual_seed(12)
        first = AgriTransformer(config)
        torch.manual_seed(12)
        second = AgriTransformer(config)
        for key, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
        ids = torch.tensor([[1, 2, 3, 4]])
        self.assertFalse(torch.equal(first(ids), first(ids)))
        torch.manual_seed(44)
        output = first(ids)
        torch.manual_seed(44)
        torch.testing.assert_close(output, second(ids), rtol=0, atol=0)
        first.eval()
        torch.testing.assert_close(first(ids), first(ids), rtol=0, atol=0)

    def test_exact_state_dict_roundtrip(self):
        inputs = self.mixed()
        original = self.model(**inputs)
        stream = BytesIO()
        torch.save(self.model.state_dict(), stream)
        stream.seek(0)
        restored = AgriTransformer(self.config).eval()
        restored.load_state_dict(torch.load(stream, weights_only=True))
        torch.testing.assert_close(original, restored(**inputs), rtol=0, atol=0)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)

    def test_input_validation_shapes_dtypes_ranges_and_devices(self):
        inputs = self.mixed()
        bad_inputs = [
            ("input_ids", [[1]]),
            ("input_ids", torch.ones(6, dtype=torch.long)),
            ("input_ids", torch.ones(1, 6, 1, dtype=torch.long)),
            ("input_ids", torch.ones(1, 6)),
            ("input_ids", torch.ones(1, 6, dtype=torch.int32)),
            ("input_ids", torch.full((1, 6), -1)),
            ("input_ids", torch.full((1, 6), 19)),
            ("input_ids", torch.ones(1, 6, dtype=torch.long, device="meta")),
            ("structured_ids", torch.zeros(1, 5, dtype=torch.long)),
            ("structured_ids", torch.zeros(1, 6)),
            ("structured_ids", torch.full((1, 6), 29)),
            ("structured_ids", torch.full((1, 6), -1)),
            ("attention_mask", torch.ones(1, 6, dtype=torch.long)),
            ("attention_mask", torch.ones(6, dtype=torch.bool)),
            ("attention_mask", torch.ones(1, 6, dtype=torch.bool, device="meta")),
            ("numeric_features", torch.zeros(1, 6)),
            ("numeric_features", torch.zeros(1, 5, dtype=torch.long)),
            ("numeric_features", torch.full((1, 6), -2)),
            ("numeric_features", torch.full((1, 6), 13)),
            ("numeric_features", torch.zeros(1, 6, dtype=torch.long, device="meta")),
            ("numeric_values", torch.zeros(1, 6, 6)),
            ("numeric_values", torch.zeros(1, 6, 7, dtype=torch.long)),
            ("numeric_values", torch.zeros(1, 6, 7, dtype=torch.complex64)),
            ("numeric_values", torch.zeros(1, 6, 7, device="meta")),
        ]
        for name, value in bad_inputs:
            with self.subTest(name=name, value=str(value)), self.assertRaises((ValueError, TypeError)):
                self.model(**{**inputs, name: value})
        for value in (float("nan"), float("inf"), -float("inf")):
            values = inputs["numeric_values"].clone()
            values[0, -1, 0] = value
            with self.subTest(nonfinite=value), self.assertRaises(ValueError):
                self.model(**{**inputs, "numeric_values": values})

    def test_invalid_namespace_and_numeric_alignment(self):
        with self.assertRaises(ValueError):
            self.model(torch.tensor([[1]]), structured_ids=torch.tensor([[1]]))
        with self.assertRaises(ValueError):
            self.model(torch.tensor([[0]]), attention_mask=torch.tensor([[True]]))
        inputs = self.mixed()
        for name in ("numeric_values", "numeric_features"):
            with self.subTest(missing=name), self.assertRaises(ValueError):
                self.model(**{key: value for key, value in inputs.items() if key != name})
        for index in (0, 5):
            features = inputs["numeric_features"].clone()
            features[0, index] = 2
            with self.assertRaises(ValueError):
                self.model(**{**inputs, "numeric_features": features})
        mask = torch.tensor([[True, False, True, True, True, False]])
        with self.assertRaises(ValueError):
            self.model(**inputs, attention_mask=mask)

    def test_context_boundary_and_empty_input(self):
        self.assertEqual(self.model(torch.ones(1, 12, dtype=torch.long)).shape, (1, 12, 19))
        for shape in ((1, 13), (1, 0), (0, 4)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.model(torch.zeros(shape, dtype=torch.long))

    def test_shifted_loss_ignored_labels_and_mask(self):
        logits = torch.randn(2, 5, 19, requires_grad=True)
        labels = torch.tensor([[1, -100, 3, 4, 5], [2, 3, 4, -100, 6]])
        mask = torch.tensor([[True, True, True, False, True], [False, True, True, True, True]])
        expected = F.cross_entropy(torch.stack((logits[0, 1], logits[1, 1], logits[1, 3])),
                                   torch.tensor([3, 4, 6]))
        actual = causal_lm_loss(logits, labels, mask)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertEqual(logits.grad[:, -1].count_nonzero().item(), 0)
        self.assertEqual(logits.grad[0, 0].count_nonzero().item(), 0)
        self.assertEqual(logits.grad[0, 2:4].count_nonzero().item(), 0)
        expected_unmasked = F.cross_entropy(logits[:, :-1].reshape(-1, 19),
                                            labels[:, 1:].reshape(-1), ignore_index=-100)
        torch.testing.assert_close(causal_lm_loss(logits, labels), expected_unmasked)

    def test_loss_empty_supervision_and_all_padding_gradients(self):
        for length in (1, 4):
            for masked in (False, True):
                logits = torch.randn(2, length, 19, requires_grad=True)
                labels = torch.full((2, length), 2 if masked else -100, dtype=torch.long)
                mask = torch.zeros(2, length, dtype=torch.bool) if masked else None
                loss = causal_lm_loss(logits, labels, mask)
                self.assertEqual(loss.item(), 0.0)
                self.assertTrue(loss.requires_grad)
                loss.backward()
                self.assertEqual(logits.grad.count_nonzero().item(), 0)
        logits = self.model(torch.zeros(2, 4, dtype=torch.long))
        causal_lm_loss(logits, torch.full((2, 4), -100, dtype=torch.long)).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters()
                            if p.grad is not None))

    def test_loss_validation_including_masked_targets(self):
        logits = torch.randn(1, 3, 19)
        labels = torch.tensor([[1, 2, 3]])
        cases = [
            (logits.long(), labels, None),
            (logits[:, :, 0], labels, None),
            (logits[:, :0], labels[:, :0], None),
            (logits, labels.float(), None),
            (logits, labels[:, :2], None),
            (logits, labels.to("meta"), None),
            (logits, labels, torch.ones(1, 3)),
            (logits, labels, torch.ones(1, 2, dtype=torch.bool)),
            (logits, labels, torch.ones(1, 3, dtype=torch.bool, device="meta")),
        ]
        for args in cases:
            with self.subTest(args=str(args)), self.assertRaises((TypeError, ValueError)):
                causal_lm_loss(*args)
        for target in (-101, -1, 19):
            bad_labels = labels.clone()
            bad_labels[0, 0] = target
            with self.assertRaises(ValueError):
                causal_lm_loss(logits, bad_labels, torch.zeros(1, 3, dtype=torch.bool))
        for value in (float("inf"), float("nan")):
            bad_logits = logits.clone()
            bad_logits[0, 0, 0] = value
            with self.assertRaises(ValueError):
                causal_lm_loss(bad_logits, labels)


if __name__ == "__main__":
    unittest.main()
