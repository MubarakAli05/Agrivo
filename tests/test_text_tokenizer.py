"""Focused behavioral and artifact-validation tests for custom byte-level BPE."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
import shutil
import unittest
import uuid

from tokenizer import BPETokenizer, SPECIAL_TOKENS

OFFSET = len(SPECIAL_TOKENS)
BASE_SIZE = OFFSET + 256


def byte_id(text: str) -> int:
    return text.encode("utf-8")[0] + OFFSET


def merge_rows(model: BPETokenizer) -> list[tuple[int, int]]:
    return [tuple(map(int, line.split())) for line in model.artifacts()["merges.txt"].decode().splitlines()[1:]]


class TextTokenizerTests(unittest.TestCase):
    def test_fixed_special_ids_and_full_byte_alphabet(self) -> None:
        expected = (
            "<pad>", "<unk>", "<bos>", "<eos>", "<sep>", "<mask>",
            "<question>", "<context>", "<answer>", "<unknown>", "<source>",
        )
        self.assertEqual(SPECIAL_TOKENS, expected)
        model = BPETokenizer(BASE_SIZE).fit([])
        self.assertEqual(model.vocab_size, BASE_SIZE)
        self.assertEqual([model.token_to_id(token) for token in SPECIAL_TOKENS], list(range(OFFSET)))
        vocab = json.loads(model.artifacts()["vocab.json"])
        self.assertEqual(vocab["byte_tokens"], {bytes([value]).hex(): OFFSET + value for value in range(256)})
        self.assertEqual(model.encode("Aé"), [OFFSET + value for value in "Aé".encode()])

    def test_normalization_preserves_case_and_whitespace(self) -> None:
        self.assertEqual(BPETokenizer.normalize("  Café\r\nCAFE\u0301\r\tX\n\u00a0"), "  Café\nCAFÉ\n\tX\n\u00a0")
        self.assertEqual(BPETokenizer.normalize("Ａ① ﬁ"), "Ａ① ﬁ")

    def test_roundtrip_multilingual_numbers_units_and_ranges(self) -> None:
        samples = [
            "English हिन्दी বাংলা தமிழ் العربية 中文 日本語 русский",
            "🌾👩🏽‍🌾 🌧️ 👨‍👩‍👧‍👦 🚜", "Cafe\u0301 a\u0323\u0301 क\u093f",
            "Apply 25–30 kg/ha; 12.5% at 20-25 °C, pH 6.0–7.5. ०१२३ ١٢٣",
            "Leading\t\tand  repeated\n\n whitespace \r\n END ",
            "\x00\x01\x7f\u200d\u2028\U0010ffff", "", "_snake_case_",
        ]
        model = BPETokenizer(400).fit(samples * 3)
        for sample in samples:
            with self.subTest(sample=sample):
                encoded = model.encode(sample)
                self.assertEqual(model.decode(encoded), model.normalize(sample))
                self.assertTrue(all(OFFSET <= token_id < model.vocab_size for token_id in encoded))
                self.assertEqual(model.decode(model.encode(sample, True)), model.normalize(sample))

    def test_random_valid_unicode_roundtrip(self) -> None:
        rng = random.Random(23)
        alphabet = [chr(value) for value in range(256)] + list("कृषि中文é🌾\u0301\u200d\U0010ffff")
        texts = ["".join(rng.choices(alphabet, k=120)) for _ in range(20)]
        model = BPETokenizer(350).fit(texts[:10])
        for text in texts:
            self.assertEqual(model.decode(model.encode(text)), model.normalize(text))

    def test_deterministic_ties_choose_lowest_id_pair(self) -> None:
        model = BPETokenizer(BASE_SIZE + 1).fit(["cd", "ab", "cd", "ab"])
        self.assertEqual(merge_rows(model), [(byte_id("a"), byte_id("b"))])
        self.assertEqual(model.encode("ab cd"), [BASE_SIZE, byte_id(" "), byte_id("c"), byte_id("d")])

    def test_frequency_is_weighted_by_repeated_segments(self) -> None:
        model = BPETokenizer(BASE_SIZE + 1).fit(["ab cd cd", "ab cd"])
        self.assertEqual(merge_rows(model), [(byte_id("c"), byte_id("d"))])

    def test_frequency_counts_overlapping_pairs(self) -> None:
        model = BPETokenizer(BASE_SIZE + 1).fit(["aaa"])
        self.assertEqual(merge_rows(model), [(byte_id("a"), byte_id("a"))])
        self.assertEqual(model.encode("aaa"), [BASE_SIZE, byte_id("a")])

    def test_minimum_pair_frequency_two(self) -> None:
        model = BPETokenizer().fit(["abc"])
        self.assertEqual(model.vocab_size, BASE_SIZE)
        self.assertEqual(merge_rows(model), [])

    def test_ranked_merges_and_compression(self) -> None:
        model = BPETokenizer(BASE_SIZE + 3).fit(["abab", "abab"])
        self.assertEqual(merge_rows(model), [(byte_id("a"), byte_id("b")), (BASE_SIZE, BASE_SIZE)])
        self.assertEqual(model.vocab_size, BASE_SIZE + 2)
        self.assertEqual(model.encode("abab"), [BASE_SIZE + 1])
        self.assertEqual(model.encode("ababa"), [BASE_SIZE + 1, byte_id("a")])

    def test_encoding_uses_rank_not_leftmost_pair(self) -> None:
        model = BPETokenizer(BASE_SIZE + 2).fit(["bc"] * 4 + ["ab"] * 2)
        self.assertEqual(merge_rows(model), [(byte_id("b"), byte_id("c")), (byte_id("a"), byte_id("b"))])
        self.assertEqual(model.encode("abc"), [byte_id("a"), BASE_SIZE])

    def test_does_not_merge_across_documents(self) -> None:
        model = BPETokenizer().fit(["a", "b"] * 20)
        self.assertEqual(model.vocab_size, BASE_SIZE)

    def test_does_not_merge_across_pretoken_boundaries(self) -> None:
        model = BPETokenizer().fit(["a1-b_ "] * 20)
        self.assertEqual(model.vocab_size, BASE_SIZE)
        self.assertEqual(model.encode("a1-b_ "), [OFFSET + value for value in b"a1-b_ "])

    def test_unicode_normalization_happens_before_learning(self) -> None:
        one = BPETokenizer().fit(["Cafe\u0301\r\n"] * 3)
        two = BPETokenizer().fit(["Café\n"] * 3)
        self.assertEqual(one.artifacts(), two.artifacts())

    def test_training_order_and_iterable_type_do_not_change_artifacts(self) -> None:
        documents = ["banana bandana", "cab cab", "कृषि 🌾", "cab cab", "banana bandana"]
        one = BPETokenizer(350).fit(documents)
        two = BPETokenizer(350).fit(reversed(documents))
        three = BPETokenizer(350).fit(text for text in documents)
        self.assertEqual(one.artifacts(), two.artifacts())
        self.assertEqual(one.artifacts(), three.artifacts())

    def test_literal_controls_cannot_inject_reserved_ids(self) -> None:
        text = "".join(SPECIAL_TOKENS)
        model = BPETokenizer(450).fit([text] * 5)
        ids = model.encode(text)
        self.assertTrue(all(value >= OFFSET for value in ids))
        self.assertEqual(model.decode(ids), text)
        self.assertEqual(model.decode(ids, False), text)
        with_controls = model.encode(text, True)
        self.assertEqual(with_controls, [2, *ids, 3])
        self.assertEqual(model.decode(with_controls, False), "<bos>" + text + "<eos>")

    def test_explicit_controls_and_generator_decoding(self) -> None:
        model = BPETokenizer().fit([])
        self.assertEqual(model.decode(iter(range(OFFSET))), "")
        self.assertEqual(model.decode(range(OFFSET), False), "".join(SPECIAL_TOKENS))
        with self.assertRaises(KeyError):
            model.token_to_id("ordinary")
        with self.assertRaises(TypeError):
            model.token_to_id(None)  # type: ignore[arg-type]

    def test_empty_training_and_text(self) -> None:
        for texts in ([], [""], iter([])):
            model = BPETokenizer().fit(texts)
            self.assertEqual(model.vocab_size, BASE_SIZE)
            self.assertEqual(model.encode(""), [])
            self.assertEqual(model.encode("", True), [2, 3])
            self.assertEqual(model.decode([]), "")
            self.assertEqual(model.decode(model.encode("Held-out 🌾")), "Held-out 🌾")

    def test_held_out_encoding_does_not_mutate_training(self) -> None:
        model = BPETokenizer(350).fit(["train train train"])
        before = model.artifacts()
        size = model.vocab_size
        for text in ["validation 未见 🌾", "test 123 kg/ha", "<answer>"]:
            self.assertEqual(model.decode(model.encode(text)), text)
        self.assertEqual(model.artifacts(), before)
        self.assertEqual(model.vocab_size, size)

    def test_refit_replaces_state_and_returns_self(self) -> None:
        model = BPETokenizer(350)
        self.assertIs(model.fit(["abc"] * 3), model)
        self.assertIs(model.fit([]), model)
        self.assertEqual(model.vocab_size, BASE_SIZE)
        self.assertEqual(model.encode("abc"), [byte_id(value) for value in "abc"])

    def test_invalid_fit_leaves_existing_state_unchanged(self) -> None:
        model = BPETokenizer().fit(["abc"] * 3)
        before = model.artifacts()
        with self.assertRaises(TypeError):
            model.fit(["changed", None])  # type: ignore[list-item]
        self.assertEqual(model.artifacts(), before)

    def test_constructor_validation(self) -> None:
        for value in (True, False, 300.0, "300", None):
            with self.subTest(value=value), self.assertRaises(TypeError):
                BPETokenizer(value)  # type: ignore[arg-type]
        for value in (-1, 0, BASE_SIZE - 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                BPETokenizer(value)

    def test_unfitted_operations_raise(self) -> None:
        model = BPETokenizer()
        operations = [
            lambda: model.vocab_size, lambda: model.encode(""), lambda: model.decode([]),
            lambda: model.token_to_id("<bos>"), lambda: model.artifacts(),
            lambda: model.save(Path.cwd() / "must-not-create-bpe-artifacts"),
        ]
        for operation in operations:
            with self.assertRaises(RuntimeError):
                operation()

    def test_invalid_training_inputs(self) -> None:
        for value in (None, 4, "abc", b"abc", ["abc", 5], [None]):
            with self.subTest(value=value), self.assertRaises(TypeError):
                BPETokenizer().fit(value)  # type: ignore[arg-type]

    def test_invalid_text_and_flags(self) -> None:
        model = BPETokenizer().fit([])
        for value in (None, b"a", 1, [], True):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    model.encode(value)  # type: ignore[arg-type]
                with self.assertRaises(TypeError):
                    model.normalize(value)  # type: ignore[arg-type]
        for value in (0, 1, None, "yes"):
            with self.assertRaises(TypeError):
                model.encode("", value)  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                model.decode([], value)  # type: ignore[arg-type]
        for text in ("\ud800", "\udfff"):
            with self.assertRaises(ValueError):
                model.encode(text)
            with self.assertRaises(ValueError):
                model.fit([text])

    def test_invalid_token_ids(self) -> None:
        model = BPETokenizer().fit([])
        for ids in ([True], [False], [1.0], [None], ["12"], "12", b"12", bytearray(b"12"), None, 2):
            with self.subTest(ids=ids), self.assertRaises(TypeError):
                model.decode(ids)  # type: ignore[arg-type]
        for token_id in (-1, model.vocab_size, 10**30):
            with self.assertRaises(ValueError):
                model.decode([token_id])
        with self.assertRaises(UnicodeDecodeError):
            model.decode([OFFSET + 255])

    def test_pair_counts_match_naive_reference_training(self) -> None:
        rng = random.Random(812)
        documents = ["".join(rng.choices("abcde", k=15)) for _ in range(25)]
        documents += documents[:10]
        sequences = [tuple(OFFSET + value for value in text.encode()) for text in documents]
        expected = []
        for next_id in range(BASE_SIZE, BASE_SIZE + 30):
            counts = Counter(pair for sequence in sequences for pair in zip(sequence, sequence[1:]))
            if not counts:
                break
            pair = min(counts, key=lambda item: (-counts[item], item))
            if counts[pair] < 2:
                break
            expected.append(pair)
            updated = []
            for sequence in sequences:
                output = []
                index = 0
                while index < len(sequence):
                    if sequence[index:index + 2] == pair:
                        output.append(next_id)
                        index += 2
                    else:
                        output.append(sequence[index])
                        index += 1
                updated.append(tuple(output))
            sequences = updated
        model = BPETokenizer(BASE_SIZE + 30).fit(documents)
        self.assertEqual(merge_rows(model), expected)
        self.assertEqual([model.encode(text) for text in documents], [list(sequence) for sequence in sequences])


class TextTokenizerArtifactsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path.cwd() / (".bpe-test-artifacts-" + uuid.uuid4().hex)
        self.directory.mkdir()
        self.addCleanup(shutil.rmtree, self.directory)
        self.model = BPETokenizer(350).fit(["banana banana 🌾", "कृषि 25–30 kg/ha"] * 4)
        self.model.save(self.directory)

    def write_json(self, name: str, value: object) -> None:
        (self.directory / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def read_json(self, name: str) -> dict:
        return json.loads((self.directory / name).read_text(encoding="utf-8"))

    def assert_corrupt(self) -> None:
        with self.assertRaises(ValueError):
            BPETokenizer.load(self.directory)

    def test_save_load_roundtrip_and_artifact_bytes(self) -> None:
        artifacts = self.model.artifacts()
        self.assertEqual(set(artifacts), {"config.json", "vocab.json", "merges.txt"})
        self.assertTrue(all(isinstance(value, bytes) for value in artifacts.values()))
        for name, value in artifacts.items():
            self.assertEqual((self.directory / name).read_bytes(), value)
        restored = BPETokenizer.load(self.directory)
        self.assertEqual(restored.artifacts(), artifacts)
        self.assertEqual(restored.vocab_size, self.model.vocab_size)
        for text in ("banana 🌾", "Held-out café 3.5 kg", "", "<context>", "a\r\nb"):
            self.assertEqual(restored.encode(text), self.model.encode(text))
            self.assertEqual(restored.decode(restored.encode(text)), self.model.normalize(text))
        restored.save(self.directory / "nested" / "model")
        self.assertEqual(BPETokenizer.load(self.directory / "nested" / "model").artifacts(), artifacts)

    def test_empty_model_save_load(self) -> None:
        empty = BPETokenizer().fit([])
        empty.save(self.directory)
        self.assertEqual(BPETokenizer.load(self.directory).artifacts(), empty.artifacts())

    def test_missing_artifact(self) -> None:
        for name in self.model.artifacts():
            self.model.save(self.directory)
            (self.directory / name).unlink()
            with self.assertRaises(FileNotFoundError):
                BPETokenizer.load(self.directory)

    def test_duplicate_json_keys_including_nested_tables(self) -> None:
        for name, text in (
            ("config.json", '{"format_version":1,"format_version":1}'),
            ("vocab.json", '{"special_tokens":{"<pad>":0,"<pad>":0},"byte_tokens":{}}'),
            ("vocab.json", '{"special_tokens":{},"byte_tokens":{},"byte_tokens":{}}'),
        ):
            with self.subTest(name=name, text=text):
                self.model.save(self.directory)
                (self.directory / name).write_text(text, encoding="utf-8")
                self.assert_corrupt()

    def test_invalid_json_and_encoding(self) -> None:
        for name in ("config.json", "vocab.json"):
            for content in (b"{", b"[]", b"null", b"NaN", b"Infinity", b"\xff"):
                with self.subTest(name=name, content=content):
                    self.model.save(self.directory)
                    (self.directory / name).write_bytes(content)
                    self.assert_corrupt()

    def test_config_corruption(self) -> None:
        original = self.read_json("config.json")
        changes = {
            "format_version": [True, 1.0, 2], "algorithm": ["external"],
            "requested_vocab_size": [False, 1, BASE_SIZE, 350.0],
            "vocab_size": [True, 300.0, self.model.vocab_size + 1],
            "special_tokens": [list(reversed(SPECIAL_TOKENS))],
            "normalization": ["NFKC"], "pretoken_pattern": [".*"],
            "min_pair_frequency": [True, 1], "extra": ["field"],
        }
        for key, values in changes.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    config = dict(original)
                    config[key] = value
                    self.write_json("config.json", config)
                    self.assert_corrupt()
        for key in original:
            config = dict(original)
            del config[key]
            self.write_json("config.json", config)
            self.assert_corrupt()

    def test_vocab_corruption(self) -> None:
        for mutate in (
            lambda value: value["special_tokens"].__setitem__("<pad>", False),
            lambda value: value["special_tokens"].__setitem__("<bos>", 3),
            lambda value: value["byte_tokens"].__setitem__("00", OFFSET + 1),
            lambda value: value["byte_tokens"].__setitem__("00", float(OFFSET)),
            lambda value: value["byte_tokens"].__delitem__("ff"),
            lambda value: value["byte_tokens"].__setitem__("not-hex", 999),
            lambda value: value.__setitem__("byte_tokens", []),
            lambda value: value.__setitem__("extra", {}),
        ):
            self.model.save(self.directory)
            vocab = self.read_json("vocab.json")
            mutate(vocab)
            self.write_json("vocab.json", vocab)
            self.assert_corrupt()

    def test_merge_corruption(self) -> None:
        header = "# custom-byte-bpe v1\n"
        first = f"{byte_id('a')} {byte_id('b')}\n"
        invalid = [
            "", "wrong header\n", header + "\n", header + "1 2\n",
            header + f"{BASE_SIZE} 12\n", header + "12 -1\n", header + "12 13 14\n",
            header + "012 13\n", header + "12.0 13\n", header + "True 13\n",
            header + first + first, header, header + "12 13\n",
        ]
        for text in invalid:
            with self.subTest(text=text):
                (self.directory / "merges.txt").write_text(text, encoding="utf-8")
                self.assert_corrupt()

    def test_duplicate_bytes_via_distinct_merge_derivations(self) -> None:
        a, b, c = (byte_id(value) for value in "abc")
        text = f"# custom-byte-bpe v1\n{a} {b}\n{b} {c}\n{BASE_SIZE} {c}\n{a} {BASE_SIZE + 1}\n"
        (self.directory / "merges.txt").write_text(text, encoding="utf-8")
        self.assert_corrupt()

    def test_rank_order_disagrees_with_vocab(self) -> None:
        lines = self.model.artifacts()["merges.txt"].decode().splitlines()
        lines[1], lines[2] = lines[2], lines[1]
        (self.directory / "merges.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.assert_corrupt()


if __name__ == "__main__":
    unittest.main()
