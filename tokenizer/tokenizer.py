"""Dependency-free, deterministic byte-level BPE with explicit control tokens."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
import heapq
import json
from pathlib import Path
import re
import unicodedata

SPECIAL_TOKENS = (
    "<pad>", "<unk>", "<bos>", "<eos>", "<sep>", "<mask>",
    "<question>", "<context>", "<answer>", "<unknown>", "<source>",
)
_BYTE_OFFSET = len(SPECIAL_TOKENS)
_BASE_SIZE = _BYTE_OFFSET + 256
_PRETOKEN_PATTERN = r"[^\W\d_]+|\d+|[^\w\s]+|_+|\s+"
_PRETOKEN = re.compile(_PRETOKEN_PATTERN)
_MERGE_HEADER = "# custom-byte-bpe v1"
Pair = tuple[int, int]


def _pairs(sequence: tuple[int, ...]) -> Counter[Pair]:
    return Counter(zip(sequence, sequence[1:]))


def _merge(sequence: tuple[int, ...], pair: Pair, token_id: int) -> tuple[int, ...]:
    result: list[int] = []
    index = 0
    while index < len(sequence):
        if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
            result.append(token_id)
            index += 2
        else:
            result.append(sequence[index])
            index += 1
    return tuple(result)


def _json_object(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> object:
    raise ValueError(f"Invalid JSON constant: {value}")


def _read_json(path: Path) -> object:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_json_object,
        parse_constant=_invalid_constant,
    )


class BPETokenizer:
    """Learn within regex segments; break frequency ties by ascending token-ID pair.

    IDs 0..10 are controls, followed by all 256 bytes in byte order. Text never
    interprets control spellings. Empty training data produces the byte alphabet.
    Decoding an arbitrary ID sequence that is not valid UTF-8 raises ValueError.
    """

    def __init__(self, vocab_size: int = 2048) -> None:
        if type(vocab_size) is not int:
            raise TypeError("vocab_size must be an integer")
        if vocab_size < _BASE_SIZE:
            raise ValueError(f"vocab_size must be at least {_BASE_SIZE}")
        self._requested_vocab_size = vocab_size
        self._tokens: list[bytes] | None = None
        self._merges: list[Pair] = []
        self._ranks: dict[Pair, tuple[int, int]] = {}

    @staticmethod
    def normalize(text: str) -> str:
        """Apply NFC and convert CRLF/CR to LF, preserving other whitespace/case."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        normalized = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
        normalized.encode("utf-8")  # Reject lone surrogates, not valid Unicode text.
        return normalized

    def _require_fitted(self) -> list[bytes]:
        if self._tokens is None:
            raise RuntimeError("Tokenizer must be fitted or loaded first")
        return self._tokens

    @property
    def vocab_size(self) -> int:
        """Actual vocabulary size, including control IDs (requires fitting)."""
        return _BYTE_OFFSET + len(self._require_fitted())

    def token_to_id(self, token: str) -> int:
        """Look up an explicit control token; unknown spellings raise KeyError."""
        self._require_fitted()
        if not isinstance(token, str):
            raise TypeError("token must be a string")
        try:
            return SPECIAL_TOKENS.index(token)
        except ValueError:
            raise KeyError(token) from None

    def fit(self, texts: Iterable[str]) -> BPETokenizer:
        """Replace learned state using only texts supplied here, never encode input."""
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be an iterable of strings, not a string")
        segments: Counter[tuple[int, ...]] = Counter()
        for text in texts:
            for match in _PRETOKEN.finditer(self.normalize(text)):
                segments[tuple(byte + _BYTE_OFFSET for byte in match.group().encode("utf-8"))] += 1
        sequences = list(segments)
        weights = list(segments.values())
        counts: Counter[Pair] = Counter()
        locations: dict[Pair, set[int]] = defaultdict(set)
        for index, sequence in enumerate(sequences):
            for pair, count in _pairs(sequence).items():
                counts[pair] += count * weights[index]
                locations[pair].add(index)
        heap = [(-count, pair) for pair, count in counts.items() if count >= 2]
        heapq.heapify(heap)
        tokens = [bytes([byte]) for byte in range(256)]
        known_tokens = set(tokens)
        merges: list[Pair] = []
        while heap and len(tokens) + _BYTE_OFFSET < self._requested_vocab_size:
            negative_count, pair = heapq.heappop(heap)
            if counts[pair] != -negative_count:
                continue
            token = tokens[pair[0] - _BYTE_OFFSET] + tokens[pair[1] - _BYTE_OFFSET]
            if token in known_tokens:
                continue
            new_id = _BYTE_OFFSET + len(tokens)
            tokens.append(token)
            known_tokens.add(token)
            merges.append(pair)
            changed: set[Pair] = set()
            # Update only affected unique segments; corpus repetitions are weights.
            for index in sorted(locations[pair]):
                before = _pairs(sequences[index])
                sequences[index] = _merge(sequences[index], pair, new_id)
                after = _pairs(sequences[index])
                for affected in before.keys() | after.keys():
                    difference = after[affected] - before[affected]
                    if difference:
                        counts[affected] += difference * weights[index]
                        changed.add(affected)
                    if affected not in after:
                        locations[affected].discard(index)
                    elif affected not in before:
                        locations[affected].add(index)
            for affected in sorted(changed):
                if counts[affected] >= 2:
                    heapq.heappush(heap, (-counts[affected], affected))
        self._tokens = tokens
        self._merges = merges
        self._ranks = {pair: (rank, _BASE_SIZE + rank) for rank, pair in enumerate(merges)}
        return self

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode normalized UTF-8, optionally enclosing it in BOS/EOS controls."""
        self._require_fitted()
        if type(add_special_tokens) is not bool:
            raise TypeError("add_special_tokens must be a boolean")
        result: list[int] = []
        for match in _PRETOKEN.finditer(self.normalize(text)):
            sequence = tuple(byte + _BYTE_OFFSET for byte in match.group().encode("utf-8"))
            while len(sequence) > 1:
                candidates = (self._ranks[pair] for pair in zip(sequence, sequence[1:]) if pair in self._ranks)
                best = min(candidates, default=None)
                if best is None:
                    break
                rank, token_id = best
                sequence = _merge(sequence, self._merges[rank], token_id)
            result.extend(sequence)
        if add_special_tokens:
            return [SPECIAL_TOKENS.index("<bos>"), *result, SPECIAL_TOKENS.index("<eos>")]
        return result

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        """Decode valid UTF-8; control tokens are skipped unless explicitly requested."""
        tokens = self._require_fitted()
        if type(skip_special_tokens) is not bool:
            raise TypeError("skip_special_tokens must be a boolean")
        if isinstance(ids, (str, bytes, bytearray)):
            raise TypeError("ids must be an iterable of integer token IDs")
        output = bytearray()
        for token_id in ids:
            if type(token_id) is not int:
                raise TypeError("token IDs must be integers, not booleans")
            if not 0 <= token_id < _BYTE_OFFSET + len(tokens):
                raise ValueError(f"Token ID out of range: {token_id}")
            if token_id < _BYTE_OFFSET:
                if not skip_special_tokens:
                    output.extend(SPECIAL_TOKENS[token_id].encode("utf-8"))
            else:
                output.extend(tokens[token_id - _BYTE_OFFSET])
        return output.decode("utf-8")

    def _config(self) -> dict[str, object]:
        return {
            "format_version": 1,
            "algorithm": "custom-byte-bpe",
            "requested_vocab_size": self._requested_vocab_size,
            "vocab_size": self.vocab_size,
            "special_tokens": list(SPECIAL_TOKENS),
            "normalization": "NFC+LF",
            "pretoken_pattern": _PRETOKEN_PATTERN,
            "min_pair_frequency": 2,
        }

    def artifacts(self) -> dict[str, bytes]:
        """Return deterministic UTF-8 artifacts for the caller's atomic publication."""
        tokens = self._require_fitted()
        vocab = {
            "special_tokens": {token: index for index, token in enumerate(SPECIAL_TOKENS)},
            "byte_tokens": {token.hex(): index + _BYTE_OFFSET for index, token in enumerate(tokens)},
        }
        def serialize(value: object) -> bytes:
            return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
        merges = _MERGE_HEADER + "\n" + "".join(f"{left} {right}\n" for left, right in self._merges)
        return {"vocab.json": serialize(vocab), "merges.txt": merges.encode("utf-8"), "config.json": serialize(self._config())}

    def save(self, directory: Path | str) -> None:
        """Write the three artifacts; use artifacts() for atomic directory publication."""
        artifacts = self.artifacts()
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        for name, content in artifacts.items():
            (path / name).write_bytes(content)

    @classmethod
    def load(cls, directory: Path | str) -> BPETokenizer:
        """Reconstruct and cross-check strictly validated JSON and ranked merges."""
        path = Path(directory)
        config = _read_json(path / "config.json")
        if not isinstance(config, dict):
            raise ValueError("config.json must contain an object")
        for key in ("format_version", "requested_vocab_size", "vocab_size", "min_pair_frequency"):
            if type(config.get(key)) is not int:
                raise ValueError(f"Invalid config integer: {key}")
        requested = config["requested_vocab_size"]
        if not isinstance(requested, int) or requested < _BASE_SIZE:
            raise ValueError("Invalid requested vocabulary size")
        model = cls(requested)
        lines = (path / "merges.txt").read_text(encoding="utf-8").splitlines()
        if not lines or lines[0] != _MERGE_HEADER:
            raise ValueError("Invalid merges header")
        tokens = [bytes([byte]) for byte in range(256)]
        known_tokens = set(tokens)
        merges: list[Pair] = []
        seen_pairs: set[Pair] = set()
        for line in lines[1:]:
            if re.fullmatch(r"(0|[1-9][0-9]*) (0|[1-9][0-9]*)", line) is None:
                raise ValueError("Invalid merge row")
            left, right = map(int, line.split())
            next_id = _BYTE_OFFSET + len(tokens)
            if not (_BYTE_OFFSET <= left < next_id and _BYTE_OFFSET <= right < next_id):
                raise ValueError("Merge references a control or unavailable token")
            pair = (left, right)
            token = tokens[left - _BYTE_OFFSET] + tokens[right - _BYTE_OFFSET]
            if pair in seen_pairs or token in known_tokens:
                raise ValueError("Duplicate merge or byte token")
            merges.append(pair)
            seen_pairs.add(pair)
            tokens.append(token)
            known_tokens.add(token)
        model._tokens = tokens
        model._merges = merges
        model._ranks = {pair: (rank, _BASE_SIZE + rank) for rank, pair in enumerate(merges)}
        if model.vocab_size > requested or config != model._config():
            raise ValueError("Config and merges disagree")
        vocab = _read_json(path / "vocab.json")
        if not isinstance(vocab, dict) or set(vocab) != {"special_tokens", "byte_tokens"}:
            raise ValueError("Invalid vocabulary structure")
        for table in vocab.values():
            if not isinstance(table, dict) or any(type(value) is not int for value in table.values()):
                raise ValueError("Invalid vocabulary IDs")
        expected = {
            "special_tokens": {token: index for index, token in enumerate(SPECIAL_TOKENS)},
            "byte_tokens": {token.hex(): index + _BYTE_OFFSET for index, token in enumerate(tokens)},
        }
        if vocab != expected:
            raise ValueError("Vocabulary and merges disagree")
        return model
