"""Train-only quantile tokenization of structured soil measurements."""

from bisect import bisect_left
from collections.abc import Iterable, Mapping, Sequence
import json
import math
from numbers import Real
from pathlib import Path


DEFAULT_FEATURES = ("ph", "nitrogen", "phosphorus", "potassium")


class SoilTokenizer:
    """Encode each feature as a missing token or a feature-specific numeric bin."""

    pad_token_id = 0

    def __init__(
        self, features: Sequence[str] = DEFAULT_FEATURES, n_bins: int = 16
    ) -> None:
        if isinstance(features, (str, bytes)):
            raise ValueError("features must be a sequence of feature names")
        self.features = tuple(features)
        if (
            not self.features
            or any(not isinstance(name, str) or not name.strip() for name in self.features)
            or len(set(self.features)) != len(self.features)
        ):
            raise ValueError("features must contain unique, nonempty names")
        if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
            raise ValueError("n_bins must be a positive integer")
        self.n_bins = n_bins
        self._edges: dict[str, tuple[float, ...]] | None = None

    @property
    def vocab_size(self) -> int:
        """Embedding table size, including padding and unused numeric bins."""
        return 1 + len(self.features) * (self.n_bins + 1)

    @staticmethod
    def _number(value: object, feature: str) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (Real, str)):
            raise ValueError(f"{feature}: expected a number or missing value")
        if isinstance(value, str) and not value.strip():
            return None
        try:
            number = float(value)
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"{feature}: invalid numeric value {value!r}") from exc
        if math.isnan(number):
            return None
        if not math.isfinite(number):
            raise ValueError(f"{feature}: infinity is not supported")
        return number

    def fit(self, rows: Iterable[Mapping[str, object]]) -> "SoilTokenizer":
        """Learn empirical quantile boundaries from training rows only."""
        values: dict[str, list[float]] = {name: [] for name in self.features}
        for row in rows:
            for feature in self.features:
                number = self._number(row.get(feature), feature)
                if number is not None:
                    values[feature].append(number)

        edges: dict[str, tuple[float, ...]] = {}
        for feature, samples in values.items():
            if not samples:
                raise ValueError(f"{feature}: training requires at least one numeric value")
            samples.sort()
            # Nearest-rank quantiles keep ties together; constant columns need no cuts.
            cuts = {
                samples[(i * len(samples) + self.n_bins - 1) // self.n_bins - 1]
                for i in range(1, self.n_bins)
            }
            edges[feature] = tuple(sorted(cut for cut in cuts if cut < samples[-1]))
        self._edges = edges
        return self

    def encode(self, row: Mapping[str, object]) -> list[int]:
        """Return one token per configured feature, in the configured order."""
        if self._edges is None:
            raise RuntimeError("Call fit() or load() before encoding")
        tokens = []
        for index, feature in enumerate(self.features):
            base = 1 + index * (self.n_bins + 1)
            value = self._number(row.get(feature), feature)
            token = base if value is None else base + 1 + bisect_left(self._edges[feature], value)
            tokens.append(token)
        return tokens

    def transform(self, rows: Iterable[Mapping[str, object]]) -> list[list[int]]:
        """Encode a batch without modifying learned boundaries."""
        if self._edges is None:
            raise RuntimeError("Call fit() or load() before encoding")
        return [self.encode(row) for row in rows]

    def fit_transform(self, rows: Iterable[Mapping[str, object]]) -> list[list[int]]:
        batch = list(rows)
        return self.fit(batch).transform(batch)

    def save(self, path: str | Path) -> None:
        """Save feature order and learned boundaries as versioned JSON."""
        if self._edges is None:
            raise RuntimeError("Cannot save an unfitted tokenizer")
        payload = {
            "version": 1,
            "features": self.features,
            "n_bins": self.n_bins,
            "edges": self._edges,
        }
        Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SoilTokenizer":
        """Load a tokenizer, rejecting malformed or unsupported model data."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or type(payload.get("version")) is not int or payload["version"] != 1:
            raise ValueError("Unsupported tokenizer format version")
        try:
            tokenizer = cls(payload["features"], payload["n_bins"])
            raw_edges = payload["edges"]
            if not isinstance(raw_edges, dict) or set(raw_edges) != set(tokenizer.features):
                raise ValueError("Invalid feature boundaries")
            edges = {}
            for feature in tokenizer.features:
                cuts = raw_edges[feature]
                if not isinstance(cuts, list) or len(cuts) >= tokenizer.n_bins:
                    raise ValueError(f"{feature}: invalid boundary count")
                if any(type(cut) not in (int, float) or not math.isfinite(cut) for cut in cuts):
                    raise ValueError(f"{feature}: boundaries must be finite numbers")
                if any(left >= right for left, right in zip(cuts, cuts[1:])):
                    raise ValueError(f"{feature}: boundaries must be strictly increasing")
                edges[feature] = tuple(float(cut) for cut in cuts)
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError("Invalid tokenizer data") from exc
        tokenizer._edges = edges
        return tokenizer
