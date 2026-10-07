"""Offline synthetic evidence retrieval; no external observations or recommendations."""

from retrieval.index import build_index, search, validate_index

__all__ = ["build_index", "search", "validate_index"]
