"""Offline, approval-gated external snapshot adapters (no network clients)."""

from sources.ingestion import adapter_schema, ingest_snapshot, inspect_adapter

__all__ = ["adapter_schema", "ingest_snapshot", "inspect_adapter"]
