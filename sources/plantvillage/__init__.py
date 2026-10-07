"""Phase 10: PlantVillage metadata adapter. Use sources.ingestion for gated I/O."""

from sources._plants import normalize as _normalize
from sources._plants import schema


def _schema() -> dict:
    return schema("plantvillage")
