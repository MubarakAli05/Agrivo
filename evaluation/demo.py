"""Required demonstration questions and an evidence-aware response contract."""

from typing import Any


DEMO_QUESTIONS = (
    "What is soil pH?",
    "What soil properties can SoilGrids provide?",
    "What is the soil pH at this coordinate?",
    "What is the uncertainty of this soil estimate?",
    "What soil information is available for this region?",
    "What diseases are present in PlantVillage?",
    "Which crops are covered by PlantVillage?",
    "What does PlantDoc contain?",
    "Which apple disease datasets are available?",
    "What agriculture datasets are available from the Indian government?",
    "Which source should I use for US soil survey information?",
    "What is the source of this answer?",
    "Is this value measured or estimated?",
    "What information is missing?",
    "Can you answer this without making assumptions?",
    "I uploaded a plant image. What disease does it resemble?",
    "What evidence supports this plant prediction?",
    "Could this be an unknown disease?",
    "Compare two soil locations.",
    "Which crop is more suitable based on retrieved soil properties?",
)


def validate_response(response: dict[str, Any]) -> None:
    required = {"answer", "data_used", "sources", "model_version", "confidence", "unknown"}
    missing = required - response.keys()
    if missing:
        raise ValueError(f"Incomplete answer contract: {sorted(missing)}")
    if not isinstance(response["answer"], str) or not response["answer"].strip():
        raise ValueError("Every response requires a nonempty answer")
    if not isinstance(response["unknown"], bool):
        raise ValueError("UNKNOWN must be an explicit boolean")
    if response["confidence"] is not None:
        raise ValueError("This synthetic proof has no calibrated confidence; use null")
    if not response["unknown"] and not response["sources"]:
        raise ValueError("Known answers require source attribution")
    if not response["unknown"] and not response["data_used"]:
        raise ValueError("Known answers require disclosed evidence")
