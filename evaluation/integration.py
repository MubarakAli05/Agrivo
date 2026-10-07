"""Persistent synthetic end-to-end checks, not agricultural quality certification."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from evaluation.demo import DEMO_QUESTIONS, validate_response


KNOWN_QUESTION = "What is pH at 0-5 cm for synthetic-site-02?"


def integration_report_path(root: Path) -> Path:
    return root / "evaluation" / "reports" / "phase16.json"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprints(root: Path) -> dict[str, str]:
    code_root = Path(__file__).resolve().parent.parent
    files = {}
    for package in ("agri", "tokenizer", "models", "training", "retrieval", "sources", "app", "evaluation"):
        for path in sorted((code_root / package).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".html", ".css", ".js"}:
                files["code:" + path.relative_to(code_root).as_posix()] = _digest(path)
    for directory in ("data/releases", "tokenizer/releases", "checkpoints", "models/transformer/reports"):
        for path in sorted(root.joinpath(*directory.split("/")).rglob("*")):
            if path.is_file() and path.name != ".gitkeep":
                files["artifact:" + path.relative_to(root).as_posix()] = _digest(path)
    for relative in ("configs/agri-mini.json", "data/source_registry.yaml", "licenses/registry.json", "data/indexed/synthetic-v1.json"):
        path = root.joinpath(*relative.split("/"))
        if path.exists():
            files["artifact:" + relative] = _digest(path)
    return files


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _validate_cases(cases: Any, training_ready: bool) -> dict[str, Any]:
    if not isinstance(cases, list) or any(not isinstance(case, dict) for case in cases):
        raise ValueError("Malformed integration cases")
    if [case.get("question") for case in cases] != [*DEMO_QUESTIONS, KNOWN_QUESTION]:
        raise ValueError("Integration question set mismatch")
    for case in cases:
        response = case.get("response")
        if not isinstance(response, dict):
            raise ValueError("Malformed integration response")
        validate_response(response)
        if response.get("status") not in {"GREEN", "YELLOW"}:
            raise ValueError("Integration report contains a failed response")
        if not isinstance(response.get("model_used"), bool):
            raise ValueError("Missing model execution indicator")
        if not training_ready and (response["unknown"] is not True or response["model_used"] is not False):
            raise ValueError("Deferred training requires safe abstention without model execution")
    known = cases[-1]["response"]
    if training_ready:
        if (known["unknown"] or known["model_used"] is not True
                or known.get("answer_origin") != "deterministic_evidence_renderer"):
            raise ValueError("Integration report lacks checkpoint-backed evidence inference")
        draft = known.get("neural_draft")
        if not isinstance(draft, dict) or draft.get("used_as_factual_answer") is not False:
            raise ValueError("Unverified neural draft must not be presented as factual evidence")
    elif (known.get("reason") != "checkpoint_missing_or_unavailable" or not known["sources"]
          or known.get("answer_origin") != "abstention"):
        raise ValueError("Deferred integration must demonstrate retrieved evidence and missing-checkpoint abstention")
    return {"response_contracts_passed": len(cases), "original_demo_questions": len(DEMO_QUESTIONS),
            "unknown_responses": sum(case["response"]["unknown"] for case in cases),
            "model_executed_responses": sum(case["response"]["model_used"] for case in cases),
            "agricultural_accuracy": None}


def run_integration(root: Path) -> dict[str, Any]:
    from agri.model_check import validate_model
    from agri.source_registry import inspect_sources
    from models.vision_baseline.workflow import validate_vision
    from retrieval.index import validate_index
    from retrieval.qa import answer_query
    from training.trainer import training_report_path, validate_training

    root = root.resolve()
    start, cpu_start = time.perf_counter(), time.process_time()
    before = _fingerprints(root)
    training_path = training_report_path(root)
    training_ready = training_path.exists()
    if training_ready:
        training = validate_training(root)
        if training["status"] != "GREEN":
            raise ValueError("Integration prerequisite is not GREEN: training")
    else:
        if training_path.parent.exists() and any(
                path.name != ".gitkeep" for path in training_path.parent.iterdir()):
            raise ValueError("Incomplete training artifacts: report missing")
        training = {"phase": 6, "status": "YELLOW", "reason": "training_checkpoint_absent",
                    "training": "Not started; no trained transformer checkpoint"}
    dependencies = {
        "transformer": validate_model(root), "training": training,
        "retrieval": validate_index(root), "vision": validate_vision(root),
        "sources": inspect_sources(root),
    }
    for name in ("transformer", "retrieval"):
        if dependencies[name]["status"] != "GREEN":
            raise ValueError(f"Integration prerequisite is not GREEN: {name}")
    if dependencies["vision"].get("synthetic_model_mechanics") != "GREEN":
        raise ValueError("Run the synthetic vision mechanics check first")
    cases = []
    for question in (*DEMO_QUESTIONS, KNOWN_QUESTION):
        response = answer_query(root, question, max_new_tokens=8)
        validate_response(response)
        if response["status"] == "RED":
            raise ValueError(f"Demo execution failed: {question}: {response.get('reason')}")
        cases.append({"question": question, "response": response})
    metrics = _validate_cases(cases, training_ready)
    if before != _fingerprints(root):
        raise ValueError("Dependencies changed during integration; rerun after other work finishes")
    report = {
        "phase": 16, "status": "YELLOW",
        "integration_mechanics": "GREEN" if training_ready else "PARTIAL",
        "model_execution": "GREEN" if training_ready else "BLOCKED",
        "scope": "Synthetic offline integration only; not agricultural quality certification",
        "data_used": "Original synthetic soil/plant metadata/QA and geometric image tensors only",
        "model": "Scratch AgriTransformer-v0 plus 507-parameter synthetic-pattern CNN",
        "training": dependencies["training"],
        "metrics": metrics,
        "source_status": dependencies["sources"],
        "what_works": ["Checkpoint-backed inference" if training_ready else "Safe missing-checkpoint abstention",
                       "Exact local evidence and source attribution", "Explicit UNKNOWN",
                       "Original 20-question response contract", "Synthetic CNN mechanics"],
        "what_failed": [],
        "blockers": (["Transformer training deferred; checkpoint-backed integration blocked until RAM is available"]
                     if not training_ready else []) + ["External data approvals remain separate",
                     "No real-image disease model", "No calibrated confidence or field-quality evaluation"],
        "next_required_step": ("Complete bounded transformer training when RAM permits, then rerun integration"
                               if not training_ready else
                               "Review source-specific rights and intended use before any real-data integration"),
        "resource_usage": {"wall_seconds": time.perf_counter() - start,
                           "cpu_seconds": time.process_time() - cpu_start,
                           "ram": "Peak process RAM not measured", "gpu": "Not used",
                           "generation_budget": "21 responses, at most eight new tokens each; no optimizer steps"},
        "fingerprints": before, "cases": cases,
    }
    envelope = {"report": report, "sha256": hashlib.sha256(_canonical(report)).hexdigest()}
    target = integration_report_path(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(_canonical(envelope) + b"\n")
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return report


def validate_integration(root: Path) -> dict[str, Any]:
    root = root.resolve()
    envelope = json.loads(integration_report_path(root).read_text(encoding="utf-8"))
    if not isinstance(envelope, dict) or set(envelope) != {"report", "sha256"}:
        raise ValueError("Malformed integration report envelope")
    report = envelope["report"]
    if hashlib.sha256(_canonical(report)).hexdigest() != envelope["sha256"]:
        raise ValueError("Integration report checksum mismatch")
    if (not isinstance(report, dict) or report.get("phase") != 16 or report.get("status") != "YELLOW"
            or report.get("integration_mechanics") not in {"GREEN", "PARTIAL"}):
        raise ValueError("Invalid integration report")
    if report.get("fingerprints") != _fingerprints(root):
        raise ValueError("Integration report is stale; rerun integration")
    training_ready = report["integration_mechanics"] == "GREEN"
    training = report.get("training")
    if (not isinstance(training, dict) or training.get("status") != ("GREEN" if training_ready else "YELLOW")
            or report.get("model_execution") != ("GREEN" if training_ready else "BLOCKED")
            or (not training_ready and training.get("reason") != "training_checkpoint_absent")):
        raise ValueError("Inconsistent integration/training status")
    metrics = _validate_cases(report.get("cases"), training_ready)
    if report.get("metrics") != metrics:
        raise ValueError("Integration metrics do not match recorded responses")
    return report
