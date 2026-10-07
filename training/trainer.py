"""Real causal-LM optimization, immutable snapshots, and provenance validation.

Only the already-fitted, verified train-only tokenizer release is loaded. No
validation/test examples enter an optimizer batch. Validation loss is a
supervised-token-weighted mean, not an agricultural answer-quality score.
An existing completed run is verified/reused; incomplete runs are not resumed
or silently restarted, since that would reset the approved optimization budget.
"""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time
from typing import Any, Generator
from uuid import uuid4

from agri.config import load_config
from agri.model_check import _prepare


CODE_ROOT = Path(__file__).resolve().parent.parent
CAVEATS = [
    "Synthetic-only, randomly initialized tiny baseline; no pretrained weights or external data.",
    "Loss/perplexity measure language-model mechanics, not agricultural correctness or generalization.",
    "Generated text is untrusted and must not be used as factual farming recommendations.",
]


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate checkpoint JSON key: {key}")
        result[key] = value
    return result


def _read(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique)
    if not isinstance(result, dict):
        raise ValueError("Checkpoint metadata must be an object")
    return result


def _run_path(root: Path) -> Path:
    seed = load_config(root / "configs" / "agri-mini.json")["seed"]
    return root / "checkpoints" / f"agri-transformer-v0-seed{seed}"


def training_report_path(root: Path) -> Path:
    """Return the stable published-report locator for optional status discovery.

    This JSON pointer names the immutable step manifest containing ``report``.
    Test ``training_report_path(root).exists()`` before ``validate_training``;
    neither operation imports Torch or deserializes tensor weights.
    """
    return _run_path(root.resolve()) / "latest.json"


def checkpoint_path(root: Path) -> Path:
    """Return the latest published snapshot path, never an unpublished partial file."""
    directory = _run_path(root.resolve())
    pointer = _read(directory / "latest.json")
    step = pointer.get("directory")
    if not isinstance(step, str) or not re.fullmatch(r"step-[0-9]{6}", step):
        raise ValueError("Invalid checkpoint directory pointer")
    return directory / step / "weights.pt"


def _guard(config: dict[str, Any]) -> None:
    settings = config["training"]
    if settings["allow_large_runs"]:
        raise ValueError("Large runs are not approved; allow_large_runs must remain false")
    if (settings["max_steps"] > 20 or settings["epochs"] > 1 or settings["batch_size"] > 2
            or settings["learning_rate"] > 0.001):
        raise ValueError("Training exceeds approved CPU limits: 20 steps, 1 epoch, batch 2, learning rate <= 0.001")


def _prepare_training(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _guard(load_config(root / "configs" / "agri-mini.json"))
    config, base, packed = _prepare(root)
    provenance = deepcopy(base["provenance"])
    provenance["config"] = config
    provenance["training_code_sha256"] = _hash(CODE_ROOT / "training" / "trainer.py")
    provenance["tokenizer_code_sha256"] = {
        name: _hash(CODE_ROOT / "tokenizer" / name)
        for name in ("tokenizer.py", "structured.py", "train_tokenizer.py")
    }
    provenance["objective"] = "input-aligned causal next-token cross entropy; structured/padding targets ignored"
    provenance["optimizer_split"] = "train"
    provenance["tokenizer_fitting_split"] = "train (verified existing artifacts; never refitted here)"
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if {row["group_id"] for row in packed[left]} & {row["group_id"] for row in packed[right]}:
            raise ValueError("Cross-split group leakage")
    return config, provenance, packed


def _plan(config: dict[str, Any], provenance: dict[str, Any], packed: dict[str, Any]) -> dict[str, Any]:
    arch, settings = provenance["architecture"], config["training"]
    h, f, layers = arch["hidden_size"], arch["feed_forward_size"], arch["layers"]
    parameters = h * (2 * arch["vocab_size"] + arch["structured_vocab_size"] + arch["context_length"] + 104)
    parameters += layers * (4 * h * h + 2 * h * f + 9 * h + f) + 2 * h
    steps = min(settings["max_steps"], settings["epochs"] * math.ceil(len(packed["train"]) / settings["batch_size"]))
    length, batch = arch["context_length"], settings["batch_size"]
    attention = 4 * batch * arch["attention_heads"] * length * length
    activation_budget = layers * (8 * attention + 4 * batch * length * (12 * h + 4 * f))
    estimates = {
        "parameters": parameters, "parameter_bytes_float32": parameters * 4,
        "optimizer_and_parameter_bytes_estimate": parameters * 16,
        "activation_bytes_estimate": activation_budget,
        "process_ram_bytes_estimate": parameters * 16 + activation_budget + 512 * 1024 * 1024,
        "estimate_caveat": "Conservative planning estimate, not measured RSS or a hard memory guarantee.",
        "one_attention_matrix_bytes": attention, "cpu_threads": 1, "gpu_bytes": 0,
        "cpu_time": "Hardware-dependent; bounded full-context CPU steps may take minutes. No timing guarantee.",
        "optimizer_steps": steps, "epochs_limit": settings["epochs"], "batch_size": batch,
        "checkpoint_every_steps": settings["checkpoint_every_steps"],
        "checkpoint_count": math.ceil(steps / settings["checkpoint_every_steps"]),
        "checkpoint_storage_bytes_estimate": math.ceil(steps / settings["checkpoint_every_steps"]) * (parameters * 4 + 262144),
        "train_examples": len(packed["train"]), "validation_examples": len(packed["validation"]),
        "test_examples_used_for_optimization_or_evaluation": 0,
        "max_packed_positions": {split: max(len(row["input_ids"]) for row in rows) for split, rows in packed.items()},
    }
    return {"phase": 6, "status": "GREEN", "plan_only": True, "artifacts_written": False,
            "model": "AgriTransformer-v0; custom architecture; no pretrained weights",
            "training": "Plan only; no model allocated or optimizer steps executed",
            "source_status": "Synthetic fixture data only; no external data used",
            "next_required_step": "Run the tiny dry-run preflight, then explicitly request the bounded training run",
            "blockers": ["No agricultural answer-quality evidence", "Real-source generalization is not evaluated"],
            "resource_usage": {"scope": "Planning only; resource estimates are separate",
                               "device": None, "cpu_threads": None, "gpu_bytes": 0,
                               "peak_ram_bytes": None, "wall_seconds": None, "storage_bytes_written": 0},
            "resource_estimates": estimates, "provenance": provenance,
            "metrics": None, "caveats": list(CAVEATS), "resume_supported": False}


def training_plan(root: Path) -> dict[str, Any]:
    """Validate inputs and estimate resources without allocating a Torch model/tensor."""
    return _plan(*_prepare_training(root.resolve()))


@contextmanager
def cpu_session(seed: int) -> Generator[None, None, None]:
    """Use one CPU thread and restore caller RNG/thread settings even on failure."""
    import torch

    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.random.default_generator.manual_seed(seed)
            yield
    finally:
        torch.set_num_threads(threads)


def _forward(model: Any, batch: dict[str, Any]) -> Any:
    return model(**{key: value for key, value in batch.items() if key != "labels"})


def _evaluate(model: Any, rows: list[dict[str, Any]], batch_size: int) -> dict[str, Any]:
    import torch
    from models.transformer.model import causal_lm_loss
    from models.transformer.packing import collate_examples

    was_training = model.training
    total, targets = 0.0, 0
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                batch = collate_examples(rows[start:start + batch_size])
                loss = causal_lm_loss(_forward(model, batch), batch["labels"], batch["attention_mask"])
                count = int(((batch["labels"][:, 1:] != -100) & batch["attention_mask"][:, 1:]
                             & batch["attention_mask"][:, :-1]).sum().item())
                total += float(loss.item()) * count
                targets += count
    finally:
        model.train(was_training)
    if not targets or not math.isfinite(total):
        raise ValueError("Evaluation requires finite loss and supervised tokens")
    loss = total / targets
    perplexity = math.exp(loss)
    if not math.isfinite(perplexity):
        raise ValueError("Nonfinite evaluation perplexity")
    return {"loss": loss, "perplexity": perplexity, "supervised_tokens": targets, "examples": len(rows)}


def _publish(root: Path, model: Any, report: dict[str, Any]) -> Path:
    import torch

    directory = _run_path(root)
    directory.mkdir(parents=True, exist_ok=True)
    name = f"step-{report['metrics']['optimizer_steps']:06d}"
    destination = directory / name
    if destination.exists():
        raise ValueError("Checkpoint already exists; refusing to overwrite")
    staging = directory / f".publishing-{uuid4().hex}"
    pointer = directory / f".latest-{uuid4().hex}.json"
    staging.mkdir()
    try:
        with (staging / "weights.pt").open("wb") as stream:
            torch.save({"format_version": 1, "model": model.state_dict()}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        report = deepcopy(report)
        weights_size = (staging / "weights.pt").stat().st_size
        report["resource_usage"]["checkpoint_weights_bytes"] = weights_size
        report["resource_usage"]["run_checkpoint_weights_bytes"] = weights_size + sum(
            path.stat().st_size for path in directory.glob("step-*/weights.pt")
        )
        content = {"format_version": 1, "report": report, "weights_sha256": _hash(staging / "weights.pt"),
                   "weights_size_bytes": weights_size}
        with (staging / "manifest.json").open("wb") as stream:
            stream.write(_bytes(content))
            stream.flush()
            os.fsync(stream.fileno())
        manifest_hash = _hash(staging / "manifest.json")
        os.replace(staging, destination)
        with pointer.open("wb") as stream:
            stream.write(_bytes({"directory": name, "manifest_sha256": manifest_hash}))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pointer, directory / "latest.json")
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        pointer.unlink(missing_ok=True)
    return destination / "weights.pt"


def _validated(root: Path, plan: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    path = checkpoint_path(root)
    pointer = _read(_run_path(root) / "latest.json")
    if set(pointer) != {"directory", "manifest_sha256"} or pointer["manifest_sha256"] != _hash(path.parent / "manifest.json"):
        raise ValueError("Checkpoint manifest checksum mismatch")
    manifest = _read(path.parent / "manifest.json")
    if set(manifest) != {"format_version", "report", "weights_sha256", "weights_size_bytes"} or type(manifest["format_version"]) is not int or manifest["format_version"] != 1:
        raise ValueError("Invalid checkpoint manifest format")
    if manifest["weights_size_bytes"] != path.stat().st_size or manifest["weights_sha256"] != _hash(path):
        raise ValueError("Checkpoint weights checksum/size mismatch")
    report = manifest["report"]
    if not isinstance(report, dict) or _bytes(report.get("provenance")) != _bytes(plan["provenance"]):
        raise ValueError("Checkpoint provenance/configuration mismatch; refusing stale weights")
    if _bytes(report.get("resource_estimates")) != _bytes(plan["resource_estimates"]):
        raise ValueError("Checkpoint resource estimates mismatch")
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Missing training metrics")
    steps = metrics.get("optimizer_steps")
    if type(steps) is not int or not 1 <= steps <= plan["resource_estimates"]["optimizer_steps"] or path.parent.name != f"step-{steps:06d}":
        raise ValueError("Invalid checkpoint optimizer step count")
    if type(report.get("complete")) is not bool or report["complete"] != (steps == plan["resource_estimates"]["optimizer_steps"]):
        raise ValueError("Invalid checkpoint completion state")
    if report.get("status") != ("GREEN" if report["complete"] else "YELLOW"):
        raise ValueError("Checkpoint status disagrees with completion state")
    losses = metrics.get("train_step_losses")
    if not isinstance(losses, list) or len(losses) != steps or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in losses):
        raise ValueError("Invalid recorded training losses")
    for key in ("train_probe_before", "train_probe_after", "validation_before", "validation_after"):
        value = metrics.get(key)
        if not isinstance(value, dict) or any(type(value.get(field)) not in (float, int) or not math.isfinite(value[field]) or value[field] < 0 for field in ("loss", "perplexity")):
            raise ValueError("Invalid finite evaluation metrics")
        if not math.isclose(value["perplexity"], math.exp(value["loss"]), rel_tol=1e-12):
            raise ValueError("Perplexity differs from exp(loss)")
    return report, path


def validate_training(root: Path) -> dict[str, Any]:
    """Read-only verification of dataset/tokenizers/config/code and snapshot hashes.

    No Torch imports, tensor deserialization, fitting, model creation, or writes.
    The returned usage metrics describe the saved run, not this validation call.
    """
    root = root.resolve()
    report, _ = _validated(root, training_plan(root))
    return report


def load_checkpoint(root: Path) -> tuple[Any, dict[str, Any]]:
    """Load only verified local tensor weights; caller should use cpu_session."""
    import torch
    from models.transformer.model import AgriTransformer, TransformerConfig

    root = root.resolve()
    report, path = _validated(root, training_plan(root))
    if not report["complete"]:
        raise ValueError("Training checkpoint is incomplete; finish the approved run before inference")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"format_version", "model"} or payload["format_version"] != 1:
        raise ValueError("Invalid tensor checkpoint format")
    model = AgriTransformer(TransformerConfig(**report["provenance"]["architecture"]))
    try:
        model.load_state_dict(payload["model"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("Checkpoint tensor shapes/actual vocabulary mismatch") from exc
    if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise ValueError("Checkpoint contains nonfinite model parameters")
    model.eval()
    return model, report


def train_model(root: Path, dry_run: bool = False) -> dict[str, Any]:
    """Run a tiny smoke test or the explicitly requested, strictly bounded budget.

    ``dry_run`` is authoritative; the config flag is a CLI default, not a hidden
    override of an explicit actual-run call. No option authorizes larger runs.
    """
    if type(dry_run) is not bool:
        raise ValueError("dry_run must be a boolean")
    root = root.resolve()
    config, provenance, packed = _prepare_training(root)
    plan = _plan(config, provenance, packed)
    print(json.dumps({"training_resource_estimates": plan["resource_estimates"], "dry_run": dry_run}), file=sys.stderr, flush=True)
    if not dry_run and (_run_path(root) / "latest.json").exists():
        existing, _ = _validated(root, plan)
        if not existing["complete"]:
            raise ValueError("Incomplete checkpoint exists; resume is unsupported and the budget cannot be silently restarted")
        return existing
    import torch
    from models.transformer.model import AgriTransformer, TransformerConfig, causal_lm_loss
    from models.transformer.packing import collate_examples

    started = time.perf_counter()
    with cpu_session(config["seed"]):
        if dry_run:
            model = AgriTransformer(TransformerConfig(vocab_size=provenance["architecture"]["vocab_size"],
                structured_vocab_size=provenance["architecture"]["structured_vocab_size"], context_length=16,
                layers=1, hidden_size=16, attention_heads=2, feed_forward_size=32, dropout=0))
            ids = torch.full((1, 16), 2, dtype=torch.long)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
            before = causal_lm_loss(model(ids), ids)
            before.backward()
            optimizer.step()
            with torch.no_grad():
                after = causal_lm_loss(model(ids), ids)
            metrics = {"smoke_loss_before": float(before.item()), "smoke_loss_after": float(after.item()),
                       "optimizer_steps": 1, "scope": "constructed 16-position IDs only; not dataset evaluation"}
            if not all(math.isfinite(metrics[key]) for key in ("smoke_loss_before", "smoke_loss_after")):
                raise ValueError("Nonfinite smoke loss")
            elapsed = time.perf_counter() - started
            return {**plan, "plan_only": False, "dry_run": True, "metrics": metrics,
                    "training": "One in-memory optimizer smoke step on constructed IDs; no dataset training or checkpoint",
                    "next_required_step": "Explicitly request the configured bounded training run",
                    "resource_usage": {"scope": "Tiny smoke execution; peak RAM was not measured",
                                       "device": "cpu", "cpu_threads": 1, "gpu_bytes": 0,
                                       "peak_ram_bytes": None, "wall_seconds": elapsed, "storage_bytes_written": 0},
                    "tiny_parameters": sum(p.numel() for p in model.parameters()), "elapsed_seconds": elapsed}
        model = AgriTransformer(TransformerConfig(**provenance["architecture"]))
        settings = config["training"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"])
        probe = packed["train"][:settings["batch_size"]]
        before = _evaluate(model, probe, settings["batch_size"])
        validation_before = _evaluate(model, packed["validation"], settings["batch_size"])
        initial_embedding = model.text_embeddings.weight.detach().clone()
        losses: list[float] = []
        trained_ids: list[str] = []
        order = list(range(len(packed["train"])))
        random.Random(config["seed"]).shuffle(order)
        limit = plan["resource_estimates"]["optimizer_steps"]
        for start in range(0, len(order), settings["batch_size"]):
            selected = [packed["train"][index] for index in order[start:start + settings["batch_size"]]]
            batch = collate_examples(selected)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = causal_lm_loss(_forward(model, batch), batch["labels"], batch["attention_mask"])
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise ValueError("Nonfinite training gradients")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise ValueError("Nonfinite optimized weights")
            losses.append(float(loss.item()))
            trained_ids.extend(row["example_id"] for row in selected)
            step = len(losses)
            if step % settings["checkpoint_every_steps"] == 0 or step == limit:
                after = _evaluate(model, probe, settings["batch_size"])
                validation = _evaluate(model, packed["validation"], settings["batch_size"])
                metrics = {"optimizer_steps": step, "train_step_losses": list(losses),
                           "train_probe_before": before, "train_probe_after": after,
                           "train_probe_loss_decreased": after["loss"] < before["loss"],
                           "validation_before": validation_before, "validation_after": validation,
                           "validation_loss": validation["loss"], "validation_perplexity": validation["perplexity"],
                           "training_example_ids": list(trained_ids), "optimization_split": "train",
                           "test_examples_evaluated": 0,
                           "embedding_update_l2": float((model.text_embeddings.weight.detach() - initial_embedding).norm().item())}
                elapsed = time.perf_counter() - started
                report = {**plan, "plan_only": False, "dry_run": False, "artifacts_written": True,
                          "status": "GREEN" if step == limit else "YELLOW",
                          "training": f"{step}/{limit} approved optimizer steps; train split only; validation used only for evaluation",
                          "next_required_step": ("Phase 7: verify bounded checkpoint inference" if step == limit else
                                                 "Finish the active bounded run; interrupted-run resume is unsupported"),
                          "blockers": [*plan["blockers"], *([] if step == limit else ["Approved training budget incomplete"])],
                          "resource_usage": {"scope": "Training/evaluation through this step, before snapshot publication; storage counts tensor files only",
                                             "device": "cpu", "cpu_threads": 1, "gpu_bytes": 0,
                                             "peak_ram_bytes": None, "wall_seconds": elapsed},
                          "complete": step == limit, "metrics": metrics,
                          "elapsed_seconds": elapsed, "torch_version": str(torch.__version__),
                          "checkpoint": str(_run_path(root) / f"step-{step:06d}" / "weights.pt")}
                _publish(root, model, report)
            if step == limit:
                break
    return validate_training(root)
