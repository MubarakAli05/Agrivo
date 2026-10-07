"""Scratch CPU CNN for generated geometric patterns, never agricultural evidence."""

from contextlib import contextmanager
import hashlib
import struct
import threading
from typing import Any, Generator

import torch
from torch import Tensor, nn
from torch.nn import functional as F


CLASSES = (
    "SYNTHETIC_PATTERN_HORIZONTAL",
    "SYNTHETIC_PATTERN_VERTICAL",
    "SYNTHETIC_PATTERN_DIAGONAL",
)
IMAGE_SHAPE = (3, 16, 16)
PARAMETER_COUNT = 507
_LOCK = threading.RLock()


def _seed(seed: int) -> None:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")


def _cpu(device: str | torch.device) -> None:
    if str(device) != "cpu":
        raise ValueError("Vision smoke supports device=cpu only")


@contextmanager
def isolated_cpu(seed: int) -> Generator[None, None, None]:
    """Serialize our smoke runs and restore caller CPU RNG and thread settings."""
    _seed(seed)
    with _LOCK:
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                yield
        finally:
            torch.set_num_threads(threads)


def validate_images(images: Tensor) -> None:
    if not isinstance(images, Tensor):
        raise ValueError("images must be a torch tensor")
    if images.device.type != "cpu" or images.layout != torch.strided or images.dtype != torch.float32:
        raise ValueError("images must be dense CPU float32 tensors")
    if images.ndim != 4 or tuple(images.shape[1:]) != IMAGE_SHAPE or not 1 <= images.shape[0] <= 8:
        raise ValueError("images must have shape [N, 3, 16, 16] with 1 <= N <= 8")
    if not bool(torch.isfinite(images).all()) or bool(((images < 0) | (images > 1)).any()):
        raise ValueError("images must be finite and normalized to [0, 1]")


class SmallVisionCNN(nn.Module):
    """507 randomly initialized parameters; logits refer only to SYNTHETIC_PATTERN labels."""

    def __init__(self, device: str | torch.device = "cpu") -> None:
        _cpu(device)
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1, device="cpu", dtype=torch.float32)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1, device="cpu", dtype=torch.float32)
        self.classifier = nn.Linear(32, len(CLASSES), device="cpu", dtype=torch.float32)

    def forward(self, images: Tensor) -> Tensor:
        validate_images(images)
        for parameter in self.parameters():
            if parameter.device.type != "cpu" or parameter.dtype != torch.float32:
                raise ValueError("Model parameters must be CPU float32")
            if not bool(torch.isfinite(parameter).all()):
                raise ValueError("Model parameters must be finite")
        hidden = F.max_pool2d(F.relu(self.conv1(images)), 2)
        hidden = F.adaptive_avg_pool2d(F.relu(self.conv2(hidden)), (2, 2))
        logits = self.classifier(hidden.flatten(1))
        if logits.dtype != torch.float32 or not bool(torch.isfinite(logits).all()):
            raise ValueError("Model logits must be finite float32; autocast is unsupported")
        return logits


def classification_loss(logits: Tensor, targets: Tensor) -> Tensor:
    if not isinstance(logits, Tensor) or not isinstance(targets, Tensor):
        raise ValueError("logits and targets must be tensors")
    if (logits.device.type != "cpu" or logits.layout != torch.strided or logits.dtype != torch.float32
            or logits.ndim != 2 or logits.shape[1] != len(CLASSES) or not 1 <= logits.shape[0] <= 8):
        raise ValueError("logits must have CPU float32 shape [N, 3], 1 <= N <= 8")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite")
    if (targets.device.type != "cpu" or targets.layout != torch.strided or targets.dtype != torch.int64
            or targets.shape != (logits.shape[0],)):
        raise ValueError("targets must have CPU int64 shape [N]")
    if bool(((targets < 0) | (targets >= len(CLASSES))).any()):
        raise ValueError("targets must index SYNTHETIC_PATTERN classes [0, 3)")
    return F.cross_entropy(logits, targets)


def synthetic_patterns(seed: int = 42) -> tuple[Tensor, Tensor]:
    """Six generated stripe/diagonal arrays with seeded noise, not plant images."""
    _seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    images = torch.zeros((6, *IMAGE_SHAPE), dtype=torch.float32, device="cpu")
    targets = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int64, device="cpu")
    for index, label in enumerate(targets.tolist()):
        offset = index // 3
        if label == 0:
            images[index, :, 5 + offset:9 + offset, :] = 0.9
        elif label == 1:
            images[index, :, :, 5 + offset:9 + offset] = 0.9
        else:
            for row in range(16):
                images[index, :, row, (row + offset) % 16] = 0.9
    images += torch.rand(images.shape, generator=generator, device="cpu", dtype=torch.float32) * 0.05
    validate_images(images)
    return images, targets


def tensor_sha256(tensor: Tensor) -> str:
    """Canonical little-endian tensor bytes, without a NumPy bridge."""
    if tensor.device.type != "cpu" or tensor.dtype not in (torch.float32, torch.int64):
        raise ValueError("Tensor hashing supports CPU float32/int64 only")
    values = tensor.detach().contiguous().flatten().tolist()
    kind = "f" if tensor.dtype == torch.float32 else "q"
    header = f"{tensor.dtype}:{tuple(tensor.shape)}:".encode("ascii")
    return hashlib.sha256(header + struct.pack(f"<{len(values)}{kind}", *values)).hexdigest()


def state_sha256(state: dict[str, Tensor]) -> str:
    content = "\n".join(f"{key}:{tensor_sha256(value)}" for key, value in sorted(state.items()))
    return hashlib.sha256(content.encode("ascii")).hexdigest()


def run_synthetic_smoke(seed: int = 42, *, steps: int = 2, device: str = "cpu") -> tuple[dict[str, Any], dict[str, Tensor]]:
    """At most two SGD steps on one generated batch; loss is not a quality metric.

    Workflow callers must obtain vision_plan before invoking this bounded primitive.
    The returned checkpoint is only a synthetic mechanics artifact.
    """
    _seed(seed)
    _cpu(device)
    if type(steps) is not int or not 1 <= steps <= 2:
        raise ValueError("Synthetic optimizer budget is 1 or 2 steps")
    with isolated_cpu(seed):
        images, targets = synthetic_patterns(seed)
        model = SmallVisionCNN(device)
        initial = state_sha256(dict(model.state_dict()))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        losses = []
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = classification_loss(model(images), targets)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Synthetic loss is nonfinite")
            losses.append(float(loss.detach()))
            loss.backward()
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise ValueError("Synthetic gradients are missing or nonfinite")
            optimizer.step()
        with torch.no_grad():
            final_loss = classification_loss(model(images), targets)
            if not bool(torch.isfinite(final_loss)):
                raise ValueError("Synthetic final loss is nonfinite")
            losses.append(float(final_loss))
        state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        final = state_sha256(state)
        if initial == final:
            raise ValueError("Synthetic optimizer did not change weights")
        return {
            "status": "GREEN", "optimizer_steps": steps, "threads": 1, "device": "cpu",
            "examples": 6, "parameter_count": sum(p.numel() for p in model.parameters()),
            "training_batch_losses": losses, "finite_gradients": True, "weights_changed": True,
            "initial_state_sha256": initial, "final_state_sha256": final,
            "images_sha256": tensor_sha256(images), "targets_sha256": tensor_sha256(targets),
        }, state
