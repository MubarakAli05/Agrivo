"""A causal text/structured transformer built from primitive PyTorch operations.

Numeric inputs are seven already-preprocessed channels, never raw measurements.
Feature IDs 0..11 identify soil features, 12 identifies depth, and -1 means none.
Labels supplied to ``causal_lm_loss`` are input-aligned; callers must mark
structured and padding targets with -100. No tokenizer or preprocessing is hidden
inside the model.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    structured_vocab_size: int
    context_length: int = 1536
    layers: int = 4
    hidden_size: int = 256
    attention_heads: int = 4
    feed_forward_size: int = 1024
    dropout: float = 0.1

    def __post_init__(self) -> None:
        for name in (
            "vocab_size", "structured_vocab_size", "context_length", "layers",
            "hidden_size", "attention_heads", "feed_forward_size",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if (isinstance(self.dropout, bool)
                or not isinstance(self.dropout, (int, float))
                or not math.isfinite(self.dropout)
                or not 0 <= self.dropout <= 1):
            raise ValueError("dropout must be finite and between zero and one")


def _tensor(name: str, value: Tensor, shape: tuple[int, ...],
            device: torch.device, dtype: torch.dtype | None = None) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if value.device != device:
        raise ValueError(f"{name} must be on device {device}")
    if value.layout != torch.strided:
        raise ValueError(f"{name} must be a strided tensor")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")


def _range(name: str, value: Tensor, minimum: int, maximum: int) -> None:
    if bool(((value < minimum) | (value > maximum)).any()):
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")


class _LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, hidden: Tensor) -> Tensor:
        mean = hidden.mean(dim=-1, keepdim=True)
        variance = (hidden - mean).square().mean(dim=-1, keepdim=True)
        return (hidden - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias


class _CausalAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.heads = config.attention_heads
        self.head_size = config.hidden_size // config.attention_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: Tensor, mask: Tensor) -> Tensor:
        batch, length, width = hidden.shape
        qkv = self.qkv(hidden).reshape(batch, length, 3, self.heads, self.head_size)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_size)
        causal = torch.ones(length, length, dtype=torch.bool, device=hidden.device).tril()
        allowed = causal[None, None, :, :] & mask[:, None, None, :]
        scores = scores.masked_fill(~allowed, float("-inf"))
        # Empty prefixes (including all-padding rows) must not softmax all -inf.
        scores = scores.masked_fill(~allowed.any(dim=-1, keepdim=True), 0.0)
        weights = torch.softmax(scores, dim=-1).masked_fill(~allowed, 0.0)
        attended = self.attention_dropout(weights) @ value
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        return self.output_dropout(self.output(attended))


class _Block(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attention_norm = _LayerNorm(config.hidden_size)
        self.attention = _CausalAttention(config)
        self.feed_forward_norm = _LayerNorm(config.hidden_size)
        self.feed_forward_in = nn.Linear(config.hidden_size, config.feed_forward_size)
        self.feed_forward_out = nn.Linear(config.feed_forward_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: Tensor, mask: Tensor) -> Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden), mask)
        feed_forward = self.feed_forward_in(self.feed_forward_norm(hidden))
        feed_forward = self.feed_forward_out(F.gelu(feed_forward))
        hidden = hidden + self.dropout(feed_forward)
        return hidden.masked_fill(~mask.unsqueeze(-1), 0.0)


class AgriTransformer(nn.Module):
    """Return text-vocabulary logits for input-aligned text/structured positions.

    Text and structured IDs use independent tables with zero reserved for absence.
    An explicit boolean mask may hide tokens, but each unmasked position must have
    exactly one nonzero ID. Numeric features require unmasked structured tokens;
    numeric values and feature IDs must be supplied together. Floating numeric
    channels are cast to the model dtype after finite-value validation. Positions
    count active tokens, making left/right/interior padding semantically inert.
    Masked logits are exactly zero. The context limit applies to padded length T.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        if not isinstance(config, TransformerConfig):
            raise TypeError("config must be a TransformerConfig")
        self.config = config
        self.text_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.structured_embeddings = nn.Embedding(
            config.structured_vocab_size, config.hidden_size, padding_idx=0,
        )
        self.position_embeddings = nn.Embedding(config.context_length, config.hidden_size)
        self.numeric_weight = nn.Parameter(torch.empty(13, 7, config.hidden_size))
        self.numeric_bias = nn.Parameter(torch.zeros(13, config.hidden_size))
        nn.init.normal_(self.numeric_weight, mean=0.0, std=0.02)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([_Block(config) for _ in range(config.layers)])
        self.final_norm = _LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None,
                structured_ids: Tensor | None = None, numeric_values: Tensor | None = None,
                numeric_features: Tensor | None = None) -> Tensor:
        if not isinstance(input_ids, Tensor):
            raise TypeError("input_ids must be a tensor")
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have nonempty shape [B, T]")
        batch, length = input_ids.shape
        if length > self.config.context_length:
            raise ValueError("sequence exceeds context_length")
        shape = (batch, length)
        device = self.text_embeddings.weight.device
        _tensor("input_ids", input_ids, shape, device, torch.long)
        _range("input_ids", input_ids, 0, self.config.vocab_size - 1)
        if structured_ids is None:
            structured_ids = torch.zeros_like(input_ids)
        _tensor("structured_ids", structured_ids, shape, device, torch.long)
        _range("structured_ids", structured_ids, 0, self.config.structured_vocab_size - 1)
        text_active, structured_active = input_ids != 0, structured_ids != 0
        if attention_mask is None:
            attention_mask = text_active | structured_active
        _tensor("attention_mask", attention_mask, shape, device, torch.bool)
        if bool((attention_mask & ~(text_active ^ structured_active)).any()):
            raise ValueError("each active position must have either text or structured ID, not both")
        if (numeric_values is None) != (numeric_features is None):
            raise ValueError("numeric_values and numeric_features must be supplied together")
        if numeric_values is not None and numeric_features is not None:
            _tensor("numeric_values", numeric_values, (batch, length, 7), device)
            if not numeric_values.is_floating_point():
                raise TypeError("numeric_values must have floating dtype")
            if not bool(torch.isfinite(numeric_values).all()):
                raise ValueError("numeric_values must be finite")
            _tensor("numeric_features", numeric_features, shape, device, torch.long)
            _range("numeric_features", numeric_features, -1, 12)
            if bool(((numeric_features >= 0) & ~(structured_active & attention_mask)).any()):
                raise ValueError("numeric_features are valid only at active structured positions")

        positions = (attention_mask.long().cumsum(dim=1) - 1).clamp_min(0)
        hidden = self.text_embeddings(input_ids) + self.structured_embeddings(structured_ids)
        hidden = hidden + self.position_embeddings(positions)
        if numeric_values is not None and numeric_features is not None:
            feature_ids = numeric_features.clamp_min(0)
            channels = numeric_values.to(dtype=hidden.dtype)
            channels = channels.masked_fill((numeric_features < 0).unsqueeze(-1), 0.0)
            numeric = (channels.unsqueeze(-1) * self.numeric_weight[feature_ids]).sum(dim=-2)
            numeric = numeric + self.numeric_bias[feature_ids]
            hidden = hidden + numeric.masked_fill((numeric_features < 0).unsqueeze(-1), 0.0)
        hidden = self.embedding_dropout(hidden).masked_fill(~attention_mask.unsqueeze(-1), 0.0)
        for block in self.blocks:
            hidden = block(hidden, attention_mask)
        logits = self.lm_head(self.final_norm(hidden))
        return logits.masked_fill(~attention_mask.unsqueeze(-1), 0.0)


def causal_lm_loss(logits: Tensor, labels: Tensor,
                   attention_mask: Tensor | None = None) -> Tensor:
    """Mean next-token cross entropy; shift input-aligned labels internally.

    -100 targets are ignored. With a mask, both predictor and target positions
    must be active. Empty supervision returns a finite differentiable zero.
    Target validation includes ignored/masked positions (except the -100 sentinel).
    """
    if not isinstance(logits, Tensor):
        raise TypeError("logits must be a tensor")
    if logits.ndim != 3 or any(size == 0 for size in logits.shape):
        raise ValueError("logits must have nonempty shape [B, T, vocab_size]")
    _tensor("logits", logits, tuple(logits.shape), logits.device)
    if not logits.is_floating_point():
        raise TypeError("logits must have floating dtype")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite")
    batch, length, vocab_size = logits.shape
    _tensor("labels", labels, (batch, length), logits.device, torch.long)
    if bool(((labels != -100) & ((labels < 0) | (labels >= vocab_size))).any()):
        raise ValueError("labels must be -100 or a valid text vocabulary ID")
    if attention_mask is not None:
        _tensor("attention_mask", attention_mask, (batch, length), logits.device, torch.bool)
    targets = labels[:, 1:]
    valid = targets != -100
    if attention_mask is not None:
        valid = valid & attention_mask[:, :-1] & attention_mask[:, 1:]
    if not bool(valid.any()):
        return logits[..., :0].sum()
    return F.cross_entropy(logits[:, :-1][valid], targets[valid])
