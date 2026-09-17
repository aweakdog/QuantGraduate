import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional


@dataclass(frozen=True)
class FalconConfig:
    variant: str = "prototype"
    width: int = 128
    heads: int = 4
    temporal_layers: int = 2
    spatial_layers: int = 2
    prototypes: int = 8
    patch_size: int = 16
    dropout: float = 0.0
    quantiles: tuple[float, ...] = (0.01, *tuple(i / 20 for i in range(1, 20)), 0.99)
    scale_floor: float = 1e-5
    max_context: int = 8192
    max_horizon: int = 480

    def __post_init__(self):
        object.__setattr__(self, "quantiles", tuple(self.quantiles))
        if self.variant not in {"temporal", "dense", "prototype"}:
            raise ValueError("unknown model variant")
        if min(self.width, self.heads, self.temporal_layers, self.spatial_layers, self.prototypes, self.patch_size) < 1:
            raise ValueError("model dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if not self.quantiles or tuple(sorted(set(self.quantiles))) != self.quantiles:
            raise ValueError("quantiles must be strictly increasing")
        if not all(0 < q < 1 for q in self.quantiles) or 0.5 not in self.quantiles:
            raise ValueError("quantiles must lie in (0,1) and include the median")
        if not 0 <= self.dropout < 1 or self.scale_floor <= 0:
            raise ValueError("invalid dropout or scale floor")


class AttentionStack(nn.Module):
    def __init__(self, config, depth):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(config.width, config.heads, dim_feedforward=4 * config.width,
                                       dropout=config.dropout, activation="gelu", batch_first=True,
                                       norm_first=False)
            for _ in range(depth)
        ])

    def forward(self, values, valid):
        safe = valid.clone()
        safe[~safe.any(dim=-1), 0] = True
        values = values.masked_fill(~valid.unsqueeze(-1), 0)
        for layer in self.layers:
            values = layer(values, src_key_padding_mask=~safe)
            values = values.masked_fill(~valid.unsqueeze(-1), 0)
        return values


class DenseMixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = AttentionStack(config, config.spatial_layers)

    def forward(self, temporal, active):
        batch, entities, variates, patches, width = temporal.shape
        values = temporal.permute(0, 3, 1, 2, 4).reshape(batch * patches, entities * variates, width)
        mask = active[:, None].expand(batch, patches, entities, variates).reshape(batch * patches, -1)
        mixed = self.attention(values, mask)
        return mixed.reshape(batch, patches, entities, variates, width).permute(0, 2, 3, 1, 4)


class PrototypeMixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.count = config.prototypes
        self.width = config.width
        self.positive_keys = nn.Parameter(torch.randn(config.prototypes, config.width) / math.sqrt(config.width))
        self.negative_keys = nn.Parameter(torch.randn(config.prototypes, config.width) / math.sqrt(config.width))
        self.raw_lambda = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
        self.query = nn.Linear(config.width, config.width)
        self.value = nn.Linear(config.width, config.width)
        self.attention = AttentionStack(config, config.spatial_layers)
        self.request = nn.Linear(config.width, config.width)
        self.index = nn.Linear(config.width, config.width)
        self.context = nn.Linear(config.width, config.width)

    def forward(self, temporal, active):
        batch, entities, _, patches, width = temporal.shape
        query, value = self.query(temporal), self.value(temporal)
        positive = torch.einsum("bevpd,cd->bevpc", query, self.positive_keys) / math.sqrt(width)
        negative = torch.einsum("bevpd,cd->bevpc", query, self.negative_keys) / math.sqrt(width)
        weights = positive.float().softmax(dim=-1) - functional.softplus(self.raw_lambda) * negative.float().softmax(dim=-1)
        weights = (weights * active[..., None, None]).to(value.dtype)
        prototypes = torch.einsum("bevpc,bevpd->becpd", weights, value)
        values = prototypes.permute(0, 3, 1, 2, 4).reshape(batch * patches, entities * self.count, width)
        valid = active.any(dim=-1)[:, None, :, None].expand(batch, patches, entities, self.count)
        values = self.attention(values, valid.reshape(batch * patches, entities * self.count))
        prototypes = values.reshape(batch, patches, entities, self.count, width).permute(0, 2, 3, 1, 4)
        request, index, context = self.request(temporal), self.index(prototypes), self.context(prototypes)
        routing = torch.einsum("bevpd,becpd->bevpc", request, index) / math.sqrt(width)
        routing = routing.float().softmax(dim=-1).to(context.dtype)
        return torch.einsum("bevpc,becpd->bevpd", routing, context)

    def orthogonality_loss(self):
        positive = functional.normalize(self.positive_keys.float(), dim=-1)
        negative = functional.normalize(self.negative_keys.float(), dim=-1)
        return (positive @ negative.T).square().mean()


class FalconForecaster(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or FalconConfig()
        config = self.config
        inputs = 3 * config.patch_size
        self.embedding_skip = nn.Linear(inputs, config.width)
        self.embedding = nn.Sequential(nn.Linear(inputs, config.width), nn.GELU(), nn.Linear(config.width, config.width))
        self.temporal = AttentionStack(config, config.temporal_layers)
        self.head = nn.Linear(config.width, config.patch_size * len(config.quantiles))
        self.register_buffer("quantiles", torch.tensor(config.quantiles, dtype=torch.float32))
        self.gate = nn.Linear(config.width, config.width) if config.variant != "temporal" else None
        self.mixer = (PrototypeMixer(config) if config.variant == "prototype" else
                      DenseMixer(config) if config.variant == "dense" else None)

    def normalize(self, context, observed=None):
        context = context.float()
        observed = torch.isfinite(context) if observed is None else observed.bool() & torch.isfinite(context)
        count = observed.sum(dim=-1, keepdim=True)
        safe = torch.where(observed, context, 0)
        location = safe.sum(dim=-1, keepdim=True) / count.clamp_min(1)
        centered = torch.where(observed, context - location, 0)
        variance = centered.square().sum(dim=-1, keepdim=True) / count.clamp_min(1)
        scale = variance.sqrt().clamp_min(self.config.scale_floor)
        scale = torch.where(count > 0, scale, torch.ones_like(scale))
        normalized = torch.where(observed, torch.asinh(centered / scale), 0)
        return normalized, observed, location, scale

    def forward(self, context, horizon, observed=None):
        if context.ndim != 4 or min(context.shape) < 1:
            raise ValueError("context must have shape [independent_dates, entities, variates, history]")
        if observed is not None and observed.shape != context.shape:
            raise ValueError("observation mask shape differs from context")
        if not isinstance(horizon, int) or not 1 <= horizon <= self.config.max_horizon:
            raise ValueError("invalid forecast horizon")
        if context.shape[-1] > self.config.max_context:
            raise ValueError("context exceeds configured limit")
        batch, entities, variates, history = context.shape
        normalized, observed, location, scale = self.normalize(context, observed)
        active = observed.any(dim=-1)
        patch = self.config.patch_size
        total = math.ceil((history + horizon) / patch) * patch
        values = functional.pad(normalized, (0, total - history))
        mask = functional.pad(observed, (0, total - history))
        times = torch.arange(-history, total - history, device=context.device, dtype=torch.float32)
        times = (times / (history + horizon)).expand_as(values)
        features = torch.stack([values, times, mask.float()], dim=-1)
        tokens = features.reshape(batch, entities, variates, total // patch, patch * 3)
        tokens = self.embedding_skip(tokens) + self.embedding(tokens)
        patches, width = tokens.shape[-2:]
        temporal = self.temporal(tokens.reshape(-1, patches, width), active.reshape(-1, 1).expand(-1, patches))
        temporal = temporal.reshape(batch, entities, variates, patches, width)
        combined = temporal
        if self.mixer is not None:
            combined = temporal + self.gate(temporal).sigmoid() * self.mixer(temporal, active)
        normalized_prediction = self.head(combined).reshape(batch, entities, variates, total, len(self.config.quantiles))
        normalized_prediction = normalized_prediction[..., history:history + horizon, :].float()
        normalized_prediction = normalized_prediction.masked_fill(~active[..., None, None], 0)
        prediction = scale.unsqueeze(-1) * torch.sinh(normalized_prediction) + location.unsqueeze(-1)
        prediction = prediction.masked_fill(~active[..., None, None], 0)
        return {"prediction": prediction, "normalized_prediction": normalized_prediction,
                "location": location, "scale": scale, "active": active}

    def orthogonality_loss(self):
        if isinstance(self.mixer, PrototypeMixer):
            return self.mixer.orthogonality_loss()
        return self.head.weight.new_zeros(())


def quantile_loss(output, targets, quantiles, target_mask=None):
    prediction = output["normalized_prediction"]
    if targets.shape != prediction.shape[:-1]:
        raise ValueError("target shape differs from forecast")
    valid = torch.isfinite(targets) & output["active"].unsqueeze(-1)
    if target_mask is not None:
        if target_mask.shape != targets.shape:
            raise ValueError("target mask shape differs from targets")
        valid = valid & target_mask.bool()
    if not bool(valid.any()):
        raise ValueError("no observed targets with usable context")
    safe = torch.where(valid, targets.float(), output["location"].expand_as(targets))
    target = torch.asinh((safe - output["location"]) / output["scale"])
    error = target.unsqueeze(-1) - prediction
    quantiles = quantiles.to(device=prediction.device, dtype=prediction.dtype)
    loss = torch.maximum(quantiles * error, (quantiles - 1) * error)
    return (loss * valid.unsqueeze(-1)).sum() / (valid.sum() * len(quantiles))
