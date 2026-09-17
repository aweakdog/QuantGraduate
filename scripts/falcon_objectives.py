import math

OBJECTIVES = ("price_path", "price_endpoint", "return_endpoint", "return_rank")


def endpoint_returns(output, targets, context, target_mask=None):
    import torch

    if targets.shape != output["prediction"].shape[:-1] or context.shape[:3] != targets.shape[:3]:
        raise ValueError("endpoint target/context shape mismatch")
    current = context[:, :, 0, -1].float()
    actual = targets[:, :, 0, -1].float()
    valid = torch.isfinite(current) & torch.isfinite(actual) & output["active"][:, :, 0]
    if target_mask is not None:
        if target_mask.shape != targets.shape:
            raise ValueError("target mask shape mismatch")
        valid = valid & target_mask[:, :, 0, -1].bool()
    safe_current = torch.where(valid, current, 0)
    safe_actual = torch.where(valid, actual, safe_current)
    safe_prediction = torch.where(valid[..., None], output["prediction"][:, :, 0, -1].float(), safe_current[..., None])
    prediction = torch.expm1(safe_prediction - safe_current[..., None])
    realized = torch.expm1(safe_actual - safe_current)
    return prediction, realized, valid


def return_pinball(prediction, realized, valid, quantiles, scale=0.05):
    import torch

    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("return scale must be positive and finite")
    if not bool(valid.any()):
        raise ValueError("no observed endpoint targets")
    if not torch.isfinite(prediction[valid]).all() or not torch.isfinite(realized[valid]).all():
        raise ValueError("nonfinite endpoint return")
    quantiles = quantiles.to(device=prediction.device, dtype=prediction.dtype)
    prediction = torch.where(valid[..., None], prediction, 0)
    realized = torch.where(valid, realized, 0)
    error = (realized[..., None] - prediction) / scale
    loss = torch.maximum(quantiles * error, (quantiles - 1) * error)
    return (loss * valid[..., None]).sum() / (valid.sum() * len(quantiles))


def pairwise_rank_loss(scores, realized, valid, temperature=0.05):
    import torch
    from torch.nn import functional

    if scores.ndim != 2 or scores.shape != realized.shape or scores.shape != valid.shape:
        raise ValueError("ranking requires aligned [independent_dates, entities] tensors")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("rank temperature must be positive and finite")
    valid = valid.bool() & torch.isfinite(scores) & torch.isfinite(realized)
    scores = torch.where(valid, scores.float(), 0)
    realized = torch.where(valid, realized.float(), 0)
    order = torch.sign(realized[:, :, None] - realized[:, None, :])
    upper = torch.ones((scores.shape[1], scores.shape[1]), dtype=torch.bool, device=scores.device).triu(diagonal=1)
    pairs = valid[:, :, None] & valid[:, None, :] & upper & (order != 0)
    difference = (scores[:, :, None] - scores[:, None, :]) / temperature
    loss = functional.softplus(-order * difference)
    return (loss * pairs).sum() / pairs.sum().clamp_min(1)


def training_objective(output, targets, quantiles, target_mask, context, objective="price_path",
                       return_scale=0.05, rank_weight=0.1, rank_temperature=0.05):
    import torch

    from scripts.falcon_model import quantile_loss

    if objective not in OBJECTIVES:
        raise ValueError("unknown training objective")
    if target_mask is not None and target_mask.shape != targets.shape:
        raise ValueError("target mask shape mismatch")
    if not math.isfinite(rank_weight) or rank_weight < 0:
        raise ValueError("rank weight must be nonnegative and finite")
    if objective == "price_path":
        return quantile_loss(output, targets, quantiles, target_mask)
    if objective == "price_endpoint":
        endpoint_mask = torch.zeros_like(targets, dtype=torch.bool)
        endpoint_mask[:, :, 0, -1] = True
        if target_mask is not None:
            endpoint_mask = endpoint_mask & target_mask.bool()
        return quantile_loss(output, targets, quantiles, endpoint_mask)
    prediction, realized, valid = endpoint_returns(output, targets, context, target_mask)
    point = return_pinball(prediction, realized, valid, quantiles, return_scale)
    if objective == "return_endpoint":
        return point
    median = (quantiles == 0.5).nonzero().flatten()
    if len(median) != 1:
        raise ValueError("ranking anchor requires one median quantile")
    ranking = pairwise_rank_loss(prediction[..., int(median.item())], realized, valid, rank_temperature)
    return point + rank_weight * ranking
