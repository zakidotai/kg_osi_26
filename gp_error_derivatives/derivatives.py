"""Differentiate prediction MSE without updating GP parameters."""

import torch


def _constant_tensor(value, name):
    tensor = torch.as_tensor(value)
    if tensor.is_complex() or tensor.dtype == torch.bool:
        raise TypeError(f"{name} must contain continuous real values.")
    tensor = torch.as_tensor(value, device="cpu", dtype=torch.float64).detach().clone()
    if tensor.numel() == 0 or not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be nonempty and finite.")
    return tensor


def _prediction_error(query_embeddings, extracted_information, gp_component, y_true):
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Call outside torch.inference_mode() to compute derivatives.")
    if not extracted_information:
        raise ValueError("extracted_information must contain at least one numeric block.")

    with torch.enable_grad():
        queries = _constant_tensor(query_embeddings, "query_embeddings")
        if queries.ndim != 2:
            raise ValueError("query_embeddings must have shape (n_queries, dimension).")
        queries.requires_grad_(True)
        information = {
            name: _constant_tensor(value, f"extracted_information.{name}").requires_grad_(True)
            for name, value in extracted_information.items()
        }
        targets = _constant_tensor(y_true, "y_true")
        if targets.ndim != 1:
            raise ValueError("y_true must be a one-dimensional target vector.")
        predictions = gp_component(queries, information)
        if not isinstance(predictions, torch.Tensor) or not predictions.is_floating_point():
            raise TypeError("gp_component must return a floating PyTorch prediction tensor.")
        if predictions.ndim != 1 or predictions.shape != targets.shape:
            raise ValueError("Predictions and y_true must be vectors with matching lengths.")
        predictions = predictions.to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(predictions).all():
            raise ValueError("gp_component returned nonfinite predictions.")
        error = (predictions - targets).square().mean()
        if not torch.isfinite(error):
            raise ValueError("Prediction MSE overflowed; check units and numerical scales.")
    return queries, information, error


def _error_derivatives(error, inputs):
    derivatives = (
        torch.autograd.grad(error, inputs, allow_unused=True)
        if error.requires_grad else (None,) * len(inputs)
    )
    for derivative in derivatives:
        if derivative is not None and not torch.isfinite(derivative).all():
            raise ValueError("A derivative is nonfinite; check the component's numerical path.")
    return derivatives


def calculate_query_derivatives(
    query_embeddings, extracted_information, gp_component, y_true
):
    """Return d(MSE)/d(query_embeddings) as a tensor, or None if disconnected."""
    queries, _, error = _prediction_error(
        query_embeddings, extracted_information, gp_component, y_true
    )
    return _error_derivatives(error, (queries,))[0]


def calculate_extracted_information_derivatives(
    query_embeddings, extracted_information, gp_component, y_true
):
    """Return d(MSE)/d(extracted information block), with None if disconnected."""
    _, information, error = _prediction_error(
        query_embeddings, extracted_information, gp_component, y_true
    )
    derivatives = _error_derivatives(error, tuple(information.values()))
    return dict(zip(information, derivatives))
