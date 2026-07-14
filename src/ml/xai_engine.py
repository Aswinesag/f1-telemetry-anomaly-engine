from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import torch
from torch import nn


FeatureImportance: TypeAlias = dict[str, float]


class ReconstructionLossAttributionModel(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        reconstructed = self.model(input_tensor)
        squared_errors = torch.pow(input_tensor - reconstructed, 2)
        return torch.mean(squared_errors, dim=1)


class XAIEngine:
    def __init__(
        self,
        model: nn.Module,
        feature_names: Sequence[str],
        *,
        n_steps: int = 32,
    ) -> None:
        if not feature_names:
            raise ValueError("feature_names must contain at least one feature.")

        self._feature_names = tuple(feature_names)
        self._n_steps = n_steps
        self._attribution_model = ReconstructionLossAttributionModel(model)
        self._attribution_model.eval()

        from captum.attr import IntegratedGradients

        self._integrated_gradients = IntegratedGradients(self._attribution_model)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    def get_feature_importance(self, input_tensor: torch.Tensor) -> FeatureImportance:
        if input_tensor.ndim == 1:
            attribution_input = input_tensor.unsqueeze(0)
        elif input_tensor.ndim == 2:
            attribution_input = input_tensor
        else:
            raise ValueError(
                "input_tensor must be one or two dimensional for autoencoder attribution."
            )

        feature_count = attribution_input.shape[-1]
        if feature_count != len(self._feature_names):
            raise ValueError(
                f"Expected {len(self._feature_names)} attribution features, "
                f"received {feature_count}."
            )

        attribution_input = attribution_input.detach().clone().requires_grad_(True)
        baseline = torch.zeros_like(attribution_input)
        attributions = self._integrated_gradients.attribute(
            attribution_input,
            baselines=baseline,
            n_steps=self._n_steps,
        )
        contribution_scores = attributions.detach().abs().mean(dim=0)
        total_contribution = torch.sum(contribution_scores).item()

        if total_contribution <= 0.0:
            normalized_scores = torch.zeros_like(contribution_scores)
        else:
            normalized_scores = contribution_scores / total_contribution

        return {
            feature_name: float(score)
            for feature_name, score in zip(
                self._feature_names,
                normalized_scores.cpu().tolist(),
                strict=True,
            )
        }
