"""Optional deep-learning transition model wrappers."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

from geas35.models.transition.base import (
    BaseTransitionModel,
    TransitionDataset,
    TransitionModelCapabilities,
    TransitionPrediction,
)
from geas35.models.transition.sklearn_models import optional_dependency_unavailable


class TCNTransitionModel(BaseTransitionModel):
    """Small torch TCN-style sequence-to-vector transition wrapper.

    The Step 9 implementation keeps the public ``TransitionDataset`` interface
    tabular. Inputs are reshaped into a sequence with ``sequence_length`` steps,
    using sequence length 1 by default for compatibility with MDP v1 frames.
    """

    model_name = "tcn"
    capabilities = TransitionModelCapabilities(
        supports_sequence_input=True,
        supports_native_multi_output=True,
        multi_output_strategy="native",
        optional_dependency="torch",
    )

    def __init__(
        self,
        *,
        sequence_length: int = 1,
        hidden_channels: int = 16,
        kernel_size: int = 2,
        epochs: int = 20,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        random_state: int | None = None,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive.")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")
        if epochs <= 0:
            raise ValueError("epochs must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.sequence_length = int(sequence_length)
        self.hidden_channels = int(hidden_channels)
        self.kernel_size = int(kernel_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.random_state = random_state
        self.model_: Any | None = None

    def fit(
        self,
        dataset: TransitionDataset,
        validation_dataset: TransitionDataset | None = None,
    ) -> "TCNTransitionModel":
        del validation_dataset
        torch, nn = _torch_modules("TCNTransitionModel")
        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))

        x = _to_sequence_array(
            dataset.x.loc[:, list(dataset.input_columns)],
            sequence_length=self.sequence_length,
        )
        y = dataset.y.loc[:, list(dataset.target_columns)].to_numpy(dtype=np.float32).copy()
        x_tensor = torch.as_tensor(x, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.float32)

        model = _SimpleTCN(
            nn=nn,
            input_channels=x.shape[1],
            output_dim=y.shape[1],
            hidden_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        model.train()
        for _ in range(self.epochs):
            for start in range(0, len(x_tensor), self.batch_size):
                end = start + self.batch_size
                batch_x = x_tensor[start:end]
                batch_y = y_tensor[start:end]
                optimizer.zero_grad()
                loss = loss_fn(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()

        self.model_ = model
        self.input_columns_ = dataset.input_columns
        self.target_columns_ = dataset.target_columns
        return self

    def predict(
        self,
        dataset_or_frame: TransitionDataset | pd.DataFrame,
    ) -> TransitionPrediction:
        input_columns, target_columns = self._require_fitted()
        if self.model_ is None:
            raise ValueError("Transition model is not fitted.")
        torch, _ = _torch_modules("TCNTransitionModel")
        frame = (
            dataset_or_frame.x
            if isinstance(dataset_or_frame, TransitionDataset)
            else dataset_or_frame
        )
        x = _to_sequence_array(
            frame.loc[:, list(input_columns)],
            sequence_length=self.sequence_length,
        )
        self.model_.eval()
        with torch.no_grad():
            values = (
                self.model_(torch.as_tensor(x, dtype=torch.float32))
                .detach()
                .cpu()
                .numpy()
            )
        return TransitionPrediction(
            next_observation=pd.DataFrame(
                values,
                columns=target_columns,
                index=frame.index,
            ),
            target_columns=target_columns,
        )


def _torch_modules(model_label: str) -> tuple[Any, Any]:
    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
    except ModuleNotFoundError as exc:
        raise optional_dependency_unavailable(model_label, "torch") from exc
    return torch, nn


def _to_sequence_array(frame: pd.DataFrame, *, sequence_length: int) -> np.ndarray:
    values = frame.to_numpy(dtype=np.float32).copy()
    if sequence_length == 1:
        return values[:, :, None]
    feature_count = values.shape[1]
    if feature_count % sequence_length != 0:
        raise ValueError(
            "TCNTransitionModel with sequence_length > 1 requires the number of "
            "input columns to be divisible by sequence_length."
        )
    channels = feature_count // sequence_length
    return values.reshape(values.shape[0], sequence_length, channels).transpose(0, 2, 1)


class _SimpleTCN:
    def __new__(
        cls,
        *,
        nn: Any,
        input_channels: int,
        output_dim: int,
        hidden_channels: int,
        kernel_size: int,
    ) -> Any:
        padding = max(kernel_size - 1, 0)
        return nn.Sequential(
            nn.Conv1d(
                input_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, output_dim),
        )


__all__ = [
    "TCNTransitionModel",
]
