"""Backend adapters for running the forecasting pipeline with TimesFM."""

from collections.abc import Mapping, Sequence

import numpy as np

from ..configs import ForecastConfig
from ..timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch


class TimesFMBackendAdapter:
  """Wrap a TimesFM model with the pipeline backend protocol."""

  def __init__(self, model) -> None:
    self.model = model

  @classmethod
  def from_pretrained(
    cls,
    model_id: str = "google/timesfm-2.5-200m-pytorch",
    *,
    max_context: int = 1024,
    max_horizon: int = 256,
    per_core_batch_size: int = 8,
    normalize_inputs: bool = True,
  ) -> "TimesFMBackendAdapter":
    model = TimesFM_2p5_200M_torch.from_pretrained(model_id)
    model.compile(
      ForecastConfig(
        max_context=max_context,
        max_horizon=max_horizon,
        normalize_inputs=normalize_inputs,
        per_core_batch_size=per_core_batch_size,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
        return_backcast=True,
      )
    )
    return cls(model)

  def forecast(
    self,
    horizon: int,
    inputs: list[np.ndarray],
  ) -> tuple[np.ndarray, np.ndarray]:
    point, quantiles = self.model.forecast(horizon=horizon, inputs=inputs)
    return point[:, -horizon:], quantiles[:, -horizon:, :]

  def forecast_with_covariates(
    self,
    horizon: int,
    inputs: list[np.ndarray],
    dynamic_numerical_covariates: Mapping[str, list[np.ndarray]] | None = None,
    dynamic_categorical_covariates: Mapping[str, list[np.ndarray]] | None = None,
    static_numerical_covariates: Mapping[str, Sequence[float]] | None = None,
    static_categorical_covariates: Mapping[str, Sequence[str]] | None = None,
  ) -> tuple[np.ndarray, np.ndarray]:
    point, quantiles = self.model.forecast_with_covariates(
      inputs=inputs,
      dynamic_numerical_covariates=dynamic_numerical_covariates,
      dynamic_categorical_covariates=dynamic_categorical_covariates,
      static_numerical_covariates=static_numerical_covariates,
      static_categorical_covariates=static_categorical_covariates,
      xreg_mode="xreg + timesfm",
    )
    point_array = np.array(point, dtype=np.float32)
    quantile_array = np.array(quantiles, dtype=np.float32)
    return point_array[:, :horizon], quantile_array[:, :horizon, :]
