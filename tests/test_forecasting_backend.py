import numpy as np

from timesfm.forecasting.backends import TimesFMBackendAdapter


class FakeModel:

  def __init__(self):
    self.forecast_calls = []
    self.covariate_calls = []

  def forecast(self, horizon, inputs):
    self.forecast_calls.append((horizon, len(inputs)))
    point = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    quantiles = np.zeros((1, 4, 10), dtype=np.float32)
    quantiles[:, :, 5] = point
    return point, quantiles

  def forecast_with_covariates(self, **kwargs):
    self.covariate_calls.append(kwargs)
    point = [np.array([5.0, 6.0], dtype=np.float32)]
    quantiles = [np.zeros((2, 10), dtype=np.float32)]
    quantiles[0][:, 5] = point[0]
    return point, quantiles


def test_backend_adapter_trims_backcast_from_baseline_outputs():
  model = FakeModel()
  backend = TimesFMBackendAdapter(model)

  point, quantiles = backend.forecast(2, [np.array([10.0, 11.0], dtype=np.float32)])

  assert model.forecast_calls == [(2, 1)]
  assert point.shape == (1, 2)
  assert point[0].tolist() == [3.0, 4.0]
  assert quantiles.shape == (1, 2, 10)


def test_backend_adapter_passes_covariates_through():
  model = FakeModel()
  backend = TimesFMBackendAdapter(model)

  point, quantiles = backend.forecast_with_covariates(
    2,
    [np.array([10.0, 11.0], dtype=np.float32)],
    dynamic_numerical_covariates={
      "flight_frequency": [np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)]
    },
  )

  assert len(model.covariate_calls) == 1
  assert point.shape == (1, 2)
  assert quantiles.shape == (1, 2, 10)
