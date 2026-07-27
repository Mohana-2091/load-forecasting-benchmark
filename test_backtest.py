"""
Tests for the benchmark harness.

These do not test whether the models are accurate - accuracy is the thing the
benchmark measures, and asserting a particular MASE would just freeze whatever
the code happened to produce. They test the properties the *comparison* depends
on, because those are what make the results mean anything:

  * no model can see data from at or after the forecast origin
  * every model is evaluated on exactly the same folds
  * forecast origins cover all seven weekdays
  * the metrics behave correctly on known inputs

Two of these correspond to bugs that were actually found during development and
each of which reversed the headline ranking: origins all landing on one weekday,
and the sequence models receiving fewer features than the tabular ones. Tests
now pin both.

Everything runs on a synthetic series, so CI needs no data download.
"""

import numpy as np
import pandas as pd
import pytest

import backtest as B
import models as M


# ---------------------------------------------------------------- fixtures

def synthetic_series(years=3, base=30000.0, seed=0):
    """Hourly load with daily, weekly and annual cycles plus noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2017-01-01", periods=years * 8760, freq="h", tz="UTC")
    t = np.arange(len(idx))

    daily = 0.15 * np.sin(2 * np.pi * t / 24 - 1.2)
    weekly = -0.08 * (idx.dayofweek.values >= 5)
    annual = 0.12 * np.cos(2 * np.pi * t / 8766)
    noise = rng.normal(0, 0.02, len(idx))

    return pd.Series(base * (1 + daily + weekly + annual + noise), index=idx)


@pytest.fixture(scope="module")
def series():
    return synthetic_series()


@pytest.fixture(scope="module")
def fold(series):
    origin = pd.Timestamp("2019-06-12 00:00", tz="UTC")
    train = series.loc[:origin - pd.Timedelta(hours=1)]
    return train, origin


# ---------------------------------------------------------------- metrics

def test_perfect_forecast_scores_zero():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert B.mae(y, y) == 0.0
    assert B.rmse(y, y) == 0.0
    assert B.smape(y, y) == 0.0


def test_rmse_penalises_large_errors_more_than_mae():
    y = np.zeros(4)
    spread = np.array([0.0, 0.0, 0.0, 4.0])   # one big miss
    even = np.ones(4)                          # same total error, spread out
    assert B.mae(y, spread) == B.mae(y, even)
    assert B.rmse(y, spread) > B.rmse(y, even)


def test_mase_scale_is_seasonal_naive_error(series):
    v = series.values[:5000]
    expected = np.mean(np.abs(v[B.SEASON:] - v[:-B.SEASON]))
    assert B.mase_scale(v) == pytest.approx(expected)


def test_seasonal_naive_scores_near_one(series):
    """A seasonal naive forecast should sit around MASE 1.0 by construction."""
    origin = pd.Timestamp("2019-06-12 00:00", tz="UTC")
    train, actual = B.build_fold(series, origin)
    pred = M.SeasonalNaive().predict(train)
    score = B.mae(actual.values, pred) / B.mase_scale(train.values)
    assert 0.3 < score < 3.0


# ---------------------------------------------------------------- leakage

def test_features_never_reference_the_future(fold):
    """
    The decisive test. Poison every value from the origin onward, build the
    design matrix for the 24 target hours from the training slice only, and
    assert the poison never appears. If any feature reached forward in time it
    would show up here.
    """
    train, origin = fold
    poisoned = train.copy()
    future = pd.date_range(origin, periods=48, freq="h", tz="UTC")
    poisoned = pd.concat([poisoned, pd.Series(9.9e12, index=future)])

    idx = pd.date_range(origin, periods=B.HORIZON, freq="h", tz="UTC")
    X = M.design_matrix(poisoned.loc[:origin - pd.Timedelta(hours=1)], idx)

    assert np.isfinite(X.values).all()
    assert X.values.max() < 1e9, "a feature reached past the forecast origin"


def test_all_lags_are_at_least_24_hours(fold):
    """
    Every lag must be >= 24h. A day-ahead forecast issued at midnight cannot
    use hour 23 of the forecast day, so a lag of 12 would be unusable in
    deployment even though it would improve backtest scores.
    """
    assert min(M.LAGS) >= 24


def test_scaler_is_fit_on_training_data_only(fold):
    """The sequence models normalise with statistics from the training slice."""
    train, _ = fold
    model = M.DLinear(epochs=1, stride=48)
    model.fit(train)
    assert model.mu == pytest.approx(np.nanmean(
        train.iloc[-model.max_rows:].reindex(
            pd.date_range(train.index[-min(model.max_rows, len(train))],
                          train.index[-1], freq="h", tz="UTC")).values), rel=0.05)


# ---------------------------------------------------------------- folds

def test_incomplete_target_window_is_rejected(series):
    origin = pd.Timestamp("2019-06-12 00:00", tz="UTC")
    holed = series.drop(origin + pd.Timedelta(hours=5))
    assert B.build_fold(holed, origin) is None


def test_gap_in_feature_history_is_rejected(series):
    """
    A gap anywhere in the required history window disqualifies the fold - for
    every model, not just the ones that need long lags. Checking this per model
    would let the baselines keep folds the lag-based models must drop, and the
    comparison would stop being like for like.
    """
    origin = pd.Timestamp("2019-06-12 00:00", tz="UTC")
    assert B.build_fold(series, origin) is not None

    holed = series.drop(origin - pd.Timedelta(hours=400))
    assert B.build_fold(holed, origin) is None


def test_short_history_is_rejected(series):
    origin = pd.Timestamp("2017-03-01 00:00", tz="UTC")
    assert B.build_fold(series, origin) is None


def test_fold_eligibility_does_not_depend_on_the_model(series):
    """build_fold takes no model argument - eligibility cannot vary by method."""
    import inspect
    params = inspect.signature(B.build_fold).parameters
    assert set(params) == {"series", "origin"}


# ---------------------------------------------------------------- origins

def test_origins_cover_every_weekday(series):
    """
    Pins the first bug found in development. A seven-day origin step put every
    origin on a Tuesday, which made the daily naive baseline appear to beat the
    weekly seasonal naive (0.76 vs 1.01). With all weekdays covered that
    reversed (1.33 vs 1.06).
    """
    origins = B.make_origins(series.index)
    weekdays = pd.Series([o.dayofweek for o in origins]).value_counts()

    assert len(weekdays) == 7, "origins do not cover all weekdays"
    assert weekdays.max() / weekdays.min() < 1.5, "weekday coverage is skewed"


def test_origin_step_is_not_a_multiple_of_a_week():
    assert B.ORIGIN_STEP % 168 != 0


def test_origins_lie_in_the_test_year(series):
    for o in B.make_origins(series.index):
        assert o.year == B.TEST_YEAR


# ---------------------------------------------------------------- models

@pytest.mark.parametrize("name", [
    "seasonal_naive_168h", "daily_naive_24h", "seasonal_mean_4w",
    "harmonic_ols", "ridge_lags",
])
def test_model_returns_finite_horizon(fold, name):
    train, _ = fold
    model = M.build([name])[0]
    if model.needs_fit:
        model.fit(train)
    pred = np.asarray(model.predict(train, B.HORIZON), dtype=float)

    assert pred.shape == (B.HORIZON,)
    assert np.isfinite(pred).all()
    # Sanity band: a forecast three times the historical maximum is a bug.
    assert pred.max() < 3 * train.max()
    assert pred.min() > 0


def test_sequence_models_receive_calendar_features(fold):
    """
    Pins the second bug found in development. The sequence models originally
    saw only raw load while the tabular models got calendar features, which
    made the LSTM look far worse than it was. Both must now consume a non-empty
    calendar block.
    """
    train, _ = fold
    for model in (M.DLinear(epochs=1, stride=48), M.LSTMForecaster(epochs=1, stride=48)):
        model.fit(train)
        assert model.n_cal is not None and model.n_cal > 0


def test_dlinear_and_lstm_share_identical_training_setup():
    """
    The DLinear-vs-LSTM comparison only isolates architecture if everything
    else matches. Any divergence in these defaults would invalidate it.
    """
    d, l = M.DLinear(), M.LSTMForecaster()
    for attr in ("lookback", "epochs", "batch", "lr", "stride",
                 "max_rows", "weight_decay", "seed"):
        assert getattr(d, attr) == getattr(l, attr), attr


def test_ridge_and_xgboost_share_identical_features(fold):
    """Any accuracy difference between them is functional form, not inputs."""
    train, _ = fold
    r, x = M.RidgeLags(), M.XGBLags()
    r.fit(train)
    x.fit(train)
    assert r.columns == x.columns


def test_deterministic_models_are_deterministic(fold):
    train, _ = fold
    a = M.RidgeLags().fit(train).predict(train)
    b = M.RidgeLags().fit(train).predict(train)
    np.testing.assert_allclose(a, b)


def test_inputs_actually_move_the_forecast(fold):
    """Guards against a broken feature pipeline returning a constant."""
    train, _ = fold
    model = M.RidgeLags().fit(train)
    normal = model.predict(train)
    shifted = model.predict(train * 1.3)
    assert not np.allclose(normal, shifted)
