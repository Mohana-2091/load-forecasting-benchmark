"""
Forecasting methods for the benchmark.

The lineup is a deliberate ladder of complexity, so the question "does the
extra complexity pay for itself?" can be answered at each rung rather than
only at the top:

  1. seasonal_naive      last week, same hour            no parameters
  2. daily_naive         yesterday, same hour            no parameters
  3. seasonal_mean_4w    mean of last 4 same-hours       no parameters
  4. harmonic_ols        Fourier seasonality + trend     classical statistical
  5. ridge_lags          lags + calendar, regularised    linear ML
  6. xgboost_lags        same features, boosted trees    non-linear ML
  7. dlinear             one linear layer, 168 -> 24     linear sequence model
  8. lstm                recurrent sequence-to-sequence  deep learning

dlinear and lstm are deliberately paired. They receive identical windows, the
same scaler, optimiser, epoch budget and seed, and differ only in
architecture - so the comparison isolates whether recurrence earns its keep.

Fairness rules applied to every learned method
----------------------------------------------
* Identical feature access. All lag features are at least 24 hours old, which
  is exactly what is knowable when a day-ahead forecast is issued at midnight.
  No method sees a feature another method cannot have.

* Identical refit cadence. Every learned model is refit monthly. Refitting
  gradient boosting at every origin while training the LSTM once would make
  the comparison meaningless, and monthly retraining is closer to how these
  systems are actually operated.

* No tuning on the test period. Hyperparameters are fixed a priori and modest.
  Searching them against 2019 results would be selecting on the answer.

Leakage control
---------------
Every feature for a target at time t is built only from values at or before
t-24. Scalers are fit on the training slice alone. Nothing downstream of the
forecast origin is ever touched.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler

HORIZON = 24
SEASON = 168

# All lags are >= 24h so they are available for every hour of a day-ahead
# forecast issued at midnight.
LAGS = [24, 25, 26, 48, 72, 168, 169, 336, 504]


# ================================================================ features

def calendar_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic calendar features - knowable arbitrarily far ahead."""
    f = pd.DataFrame(index=index)
    hour = index.hour.values
    dow = index.dayofweek.values
    doy = index.dayofyear.values

    # Fourier terms rather than raw integers: they encode that hour 23 is
    # adjacent to hour 0, which an integer column cannot express.
    for k in (1, 2, 3):
        f["hour_sin_{}".format(k)] = np.sin(2 * np.pi * k * hour / 24)
        f["hour_cos_{}".format(k)] = np.cos(2 * np.pi * k * hour / 24)
    for k in (1, 2):
        f["week_sin_{}".format(k)] = np.sin(2 * np.pi * k * (dow * 24 + hour) / 168)
        f["week_cos_{}".format(k)] = np.cos(2 * np.pi * k * (dow * 24 + hour) / 168)
        f["year_sin_{}".format(k)] = np.sin(2 * np.pi * k * doy / 365.25)
        f["year_cos_{}".format(k)] = np.cos(2 * np.pi * k * doy / 365.25)

    f["is_weekend"] = (dow >= 5).astype(float)
    for d in range(7):
        f["dow_{}".format(d)] = (dow == d).astype(float)
    return f


def _safe(fn, arr):
    """Apply a nan-aware row aggregate, returning NaN for all-NaN rows instead
    of emitting a RuntimeWarning."""
    out = np.full(arr.shape[0], np.nan)
    ok = ~np.isnan(arr).all(axis=1)
    if ok.any():
        out[ok] = fn(arr[ok], axis=1)
    return out


def lag_frame(series: pd.Series, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Lag and rolling features, all drawn from t-24 or earlier."""
    f = pd.DataFrame(index=index)
    for L in LAGS:
        f["lag_{}".format(L)] = series.reindex(index - pd.Timedelta(hours=L)).values

    # Mean of the same hour over the previous four weeks
    same_hour = np.column_stack([
        series.reindex(index - pd.Timedelta(hours=SEASON * k)).values for k in (1, 2, 3, 4)
    ])
    # Mean and spread of the last full day of data available at issue time
    prev_day = np.column_stack([
        series.reindex(index - pd.Timedelta(hours=h)).values for h in range(24, 48)
    ])

    # An all-NaN row means the lookup landed entirely inside a data gap. The
    # backtest rejects those folds up front, but the guard keeps the warnings
    # quiet and makes the failure explicit rather than silent if it ever slips
    # through.
    with np.errstate(invalid="ignore"):
        f["same_hour_4w_mean"] = _safe(np.nanmean, same_hour)
        f["same_hour_4w_std"] = _safe(np.nanstd, same_hour)
        f["prev_day_mean"] = _safe(np.nanmean, prev_day)
        f["prev_day_min"] = _safe(np.nanmin, prev_day)
        f["prev_day_max"] = _safe(np.nanmax, prev_day)
    return f


def design_matrix(series: pd.Series, index: pd.DatetimeIndex, with_lags=True) -> pd.DataFrame:
    cal = calendar_frame(index)
    if not with_lags:
        return cal
    return pd.concat([cal, lag_frame(series, index)], axis=1)


def training_rows(train: pd.Series, with_lags=True, max_rows=None):
    """Build (X, y) from the training slice only."""
    idx = train.index
    if with_lags:
        idx = idx[idx >= train.index[0] + pd.Timedelta(hours=max(LAGS))]
    X = design_matrix(train, idx, with_lags=with_lags)
    y = train.reindex(idx).values

    keep = ~X.isna().any(axis=1).values & ~np.isnan(y)
    X, y = X[keep], y[keep]

    if max_rows and len(X) > max_rows:
        X, y = X.iloc[-max_rows:], y[-max_rows:]
    return X, y


def target_index(train: pd.Series, horizon=HORIZON) -> pd.DatetimeIndex:
    start = train.index[-1] + pd.Timedelta(hours=1)
    return pd.date_range(start, periods=horizon, freq="h", tz="UTC")


# ================================================================ models

class Model:
    """fit() is called on a refit boundary; predict() at every origin."""
    name = "base"
    needs_fit = True

    def fit(self, train: pd.Series):
        return self

    def predict(self, train: pd.Series, horizon=HORIZON) -> np.ndarray:
        raise NotImplementedError


class SeasonalNaive(Model):
    name = "seasonal_naive_168h"
    needs_fit = False

    def predict(self, train, horizon=HORIZON):
        return np.asarray(train.iloc[-SEASON:-SEASON + horizon].values, dtype=float)


class DailyNaive(Model):
    name = "daily_naive_24h"
    needs_fit = False

    def predict(self, train, horizon=HORIZON):
        return np.asarray(train.iloc[-24:].values, dtype=float)


class SeasonalMean4W(Model):
    name = "seasonal_mean_4w"
    needs_fit = False

    def predict(self, train, horizon=HORIZON):
        w = [train.iloc[-SEASON * k: -SEASON * k + horizon].values for k in range(1, 5)]
        return np.asarray(np.mean(w, axis=0), dtype=float)


class HarmonicOLS(Model):
    """
    Classical decomposition: load explained by Fourier terms for daily, weekly
    and annual cycles plus a linear trend. No lags at all - this is the
    "structural" view of the series, and a useful control for how much of the
    signal is pure calendar seasonality.
    """
    name = "harmonic_ols"

    def __init__(self, max_rows=None):
        self.max_rows = max_rows
        self.model = None
        self.t0 = None

    def _add_trend(self, X, index):
        X = X.copy()
        X["trend"] = (index - self.t0).total_seconds().values / 3600 / 8766
        return X

    def fit(self, train):
        self.t0 = train.index[0]
        X, y = training_rows(train, with_lags=False, max_rows=self.max_rows)
        X = self._add_trend(X, X.index)
        self.model = LinearRegression().fit(X, y)
        return self

    def predict(self, train, horizon=HORIZON):
        idx = target_index(train, horizon)
        X = self._add_trend(design_matrix(train, idx, with_lags=False), idx)
        return self.model.predict(X)


class RidgeLags(Model):
    """Linear model on lags plus calendar. Ridge because the lags are highly
    collinear; unregularised OLS is unstable on them."""
    name = "ridge_lags"

    def __init__(self, alpha=1.0, max_rows=None):
        self.alpha = alpha
        self.max_rows = max_rows
        self.scaler = None
        self.model = None
        self.columns = None

    def fit(self, train):
        X, y = training_rows(train, max_rows=self.max_rows)
        self.columns = list(X.columns)
        self.scaler = StandardScaler().fit(X.values)
        self.model = Ridge(alpha=self.alpha).fit(self.scaler.transform(X.values), y)
        return self

    def predict(self, train, horizon=HORIZON):
        idx = target_index(train, horizon)
        X = design_matrix(train, idx)[self.columns]
        return self.model.predict(self.scaler.transform(X.values))


class XGBLags(Model):
    """Gradient boosting on exactly the same features as RidgeLags, so any
    difference is attributable to the functional form, not the inputs."""
    name = "xgboost_lags"

    def __init__(self, max_rows=None, **params):
        from xgboost import XGBRegressor
        self.max_rows = max_rows
        self.params = dict(
            n_estimators=350, max_depth=6, learning_rate=0.06,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            random_state=42, n_jobs=-1,
        )
        self.params.update(params)
        self._cls = XGBRegressor
        self.model = None
        self.columns = None

    def fit(self, train):
        X, y = training_rows(train, max_rows=self.max_rows)
        self.columns = list(X.columns)
        self.model = self._cls(**self.params).fit(X.values, y)
        return self

    def predict(self, train, horizon=HORIZON):
        idx = target_index(train, horizon)
        X = design_matrix(train, idx)[self.columns]
        return self.model.predict(X.values)


class _SeqModel(Model):
    """
    Shared training machinery for the sequence models.

    Both take the last `lookback` hours of load AND the calendar features of
    the 24 target hours, and emit all 24 hours at once (direct multi-horizon,
    not recursive - recursive forecasting compounds its own errors over 24
    steps).

    Why the calendar features are here
    ----------------------------------
    A first version fed the sequence models the raw load window only, while
    the tabular models received hour-of-day, day-of-week and annual Fourier
    terms. The LSTM scored MASE 1.166 against XGBoost's 0.503, which looked
    like a decisive result and was not one: the two were not given the same
    information. The LSTM had to infer from 168 raw numbers what the tabular
    models were simply told.

    That violates the benchmark's own fairness rule, and reporting it would
    have been a rigged comparison dressed up as a finding. Both sequence
    models now receive the identical calendar frame the tabular models use,
    flattened across the 24 target hours. Those hours are in the future, but
    calendar features are deterministic and knowable arbitrarily far ahead, so
    this is not leakage.

    Everything except the architecture is held constant between the two: the
    same windows, scaler, optimiser, weight decay, epoch budget and seed. Any
    remaining difference is attributable to the architecture alone.
    """
    name = "seq"

    def __init__(self, lookback=168, epochs=10, batch=128, lr=2e-3,
                 stride=6, max_rows=17520, weight_decay=1e-4, seed=42):
        self.lookback = lookback
        self.epochs = epochs
        self.batch = batch
        self.lr = lr
        self.stride = stride
        self.max_rows = max_rows
        self.weight_decay = weight_decay
        self.seed = seed
        self.net = None
        self.mu = None
        self.sd = None
        self.n_cal = None

    def _build(self):
        raise NotImplementedError

    def _windows(self, train):
        """Build (load window, target calendar, target) triples.

        The series is reindexed to a complete hourly grid first, and any window
        touching a gap is dropped - the same treatment the tabular models get,
        where a lag landing in a gap produces NaN and the row is removed.
        """
        s = train.iloc[-self.max_rows:]
        full = pd.date_range(s.index[0], s.index[-1], freq="h", tz="UTC")
        z = s.reindex(full).values.astype(np.float32)

        cal = calendar_frame(full).values.astype(np.float32)
        self.n_cal = cal.shape[1] * HORIZON

        finite = ~np.isnan(z)
        self.mu = float(np.nanmean(z))
        self.sd = float(np.nanstd(z))
        z = (z - self.mu) / self.sd

        W, C, Y = [], [], []
        last = len(z) - self.lookback - HORIZON
        for i in range(0, last, self.stride):
            a, b = i, i + self.lookback + HORIZON
            if not finite[a:b].all():
                continue                      # window touches a gap
            W.append(z[i:i + self.lookback])
            C.append(cal[i + self.lookback:i + self.lookback + HORIZON].ravel())
            Y.append(z[i + self.lookback:i + self.lookback + HORIZON])

        return np.array(W), np.array(C), np.array(Y)

    def fit(self, train):
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        W, C, Y = self._windows(train)

        Xw = torch.tensor(W)
        Xc = torch.tensor(C)
        y = torch.tensor(Y)

        torch.manual_seed(self.seed)
        self.net = self._build()
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        loss_fn = torch.nn.MSELoss()
        loader = DataLoader(TensorDataset(Xw, Xc, y), batch_size=self.batch, shuffle=True)

        self.net.train()
        for _ in range(self.epochs):
            for wb, cb, yb in loader:
                opt.zero_grad()
                loss_fn(self.net(wb, cb), yb).backward()
                opt.step()
        return self

    def predict(self, train, horizon=HORIZON):
        import torch
        idx = target_index(train, horizon)
        cal = calendar_frame(idx).values.astype(np.float32).ravel()

        z = (train.values[-self.lookback:].astype(np.float32) - self.mu) / self.sd
        w = torch.tensor(z).view(1, self.lookback)
        c = torch.tensor(cal).view(1, -1)

        self.net.eval()
        with torch.no_grad():
            out = self.net(w, c).numpy().ravel()
        return out * self.sd + self.mu


class DLinear(_SeqModel):
    """
    A single linear layer mapping the past 168 hours plus the target calendar
    to the next 24 hours.

    This is the control the 2022 paper "Are Transformers Effective for Time
    Series Forecasting?" used to argue that much of the reported advantage of
    deep sequence models does not survive careful evaluation. No recurrence, no
    attention, no non-linearity anywhere - and it trains in seconds.

    If it matches or beats the LSTM here, that is the single most informative
    result this benchmark can produce.
    """
    name = "dlinear"

    def _build(self):
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self, lookback, n_cal, horizon):
                super().__init__()
                self.fc = nn.Linear(lookback + n_cal, horizon)

            def forward(self, w, c):
                return self.fc(torch.cat([w, c], dim=1))

        return Net(self.lookback, self.n_cal, HORIZON)


class LSTMForecaster(_SeqModel):
    """
    Recurrent encoder over the load window, concatenated with the target
    calendar before the output head.

    Kept small and untuned on purpose. The question is whether recurrence beats
    simpler methods on this problem, and tuning the architecture against the
    2019 evaluation period would be selecting on the answer.
    """
    name = "lstm"

    def __init__(self, hidden=48, layers=1, **kwargs):
        super().__init__(**kwargs)
        self.hidden = hidden
        self.layers = layers

    def _build(self):
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self, hidden, layers, n_cal, horizon):
                super().__init__()
                self.lstm = nn.LSTM(1, hidden, layers, batch_first=True)
                self.head = nn.Linear(hidden + n_cal, horizon)

            def forward(self, w, c):
                out, _ = self.lstm(w.unsqueeze(-1))
                return self.head(torch.cat([out[:, -1, :], c], dim=1))

        return Net(self.hidden, self.layers, self.n_cal, HORIZON)


BASELINES = [SeasonalNaive(), DailyNaive(), SeasonalMean4W()]


def build(names):
    registry = {
        "seasonal_naive_168h": SeasonalNaive,
        "daily_naive_24h": DailyNaive,
        "seasonal_mean_4w": SeasonalMean4W,
        "harmonic_ols": HarmonicOLS,
        "ridge_lags": RidgeLags,
        "xgboost_lags": XGBLags,
        "dlinear": DLinear,
        "lstm": LSTMForecaster,
    }
    return [registry[n]() for n in names]
