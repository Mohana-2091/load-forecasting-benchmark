"""
Backtest runner for the forecasting benchmark.

The evaluation design is defined here and was written and validated before any
learned model was added, because the design decides whether the comparison
means anything.

Design decisions
----------------

1. HORIZON: 24 hours ahead.
   Day-ahead is the real operational problem - grid operators commit
   generation for the following day. One-step-ahead forecasting looks
   impressive and is close to useless, because at h=1 a "last value" forecast
   is already very strong.

2. ROLLING ORIGIN, not a single split.
   One split gives one number per model from one arbitrary period. The origin
   rolls forward through 2019, giving ~60 origins per series and ~360 across
   the panel, so consistency of a ranking can be tested rather than asserted.

3. ORIGIN SPACING OF SIX DAYS, NOT SEVEN.
   A weekly step looks natural and is a trap. The first origin falls on
   1 January 2019, a Tuesday, so a seven-day step makes every origin a
   Tuesday - the benchmark would only ever test weekday-to-weekday
   transitions and never a Saturday, Sunday, or the Sunday-to-Monday jump.
   A first run with a seven-day step showed the daily naive baseline beating
   the weekly seasonal naive by a wide margin (MASE 0.76 vs 1.01). With a
   six-day step, which advances the weekday by one each time, that reversed
   completely (1.33 vs 1.06). The original result was an artifact of the
   spacing, not a finding.

4. EXPANDING TRAINING WINDOW, MONTHLY REFIT.
   Each model trains on everything up to the origin. Models are refit when the
   calendar month changes, and the same cadence applies to every method -
   refitting gradient boosting at every origin while training the LSTM once
   would invalidate the comparison.

5. MASE AS THE HEADLINE METRIC.
   MAPE is the usual default and is a poor choice here: asymmetric, unstable
   at low load, and not comparable across six grids of very different size.
   MASE scales by the in-sample error of a seasonal naive forecast, so MASE<1
   means "beats seasonal naive" and values are comparable across series.

6. THE BASELINE IS A REAL COMPETITOR.
   Seasonal naive at lag 168 is genuinely hard to beat on hourly load. Much of
   the point of this benchmark is to find out how much of the reported
   advantage of complex methods survives against it.
"""

import argparse
import os
import time
import numpy as np
import pandas as pd

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import models as M

HORIZON = 24
SEASON = 168
ORIGIN_STEP = 6 * 24
TEST_YEAR = 2019
MIN_TRAIN_HOURS = 365 * 24

# Longest lag any model uses (504h) plus the 24h it is offset by. A fold is
# only usable if this window before the origin is complete and contiguous.
HISTORY_REQUIRED = 504 + 24


# ---------------------------------------------------------------- metrics

def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def smape(y, p):
    denom = (np.abs(y) + np.abs(p)) / 2
    return float(100 * np.mean(np.abs(y - p) / denom))


def mase_scale(train_values, season=SEASON):
    """In-sample MAE of seasonal naive. Training data only."""
    return float(np.mean(np.abs(train_values[season:] - train_values[:-season])))


# ---------------------------------------------------------------- splitter

def make_origins(index, test_year=TEST_YEAR):
    start = pd.Timestamp("{}-01-01 00:00".format(test_year), tz="UTC")
    end = pd.Timestamp("{}-12-31 00:00".format(test_year), tz="UTC")
    origins, t = [], start
    while t <= end:
        if t in index:
            origins.append(t)
        t += pd.Timedelta(hours=ORIGIN_STEP)
    return origins


def build_fold(series, origin):
    """
    Return (train, actual) or None if the fold is unusable.

    Three conditions, all checked identically for every model so that no method
    is evaluated on an easier subset of folds than another:

      1. enough training history overall
      2. the 24-hour target window is complete
      3. the HISTORY_REQUIRED hours before the origin are complete and
         contiguous, so every lag feature any model uses is available

    Condition 3 matters more than it looks. Series were dropna'd, so hours lost
    to ENTSO-E outages are absent from the index entirely and a lag landing in
    one silently returns NaN. Checking it per model would let the baselines -
    which need no lags - keep folds that the lag-based models must skip, and
    the comparison would no longer be like for like.
    """
    train = series.loc[:origin - pd.Timedelta(hours=1)]
    if len(train) < MIN_TRAIN_HOURS:
        return None

    history = pd.date_range(origin - pd.Timedelta(hours=HISTORY_REQUIRED),
                            origin - pd.Timedelta(hours=1), freq="h", tz="UTC")
    if series.reindex(history).isna().any():
        return None

    idx = pd.date_range(origin, periods=HORIZON, freq="h", tz="UTC")
    actual = series.reindex(idx)
    if actual.isna().any():
        return None

    return train, actual


# ---------------------------------------------------------------- runner

def run(model_names, path="data/load_clean.parquet", use_mlflow=True):
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    wide = df.pivot(index="timestamp", columns="series", values="load_mw").sort_index()

    mlflow = None
    if use_mlflow:
        try:
            import mlflow as _mlflow
            mlflow = _mlflow
            mlflow.set_experiment("load-forecasting-benchmark")
        except Exception as e:
            print("MLflow unavailable ({}), continuing without tracking.".format(e))

    records = []
    fit_times = []

    for spec in model_names:
        if isinstance(spec, str):
            model = M.build([spec])[0]
            name = spec
        else:
            model = spec
            name = model.name
        t_model = time.time()
        print("\n{}".format(name))

        for code in wide.columns:
            s = wide[code].dropna()
            last_refit = None
            n_fits = 0
            n_used = 0
            candidates = make_origins(s.index)

            for origin in candidates:
                fold = build_fold(s, origin)
                if fold is None:
                    continue
                train, actual = fold
                n_used += 1

                if model.needs_fit:
                    key = (origin.year, origin.month)
                    if key != last_refit:
                        t0 = time.time()
                        model.fit(train)
                        fit_times.append(time.time() - t0)
                        last_refit = key
                        n_fits += 1
                        print("    {} refit {:>2} ({:.0f}s)".format(code, n_fits, fit_times[-1]),
                              end="\r", flush=True)

                pred = np.asarray(model.predict(train, HORIZON), dtype=float)
                y = actual.values
                scale = mase_scale(train.values)

                records.append({
                    "series": code, "origin": origin, "model": name,
                    "MASE": mae(y, pred) / scale,
                    "MAE": mae(y, pred),
                    "RMSE": rmse(y, pred),
                    "sMAPE": smape(y, pred),
                })

            print("  {}  {}/{} folds, {} refits".format(
                code, n_used, len(candidates), n_fits))

        elapsed = time.time() - t_model
        print("  total {:.1f}s".format(elapsed))

        if mlflow:
            sub = pd.DataFrame([r for r in records if r["model"] == name])
            with mlflow.start_run(run_name=name):
                mlflow.log_param("model", name)
                mlflow.log_param("horizon_hours", HORIZON)
                mlflow.log_param("origin_step_hours", ORIGIN_STEP)
                mlflow.log_param("test_year", TEST_YEAR)
                mlflow.log_metric("MASE_mean", sub.MASE.mean())
                mlflow.log_metric("MASE_median", sub.MASE.median())
                mlflow.log_metric("MAE_mean", sub.MAE.mean())
                mlflow.log_metric("RMSE_mean", sub.RMSE.mean())
                mlflow.log_metric("sMAPE_mean", sub.sMAPE.mean())
                mlflow.log_metric("runtime_seconds", elapsed)
                for series_code, g in sub.groupby("series"):
                    mlflow.log_metric("MASE_{}".format(series_code), g.MASE.mean())

    return pd.DataFrame(records)


def report(results):
    print("\n" + "=" * 74)
    print("MEAN MASE BY SERIES  (lower is better; 1.0 = seasonal naive in-sample)")
    print("=" * 74)
    pivot = results.pivot_table(index="model", columns="series", values="MASE", aggfunc="mean")
    pivot["OVERALL"] = results.groupby("model").MASE.mean()
    print(pivot.sort_values("OVERALL").round(3).to_string())

    print("\n=== WINS PER SERIES (how many series each model ranks first on) ===")
    per_series = results.groupby(["series", "model"]).MASE.mean().reset_index()
    winners = per_series.loc[per_series.groupby("series").MASE.idxmin()]
    print(winners.set_index("series").round(3).to_string())

    print("\n=== MEAN MAE IN MW ===")
    print(results.pivot_table(index="model", columns="series", values="MAE",
                              aggfunc="mean").round(0).to_string())

    print("\n=== RANK STABILITY (mean rank across the 6 series) ===")
    ranks = per_series.pivot(index="model", columns="series", values="MASE").rank()
    ranks["mean_rank"] = ranks.mean(axis=1)
    ranks["worst_rank"] = ranks.iloc[:, :-1].max(axis=1)
    print(ranks.sort_values("mean_rank").round(2).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        "seasonal_naive_168h", "daily_naive_24h", "seasonal_mean_4w",
        "harmonic_ols", "ridge_lags", "xgboost_lags",
    ])
    ap.add_argument("--out", default="data/results.parquet")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    results = run(args.models, use_mlflow=not args.no_mlflow)
    report(results)
    results.to_parquet(args.out, index=False)
    print("\nSaved {}".format(args.out))
