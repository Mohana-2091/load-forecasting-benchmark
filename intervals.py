"""
Prediction intervals for the benchmark.

Why this matters
----------------
Everything measured so far is a point forecast, and a point forecast is not
what a grid operator needs. "Demand will be 45,000 MW" is far less useful than
"45,000 MW, with an 80% chance of falling between 43,100 and 47,400" - the
reserve capacity that must be procured, and paid for, is a function of the
uncertainty, not the point estimate.

Method: split conformal prediction
----------------------------------
The same wrapper is applied to every method rather than building a bespoke
probabilistic model for each, so intervals are compared on the same footing as
the points and no method is advantaged by having native uncertainty while
another has it retrofitted.

At each forecast origin:

  1. The last CALIBRATION_DAYS days of training data are held out.
  2. The model is fit on the PROPER TRAINING SET ONLY - everything before the
     calibration window.
  3. That frozen model forecasts each day of the calibration window.
  4. Absolute residuals are collected PER LEAD TIME. Hour 1 ahead is easier
     than hour 23 ahead, so one pooled quantile would make early hours too
     wide and late hours too narrow.
  5. The conformal quantile of those residuals, at each lead time, becomes the
     interval half-width.

The held-out calibration set, and why the first version was wrong
-----------------------------------------------------------------
A first implementation calibrated on data the model had already been fit on.
The consequences were exactly what overfitting predicts, and the coverage error
tracked model flexibility almost perfectly:

    seasonal_naive  (nothing fitted)        coverage 93.8%   over-covered
    ridge_lags      (regularised linear)    coverage 86.7%   slightly high
    xgboost_lags    (flexible trees)        coverage 67.8%   badly under-covered

In-sample residuals understate real error, and they understate it most for the
most flexible model. XGBoost therefore produced the tightest intervals and the
best pinball loss while delivering 68% coverage against a nominal 80% - it won
the metric by cheating on it. For a grid operator that is not a scoring
artifact: an interval sold as 80% that delivers 68% means being under-reserved
one day in three.

Split conformal requires calibration data the model has not seen. Fitting on
the proper training set costs the point forecast about two months of data,
which is the honest price of a valid interval.

The flawed behaviour can be reproduced with --naive-calibration for comparison.

Metrics
-------
  coverage      share of actuals inside the interval; should match nominal.
  mean width    average width in MW. Lower is better, but only at equal
                coverage - an infinitely wide interval covers perfectly and is
                worth nothing.
  pinball loss  scores calibration and sharpness together, so it cannot be
                gamed by widening alone. Headline metric.

Caveat: conformal guarantees assume exchangeability. Load is autocorrelated and
seasonal, so the guarantee is approximate here. That is why coverage is
measured empirically rather than assumed.
"""

import argparse
import numpy as np
import pandas as pd

import backtest as B
import models as M

CALIBRATION_DAYS = 60
NOMINAL = 0.80                   # target central coverage


def pinball(y, lower, upper, nominal=NOMINAL):
    """Mean pinball loss across the two interval quantiles."""
    alpha = (1 - nominal) / 2
    q_lo, q_hi = alpha, 1 - alpha
    lo = np.where(y >= lower, q_lo * (y - lower), (1 - q_lo) * (lower - y))
    hi = np.where(y >= upper, q_hi * (y - upper), (1 - q_hi) * (upper - y))
    return float(np.mean((lo + hi) / 2))


def conformal_quantile(residuals, nominal=NOMINAL):
    """
    Finite-sample conformal quantile of absolute residuals, per lead time.

    For a symmetric interval with target coverage c, the half-width is the
    c-quantile of |residual| - not the (1 - alpha) quantile. The first version
    used 0.90 here, which alone pushed every interval towards 90% coverage.

    The (n+1)/n adjustment is the standard finite-sample correction.
    """
    n = residuals.shape[0]
    level = min(1.0, np.ceil((n + 1) * nominal) / n)
    return np.quantile(residuals, level, axis=0)


def fit_and_calibrate(name, train, days=CALIBRATION_DAYS, naive=False):
    """
    Returns (fitted model, per-lead-time half-widths).

    naive=True reproduces the original bug: fit on all of train, then calibrate
    on the tail of that same data.
    """
    cal_start = train.index[-1] - pd.Timedelta(hours=24 * days) + pd.Timedelta(hours=1)
    proper = train.loc[:cal_start - pd.Timedelta(hours=1)]

    if len(proper) < B.MIN_TRAIN_HOURS:
        return None, None

    model = M.build([name])[0]
    if model.needs_fit:
        model.fit(train if naive else proper)

    residuals = []
    for d in range(days):
        cut = cal_start + pd.Timedelta(hours=24 * d)
        history = train.loc[:cut - pd.Timedelta(hours=1)]

        target = pd.date_range(cut, periods=B.HORIZON, freq="h", tz="UTC")
        actual = train.reindex(target)
        if actual.isna().any():
            continue

        need = pd.date_range(cut - pd.Timedelta(hours=B.HISTORY_REQUIRED),
                             cut - pd.Timedelta(hours=1), freq="h", tz="UTC")
        if train.reindex(need).isna().any():
            continue

        try:
            pred = np.asarray(model.predict(history, B.HORIZON), dtype=float)
        except Exception:
            continue
        if not np.isfinite(pred).all():
            continue

        residuals.append(np.abs(actual.values - pred))

    if len(residuals) < 20:
        return None, None

    return model, conformal_quantile(np.array(residuals))


def run(model_names, path="data/load_clean.parquet", naive=False):
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    wide = df.pivot(index="timestamp", columns="series", values="load_mw").sort_index()

    records = []
    for name in model_names:
        print("\n{}{}".format(name, "  [naive calibration]" if naive else ""))

        for code in wide.columns:
            s = wide[code].dropna()
            last_refit = None
            model, widths = None, None
            n = 0

            for origin in B.make_origins(s.index):
                fold = B.build_fold(s, origin)
                if fold is None:
                    continue
                train, actual = fold

                key = (origin.year, origin.month)
                if key != last_refit:
                    model, widths = fit_and_calibrate(name, train, naive=naive)
                    last_refit = key
                if widths is None:
                    continue

                pred = np.asarray(model.predict(train, B.HORIZON), dtype=float)
                lower, upper = pred - widths, pred + widths
                y = actual.values
                n += 1

                records.append({
                    "series": code, "origin": origin, "model": name,
                    "coverage": float(np.mean((y >= lower) & (y <= upper))),
                    "mean_width_mw": float(np.mean(upper - lower)),
                    "rel_width": float(np.mean(upper - lower) / np.mean(y)),
                    "pinball": pinball(y, lower, upper),
                    "point_MAE": B.mae(y, pred),
                    "point_MASE": B.mae(y, pred) / B.mase_scale(train.values),
                })

            print("  {}  {} origins".format(code, n))

    return pd.DataFrame(records)


def report(res, naive=False):
    print("\n" + "=" * 80)
    print("INTERVAL QUALITY  (nominal coverage {:.0%}){}".format(
        NOMINAL, "  [NAIVE CALIBRATION - reproduces the bug]" if naive else ""))
    print("=" * 80)

    summary = res.groupby("model").agg(
        coverage=("coverage", "mean"),
        coverage_error=("coverage", lambda c: c.mean() - NOMINAL),
        mean_width_mw=("mean_width_mw", "mean"),
        rel_width=("rel_width", "mean"),
        pinball=("pinball", "mean"),
        point_MASE=("point_MASE", "mean"),
    )
    print(summary.sort_values("pinball").round(4).to_string())

    print("\nPinball rewards intervals that are both calibrated and tight;")
    print("coverage alone can be bought by widening. Compare pinball only")
    print("between models whose coverage is close to nominal.")

    print("\n=== COVERAGE BY SERIES ===")
    print(res.pivot_table(index="model", columns="series",
                          values="coverage", aggfunc="mean").round(3).to_string())

    print("\n=== WIDTH AS % OF MEAN LOAD ===")
    print((100 * res.pivot_table(index="model", columns="series",
                                 values="rel_width", aggfunc="mean")).round(1).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        "seasonal_naive_168h", "ridge_lags", "xgboost_lags",
    ])
    ap.add_argument("--naive-calibration", action="store_true",
                    help="calibrate on data the model was fit on (reproduces the original bug)")
    ap.add_argument("--out", default="data/results_intervals.parquet")
    args = ap.parse_args()

    res = run(args.models, naive=args.naive_calibration)
    report(res, naive=args.naive_calibration)
    res.to_parquet(args.out, index=False)
    print("\nSaved {}".format(args.out))
