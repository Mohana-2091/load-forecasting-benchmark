"""
Prepare hourly electricity load series for the forecasting benchmark.

Source: Open Power System Data, time series package v2020-10-06.
        Underlying data: ENTSO-E Transparency Platform.
        https://doi.org/10.25832/time_series/2020-10-06

Six national series are used rather than one. A method that wins on a single
series has not been shown to generalise, and the point of a benchmark is to
test whether a ranking holds across regimes:

  DE  Germany       - large, industrial, continental climate
  FR  France        - unusually temperature-sensitive (electric heating)
  ES  Spain         - Mediterranean, summer cooling peak
  GB  Great Britain - maritime, mild, winter peak
  IT  Italy         - Mediterranean, strong summer peak, August shutdown
  NO  Norway        - Nordic, hydro-dominated, extreme winter peak

Outlier handling and why the first attempt was wrong
----------------------------------------------------
The GB series contains a minimum of 1,183 MW against a mean of 36,453 MW - a
50x peak-to-trough ratio where the other series sit at 2-3x. National grid
demand does not fall to 3% of average; these are ENTSO-E reporting failures.

A first attempt compared each value to its hour-of-week median computed across
all five years, and flagged anything outside 55-170% of it. That rule removed
the GB dropouts correctly, but it also flagged 114 French and 27 Norwegian
values as "too high" and 85 Italian values as "too low".

Those were not errors. France and Norway run heavy electric heating, so a cold
snap genuinely pushes demand far above the annual median. Italy largely shuts
down for Ferragosto in August, so demand genuinely collapses. The rule was
removing real extremes, which for a forecasting benchmark is worse than leaving
dirt in: the extremes are precisely the events the models should be judged on.
Cleaning them out would have flattened the problem rather than cleaned it.

Two corrections follow from that:

  1. The reference is now LOCAL - a centred rolling median over the same hour
     of day within a +/-10 day window - so a January value is compared against
     other January values and seasonality is not mistaken for error.

  2. Only LOW outliers are removed. The observed failure mode is dropout to
     implausibly small values. High values are physically plausible and are the
     hardest, most interesting part of the forecasting task.
"""

import os
import pandas as pd
import numpy as np

SERIES = {
    "DE": "DE_load_actual_entsoe_transparency",
    "FR": "FR_load_actual_entsoe_transparency",
    "ES": "ES_load_actual_entsoe_transparency",
    "GB": "GB_UKM_load_actual_entsoe_transparency",
    "IT": "IT_load_actual_entsoe_transparency",
    "NO": "NO_load_actual_entsoe_transparency",
}

SRC = "data/time_series_60min_singleindex.csv"
MAX_GAP_HOURS = 3          # interpolate gaps up to this length, leave longer ones
WINDOW_DAYS = 21           # centred window for the local reference (+/-10 days)
LOW_RATIO = 0.50           # below 50% of the local same-hour median -> dropout

print("Loading...")
usecols = ["utc_timestamp"] + list(SERIES.values())
df = pd.read_csv(SRC, usecols=usecols, parse_dates=["utc_timestamp"], low_memory=False)
df = df.rename(columns={v: k for k, v in SERIES.items()})
df = df.rename(columns={"utc_timestamp": "timestamp"})

# 2015-2019 is the reliably covered span. 2020 is excluded on purpose: the
# package release truncates it, and COVID lockdowns distorted load in a way
# that would contaminate a benchmark of ordinary forecasting methods.
START, END = "2015-01-01", "2019-12-31 23:00"
df = df[(df.timestamp >= START) & (df.timestamp <= END)].copy()
df = df.set_index("timestamp")
print("Window: {} to {}  ({:,} hours)".format(START, END, len(df)))

# ------------------------------------------------------------------ outliers
print("\n=== DROPOUT DETECTION (vs local same-hour rolling median) ===")

hour = df.index.hour
rows = []

for code in SERIES:
    s = df[code]

    # Local reference: for each hour-of-day, a centred rolling median across
    # WINDOW_DAYS days. Seasonality is therefore built into the reference.
    reference = pd.Series(index=s.index, dtype=float)
    for h in range(24):
        mask = hour == h
        sub = s[mask]
        reference[mask] = sub.rolling(WINDOW_DAYS, center=True, min_periods=5).median()

    ratio = s / reference
    flagged = ratio < LOW_RATIO

    rows.append({
        "series": code,
        "flagged": int(flagged.sum()),
        "flagged_pct": round(100 * flagged.sum() / s.notna().sum(), 3),
        "min_before": round(float(s.min()), 0),
        "min_after": round(float(s[~flagged].min()), 0),
        "max_kept": round(float(s[~flagged].max()), 0),
    })

    df.loc[flagged, code] = np.nan

print(pd.DataFrame(rows).to_string(index=False))
print("\nNote: maxima are preserved by design - extreme peaks are real demand,")
print("and are the hardest part of the forecasting task.")

# ------------------------------------------------------------------ gaps
print("\n=== MISSING VALUES AFTER DROPOUT REMOVAL ===")
gap_rows = []
for code in SERIES:
    missing = df[code].isna()
    n_missing = int(missing.sum())

    if n_missing:
        runs = missing.ne(missing.shift()).cumsum()[missing]
        gap_lengths = runs.value_counts().values
    else:
        gap_lengths = np.array([])

    gap_rows.append({
        "series": code,
        "missing_hours": n_missing,
        "missing_pct": round(100 * n_missing / len(df), 3),
        "n_gaps": int(len(gap_lengths)),
        "gaps_over_{}h".format(MAX_GAP_HOURS): int((gap_lengths > MAX_GAP_HOURS).sum()) if n_missing else 0,
        "longest_gap_h": int(gap_lengths.max()) if n_missing else 0,
    })

print(pd.DataFrame(gap_rows).to_string(index=False))

# Short gaps are interpolated - defensible for a smooth load curve. Long gaps
# stay NaN so the backtest skips those windows honestly rather than training
# on fabricated values.
for code in SERIES:
    df[code] = df[code].interpolate(method="time", limit=MAX_GAP_HOURS, limit_area="inside")

df = df.reset_index()

# ------------------------------------------------------------------ long format
long = df.melt(id_vars="timestamp", var_name="series", value_name="load_mw")
long = long.dropna(subset=["load_mw"]).sort_values(["series", "timestamp"])

# Calendar features are shared by every model so no method gets an unfair
# information advantage. Anything derived from the target itself (lags, rolling
# statistics) is built inside the backtest loop, never here, to avoid leakage.
long["hour"] = long.timestamp.dt.hour
long["dayofweek"] = long.timestamp.dt.dayofweek
long["month"] = long.timestamp.dt.month
long["dayofyear"] = long.timestamp.dt.dayofyear
long["is_weekend"] = (long.dayofweek >= 5).astype(int)

os.makedirs("data", exist_ok=True)
long.to_parquet("data/load_clean.parquet", index=False)

# ------------------------------------------------------------------ summary
print("\n=== FINAL SERIES ===")
summary = long.groupby("series").agg(
    hours=("load_mw", "size"),
    mean_mw=("load_mw", "mean"),
    min_mw=("load_mw", "min"),
    max_mw=("load_mw", "max"),
)
summary["peak_to_trough"] = (summary.max_mw / summary.min_mw).round(2)
summary["coverage_pct"] = (100 * summary.hours / len(df)).round(2)
print(summary.round(0).to_string())

print("\n=== SEASONALITY (mean load by month, % of series annual mean) ===")
seas = long.groupby(["series", "month"]).load_mw.mean().unstack(0)
print((100 * seas / seas.mean()).round(0).to_string())

print("\nSaved data/load_clean.parquet")
