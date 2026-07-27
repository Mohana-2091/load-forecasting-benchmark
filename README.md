# Does Deep Learning Actually Win at Load Forecasting?

A benchmark of eight day-ahead forecasting methods across six European
electricity grids, evaluated over ~360 rolling forecast origins.

The short answer: **no.** A recurrent network finished seventh of eight, behind
a trivial baseline. A single linear layer beat it on every series. And two
separate bugs in the *evaluation design* — not the models — each reversed the
conclusion before they were found.

---

## The question

Deep learning is the default recommendation for time series forecasting. LSTMs
and Transformers dominate the literature and most tutorials. But a
[2022 paper](https://arxiv.org/abs/2205.13504) showed that a single linear
layer outperformed many Transformer architectures on standard benchmarks,
suggesting a lot of reported gains do not survive careful evaluation.

This project tests that on a domain where forecasting has real operational
stakes: day-ahead electricity demand, the forecast grid operators use to commit
generation for the following day.

---

## Data

**Source:** [Open Power System Data](https://data.open-power-system-data.org/time_series/),
time series package v2020-10-06, sourced from the ENTSO-E Transparency Platform.

Six national series, 2015–2019, hourly — 262,000 observations. They were chosen
for genuinely different demand behaviour, because a method that wins on one
series has not been shown to generalise:

| Series | Grid | Regime | Jan / Jul load |
|---|---|---|---|
| DE | Germany | Industrial, continental | 106% / 96% |
| FR | France | Electric heating, extreme winter | 129% / 85% |
| ES | Spain | Mediterranean, dual peak | 106% / 106% |
| GB | Great Britain | Maritime, mild winter peak | 113% / 91% |
| IT | Italy | Mediterranean, **summer** peak | 103% / 114% |
| NO | Norway | Nordic hydro, extreme winter | 127% / 74% |

2020 was excluded deliberately: COVID lockdowns distorted load in a way that
would have measured pandemic response rather than forecasting skill.

### Data cleaning, and the cleaning that was undone

The GB series contained a minimum of 1,183 MW against a mean of 36,453 MW — a
50× peak-to-trough ratio where every other series sat at 2–3×. National grid
demand does not fall to 3% of average; these are ENTSO-E reporting failures.

The first cleaning rule compared each value to its hour-of-week median across
all five years and flagged anything outside 55–170%. It removed the GB dropouts
correctly. It also flagged 114 French and 27 Norwegian values as "too high" and
85 Italian values as "too low".

Those were not errors. France and Norway run heavy electric heating, so a cold
snap genuinely pushes demand far above the annual median. Italy largely shuts
down for Ferragosto in August, so demand genuinely collapses. **The rule was
removing real extremes — which for a forecasting benchmark is worse than
leaving dirt in, because the extremes are exactly what the models should be
judged on.** It was flattening the problem, not cleaning it.

The corrected rule compares each value to a *local* reference — a centred
rolling median over the same hour within a ±10 day window — and removes low
outliers only. Result: 33 points removed out of 262,000, and every maximum
preserved.

---

## Evaluation design

This is where the project spends most of its care, because the design decides
whether the comparison means anything.

| Choice | Why |
|---|---|
| **24-hour horizon** | Day-ahead is the real operational problem. One-step-ahead looks impressive and is nearly useless — at h=1 a "last value" forecast is already very strong. |
| **Rolling origin, ~60 per series** | A single train/test split gives one number from one arbitrary period. 360 origins let a ranking be tested for consistency rather than asserted. |
| **Six-day origin spacing** | See below. This one mattered. |
| **Expanding window, monthly refit — same cadence for all** | Refitting gradient boosting at every origin while training the LSTM once would invalidate the comparison. |
| **MASE as headline metric** | MAPE is asymmetric, unstable at low load, and not comparable across grids of different size. MASE scales by seasonal-naive in-sample error, so 1.0 = seasonal naive and values compare across series. |
| **Identical features for every learned model** | All lags ≥24h, so every feature is knowable when a midnight day-ahead forecast is issued. |
| **No tuning on the test period** | Hyperparameters fixed a priori. Searching them against 2019 would be selecting on the answer. |

### Bug 1 — weekly origin spacing

The first version rolled the origin forward seven days at a time. That looks
natural. It is a trap: 1 January 2019 was a Tuesday, so **every origin was a
Tuesday.** The benchmark only ever tested weekday-to-weekday transitions, and
never once evaluated a Saturday, a Sunday, or the Sunday-to-Monday jump — the
hardest cases for any method leaning on yesterday.

The result flipped completely once fixed:

| Model | 7-day origins | 6-day origins |
|---|---|---|
| daily_naive_24h | **0.763** (best) | **1.340** (worst) |
| seasonal_naive_168h | 1.011 | 1.040 (best baseline) |

A six-day step advances the weekday by one each time, covering all seven evenly
(47–54 origins each). The original finding was an artifact of a single parameter.

### Bug 2 — unequal feature access

The first sequence-model run gave the LSTM 168 raw load values while the tabular
models received hour-of-day, day-of-week and annual Fourier terms. The LSTM
scored MASE 1.166 against XGBoost's 0.503, which looked decisive and was not:
the LSTM had to *infer* from raw numbers what the other models were simply told.

That violated the benchmark's own fairness rule. Both sequence models now
receive the identical calendar frame, flattened across the 24 target hours.
Calendar features are deterministic and knowable arbitrarily far ahead, so this
is not leakage.

Fixing it changed the LSTM substantially per series — DE 1.200→0.922, ES
1.047→0.754, IT 1.132→0.821 — while GB got worse (1.249→2.139). The overall
average barely moved, which is itself a warning about reporting panel averages
without per-series detail.

### Fold eligibility is model-independent

Series were `dropna`'d, so hours lost to ENTSO-E outages are absent from the
index and a lag landing in one silently returns NaN. Skipping those folds
*inside each model* would let the baselines — which need no lags — keep folds
the lag-based models had to drop, and the models would no longer be scored on
the same data.

Instead a fold is usable only if the 528 hours before the origin are complete
and contiguous, checked once and identically for every method. Folds used:
DE 61/61, ES 61/61, IT 61/61, FR 58/61, NO 57/61, GB 49/61.

---

## Results

Mean MASE across all folds. Lower is better; 1.0 = seasonal naive.

| Rank | Model | MASE | vs naive | Runtime | Type |
|---|---|---|---|---|---|
| 1 | **xgboost_lags** | **0.503** | −52% | 71s | Gradient boosting |
| 2 | ridge_lags | 0.685 | −34% | 11s | Linear + lags |
| 3 | dlinear | 0.804 | −23% | 25s | One linear layer |
| 4 | seasonal_naive_168h | 1.040 | — | 0.2s | Baseline |
| 5 | **lstm** | **1.077** | **+4%** | ~590s | Recurrent network |
| 6 | seasonal_mean_4w | 1.181 | +14% | 0.2s | Baseline |
| 7 | harmonic_ols | 1.248 | +20% | 4s | Classical decomposition |
| 8 | daily_naive_24h | 1.340 | +29% | 0.2s | Baseline |

Sequence-model figures are means over three seeds.

### Per series

| Model | DE | ES | FR | GB | IT | NO |
|---|---|---|---|---|---|---|
| xgboost_lags | **0.586** | **0.439** | **0.410** | **0.721** | **0.439** | **0.458** |
| ridge_lags | 0.796 | 0.710 | 0.524 | 0.790 | 0.768 | 0.526 |
| dlinear | 0.736 | 0.570 | 0.792 | 1.374 | 0.644 | 0.824 |
| seasonal_naive_168h | 1.239 | 0.980 | 0.950 | 1.089 | 1.076 | 0.902 |
| lstm | 0.886 | 0.715 | 1.054 | 2.073 | 0.800 | 1.133 |
| harmonic_ols | 1.475 | 1.313 | 1.066 | 1.187 | 1.497 | 0.906 |
| daily_naive_24h | 2.119 | 1.366 | 0.850 | 1.207 | 1.699 | 0.710 |

**XGBoost ranks first on all six series and Ridge second on all six.** That
consistency is what makes the top of the table a result rather than a lucky draw.

---

## Findings

### 1. The recurrent network lost to a trivial baseline

The LSTM finished at 1.077 — worse than copying last week's values (1.040) —
while taking roughly 590 seconds against 0.2. It was the slowest method in the
benchmark and one of the least accurate.

### 2. Removing recurrence made the network better

DLinear and the LSTM received identical windows, identical calendar features,
the same scaler, optimiser, weight decay, epoch budget and seed. The only
difference is architecture.

| Series | DLinear | LSTM | Gap ÷ seed noise |
|---|---|---|---|
| DE | 0.736 | 0.886 | 2.3× |
| ES | 0.570 | 0.715 | 4.2× |
| FR | 0.792 | 1.054 | 3.4× |
| GB | 1.374 | 2.073 | 8.8× |
| IT | 0.644 | 0.800 | 8.0× |
| NO | 0.824 | 1.133 | 2.3× |
| **Overall** | **0.804** | **1.077** | **2.2×** |

DLinear wins on every series by a margin exceeding seed noise everywhere. This
independently reproduces the DLinear paper's central claim in a different domain.

### 3. Feature engineering beat representation learning

XGBoost at 0.503 and Ridge at 0.685 both use the same ~30 hand-built features:
lags at 24/25/26/48/72/168/169/336/504 hours, rolling statistics, and calendar
Fourier terms. Ridge — a closed-form linear solution fitting in 11 seconds —
outperformed the LSTM by 36%.

Ridge and XGBoost use *identical* inputs, so the 0.18 MASE between them is
attributable purely to functional form. The non-linearity is worth something; it
is worth far less than the features.

### 4. Complexity cost reproducibility, not just accuracy

| Model | Seed std | Seed range |
|---|---|---|
| DLinear | 0.0081 | 0.0147 |
| LSTM | 0.0635 | 0.1248 |

**The LSTM is 8× more seed-variable.** On NO it ranged 0.994–1.268 across three
seeds — a spread wide enough to contain the entire DLinear-vs-LSTM gap. In
production that means a retrain can land anywhere in that band with no change to
code or data.

### 5. Pure calendar structure is not enough

`harmonic_ols` — Fourier terms for daily, weekly and annual cycles plus a linear
trend, the textbook decomposition, no lags — finished seventh at 1.248, losing
to seasonal naive. Recent history matters far more than calendar structure on
this problem, which is why every competitive method here is lag-based.

### 6. Regime changes the ranking at the bottom, not the top

`daily_naive` was worst overall (1.340) but ranked third on FR (0.850) and NO
(0.710) — the two extreme winter-peaked series, where day-to-day persistence is
strong. On the other four it ranked last. The top two positions never moved.

## Prediction intervals

Point forecasts are not what a grid operator needs. The reserve capacity that
must be procured — and paid for — is a function of uncertainty, not of the
point estimate. Split conformal prediction is applied identically to every
method, with residual quantiles computed **per lead time** (hour 1 ahead is
easier than hour 23, so a pooled quantile would be too wide early and too
narrow late).

| Model | Coverage | Width (% of load) | Pinball | Point MASE |
|---|---|---|---|---|
| xgboost_lags | 79.1% | **8.5%** | **281** | 0.509 |
| ridge_lags | 79.5% | 11.2% | 376 | 0.682 |
| seasonal_naive_168h | 78.3% | 17.6% | 624 | 1.039 |

Nominal coverage is 80%; all three land within 1.7 points. Because coverage is
comparable, the widths are comparable: **XGBoost achieves the same reliability
as seasonal naive at half the interval width** — same confidence, half the
reserve capacity.

### The calibration set has to be held out, and the failure is instructive

A first implementation calibrated on data the model had already been fit on.
The coverage error then tracked model flexibility almost perfectly:

| Model | Held-out calibration | In-sample calibration | Error |
|---|---|---|---|
| seasonal_naive_168h (nothing fitted) | 78.3% | 78.3% | 0.0 |
| ridge_lags (regularised linear) | 79.5% | 79.0% | −0.5 |
| xgboost_lags (flexible trees) | 79.1% | **58.5%** | **−20.6** |

In-sample residuals understate real error, and they understate it most for the
most flexible model. XGBoost produced intervals half as wide (4.8% of load
against 8.5%) and the best pinball loss on the flawed run — it won the metric
by cheating on it. On NO, coverage fell to 51%: a coin flip sold as 80%
confidence.

For an operator this is not a scoring artifact. An interval advertised at 80%
that delivers 58% means being under-reserved two days in five.

Fixing it costs the point forecast about two months of training data, and the
point MASE barely moved (0.509 against 0.503 in the main benchmark). That is a
cheap price for an interval that means what it says. `--naive-calibration`
reproduces the flawed behaviour for comparison.
---

## What this does not show

- **Not a claim that deep learning never wins.** It is a claim that on day-ahead
  load forecasting, with modest untuned architectures on CPU, it did not — and
  that the baselines it is usually compared against are stronger than they are
  given credit for.
- **No architecture search.** A tuned LSTM, a Transformer, or N-BEATS might do
  better. Tuning against the 2019 evaluation period would have been selecting on
  the answer, so it was not done; that is a deliberate limitation, not an oversight.
- **No exogenous variables.** Temperature drives electricity demand heavily and
  is absent here. Its inclusion could change the ranking, most plausibly by
  favouring models that handle non-linear interactions.
- **Runtimes are indicative only.** One `lstm` seed took 4,660s against ~590s for
  the other two on identical code and data — thermal throttling on a laptop, not
  a model property. The architectural argument (recurrence is inherently
  sequential; a linear layer is one matmul) holds; the exact multiplier does not.
- **One package version, one five-year window, six European grids.** Generality
  beyond that is untested.

---

## Running it

```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download time_series_60min_singleindex.csv from
# https://data.open-power-system-data.org/time_series/2020-10-06/
# and place it in data/

python prep_data.py                                 # clean, scope, outlier removal
python backtest.py                                  # baselines + tabular models
python backtest.py --models dlinear --out data/results_dlinear.parquet
python backtest.py --models lstm    --out data/results_lstm.parquet
python seed_check.py --kinds dlinear lstm --seeds 42 7 2024

mlflow ui                                           # experiment tracking
```

---

## Repository

| File | Purpose |
|---|---|
| `prep_data.py` | Series selection, windowing, dropout detection, gap handling |
| `backtest.py` | Rolling-origin harness, metrics, fold eligibility, MLflow logging |
| `models.py` | All eight methods, shared feature construction, leakage control |
| `seed_check.py` | Multi-seed stability check for the stochastic models |

**Stack:** Python · pandas · NumPy · scikit-learn · XGBoost · PyTorch · MLflow · parquet

---

## Data attribution

Open Power System Data (2020). *Data Package Time series. Version 2020-10-06.*
https://doi.org/10.25832/time_series/2020-10-06
Primary source: ENTSO-E Transparency Platform.
