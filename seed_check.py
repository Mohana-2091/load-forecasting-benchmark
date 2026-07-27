"""
Seed stability check for the two sequence models.

Why this exists
---------------
The headline comparison in this benchmark is DLinear at MASE 0.808 against the
LSTM at 1.133 - a single linear layer beating a recurrent network on every one
of the six series, using identical inputs and 27x less compute.

That is a strong claim, and it rests on one training run of each. Neural
network training is stochastic: weight initialisation, batch shuffling and
optimiser trajectory all depend on the random seed. If re-seeding moved either
model by more than the gap between them, the ranking would be noise dressed up
as a result, and the honest conclusion would be "these two are
indistinguishable on this problem".

The tree and linear models do not need this check - XGBoost is seeded and
deterministic given fixed data, and Ridge has a closed-form solution.

This script re-runs both sequence models under several seeds and reports the
spread. The comparison is only reportable if the between-model gap is clearly
larger than the within-model seed spread.
"""

import argparse
import numpy as np
import pandas as pd

import backtest as B
import models as M


def build_seeded(kind, seed):
    model = M.DLinear(seed=seed) if kind == "dlinear" else M.LSTMForecaster(seed=seed)
    model.name = "{}_seed{}".format(kind, seed)
    return model


def main(kinds, seeds, out):
    instances = [build_seeded(k, s) for k in kinds for s in seeds]
    results = B.run(instances, use_mlflow=False)

    results["kind"] = results.model.str.rsplit("_seed", n=1).str[0]
    results["seed"] = results.model.str.rsplit("_seed", n=1).str[1].astype(int)

    print("\n" + "=" * 74)
    print("MASE BY SEED")
    print("=" * 74)
    by_seed = results.groupby(["kind", "seed"]).MASE.mean().unstack()
    by_seed["mean"] = by_seed.mean(axis=1)
    by_seed["std"] = results.groupby(["kind", "seed"]).MASE.mean().groupby("kind").std()
    by_seed["range"] = by_seed.iloc[:, :len(seeds)].max(axis=1) - by_seed.iloc[:, :len(seeds)].min(axis=1)
    print(by_seed.round(4).to_string())

    print("\n=== PER-SERIES MEAN AND SPREAD ACROSS SEEDS ===")
    per = results.groupby(["kind", "series", "seed"]).MASE.mean().reset_index()
    tbl = per.groupby(["kind", "series"]).MASE.agg(["mean", "std", "min", "max"])
    tbl["range"] = tbl["max"] - tbl["min"]
    print(tbl.round(3).to_string())

    if len(kinds) == 2:
        means = by_seed["mean"]
        gap = abs(means.iloc[0] - means.iloc[1])
        widest = by_seed["range"].max()
        print("\n" + "=" * 74)
        print("Between-model gap : {:.4f} MASE".format(gap))
        print("Widest seed range : {:.4f} MASE".format(widest))
        if gap > 2 * widest:
            print("VERDICT: gap is more than twice the seed spread - the ranking holds.")
        elif gap > widest:
            print("VERDICT: gap exceeds the seed spread, but not by much - report with caution.")
        else:
            print("VERDICT: gap is within seed noise - the two are not distinguishable here.")

        print("\n=== PER-SERIES: DOES THE RANKING HOLD EVERYWHERE? ===")
        wide = per.pivot_table(index="series", columns="kind", values="MASE", aggfunc="mean")
        spread = per.pivot_table(index="series", columns="kind", values="MASE", aggfunc="std")
        wide.columns = [c + "_mean" for c in wide.columns]
        spread.columns = [c + "_std" for c in spread.columns]
        combined = pd.concat([wide, spread], axis=1)
        a, b = kinds
        combined["gap"] = combined["{}_mean".format(b)] - combined["{}_mean".format(a)]
        combined["gap_vs_noise"] = (combined["gap"].abs() /
                                    combined[["{}_std".format(a), "{}_std".format(b)]].max(axis=1))
        print(combined.round(3).to_string())

    results.to_parquet(out, index=False)
    print("\nSaved {}".format(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["dlinear", "lstm"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 2024])
    ap.add_argument("--out", default="data/results_seeds.parquet")
    args = ap.parse_args()
    main(args.kinds, args.seeds, args.out)
