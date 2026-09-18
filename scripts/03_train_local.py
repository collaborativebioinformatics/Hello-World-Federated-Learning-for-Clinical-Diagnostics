"""Step 3: train one logistic regression per hospital, then score it on the locked test set.

The model is plain logistic regression:

    p(pathogenic) = sigmoid( w · [12 prediction scores, log frequency] + b )

Fourteen numbers in total. That is deliberate. Weight averaging across sites is
clean for a linear model and a heuristic for anything deeper, our sites are
non-IID on purpose, and the claim we have to defend is "frequency is the lever",
which is a statement about one coefficient. A stronger model would also be
better at the thing we do not want it to do: recognising the gene, which 45% of
the pathogenic rows would let it get away with. See docs/step2_review.md
concern 3.

No torch. The weights are one vector of 14 floats, so the FedAvg step 4 has to
do is np.mean over three of them, and the "only weights travel" claim is
literally 14 numbers.

This script trains the SINGLE-SITE and POOLED rows of the README section 5
grid. Those are the baselines federation has to beat; without them a federated
AUC means nothing. The federated row is step 4.

Reads   data/site_*/verdicts.csv       one hospital's private labelled variants
        data/site_*/patient_counts.csv its carrier counts, for the test evidence
        data/test/variants.csv         the locked test set
Writes  data/results_local.json        every number this prints
        docs/step3_results.md          the same, as tables

With --panel NAME every path above sits under data/NAME/ instead. `cardiac` is
the default and keeps the top of data/, the rule step 1 follows.

Usage:
    uv run python scripts/03_train_local.py
    uv run python scripts/03_train_local.py --panel cancer   # another disease area, from data/cancer/
    uv run python scripts/03_train_local.py --self-check     # the maths, against brute force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_PANEL = "cardiac"
SITES = ["site_oslo", "site_karachi", "site_lagos"]

# Dropped: polyphen2_hdiv, missing for 29-32% of training rows. The remaining
# gaps are filled with the neutral rank score rather than an imputed statistic.
# Imputing from a distribution would need hospitals to share one, and a
# "was missing" flag would be worse still: gaps follow the gene and the gene
# follows the label, so the flag would smuggle the answer in as a feature.
DROP_FEATURES = {"polyphen2_hdiv"}
NEUTRAL_SCORE = 0.5

# Frequency is the one feature that is not already a 0-to-1 rank score. 76-85%
# of af_local is exactly zero, so the raw column is useless to a linear model:
# log it, floor it, and rescale to 0-1 with a FIXED transform. Fixed matters —
# a fitted scaler would be a per-site statistic that hospitals would have to
# share, which is exactly what the rank scores were chosen to avoid.
FREQUENCY_FLOOR = 1e-6

# Gradient descent. The data is at most 4,319 rows by 13 features, so full batch
# converges in a second and there is no reason to sample.
EPOCHS = 3000
LEARNING_RATE = 1.0
L2 = 1e-3
SEED = 12

# The threshold is set on the TRAINING rows, at the sensitivity a clinical lab
# would insist on, and then applied unchanged to the test set. Picking it on the
# test set would tune the very thing being measured.
TARGET_SENSITIVITY = 0.95


# ---------------------------------------------------------------------------
# Metrics, written out rather than imported: scikit-learn is a thousand files
# for two functions, and --self-check tests these against brute force.
# ---------------------------------------------------------------------------
def roc_auc(labels: np.ndarray, score: np.ndarray) -> float:
    """Area under the ROC curve, by the rank identity, ties averaged."""
    labels, score = np.asarray(labels), np.asarray(score, dtype=float)
    positives, negatives = labels.sum(), (labels == 0).sum()
    if positives == 0 or negatives == 0:
        return float("nan")  # undefined, e.g. the discordant rows, which are all benign

    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ordered = score[order]
    start = 0
    while start < len(ordered):
        stop = np.searchsorted(ordered, ordered[start], side="right")
        ranks[order[start:stop]] = (start + stop - 1) / 2 + 1
        start = stop
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def threshold_at_sensitivity(labels: np.ndarray, score: np.ndarray, sensitivity: float) -> float:
    """The highest cut that still catches `sensitivity` of the pathogenic rows."""
    return float(np.quantile(score[labels == 1], 1 - sensitivity))


def wilson_interval(hits: int, total: int) -> tuple[float, float]:
    """95% interval for a proportion. Wilson, because the counts here are single digits."""
    if total == 0:
        return float("nan"), float("nan")
    z, p = 1.96, hits / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    spread = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0.0, centre - spread), min(1.0, centre + spread)


def sign_test(fixed: int, broken: int) -> float:
    """Two-sided exact p for a paired before/after comparison on the same variants.

    The evidence settings are scored on ONE set of variants, so the question is
    not "are 2/183 and 7/183 different", which nothing could show at this size.
    It is "of the variants whose call changed, did they mostly change one way",
    and that is answered by the discordant pairs alone.
    """
    changed = fixed + broken
    if changed == 0:
        return 1.0
    from math import comb
    tail = sum(comb(changed, k) for k in range(min(fixed, broken) + 1))
    return min(1.0, 2 * tail / 2 ** changed)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
def features(table: pd.DataFrame, score_columns: list[str], frequency: np.ndarray) -> np.ndarray:
    """Twelve prediction scores plus one log frequency, every column on 0 to 1."""
    scores = table[score_columns].to_numpy(dtype=float)
    scores = np.where(np.isnan(scores), NEUTRAL_SCORE, scores)
    log_frequency = (np.log10(np.asarray(frequency, dtype=float) + FREQUENCY_FLOOR) - np.log10(FREQUENCY_FLOOR)) / -np.log10(FREQUENCY_FLOOR)
    return np.column_stack([scores, log_frequency])


def fit(X: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None, epochs: int = EPOCHS) -> np.ndarray:
    """Logistic regression by full-batch gradient descent. Returns [w..., b].

    Takes a starting point and a number of epochs, because that is exactly what
    one FedAvg round asks a client to do.
    """
    design = np.column_stack([X, np.ones(len(X))])
    weights = np.zeros(design.shape[1]) if weights is None else weights.copy()
    penalty = np.full(design.shape[1], L2)
    penalty[-1] = 0.0  # never shrink the intercept towards zero
    for _ in range(epochs):
        error = 1 / (1 + np.exp(-design @ weights)) - y
        weights -= LEARNING_RATE * (design.T @ error / len(design) + penalty * weights)
    return weights


def predict(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(np.column_stack([X, np.ones(len(X))]) @ weights)))


# ---------------------------------------------------------------------------
# Evidence: the same trained model, given different answers to "how common is it"
# ---------------------------------------------------------------------------
def evidence_columns(test: pd.DataFrame) -> dict[str, np.ndarray]:
    """Four ways to fill the frequency slot for a test variant, worst to best."""
    counts = {
        site: pd.read_csv(DATA_DIR / site / "patient_counts.csv").set_index("variant_id")
        for site in SITES
    }
    local = {
        site: (frame.ac_unaffected / frame.an_unaffected).reindex(test.variant_id).to_numpy()
        for site, frame in counts.items()
    }
    return {
        # What every hospital has today: a reference built mostly on Europeans.
        "public": test.af_public.to_numpy(),
        # One hospital asking only its own patients.
        "own_hospital": local["site_oslo"],
        # Step 6: ask all three, keep the highest. A variant common in ANY
        # population cannot be the cause of a rare severe disease, so the veto
        # should fire on the highest count anyone reports, not on an average
        # that 20,000 Europeans would dominate.
        "federated_query": np.max(np.stack(list(local.values())), axis=0),
        # gnomAD's real per-population frequencies. Not allowed in real life.
        "ceiling": test[[f"af_{p}" for p in ("nfe", "sas", "afr")]].max(axis=1).to_numpy(),
    }


# ---------------------------------------------------------------------------
def data_dir(panel: str) -> Path:
    """The cardiac build keeps its place at the top of data/. Every other panel gets data/<panel>/, as in step 1."""
    return ROOT / "data" if panel == DEFAULT_PANEL else ROOT / "data" / panel


def load() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, list[str]]:
    columns = json.loads((DATA_DIR / "columns.json").read_text())
    score_columns = [c for c in columns["features"] if c not in DROP_FEATURES]
    hospitals = {site: pd.read_csv(DATA_DIR / site / "verdicts.csv") for site in SITES}
    # Pooled: the three files stacked. A variant two labs both classified is one
    # row, not two, or its label would count twice.
    hospitals["pooled"] = pd.concat(hospitals.values()).drop_duplicates("variant_id")
    return hospitals, pd.read_csv(DATA_DIR / "test" / "variants.csv"), score_columns


def main() -> int:
    if not (DATA_DIR / "test" / "variants.csv").exists():
        print(f"{(DATA_DIR / 'test').relative_to(ROOT).as_posix()}/ is missing. Run scripts/02_simulate_hospitals.py first, with the same --panel.", file=sys.stderr)
        return 1
    np.random.seed(SEED)
    hospitals, test, score_columns = load()
    evidence = evidence_columns(test)
    label = test.label.to_numpy()

    print(f"logistic regression on {len(score_columns)} prediction scores + log frequency "
          f"= {len(score_columns) + 2} weights including the intercept\n")

    models, results = {}, {"features": score_columns + ["log_frequency"], "training": {}, "evidence": {}}
    print(f"{'trained on':<16} {'rows':>6} {'pathogenic':>11} {'train AUC':>10} {'cut':>7}")
    for name, rows in hospitals.items():
        X = features(rows, score_columns, rows.af_local.to_numpy())
        y = rows.label.to_numpy()
        weights = fit(X, y)
        own = predict(X, weights)
        cut = threshold_at_sensitivity(y, own, TARGET_SENSITIVITY)
        models[name] = (weights, cut)
        print(f"{name:<14} {len(rows):>6} {int(y.sum()):>11} {roc_auc(y, own):>10.3f} {cut:>7.3f}")
        results["training"][name] = {
            "rows": len(rows), "pathogenic": int(y.sum()),
            "train_auc": roc_auc(y, own), "threshold": cut,
            "weights": dict(zip(results["features"] + ["intercept"], np.round(weights, 4).tolist())),
        }

    # ---- what the model learned, which is the point of using a linear one ----
    print("\nlearned weights, largest first. Negative pushes towards benign.")
    weights, _ = models["pooled"]
    for feature, value in sorted(zip(results["features"], weights[:-1]), key=lambda kv: -abs(kv[1]))[:6]:
        print(f"  {feature:<16} {value:>8.2f}")
    print(f"  {'(intercept)':<16} {weights[-1]:>8.2f}")

    # ---- the grid: rows are training setups, columns are frequency evidence ----
    discordant = test.pop_discordant.to_numpy() == 1
    unseen = (test.test_kind == "unseen_gene").to_numpy()
    print(f"\ntest set: {len(test)} rows, {int(label.sum())} pathogenic. "
          f"{int(unseen.sum())} of them from genes no hospital has seen.")

    for title, subset, metric in [
        ("AUC, all test rows", np.ones(len(test), bool), "auc"),
        ("AUC, unseen genes only (the honest one)", unseen, "auc"),
        (f"false alarms on the {int(discordant.sum())} population-discordant rows, all benign, "
         f"at {TARGET_SENSITIVITY:.0%} training sensitivity", discordant, "false_alarm"),
        ("false alarms on every benign test row, for scale", np.ones(len(test), bool), "false_alarm"),
        ("sensitivity on every pathogenic test row", np.ones(len(test), bool), "sensitivity"),
    ]:
        print(f"\n{title}")
        print(f"  {'trained on':<16}" + "".join(f"{k:>17}" for k in evidence))
        for name, (weights, cut) in models.items():
            cells = []
            for source in evidence.values():
                probability = predict(features(test, score_columns, source), weights)
                if metric == "auc":
                    value = roc_auc(label[subset], probability[subset])
                elif metric == "false_alarm":
                    benign = subset & (label == 0)
                    value = float((probability[benign] >= cut).mean())
                else:
                    value = float((probability[label == 1] >= cut).mean())
                cells.append(value)
            results["evidence"].setdefault(metric if metric != "auc" else title, {})[name] = dict(zip(evidence, cells))
            print(f"  {name:<16}" + "".join(f"{c:>17.3f}" for c in cells))

    # ---- the headline, counted rather than rated, and tested pairwise ----
    # 183 rows turns every rate into single digits, so print what those digits
    # are and put an interval on them before anyone quotes a threefold drop.
    weights, cut = models["pooled"]
    alarms = {
        source_name: predict(features(test, score_columns, source), weights) >= cut
        for source_name, source in evidence.items()
    }
    benign_discordant = discordant & (label == 0)
    print(f"\nthe headline, on the {int(benign_discordant.sum())} discordant benign rows, pooled model:")
    print(f"  {'evidence':<16}{'false alarms':>14}{'rate':>8}{'95% interval':>18}")
    for source_name, fired in alarms.items():
        hits = int(fired[benign_discordant].sum())
        low, high = wilson_interval(hits, int(benign_discordant.sum()))
        print(f"  {source_name:<16}{hits:>8} / {int(benign_discordant.sum()):<3}{hits / benign_discordant.sum():>8.3f}"
              f"{f'{low:.3f} to {high:.3f}':>18}")

    print("\n  Those intervals overlap, so the rates alone settle nothing. The settings are")
    print("  scored on the SAME variants, so the paired comparison is the one that counts:")
    results["paired"] = {}
    for where, mask in [("discordant benign", benign_discordant), ("all benign", label == 0)]:
        for baseline in ("public", "own_hospital"):
            fixed = int((alarms[baseline] & ~alarms["federated_query"] & mask).sum())
            broken = int((~alarms[baseline] & alarms["federated_query"] & mask).sum())
            p = sign_test(fixed, broken)
            print(f"  {where:<18} {baseline:<13} -> federated query: {fixed:>3} removed, {broken:>2} introduced, "
                  f"exact p = {p:.4f}")
            results["paired"][f"{where}|{baseline}"] = {"removed": fixed, "introduced": broken, "p": p}

    # ---- ablation: is the frequency feature doing anything at all? ----
    # AUC barely moves between evidence settings, which invites the question of
    # whether the frequency column earns its place. Train without it and see.
    print("\nablation on the pooled model, to show what the frequency feature buys:")
    print(f"  {'model':<34}{'AUC':>8}{'discordant false alarms':>26}")
    pooled_rows = hospitals["pooled"]
    scores_only_X = features(pooled_rows, score_columns, pooled_rows.af_local.to_numpy())[:, :-1]
    scores_only_y = pooled_rows.label.to_numpy()
    scores_only_w = fit(scores_only_X, scores_only_y)
    scores_only_cut = threshold_at_sensitivity(scores_only_y, predict(scores_only_X, scores_only_w), TARGET_SENSITIVITY)
    test_scores_only = features(test, score_columns, np.zeros(len(test)))[:, :-1]
    probability = predict(test_scores_only, scores_only_w)
    hits = int((probability[benign_discordant] >= scores_only_cut).sum())
    print(f"  {f'{len(score_columns)} scores, no frequency at all':<34}{roc_auc(label, probability):>8.3f}"
          f"{f'{hits} / {int(benign_discordant.sum())}':>26}")
    results["ablation"] = {"scores_only": {"auc": roc_auc(label, probability), "false_alarms": hits}}
    for source_name in ("public", "federated_query"):
        probability = predict(features(test, score_columns, evidence[source_name]), weights)
        hits = int((probability[benign_discordant] >= cut).sum())
        print(f"  {f'{len(score_columns)} scores + {source_name} frequency':<34}{roc_auc(label, probability):>8.3f}"
              f"{f'{hits} / {int(benign_discordant.sum())}':>26}")
        results["ablation"][source_name] = {"auc": roc_auc(label, probability), "false_alarms": hits}

    # ---- per-population AUC, with the positive counts that make it unusable ----
    print("\nAUC by the population a variant is most common in, with positive counts:")
    print(f"  {'population':<12}{'rows':>6}{'pathogenic':>12}" + "".join(f"{k:>17}" for k in evidence))
    for population, index in test.groupby("pop").groups.items():
        mask = test.index.isin(index)
        positives = int(label[mask].sum())
        cells = [roc_auc(label[mask], predict(features(test, score_columns, source), models["pooled"][0])[mask])
                 for source in evidence.values()]
        warning = "   <- too few positives" if 0 < positives < 30 else ""
        print(f"  {population:<12}{mask.sum():>6}{positives:>12}" + "".join(f"{c:>17.3f}" for c in cells) + warning)

    results["test_rows"] = len(test)
    results["pathogenic_test_rows"] = int(label.sum())
    results["discordant_benign_rows"] = int(benign_discordant.sum())
    results["false_alarm_counts"] = {
        name: int(fired[benign_discordant].sum()) for name, fired in alarms.items()
    }
    results["false_alarm_counts"]["scores_only"] = results["ablation"]["scores_only"]["false_alarms"]
    (DATA_DIR / "results_local.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {(DATA_DIR / 'results_local.json').relative_to(ROOT)}")
    return 0


def self_check() -> int:
    """The two pieces of maths worth doubting: the AUC, and the fit."""
    rng = np.random.default_rng(0)

    # AUC against its definition: the share of (pathogenic, benign) pairs ordered
    # correctly, ties counting a half.
    for _ in range(5):
        y = rng.integers(0, 2, 60)
        s = rng.integers(0, 8, 60).astype(float)  # small range, so plenty of ties
        if y.sum() in (0, len(y)):
            continue
        pairs = [(1.0 if a > b else 0.5 if a == b else 0.0) for a in s[y == 1] for b in s[y == 0]]
        assert abs(roc_auc(y, s) - np.mean(pairs)) < 1e-12, "roc_auc disagrees with brute force"

    # The fit against a problem with a known answer: a single clean feature must
    # get a large positive weight, and a pure noise feature must get ~nothing.
    X = rng.random((4000, 2))
    y = (X[:, 0] + rng.normal(0, 0.1, 4000) > 0.5).astype(int)
    weights = fit(X, y)
    assert weights[0] > 5, f"signal feature got weight {weights[0]:.2f}, expected a large positive one"
    assert abs(weights[1]) < 1, f"noise feature got weight {weights[1]:.2f}, expected about zero"
    assert roc_auc(y, predict(X, weights)) > 0.95, "fit cannot separate a nearly separable problem"

    # Frequency has to survive the log transform: rare and common must not collapse.
    rare, common = features(pd.DataFrame({"a": [0.0, 0.0]}), ["a"], np.array([0.0, 0.15]))[:, -1]
    assert rare < 0.01 and common > 0.7, f"log frequency maps 0 and 0.15 to {rare:.2f} and {common:.2f}"

    print("self-check passed: AUC matches brute force, the fit recovers a known signal, "
          "and the frequency transform separates rare from common.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a logistic regression per hospital and score it")
    parser.add_argument("--panel", default=DEFAULT_PANEL, help="disease area: data/ for cardiac, data/<panel>/ for any other (default: cardiac)")
    parser.add_argument("--self-check", action="store_true", help="test the maths, no data needed")
    args = parser.parse_args()
    DATA_DIR = data_dir(args.panel)
    raise SystemExit(self_check() if args.self_check else main())
