"""Step 3, checked: is the headline sturdy, and what does it cost?

Four questions about scripts/03_train_local.py, answered without changing it:

    1. Did its gradient descent reach the optimum?
       Solve the same penalised objective exactly and compare.
    2. Is "2 false alarms" the luck of one simulated cohort?
       Re-draw every hospital's patients many times, model held fixed.
    3. What does the extra frequency evidence cost among pathogenic variants?
    4. What if the hospitals' answers are ADDED to the public reference
       instead of replacing it, which is what a real hospital would do?

It ends with every evidence setting side by side, which is Table 3 of
Manuscript.md, and with how far a single score gets on its own.

Reads   the same files as step 3
Writes  data/results_checks.json   every number this prints

Usage:
    uv run python scripts/03_check_results.py
    uv run python scripts/03_check_results.py --panel cancer   # another disease area, from data/cancer/
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

COHORTS = 200  # fresh sets of simulated patients
SEED = 12
NEWTON_STEPS = 50


def load_step3():
    """Import 03_train_local.py, whose name starts with a digit, so that its own functions are the ones checked."""
    spec = importlib.util.spec_from_file_location("step3", ROOT / "scripts" / "03_train_local.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact_fit(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    """The optimum of step 3's objective by Newton's method, intercept unpenalised as there."""
    design = np.column_stack([X, np.ones(len(X))])
    penalty = np.full(design.shape[1], l2)
    penalty[-1] = 0.0
    weights = np.zeros(design.shape[1])
    for _ in range(NEWTON_STEPS):
        p = 1 / (1 + np.exp(-design @ weights))
        gradient = design.T @ (p - y) / len(y) + penalty * weights
        hessian = (design * (p * (1 - p))[:, None]).T @ design / len(y) + np.diag(penalty)
        weights -= np.linalg.solve(hessian, gradient)
    return weights


def redraw_cohort(test: pd.DataFrame, gene_copies: dict[str, int], population: dict[str, str],
                  rng: np.random.Generator) -> dict[str, np.ndarray]:
    """New unaffected patients at every hospital, drawn the way step 2 draws them."""
    return {
        site: rng.binomial(gene_copies[site], test[population[site]].to_numpy()) / gene_copies[site]
        for site in gene_copies
    }


def summarise(counts: list[int]) -> dict[str, int]:
    return {"median": int(np.median(counts)), "lowest": int(min(counts)), "highest": int(max(counts))}


def main(panel: str) -> int:
    global DATA_DIR
    step3 = load_step3()
    # Step 3's load() and evidence_columns() read from its own DATA_DIR, so both scripts follow the same panel.
    DATA_DIR = step3.DATA_DIR = step3.data_dir(panel)
    hospitals, test, score_columns = step3.load()
    evidence = step3.evidence_columns(test)
    label = test.label.to_numpy()
    benign, pathogenic = label == 0, label == 1
    discordant_benign = (test.pop_discordant.to_numpy() == 1) & benign
    results: dict = {}

    pooled = hospitals["pooled"]
    X = step3.features(pooled, score_columns, pooled.af_local.to_numpy())
    y = pooled.label.to_numpy()
    weights = step3.fit(X, y)
    cut = step3.threshold_at_sensitivity(y, step3.predict(X, weights), step3.TARGET_SENSITIVITY)

    def fires(frequency: np.ndarray, model: np.ndarray = weights, threshold: float = cut) -> np.ndarray:
        return step3.predict(step3.features(test, score_columns, frequency), model) >= threshold

    # ---- 1. the fit ----
    exact = exact_fit(X, y, step3.L2)
    exact_cut = step3.threshold_at_sensitivity(y, step3.predict(X, exact), step3.TARGET_SENSITIVITY)
    differing = sum(int((fires(column) != fires(column, exact, exact_cut)).sum()) for column in evidence.values())
    compared = len(test) * len(evidence)
    largest_gap = float(np.abs(weights - exact).max())
    print("1. the fit")
    print(f"   largest gap between a step 3 coefficient and the exact optimum: {largest_gap:.3f}")
    print(f"   test calls that differ between the two fits: {differing} of {compared:,}")
    results["fit"] = {"largest_coefficient_gap": largest_gap, "calls_differing": differing, "calls_compared": compared}

    # ---- 2. fresh cohorts ----
    columns = json.loads((DATA_DIR / "columns.json").read_text())
    gene_copies = {
        site: int(pd.read_csv(DATA_DIR / site / "patient_counts.csv", usecols=["an_unaffected"]).an_unaffected.iloc[0])
        for site in step3.SITES
    }
    rng = np.random.default_rng(SEED)
    drawn = {"own_hospital": [], "federated_query": [], "public_plus_query": [],
             "federated_query, all benign": [], "public_plus_query, all benign": []}
    for _ in range(COHORTS):
        local = redraw_cohort(test, gene_copies, columns["af_by_site"], rng)
        highest = np.max(np.stack(list(local.values())), axis=0)
        own, query, both = fires(local["site_oslo"]), fires(highest), fires(np.maximum(evidence["public"], highest))
        drawn["own_hospital"].append(int(own[discordant_benign].sum()))
        drawn["federated_query"].append(int(query[discordant_benign].sum()))
        drawn["public_plus_query"].append(int(both[discordant_benign].sum()))
        drawn["federated_query, all benign"].append(int(query[benign].sum()))
        drawn["public_plus_query, all benign"].append(int(both[benign].sum()))

    this_build = {name: fires(column) for name, column in evidence.items()}
    public_discordant = int(this_build["public"][discordant_benign].sum())
    public_all = int(this_build["public"][benign].sum())
    own_this_build = int(this_build["own_hospital"][discordant_benign].sum())
    print(f"\n2. false alarms across {COHORTS} fresh cohorts, same trained model")
    print(f"   fixed, because they do not depend on the patients: public {public_discordant} of {int(discordant_benign.sum())}"
          f" discordant and {public_all} of {int(benign.sum())} benign;"
          f" ceiling {int(this_build['ceiling'][discordant_benign].sum())} and {int(this_build['ceiling'][benign].sum())}")
    print(f"   {'setting':<32}{'median':>8}{'lowest':>8}{'highest':>9}")
    for name, counts in drawn.items():
        row = summarise(counts)
        print(f"   {name:<32}{row['median']:>8}{row['lowest']:>8}{row['highest']:>9}")
    beats_public = float(np.mean(np.array(drawn["federated_query"]) < public_discordant))
    as_bad_as_build = float(np.mean(np.array(drawn["own_hospital"]) >= own_this_build))
    print(f"   cohorts where the federated query beats the public reference on discordant rows: {beats_public:.0%}")
    print(f"   cohorts where own hospital is as bad as in this build ({own_this_build}): {as_bad_as_build:.1%}")
    results["fresh_cohorts"] = {
        "cohorts": COHORTS, "public_discordant": public_discordant, "public_all_benign": public_all,
        "own_hospital_this_build": own_this_build, "share_as_bad_as_this_build": as_bad_as_build,
        "share_query_beats_public": beats_public, **{name: summarise(counts) for name, counts in drawn.items()},
    }

    # ---- 3 and 4. the cost, and adding the answers to the public reference ----
    this_build["public_plus_query"] = fires(np.maximum(evidence["public"], evidence["federated_query"]))
    print("\n3 and 4. calls that change once the hospitals answer, against the public reference alone")
    print(f"   {'setting':<20}{'benign: alarms removed':>24}{'introduced':>12}{'pathogenic: flags lost':>25}{'gained':>8}")
    results["against_public"] = {}
    for name in ("federated_query", "public_plus_query"):
        before, after = this_build["public"], this_build[name]
        row = {
            "alarms_removed": int((before & ~after & benign).sum()),
            "alarms_introduced": int((~before & after & benign).sum()),
            "flags_lost": int((before & ~after & pathogenic).sum()),
            "flags_gained": int((~before & after & pathogenic).sum()),
            "false_alarms_all_benign": int(after[benign].sum()),
            "sensitivity": float(after[pathogenic].mean()),
        }
        results["against_public"][name] = row
        print(f"   {name:<20}{row['alarms_removed']:>24}{row['alarms_introduced']:>12}{row['flags_lost']:>25}{row['flags_gained']:>8}")
    lost = this_build["public"] & ~this_build["public_plus_query"] & pathogenic
    results["against_public"]["pathogenic_flags_lost"] = test.loc[lost, "name"].tolist()
    print(f"   pathogenic variants that lose their flag: {', '.join(test.loc[lost, 'name'])}")

    # ---- the whole comparison in one table, pooled model ----
    # "no frequency" is a separate model trained on the scores alone, as in step 3's ablation.
    unseen = (test.test_kind == "unseen_gene").to_numpy()
    scores_only = step3.fit(X[:, :-1], y)
    scores_only_cut = step3.threshold_at_sensitivity(y, step3.predict(X[:, :-1], scores_only), step3.TARGET_SENSITIVITY)
    test_scores = step3.features(test, score_columns, np.zeros(len(test)))[:, :-1]
    probability = {"no_frequency": step3.predict(test_scores, scores_only)}
    fired = {"no_frequency": probability["no_frequency"] >= scores_only_cut}
    evidence["public_plus_query"] = np.maximum(evidence["public"], evidence["federated_query"])
    for name in ("public", "own_hospital", "federated_query", "public_plus_query", "ceiling"):
        probability[name] = step3.predict(step3.features(test, score_columns, evidence[name]), weights)
        fired[name] = probability[name] >= cut

    print(f"\n5. every setting side by side, pooled model, {int(discordant_benign.sum())} discordant benign, "
          f"{int(benign.sum())} benign and {int(pathogenic.sum())} pathogenic test variants")
    print(f"   {'frequency evidence':<20}{'AUC all':>9}{'AUC unseen genes':>18}{'FP discordant':>15}{'FP benign':>11}{'sensitivity':>13}")
    results["settings"] = {}
    for name, scored in probability.items():
        row = {
            "auc_all": step3.roc_auc(label, scored),
            "auc_unseen_genes": step3.roc_auc(label[unseen], scored[unseen]),
            "false_positives_discordant": int(fired[name][discordant_benign].sum()),
            "false_positives_benign": int(fired[name][benign].sum()),
            "sensitivity": float(fired[name][pathogenic].mean()),
        }
        results["settings"][name] = row
        print(f"   {name:<20}{row['auc_all']:>9.3f}{row['auc_unseen_genes']:>18.3f}{row['false_positives_discordant']:>15}"
              f"{row['false_positives_benign']:>11}{row['sensitivity']:>13.3f}")

    # ---- how far does one score get on its own? ----
    alone = {
        column: step3.roc_auc(label, test[column].fillna(step3.NEUTRAL_SCORE).to_numpy()) for column in score_columns
    }
    best = sorted(alone.items(), key=lambda item: -item[1])[:3]
    print("\n6. AUC of a single score on the test set, best three: " + ", ".join(f"{k} {v:.3f}" for k, v in best))
    results["single_score_auc"] = alone

    (DATA_DIR / "results_checks.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {(DATA_DIR / 'results_checks.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check how sturdy the step 3 headline is, and what it costs")
    parser.add_argument("--panel", default="cardiac", help="disease area: data/ for cardiac, data/<panel>/ for any other (default: cardiac)")
    raise SystemExit(main(parser.parse_args().panel))
