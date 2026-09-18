"""Step 6, checked: what the per-gene line changes, and that the query scores exactly as step 3 does.

Every number is computed at run time. Nothing under data/ is written.

    1. Rule 1's line per gene, for every built disease area: how many genes need two bad copies,
       and among the variants in those genes, how many harmful and harmless ones rule 1 clears on
       the old 0.1% line and on the gene's own 1% line. Rule 1 alone, all hospitals plus the public
       database, counts under 5 hidden, so that the counts among sick patients play no part.
    2. The model in the query against step 3: the heart test set is scored through the query's own
       feature pipeline and the numbers step 3 saved in results_local.json are reproduced. The saved
       weights are rounded to four decimals, so the pooled model is also refitted with step 3's own
       fit() and its unrounded weights sent through the same pipeline.

Reads   config/<area>_gene_panel.txt, data/<area>/ for every built area, data/test/variants.csv,
        data/results_local.json, and scripts/03_train_local.py (imported, read only)
Usage:
    uv run python scripts/06_check_query_rules.py
    uv run python scripts/06_check_query_rules.py --json PATH   # the same numbers as a file
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import hospital_query as hq

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. The per-gene line
# ---------------------------------------------------------------------------
def verdicts_of(area: str) -> pd.Series:
    """variant_id -> ClinVar verdict, from the area's missense table and its other-types table when built."""
    tables = [pd.read_csv(hq.area_folder(area) / "variants.csv", usecols=["variant_id", "verdict"])]
    other = hq.other_types_folder(area) / "variants.csv"
    if other.exists():
        tables.append(pd.read_csv(other, usecols=["variant_id", "verdict"]))
    return pd.concat(tables, ignore_index=True).drop_duplicates("variant_id").set_index("variant_id").verdict


def line_check(area: str) -> dict:
    hq.set_area(area)
    two_copy = sorted(gene for gene in hq._gene_modes() if hq.needs_two_copies(gene))
    table = hq.overview(None, hq.DEFAULT_MIN_COUNT)
    public = pd.Series(hq._public_reference()).reindex(table.index).fillna(0.0)
    rows = table[table.gene.isin(two_copy)].assign(
        most_reported=pd.concat([table.best_frequency, public], axis=1).max(axis=1),  # the most any source reports
        verdict=verdicts_of(area).reindex(table.index),
    )
    out = {"genes": len(hq._gene_modes()), "two_copy_genes": len(two_copy), "variants_in_two_copy_genes": len(rows)}
    for verdict, word in (("pathogenic", "harmful"), ("benign", "harmless")):
        of_kind = rows[rows.verdict == verdict]
        old = of_kind[of_kind.most_reported >= hq.TOO_COMMON]
        new = of_kind[of_kind.most_reported >= hq.TOO_COMMON_TWO_COPIES]
        out[word] = {"variants": len(of_kind), "cleared_on_0.1%": len(old), "cleared_on_1%": len(new)}
        if verdict == "pathogenic":
            out[word]["names_cleared_on_0.1%"] = [
                f"{row['name']} ({row.most_reported:.2%}{', still cleared on 1%' if row.most_reported >= hq.TOO_COMMON_TWO_COPIES else ''})"
                for _, row in old.sort_values("most_reported", ascending=False).iterrows()
            ]
    return out


def print_line_check(area: str, numbers: dict) -> None:
    print(f"\n{hq.area_title(area)} ({area}): {numbers['genes']:,} genes, {numbers['two_copy_genes']:,} need two bad copies, "
          f"{numbers['variants_in_two_copy_genes']:,} variants in those genes")
    for word in ("harmful", "harmless"):
        n = numbers[word]
        print(f"  {word:<9} {n['variants']:>7,} variants   cleared by rule 1 on the old 0.1% line: {n['cleared_on_0.1%']:>6,}   "
              f"on the gene's 1% line: {n['cleared_on_1%']:>5,}")
    named = numbers["harmful"]["names_cleared_on_0.1%"]
    if named:
        shown = named if len(named) <= 12 else named[:12] + [f"and {len(named) - 12} more"]
        print("  harmful ones the old line cleared: " + ", ".join(shown))


# ---------------------------------------------------------------------------
# 2. The model in the query against step 3
# ---------------------------------------------------------------------------
def load_step3():
    spec = importlib.util.spec_from_file_location("step3", ROOT / "scripts" / "03_train_local.py")
    step3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step3)
    return step3


def model_check() -> dict:
    hq.set_area(hq.DEFAULT_AREA)
    step3 = load_step3()
    saved = json.loads((hq.area_folder() / hq.MODEL_LOCAL_FILE).read_text(encoding="utf-8"))
    test = pd.read_csv(hq.area_folder() / "test" / "variants.csv")
    label = test.label.to_numpy()
    evidence = step3.evidence_columns(test)
    discordant_benign = (test.pop_discordant.to_numpy() == 1) & (label == 0)
    unseen = (test.test_kind == "unseen_gene").to_numpy()

    model = hq._model()
    scores = model.features[:-1]

    def through_the_query(weights: hq.Model, frequency: np.ndarray) -> np.ndarray:
        return np.array([weights.probability(hq.model_scores(v, scores), float(f)) for v, f in zip(test.variant_id, frequency)])

    out = {"model_wired": model.source, "features_match_step_3": model.features == saved["features"], "settings": {}}
    mine = np.array([[hq.MODEL_NEUTRAL_SCORE if np.isnan(s) else s for s in hq.model_scores(v, scores)] + [hq.log_frequency(f)]
                     for v, f in zip(test.variant_id, evidence["federated_query"])])
    out["feature_rows_max_difference"] = float(np.abs(mine - step3.features(test, scores, evidence["federated_query"])).max())

    # step 3's pooled model refitted, so that the unrounded weights can be compared too
    hospitals, _, score_columns = step3.load()
    np.random.seed(step3.SEED)
    pooled = hospitals["pooled"]
    unrounded = step3.fit(step3.features(pooled, score_columns, pooled.af_local.to_numpy()), pooled.label.to_numpy())
    out["refit_rounds_to_the_saved_weights"] = bool(np.allclose(np.round(unrounded, 4), model.weights + [model.intercept]))
    exact = hq.Model("pooled", model.features, unrounded[:-1].tolist(), float(unrounded[-1]), model.threshold)

    for name, frequency in evidence.items():
        p_saved_weights = through_the_query(model, frequency)
        p_unrounded = through_the_query(exact, frequency)
        out["settings"][name] = {
            "false_alarms": int((p_saved_weights[discordant_benign] >= model.threshold).sum()),
            "false_alarms_saved": saved["false_alarm_counts"][name],
            "auc_unseen_genes": step3.roc_auc(label[unseen], p_saved_weights[unseen]),
            "auc_unseen_genes_saved": saved["evidence"]["AUC, unseen genes only (the honest one)"]["pooled"][name],
            "sensitivity": float((p_saved_weights[label == 1] >= model.threshold).mean()),
            "sensitivity_saved": saved["evidence"]["sensitivity"]["pooled"][name],
            "auc_all_rows_saved_weights": step3.roc_auc(label, p_saved_weights),
            "auc_all_rows_unrounded_weights": step3.roc_auc(label, p_unrounded),
            "auc_all_rows_saved": saved["evidence"]["AUC, all test rows"]["pooled"][name],
        }
    return out


def print_model_check(numbers: dict) -> None:
    print(f"\nthe model in the query: {numbers['model_wired']} model wired; feature list equals step 3's: {numbers['features_match_step_3']}; "
          f"feature rows differ from step 3's by at most {numbers['feature_rows_max_difference']:.0e}")
    print(f"refitting the pooled model with step 3's fit() gives the saved weights once rounded to 4 decimals: {numbers['refit_rounds_to_the_saved_weights']}")
    print(f"  {'evidence':<17}{'false alarms':>13}{'saved':>7}{'AUC unseen':>13}{'saved':>13}{'sensitivity':>13}{'saved':>13}"
          f"{'AUC all, saved w':>18}{'unrounded w':>18}{'saved':>18}")
    exact = True
    for name, n in numbers["settings"].items():
        ok = (n["false_alarms"] == n["false_alarms_saved"] and n["auc_unseen_genes"] == n["auc_unseen_genes_saved"]
              and n["sensitivity"] == n["sensitivity_saved"] and n["auc_all_rows_unrounded_weights"] == n["auc_all_rows_saved"])
        exact &= ok
        print(f"  {name:<17}{n['false_alarms']:>13}{n['false_alarms_saved']:>7}{n['auc_unseen_genes']:>13.10f}{n['auc_unseen_genes_saved']:>13.10f}"
              f"{n['sensitivity']:>13.10f}{n['sensitivity_saved']:>13.10f}{n['auc_all_rows_saved_weights']:>18.13f}"
              f"{n['auc_all_rows_unrounded_weights']:>18.13f}{n['auc_all_rows_saved']:>18.13f}  {'match' if ok else 'DIFFERS'}")
    print("  every saved number reproduced to the last digit" if exact else "  SOMETHING DIFFERS from what step 3 saved")


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Check the per-gene line and the model in the query")
    parser.add_argument("--json", type=Path, help="write the numbers to this file")
    args = parser.parse_args()

    results = {"line_per_area": {}, "model": {}}
    print("rule 1's line per gene: 0.1% when one bad copy is enough, 1% when both copies must be bad")
    for area in hq.available_areas():
        results["line_per_area"][area] = line_check(area)
        print_line_check(area, results["line_per_area"][area])

    if (hq.area_folder(hq.DEFAULT_AREA) / hq.MODEL_LOCAL_FILE).exists():
        results["model"] = model_check()
        print_model_check(results["model"])
    else:
        print("\ndata/results_local.json is missing: run scripts/03_train_local.py to check the model")

    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as problem:
        print(problem, file=sys.stderr)
        raise SystemExit(1)
