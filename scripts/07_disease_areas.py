"""Step 7: one federated pipeline per disease area, on the three ancestry hospitals.

The question: a patient of South Asian or African ancestry walks into Oslo. Does
Oslo's model, trained only on what Oslo classified, serve them as well as a model
trained with the other hospitals (FedAvg), or with everything pooled? Step 4
answered this for heart genes only. This repeats it for every disease area at
once, so the answer can be drawn as one plot: x = disease area, y = AUC, three
lines for Oslo only / federated / pooled.

Every area is a subset of the all-panels build (data/all): the same three
hospitals, the same patient cohorts, the same locked test set, cut down to the
genes on that area's PanelApp panels. Hospitals re-deal their verdicts under
seeds 100.. as step 4 did, so the error bars mean the same thing they meant there.
Frequency evidence is the federated count query for every arm, as in step 4's
headline, so the only thing that differs between arms is the training.

FedAvg is numpy, 20 rounds of 150 local steps, weighted by row count. Step 4
showed this matches the NVFlare run to three decimals.

Reads   data/all/variants.csv, data/all/test/, data/all/site_*/patient_counts.csv,
        config/all_gene_panel.txt
Writes  data/all/results_disease_areas.json
        docs/disease_areas_figure.png, docs/disease_areas_table.md

Usage:
    python scripts/07_disease_areas.py            # train and plot
    python scripts/07_disease_areas.py --plot-only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "all"
RESULTS = DATA_DIR / "results_disease_areas.json"
FIGURE = ROOT / "docs" / "disease_areas_figure.png"
TABLE = ROOT / "docs" / "disease_areas_table.md"


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


step2 = load("step2", "02_simulate_hospitals.py")
step3 = load("step3", "03_train_local.py")
step3.DATA_DIR = DATA_DIR
SITES = list(step2.SITES)

# Disease areas as sets of PanelApp signed-off panel ids (config/all_panel_versions.json
# has the names). A gene on any listed panel belongs to the area; genes can sit in
# more than one area, which is biology, not an error.
AREAS = {
    "cardiac": [49, 652, 842, 700, 772, 502],
    "cancer": [635, 143, 504, 503, 524, 1223, 521, 522, 648],
    "neurology": [402, 285, 466, 477, 488, 474, 1669, 96, 476, 496, 579, 567, 568, 540, 847],
    "neuromuscular": [185, 207, 225, 235, 465, 232, 542, 846],
    "metabolic": [467, 25, 528, 529, 112],
    "renal": [106, 283, 292, 487, 548, 678, 725, 149, 1537],
    "eye": [307, 509, 230, 186, 658, 722, 511],
    "haematology": [545, 518, 519, 516, 157, 1397, 508],
    "immunology": [398, 1075],
    "skeletal": [309, 196, 1471],
    "skin": [553, 554, 555, 556, 559, 560, 563, 565],
    "hearing": [126],
}
ARMS = ["oslo_only", "federated", "pooled"]
ROUNDS, LOCAL_EPOCHS = 20, 150
SEEDS = [100, 101, 102]
TRAVELLING = ("sas", "afr")  # test rows whose population is not Oslo's


def genes_by_area() -> dict[str, set[str]]:
    area_of = {pid: area for area, ids in AREAS.items() for pid in ids}
    members: dict[str, set[str]] = {area: set() for area in AREAS}
    for line in (ROOT / "config" / "all_gene_panel.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        gene, _, rest = line.partition("#")
        for token in rest.split("|")[0].replace(",", " ").split():
            if token.isdigit() and int(token) in area_of:
                members[area_of[int(token)]].add(gene.strip())
    return members


def fedavg(per_site: list[np.ndarray], rows: list[int]) -> np.ndarray:
    return np.average(np.stack(per_site), axis=0, weights=rows)


def score(test: pd.DataFrame, probability: np.ndarray, cut: float) -> dict:
    label = test.label.to_numpy()
    pop = test["pop"].to_numpy()
    travelling = np.isin(pop, TRAVELLING)
    discordant_benign = (test.pop_discordant.to_numpy() == 1) & (label == 0)
    return {
        "auc": step3.roc_auc(label, probability),
        "auc_travelling": step3.roc_auc(label[travelling], probability[travelling]),
        **{f"auc_{p}": step3.roc_auc(label[pop == p], probability[pop == p]) for p in ("nfe", "sas", "afr")},
        "false_alarms": int((probability[discordant_benign] >= cut).sum()),
        "discordant_benign": int(discordant_benign.sum()),
        "sensitivity": float((probability[label == 1] >= cut).mean()),
    }


def run() -> dict:
    columns = json.loads((DATA_DIR / "columns.json").read_text())
    score_columns = [c for c in columns["features"] if c not in step3.DROP_FEATURES]
    test = pd.read_csv(DATA_DIR / "test" / "variants.csv")
    everything = pd.read_csv(DATA_DIR / "variants.csv")
    pool = everything[~everything.variant_id.isin(set(test.variant_id))].reset_index(drop=True)
    evidence = step3.evidence_columns(test)["federated_query"]
    local = {}
    for site in SITES:
        counts = pd.read_csv(DATA_DIR / site / "patient_counts.csv").set_index("variant_id")
        local[site] = counts.ac_unaffected / counts.an_unaffected
    members = genes_by_area()

    results = {"seeds": SEEDS, "rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS,
               "evidence": "federated_query", "areas": {}}
    for area, genes in members.items():
        in_area = test.gene.isin(genes).to_numpy()
        test_a, evidence_a = test[in_area].reset_index(drop=True), evidence[in_area]
        X_test = step3.features(test_a, score_columns, evidence_a)
        label = test_a.label.to_numpy()
        info = {"genes": len(genes), "test_rows": int(in_area.sum()), "test_pathogenic": int(label.sum()),
                "test_travelling": int(test_a["pop"].isin(TRAVELLING).sum()),
                "test_travelling_pathogenic": int(label[test_a["pop"].isin(TRAVELLING)].sum()),
                "training_rows": {}, "runs": {arm: [] for arm in ARMS}}
        print(f"\n{area}: {len(genes)} genes, {info['test_rows']} test rows ({info['test_pathogenic']} pathogenic), "
              f"{info['test_travelling']} from non-European patients")
        for seed in SEEDS:
            held = step2.deal_verdicts(pool, np.random.default_rng(seed))
            hospitals = {}
            for site in SITES:
                rows = pool[held[site].to_numpy() & pool.gene.isin(genes).to_numpy()].copy()
                rows["af_local"] = local[site].reindex(rows.variant_id).to_numpy()
                hospitals[site] = (step3.features(rows, score_columns, rows.af_local.to_numpy()), rows.label.to_numpy())
            pooled = pd.concat(pool[held[s].to_numpy() & pool.gene.isin(genes).to_numpy()].assign(
                af_local=local[s].reindex(pool[held[s].to_numpy() & pool.gene.isin(genes).to_numpy()].variant_id).to_numpy())
                for s in SITES).drop_duplicates("variant_id")
            hospitals["pooled"] = (step3.features(pooled, score_columns, pooled.af_local.to_numpy()), pooled.label.to_numpy())
            info["training_rows"][seed] = {k: len(y) for k, (_, y) in hospitals.items()}

            models = {}
            models["oslo_only"] = step3.fit(*hospitals["site_oslo"])
            models["pooled"] = step3.fit(*hospitals["pooled"])
            weights = None
            for _ in range(ROUNDS):
                weights = fedavg([step3.fit(*hospitals[s], weights=weights, epochs=LOCAL_EPOCHS) for s in SITES],
                                 [len(hospitals[s][1]) for s in SITES])
            models["federated"] = weights

            for arm in ARMS:
                # Threshold as step 4: on rows the arm may see. Federated averages each site's own quantile.
                if arm == "federated":
                    cut = float(np.average(
                        [step3.threshold_at_sensitivity(y, step3.predict(X, weights), step3.TARGET_SENSITIVITY)
                         for X, y in (hospitals[s] for s in SITES)], weights=[len(hospitals[s][1]) for s in SITES]))
                else:
                    X, y = hospitals["site_oslo" if arm == "oslo_only" else "pooled"]
                    cut = step3.threshold_at_sensitivity(y, step3.predict(X, models[arm]), step3.TARGET_SENSITIVITY)
                info["runs"][arm].append(score(test_a, step3.predict(X_test, models[arm]), cut))
            print(f"  seed {seed}: rows oslo {info['training_rows'][seed]['site_oslo']:>6} pooled {info['training_rows'][seed]['pooled']:>6} | "
                  + "  ".join(f"{arm} {info['runs'][arm][-1]['auc']:.4f}/{info['runs'][arm][-1]['auc_travelling']:.4f}" for arm in ARMS))
        info["summary"] = {arm: {key: [float(np.mean([r[key] for r in info["runs"][arm]])),
                                       float(np.std([r[key] for r in info["runs"][arm]], ddof=1))]
                                 for key in info["runs"][arm][0] if key != "discordant_benign"}
                           for arm in ARMS}
        results["areas"][area] = info
    RESULTS.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {RESULTS.relative_to(ROOT)}")
    return results


def plot(results: dict) -> None:
    """Two panels: the travelling-patient AUC, which is flat, and the false alarms, which are not."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    areas = list(results["areas"])
    # Okabe-Ito blue / vermillion / bluish green. Chosen by running the six
    # categorical-palette checks against a white surface rather than by eye: the
    # previous two-greens-and-black set failed the chroma and contrast floors, and
    # this one clears the colour-vision-deficiency target on every pair, worst
    # 11.0 against a target of 8. Markers differ too, so identity is never colour alone.
    style = {"oslo_only": ("Oslo only", "#D55E00", "o"), "federated": ("Federated (FedAvg)", "#0072B2", "s"),
             "pooled": ("Everything pooled", "#009E73", "^")}
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    panels = [("auc_travelling", "A   AUC on variants of South Asian or African ancestry patients seen at Oslo", "AUC"),
              ("false_alarms", "B   False alarms among population-discordant benign variants (lower is better)", "false alarms, %")]
    for ax, (key, title, ylabel) in zip(axes, panels):
        for arm, (label, colour, marker) in style.items():
            scale = [1.0 if key != "false_alarms" else 100.0 / results["areas"][a]["runs"][arm][0]["discordant_benign"]
                     for a in areas]
            mean = [results["areas"][a]["summary"][arm][key][0] * k for a, k in zip(areas, scale)]
            sd = [results["areas"][a]["summary"][arm][key][1] * k for a, k in zip(areas, scale)]
            ax.errorbar(range(len(areas)), mean, yerr=sd, label=label, color=colour, marker=marker,
                        capsize=3, linewidth=1.5, markersize=6)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_xticks(range(len(areas)),
                       [f"{a}\nn={results['areas'][a]['test_rows']}" for a in areas], fontsize=8.5)
    axes[0].legend(frameon=False, ncol=3, loc="lower left")
    fig.suptitle("Federated training, one pipeline per clinical specialty: three ancestry hospitals, "
                 f"{len(results['seeds'])} partitions, error bars are standard deviations", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=160)
    print(f"wrote {FIGURE.relative_to(ROOT)}")


def table(results: dict) -> None:
    """The numbers behind the figure, as markdown, so the tables cannot drift from the run."""
    arms = ["oslo_only", "federated", "pooled"]
    lines = ["| Disease area | Genes | Test variants | Travelling variants | Travelling pathogenic "
             "| AUC, Oslo alone | AUC, federated | AUC, pooled "
             "| False alarms, Oslo alone | False alarms, federated | False alarms, pooled | Discordant benign |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for area, info in results["areas"].items():
        s = info["summary"]
        auc = "".join(f" {s[a]['auc_travelling'][0]:.4f} ({s[a]['auc_travelling'][1]:.4f}) |" for a in arms)
        alarms = "".join(f" {s[a]['false_alarms'][0]:.1f} |" for a in arms)
        lines.append(f"| {area} | {info['genes']} | {info['test_rows']:,} | {info['test_travelling']:,} "
                     f"| {info['test_travelling_pathogenic']} |{auc}{alarms} "
                     f"{info['runs']['oslo_only'][0]['discordant_benign']:,} |")
    TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {TABLE.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-disease-area federated learning")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    results = json.loads(RESULTS.read_text()) if args.plot_only else run()
    plot(results)
    table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
