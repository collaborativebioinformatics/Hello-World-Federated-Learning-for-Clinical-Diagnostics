"""Step 4: federated training with NVFlare, against the single-site and pooled baselines.

Runs the real thing: an NVFlare FedAvg job in simulator mode, a server and three
clients as separate processes, each client reading only its own folder. The
client script is scripts/04_fedavg_client.py and the model is step 3's logistic
regression, unchanged, so the three arms differ only in who saw which rows.

    Oslo only    one hospital trains on its own 4,300 rows
    Federated    NVFlare FedAvg, 20 aggregation rounds, only weights travel
    Pooled       every row in one pile, which no privacy law allows

Five runs. What varies is the DEAL: which hospital classified which variant.
That is the error bar the comparison needs, because the question "does federating
cost anything against pooling" is a question about partitions, not about
optimiser noise. The test set, the patient cohorts and the model are identical
across runs, so nothing else can move the numbers. The training here is
deterministic full-batch gradient descent, so repeating a run with the same deal
would reproduce it exactly and report a standard deviation of zero, which would
be a statement about the optimiser rather than about federation.

data/ is never modified: each run writes its own hospitals under a workspace.

Writes  data/results_federated.json
        docs/step4_results.md

Usage:
    uv run python scripts/04_federated_train.py
    uv run python scripts/04_federated_train.py --runs 2 --rounds 5   # a quick smoke test
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
WORKSPACE = DATA_DIR / "fedavg_runs"

ROUNDS = 20  # aggregation rounds
LOCAL_EPOCHS = 150  # gradient steps a client takes per round; 20 x 150 = the 3000 step 3 uses
RUNS = 5


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


step3 = load("step3", "03_train_local.py")
step2 = load("step2", "02_simulate_hospitals.py")


def deal(training_pool: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Re-run step 2's dealing with a different seed. Same rule, different draw."""
    return step2.deal_verdicts(training_pool, np.random.default_rng(seed))


def write_hospitals(run_dir: Path, training_pool: pd.DataFrame, held_by: pd.DataFrame,
                    score_columns: list[str]) -> dict[str, pd.DataFrame]:
    """One folder per hospital, holding only what that hospital may see."""
    hospitals = {}
    for site in step2.SITES:
        counts = pd.read_csv(DATA_DIR / site / "patient_counts.csv").set_index("variant_id")
        # af_local is a property of the hospital's cohort, not of the deal, so the
        # patient counts from data/ stay valid however the verdicts are dealt.
        local = (counts.ac_unaffected / counts.an_unaffected)
        rows = training_pool[held_by[site]].copy()
        rows["af_local"] = local.reindex(rows.variant_id).to_numpy()
        rows = rows[["variant_id", "name", "gene", "label", "af_local", *score_columns]]
        (run_dir / site).mkdir(parents=True, exist_ok=True)
        rows.to_csv(run_dir / site / "verdicts.csv", index=False, lineterminator="\n")
        hospitals[site] = rows
    (run_dir / "score_columns.json").write_text(json.dumps(score_columns), encoding="utf-8")
    return hospitals


def run_fedavg(run_dir: Path, rounds: int, n_features: int) -> np.ndarray:
    """One NVFlare FedAvg job in simulator mode. Returns the final global weights."""
    from nvflare.app_common.abstract.fl_model import FLModel
    from nvflare.app_common.workflows.fedavg import FedAvg
    from nvflare.job_config.api import FedJob
    from nvflare.job_config.script_runner import FrameworkType, ScriptRunner

    job = FedJob(name="fedavg_logreg")
    # One extra round: the aggregate of the last round is never sent to a client,
    # and the clients are how we read the global model out. Round rounds+1 hands
    # them the aggregate of `rounds` rounds, which each one saves.
    job.to_server(FedAvg(
        num_clients=len(step2.SITES),
        num_rounds=rounds + 1,
        # A plain list, not an ndarray: the initial model is written into the
        # job config as JSON, and numpy does not survive that. The client
        # converts on receipt and every later round carries real arrays.
        model=FLModel(params={"weights": [0.0] * (n_features + 1)}),
    ))
    for site in step2.SITES:
        job.to(ScriptRunner(
            script=str(ROOT / "scripts" / "04_fedavg_client.py"),
            script_args=f"--site {site} --run-dir {run_dir} --root {ROOT} --local-epochs {LOCAL_EPOCHS}",
            framework=FrameworkType.NUMPY,
        ), site)

    # Clients are already named by to(), so simulator_run must not be given n_clients.
    job.simulator_run(str(run_dir / "simulator"), threads=len(step2.SITES))

    saved = sorted(run_dir.glob("global_from_*.npy"))
    if not saved:
        raise RuntimeError(f"no client wrote a global model in {run_dir}; the simulator run failed")
    weights = [np.load(path) for path in saved]
    # Every client is sent the same global model. If they disagree, the run is not
    # what it claims to be, so check rather than trust.
    for other in weights[1:]:
        assert np.allclose(weights[0], other), "clients received different global models"
    return weights[0]


def evaluate(weights: np.ndarray, test: pd.DataFrame, score_columns: list[str],
             evidence: dict[str, np.ndarray], threshold: float, discordant_benign: np.ndarray,
             label: np.ndarray) -> dict[str, float]:
    out = {}
    for name, source in evidence.items():
        probability = step3.predict(step3.features(test, score_columns, source), weights)
        out[f"auc_{name}"] = step3.roc_auc(label, probability)
        out[f"false_alarms_{name}"] = int((probability[discordant_benign] >= threshold).sum())
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="NVFlare FedAvg against single-site and pooled")
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--keep", action="store_true", help="keep each run's workspace for inspection")
    args = parser.parse_args()

    columns = json.loads((DATA_DIR / "columns.json").read_text())
    score_columns = [c for c in columns["features"] if c not in step3.DROP_FEATURES]
    test = pd.read_csv(DATA_DIR / "test" / "variants.csv")
    label = test.label.to_numpy()
    evidence = step3.evidence_columns(test)
    discordant_benign = (test.pop_discordant.to_numpy() == 1) & (label == 0)

    # The training pool is every variant that is not in the locked test set. The
    # test set never moves, so the five runs differ only in the deal.
    everything = pd.read_csv(DATA_DIR / "variants.csv")
    training_pool = everything[~everything.variant_id.isin(set(test.variant_id))].reset_index(drop=True)
    print(f"training pool {len(training_pool)} variants, test set {len(test)}, "
          f"{len(score_columns)} scores + log frequency\n")

    arms = ["oslo_only", "federated", "pooled"]
    collected: dict[str, list[dict]] = {arm: [] for arm in arms}
    sizes: list[dict] = []

    for run in range(args.runs):
        seed = 100 + run
        run_dir = WORKSPACE / f"seed_{seed}"
        shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True)

        held_by = deal(training_pool, seed)
        hospitals = write_hospitals(run_dir, training_pool, held_by, score_columns)
        sizes.append({site: len(rows) for site, rows in hospitals.items()})
        print(f"run {run + 1}/{args.runs}  seed {seed}  "
              + "  ".join(f"{site.replace('site_', '')} {len(rows)}" for site, rows in hospitals.items()))

        models = {}
        # Two arms are ordinary local training on this deal.
        for arm, rows in [("oslo_only", hospitals["site_oslo"]),
                          ("pooled", pd.concat(hospitals.values()).drop_duplicates("variant_id"))]:
            X = step3.features(rows, score_columns, rows.af_local.to_numpy())
            models[arm] = step3.fit(X, rows.label.to_numpy())
        # The third is NVFlare.
        models["federated"] = run_fedavg(run_dir, args.rounds, len(score_columns) + 1)

        pooled_rows = pd.concat(hospitals.values()).drop_duplicates("variant_id")
        for arm in arms:
            # Each arm may only use a threshold it could actually have computed.
            if arm == "federated":
                # No site may see another's rows, so the threshold is federated too:
                # each site takes the quantile on its own rows under the global
                # model and the server averages them by row count. One number per
                # site crosses the wire, the same kind of payload as the weights.
                per_site, weights_by_rows = [], []
                for site, rows in hospitals.items():
                    X = step3.features(rows, score_columns, rows.af_local.to_numpy())
                    per_site.append(step3.threshold_at_sensitivity(
                        rows.label.to_numpy(), step3.predict(X, models[arm]), step3.TARGET_SENSITIVITY))
                    weights_by_rows.append(len(rows))
                threshold = float(np.average(per_site, weights=weights_by_rows))
            else:
                # Oslo only has its own rows; pooled is allowed everything by construction.
                rows = hospitals["site_oslo"] if arm == "oslo_only" else pooled_rows
                X = step3.features(rows, score_columns, rows.af_local.to_numpy())
                threshold = step3.threshold_at_sensitivity(
                    rows.label.to_numpy(), step3.predict(X, models[arm]), step3.TARGET_SENSITIVITY)
            collected[arm].append(evaluate(models[arm], test, score_columns, evidence,
                                           threshold, discordant_benign, label))
            collected[arm][-1]["threshold"] = threshold
            collected[arm][-1]["weights"] = models[arm].tolist()
        if not args.keep:
            shutil.rmtree(run_dir / "simulator", ignore_errors=True)
        print()

    # ---- report ----
    def summarise(arm: str, key: str) -> tuple[float, float]:
        values = [run[key] for run in collected[arm]]
        return float(np.mean(values)), float(np.std(values, ddof=1))

    title = {"oslo_only": "Oslo only", "federated": "Federated (NVFlare FedAvg)", "pooled": "Everything pooled"}
    print("=" * 78)
    print(f"{args.runs} runs, mean (standard deviation over runs). "
          f"NVFlare {__import__('nvflare').__version__}, {args.rounds} aggregation rounds.\n")

    for metric, label_text, fmt in [
        ("auc_public", "AUC, public frequency", ".4f"),
        ("auc_federated_query", "AUC, federated count query", ".4f"),
        (f"false_alarms_public", "false alarms / 183, public frequency", ".1f"),
        (f"false_alarms_federated_query", "false alarms / 183, federated counts", ".1f"),
    ]:
        print(label_text)
        for arm in arms:
            mean, std = summarise(arm, metric)
            print(f"  {title[arm]:<28} {mean:{fmt}} ({std:{fmt}})")
        print()

    results = {
        "nvflare_version": __import__("nvflare").__version__,
        "runs": args.runs, "rounds": args.rounds, "local_epochs": LOCAL_EPOCHS,
        "seeds": [100 + r for r in range(args.runs)],
        "hospital_sizes": sizes,
        "per_run": collected,
        "summary": {arm: {metric: summarise(arm, metric)
                          for metric in collected[arm][0] if metric != "weights"} for arm in arms},
    }
    (DATA_DIR / "results_federated.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {(DATA_DIR / 'results_federated.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
