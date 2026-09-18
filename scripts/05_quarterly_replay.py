"""Quarterly replay: the three simulated hospitals grow quarter by quarter, and the shared model is retrained each time.

The live pipeline page animates real numbers, so this script produces them. It takes the
step 2 build of an area, replays it as four quarters in which every hospital's sequenced
cohort and classified verdicts grow to their final size, retrains the step 3 model on each
quarter, and scores it on the locked test set under the usual evidence settings. Q4 is the
build as it stands today, so the last quarter lands on the numbers in docs/step3_results.md.

What one quarter holds, per hospital

    patients   a cumulative share of its final cohort: 25%, 50%, 75%, 100%. Q4 is exactly
               step 2's cohort, produced by replaying step 2's own random sequence with its own
               functions. Earlier quarters are nested sub-cohorts of it: the carriers among
               the first share of gene copies are drawn by sampling without replacement from
               the final counts, so a carrier seen in Q1 is still there in Q2, and each
               quarter's counts have the distribution step 2 would give a cohort of that size.
    verdicts   a cumulative share of its final classified variants, in one fixed random order.

What is measured each quarter

    the pooled model retrained on that quarter's verdicts, scored on the locked test set with
    that quarter's counts as frequency evidence: AUC, false alarms on the population-discordant
    benign rows, sensitivity, and the weight the model put on frequency. Single-site models too.
    The pooled fit stands in for the federated model, because step 4 showed FedAvg lands on it.
    NVFlare is not run here.

    the count query without any model: for a patient at each hospital, how many variants change
    call once the other hospitals answer, by the two rules of scripts/hospital_query.py on that
    quarter's counts.

Everything is a simulation. Counts among sick patients were generated from the verdict in step 2
and exist only to demonstrate the query; they never reach a model. Real hospitals do not grow
uniformly, and no number here is a claim about one.

Steps 2 and 3 are loaded from their files and their functions are called; nothing in them is
copied or changed.

Reads   data/variants.csv and the step 2 build of each area (data/ for heart, data/<area>/ otherwise)
Writes  data/quarterly/<area>/q1..q4/     one step 2 layout per quarter, so step 3 can read it
        docs/quarterly_replay.json        the numbers the live page animates
        docs/quarterly_replay.md          the same as tables, with the Q4 check and the limits

Usage:
    uv run python scripts/05_quarterly_replay.py
    uv run python scripts/05_quarterly_replay.py --areas cardiac
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "quarterly"
DOC_JSON = ROOT / "docs" / "quarterly_replay.json"
DOC_MD = ROOT / "docs" / "quarterly_replay.md"

sys.path.insert(0, str(SCRIPTS))
import hospital_query  # noqa: E402  the two count rules, and the per-gene line

# The heart build sits at the top of data/, every other area in data/<area>/. The build over
# every NHS panel is left out on purpose: it is large and another build may be changing it.
AREAS = {"cardiac": DATA_DIR, "cancer": DATA_DIR / "cancer"}
AREA_TITLES = {"cardiac": "heart", "cancer": "inherited cancer"}

# Cumulative share of each hospital's final patients and verdicts held at the end of the quarter.
QUARTERS = {"Q1": 0.25, "Q2": 0.50, "Q3": 0.75, "Q4": 1.00}

# One seed for the quarter draws: the order verdicts arrive in, and the sub-cohorts. The final
# build itself is replayed with step 2's own seed, which is printed next to this one.
QUARTER_SEED = 12

# The evidence settings, in step 3's order, under the names the page uses.
EVIDENCE = {"public": "public", "own_hospital": "own", "federated_query": "federated", "ceiling": "ceiling"}
SETTINGS = ["none", *EVIDENCE.values()]


def load_script(name: str):
    """Import a script whose name starts with a digit, unchanged, so its functions can be called."""
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), SCRIPTS / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # @dataclass looks the module up here
    spec.loader.exec_module(module)
    return module


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode()


def save(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(csv_bytes(frame))


# ---------------------------------------------------------------------------
# Q4: today's build, replayed with step 2's own functions and random sequence
# ---------------------------------------------------------------------------
def replay_build(step2, area_dir: Path) -> dict:
    """Step 2's main(), call for call, without writing: the same rng, the same order, the same files."""
    table = pd.read_csv(area_dir / "variants.csv")
    rng = np.random.default_rng(step2.SEED)
    test, training_pool, unseen_genes = step2.lock_test_set(table, rng)
    held_by = step2.deal_verdicts(training_pool, rng)
    public_reference = table[["variant_id"]].assign(af_public=table[f"af_{step2.PUBLIC_REFERENCE_POPULATION}"])
    test = test.assign(pop=step2.population_of(test)).merge(public_reference, on="variant_id")

    counts, identical = {}, {}
    for site_name, site in step2.SITES.items():
        counts[site_name] = step2.simulate_patient_counts(table, site, rng)
        view = step2.hospital_view(training_pool[held_by[site_name]], public_reference, counts[site_name])
        identical[site_name] = (
            (area_dir / site_name / "patient_counts.csv").read_bytes() == csv_bytes(counts[site_name])
            and (area_dir / site_name / "verdicts.csv").read_bytes() == csv_bytes(view)
        )
    identical["test"] = (area_dir / "test" / "variants.csv").read_bytes() == csv_bytes(test)
    return {"table": table, "test": test, "training_pool": training_pool, "held_by": held_by,
            "public_reference": public_reference, "counts": counts, "unseen_genes": [str(g) for g in unseen_genes],
            "identical_to_build": identical}


# ---------------------------------------------------------------------------
# Q1 to Q3: nested sub-cohorts and a growing pile of verdicts
# ---------------------------------------------------------------------------
def sub_cohort(counts: pd.DataFrame, patients: int, step2, rng: np.random.Generator) -> pd.DataFrame:
    """The counts among the first `patients` of a larger cohort, drawn without replacement from its counts.

    Sick and healthy patients are split as step 2 splits them. The carriers among a subset of gene
    copies follow the hypergeometric distribution, whose marginal is the binomial step 2 draws from,
    so the sub-cohort is what simulate_patient_counts() would give for that size, and it is nested.
    """
    affected = round(patients * step2.AFFECTED_SHARE)
    an_affected, an_unaffected = 2 * affected, 2 * (patients - affected)
    assert an_affected <= counts.an_affected.iloc[0] and an_unaffected <= counts.an_unaffected.iloc[0]
    healthy, sick = counts.ac_unaffected.to_numpy(), counts.ac_affected.to_numpy()
    ac_unaffected = rng.hypergeometric(healthy, counts.an_unaffected.to_numpy() - healthy, an_unaffected)
    ac_affected = rng.hypergeometric(sick, counts.an_affected.to_numpy() - sick, an_affected)
    return pd.DataFrame({
        "variant_id": counts.variant_id.to_numpy(),
        "name": counts.name.to_numpy(),
        "ac": ac_affected + ac_unaffected,
        "an": 2 * patients,
        "ac_affected": ac_affected,
        "an_affected": an_affected,
        "ac_unaffected": ac_unaffected,
        "an_unaffected": an_unaffected,
    })


def write_quarters(step2, build: dict, area: str, area_dir: Path) -> dict[str, Path]:
    """One step 2 layout per quarter under data/quarterly/<area>/, so step 3 can be pointed at it."""
    rng = np.random.default_rng(QUARTER_SEED)
    names = list(QUARTERS)
    final_counts = build["counts"]

    # Verdicts arrive in one fixed random order per hospital; a quarter holds a prefix of it.
    order = {site: rng.permutation(int(build["held_by"][site].sum())) for site in step2.SITES}
    # Cohorts: Q4 is the build. Each earlier quarter is drawn as a sub-cohort of the one after it.
    counts = {names[-1]: final_counts}
    for later, earlier in zip(reversed(names), reversed(names[:-1])):
        counts[earlier] = {
            site: sub_cohort(counts[later][site], round(QUARTERS[earlier] * step2.SITES[site].patients), step2, rng)
            for site in step2.SITES
        }

    folders = {}
    for quarter, share in QUARTERS.items():
        folder = OUT_DIR / area / quarter.lower()
        if folder.exists():
            shutil.rmtree(folder)
        save(build["test"], folder / "test" / "variants.csv")
        save(build["public_reference"], folder / "public_reference.csv")
        shutil.copyfile(area_dir / "columns.json", folder / "columns.json")
        run = {"area": area, "quarter": quarter, "share": share, "seed": step2.SEED, "quarter_seed": QUARTER_SEED,
               "sites": {}}
        for site_name, site in step2.SITES.items():
            own = build["training_pool"][build["held_by"][site_name]]
            own = own.iloc[np.sort(order[site_name][: round(share * len(own))])]  # prefix of the order, file order kept
            view = step2.hospital_view(own, build["public_reference"], counts[quarter][site_name])
            save(counts[quarter][site_name], folder / site_name / "patient_counts.csv")
            save(view, folder / site_name / "verdicts.csv")
            run["sites"][site_name] = {
                "population": site.population,
                "patients": int(counts[quarter][site_name].an.iloc[0] // 2),
                "verdicts": len(own),
                "pathogenic": int(own.label.sum()),
                "benign": int(len(own) - own.label.sum()),
                "discordant": int(own.pop_discordant.sum()),
            }
        (folder / "sites.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
        folders[quarter] = folder
    return folders


# ---------------------------------------------------------------------------
# Train and score one quarter with step 3's own functions
# ---------------------------------------------------------------------------
def train_and_score(step3, folder: Path) -> dict:
    """Step 3's grid for one quarter: every model, every evidence setting, plus the scores-only model."""
    step3.DATA_DIR = folder
    hospitals, test, score_columns = step3.load()
    evidence = {EVIDENCE[k]: v for k, v in step3.evidence_columns(test).items()}
    label = test.label.to_numpy()
    unseen = (test.test_kind == "unseen_gene").to_numpy()
    discordant_benign = (test.pop_discordant.to_numpy() == 1) & (label == 0)

    out = {"models": {}, "test_rows": len(test), "pathogenic_rows": int(label.sum()),
           "discordant_rows": int(discordant_benign.sum()), "features": score_columns + ["log_frequency"]}
    for name, rows in hospitals.items():
        X = step3.features(rows, score_columns, rows.af_local.to_numpy())
        y = rows.label.to_numpy()
        weights = step3.fit(X, y)
        cut = step3.threshold_at_sensitivity(y, step3.predict(X, weights), step3.TARGET_SENSITIVITY)
        # Step 3's ablation: the same model without the frequency column at all.
        weights_none = step3.fit(X[:, :-1], y)
        cut_none = step3.threshold_at_sensitivity(y, step3.predict(X[:, :-1], weights_none), step3.TARGET_SENSITIVITY)

        probability = {"none": step3.predict(step3.features(test, score_columns, np.zeros(len(test)))[:, :-1], weights_none)}
        fired = {"none": probability["none"] >= cut_none}
        for setting, source in evidence.items():
            probability[setting] = step3.predict(step3.features(test, score_columns, source), weights)
            fired[setting] = probability[setting] >= cut

        model = {
            "training_rows": len(rows), "training_pathogenic": int(y.sum()),
            "threshold": round(float(cut), 4),
            "frequency_weight": round(float(weights[-2]), 4),
            "auc": {s: round(step3.roc_auc(label, p), 4) for s, p in probability.items()},
            "auc_unseen_genes": {s: round(step3.roc_auc(label[unseen], p[unseen]), 4) for s, p in probability.items()},
            "false_alarms": {s: int(f[discordant_benign].sum()) for s, f in fired.items()},
            "sensitivity": {s: round(float(f[label == 1].mean()), 4) for s, f in fired.items()},
            "paired": {
                f"{baseline}_to_federated": {
                    "removed": int((fired[baseline] & ~fired["federated"] & discordant_benign).sum()),
                    "introduced": int((~fired[baseline] & fired["federated"] & discordant_benign).sum()),
                }
                for baseline in ("public", "own")
            },
        }
        for pair in model["paired"].values():
            pair["p"] = round(step3.sign_test(pair["removed"], pair["introduced"]), 4)
        out["models"][name] = model
    return out


# ---------------------------------------------------------------------------
# The count query, no model: hospital_query.overview() on one quarter's counts
# ---------------------------------------------------------------------------
def query_view(table: pd.DataFrame, counts: dict[str, pd.DataFrame], public_reference: pd.DataFrame,
               min_count: int = hospital_query.DEFAULT_MIN_COUNT) -> dict:
    """The same two rules as hospital_query.overview(), on the counts of one quarter.

    Rule 1 at the gene's line, from hospital_query.too_common_line(): 0.1% among healthy patients when
    one bad copy is enough, 1% when both copies must be bad. Rule 2 at PILE_UP_FOLD times more common
    among the sick. A count above zero and under `min_count` is hidden, as a hospital would hide it.
    Returns, for a patient at each hospital, how many variants change call once the others answer.
    """
    index = pd.Index(table.variant_id)
    line = pd.Series(table.gene.map(hospital_query.too_common_line).to_numpy(), index=index)
    common, piles_up, carried = {}, {}, pd.Series(0, index=index)
    for site, frame in counts.items():
        frame = frame.set_index("variant_id").reindex(index)
        healthy_hidden = (frame.ac_unaffected > 0) & (frame.ac_unaffected < min_count)
        sick_hidden = (frame.ac_affected > 0) & (frame.ac_affected < min_count)
        healthy = frame.ac_unaffected.mask(healthy_hidden, 0) / frame.an_unaffected
        sick = frame.ac_affected.mask(sick_hidden, 0) / frame.an_affected
        common[site] = healthy >= line
        piles_up[site] = common[site] & ~sick_hidden & (sick >= hospital_query.PILE_UP_FOLD * healthy)
        carried = carried + frame.ac
    public = pd.Series(public_reference.af_public.to_numpy(), index=pd.Index(public_reference.variant_id))
    common_in_public = public.reindex(index).fillna(0.0) >= line

    def call(among: list[str]) -> pd.Series:
        flagged = pd.concat([piles_up[s] for s in among], axis=1).any(axis=1)
        cleared = pd.concat([common[s] for s in among], axis=1).any(axis=1) | common_in_public
        calls = pd.Series("FREQUENCY SAYS NOTHING", index=index)
        calls[cleared] = "LIKELY HARMLESS"
        calls[flagged] = "KEEP FLAGGED"
        return calls

    view = pd.DataFrame({"name": table.name.to_numpy(), "carried": carried, "after": call(list(counts))}, index=index)
    for site in counts:
        view[f"before_{site}"] = call([site])
    # One row per name, the DNA change patients carry most, as overview() keeps it.
    view = view.sort_values("carried", ascending=False, kind="stable").drop_duplicates("name")
    return {
        "variants": len(view),
        "min_count": min_count,
        "changed_calls_per_site": {site: int((view[f"before_{site}"] != view.after).sum()) for site in counts},
        "calls_after_all_answer": {call_: int((view.after == call_).sum())
                                   for call_ in ("LIKELY HARMLESS", "KEEP FLAGGED", "FREQUENCY SAYS NOTHING")},
    }


def query_view_today(area: str) -> dict[str, int]:
    """hospital_query.overview() on the build as it stands, missense rows only: the Q4 yardstick."""
    hospital_query.set_area(area)
    hospital_query.clear_caches()
    changed = {}
    for site in hospital_query.list_sites():
        table = hospital_query.overview(patient_at=site)
        changed[site] = int(table[table.mutation_type == hospital_query.MISSENSE].changed.sum())
    return changed


# ---------------------------------------------------------------------------
def replay_area(step2, step3, area: str) -> dict:
    area_dir = AREAS[area]
    started = time.time()
    hospital_query.set_area(area)  # the per-gene line of rule 1 is read from this area's gene list
    hospital_query.clear_caches()
    build = replay_build(step2, area_dir)
    print(f"\n=== {area} ({AREA_TITLES[area]}): step 2 replayed with seed {step2.SEED}, "
          f"{len(build['table'])} variants, test set {len(build['test'])} rows, unseen genes {', '.join(build['unseen_genes'])}")
    print("    Q4 files identical to today's build: " + ", ".join(f"{k} {v}" for k, v in build["identical_to_build"].items()))
    folders = write_quarters(step2, build, area, area_dir)

    quarters = []
    for quarter, folder in folders.items():
        run = json.loads((folder / "sites.json").read_text(encoding="utf-8"))
        scored = train_and_score(step3, folder)
        counts = {site: pd.read_csv(folder / site / "patient_counts.csv") for site in step2.SITES}
        query = query_view(build["table"], counts, build["public_reference"])
        pooled = scored["models"]["pooled"]
        quarters.append({
            "quarter": quarter,
            "share": QUARTERS[quarter],
            "per_site": run["sites"],
            "model": {
                "stands_in_for": "federated",
                "training_rows": pooled["training_rows"],
                "training_pathogenic": pooled["training_pathogenic"],
                "threshold": pooled["threshold"],
                "auc": pooled["auc"],
                "auc_unseen_genes": pooled["auc_unseen_genes"],
                "false_alarms": pooled["false_alarms"],
                "discordant_rows": scored["discordant_rows"],
                "sensitivity": pooled["sensitivity"],
                "pathogenic_rows": scored["pathogenic_rows"],
                "paired": pooled["paired"],
                "frequency_weight": {"pooled": pooled["frequency_weight"],
                                     "per_site": {s: scored["models"][s]["frequency_weight"] for s in step2.SITES}},
            },
            "single_site_models": {s: {k: scored["models"][s][k] for k in ("training_rows", "auc", "false_alarms", "sensitivity")}
                                   for s in step2.SITES},
            "query": query,
        })
        print_quarter(quarters[-1], list(step2.SITES))

    # ---- the Q4 check: the last quarter must be today's build, in files and in numbers ----
    check = {"files_identical": build["identical_to_build"]}
    results_file = area_dir / "results_local.json"
    if results_file.exists():
        saved = json.loads(results_file.read_text(encoding="utf-8"))
        known = {"none": saved["false_alarm_counts"]["scores_only"],
                 **{EVIDENCE[k]: v for k, v in saved["false_alarm_counts"].items() if k in EVIDENCE}}
        check["false_alarms_step3"] = known
        check["false_alarms_q4"] = quarters[-1]["model"]["false_alarms"]
        check["false_alarms_match"] = known == quarters[-1]["model"]["false_alarms"]
        check["frequency_weight_step3"] = saved["training"]["pooled"]["weights"]["log_frequency"]
    check["changed_calls_overview_today"] = query_view_today(area)
    check["changed_calls_q4"] = quarters[-1]["query"]["changed_calls_per_site"]
    check["changed_calls_match"] = check["changed_calls_overview_today"] == check["changed_calls_q4"]
    seconds = round(time.time() - started, 1)
    print(f"\n    Q4 check: false alarms match step 3 {check.get('false_alarms_match')}"
          f" {check.get('false_alarms_step3')}; changed calls match overview() {check['changed_calls_match']}"
          f" {check['changed_calls_overview_today']}; {seconds} s")

    return {
        "title": AREA_TITLES[area],
        "folder": (OUT_DIR / area).relative_to(ROOT).as_posix(),
        "variants": len(build["table"]),
        "test_rows": len(build["test"]),
        "unseen_genes": build["unseen_genes"],
        "sites": [{"name": s, "population": site.population, "final_patients": site.patients,
                   "final_verdicts": int(build["held_by"][s].sum())} for s, site in step2.SITES.items()],
        "features": scored["features"],
        "quarters": quarters,
        "q4_check": check,
        "seconds": round(seconds),  # dropped from the JSON, so reruns give the same bytes
    }


def print_quarter(q: dict, sites: list[str]) -> None:
    m = q["model"]
    n = m["discordant_rows"]
    print(f"\n  {q['quarter']}  share {q['share']:.0%}")
    print(f"    {'hospital':<13}{'patients':>9}{'verdicts':>9}{'pathogenic':>11}{'benign':>8}{'query: calls changed':>22}")
    for s in sites:
        p = q["per_site"][s]
        print(f"    {s:<13}{p['patients']:>9,}{p['verdicts']:>9,}{p['pathogenic']:>11,}{p['benign']:>8,}"
              f"{q['query']['changed_calls_per_site'][s]:>22,}")
    print(f"    pooled model on {m['training_rows']:,} verdicts, frequency weight {m['frequency_weight']['pooled']:+.2f}")
    print(f"    {'evidence':<13}{'AUC':>8}{'false alarms':>16}{'sensitivity':>13}")
    for s in SETTINGS:
        alarms = f"{m['false_alarms'][s]} / {n}"
        print(f"    {s:<13}{m['auc'][s]:>8.3f}{alarms:>16}{m['sensitivity'][s]:>13.3f}")


# ---------------------------------------------------------------------------
def write_doc(result: dict) -> None:
    lines = [
        "# Quarterly replay: the hospitals grow, the shared model is retrained each quarter",
        "",
        "> **A simulation, replayed.** The three hospitals of step 2 are simulated. Here their cohorts and",
        "> verdicts are made to grow in four equal steps, and the step 3 model is retrained after each one.",
        "> Counts among sick patients were generated from the verdict and only demonstrate the query.",
        "> The pooled fit stands in for the federated model; NVFlare was not run. Nothing here describes a real hospital.",
        "",
        f"Reproduce with `uv run python scripts/05_quarterly_replay.py`. Every number below is also in "
        f"[quarterly_replay.json](quarterly_replay.json), which the live pipeline page reads. Run on {result['note']['date']}.",
        "",
        "## What was done",
        "",
        f"1. Step 2 was replayed with its own functions and its own seed, {result['note']['seed']}: the same test set, "
        "the same dealing of verdicts, the same patient counts. The files this gives were compared byte for byte with "
        "the build in `data/`, and they are the same. That build is the last quarter.",
        "2. Each hospital's final cohort was cut into nested sub-cohorts holding 25%, 50% and 75% of its patients. "
        "The carriers among the first share of gene copies were drawn without replacement from the final counts, "
        "so a carrier seen in one quarter is still there in the next, and each quarter's counts have the distribution "
        f"step 2 would give a cohort of that size. Sick and healthy patients are split as step 2 splits them. Seed {QUARTER_SEED}, "
        "a separate generator from the build's.",
        "3. Each hospital's final verdicts were put in one fixed random order, and a quarter holds the first 25%, 50%, 75% "
        "or 100% of them. `af_local` in a quarter's verdict file comes from that quarter's own counts, through step 2's "
        "`hospital_view()`.",
        "4. Every quarter was written in the step 2 layout under `data/quarterly/<area>/q1` to `q4`, and step 3's own "
        "`load()`, `evidence_columns()`, `fit()` and threshold rule were run on it unchanged. The pooled model, the three "
        "`verdicts.csv` stacked with duplicates dropped, is the federated stand-in: step 4 showed FedAvg lands on it. "
        "The scores-only model of step 3's ablation gives the `none` column. `own` is Oslo's counts alone, as in step 3.",
        "5. The count query was applied to each quarter's counts with the two rules of `scripts/hospital_query.py`: "
        "rule 1 at the gene's line from `too_common_line()`, 0.1% among healthy patients when one bad copy is enough "
        "and 1% when both copies must be bad, rule 2 at 3 times more common among the sick, counts under 5 hidden. "
        "For a patient at each hospital the table counts the variants whose call changes once the other hospitals answer. "
        "At Q4 this was checked against `hospital_query.overview()` on the build itself.",
        "",
        "The evidence settings are those of [data_contract.md](data_contract.md): `none` is the model without a frequency "
        "column, `public` the European-only reference, `own` the patient's hospital alone, `federated` the highest count "
        "any hospital reports, `ceiling` gnomAD's real per-population frequency, which no hospital may use.",
    ]
    for area, r in result["areas"].items():
        sites = r["sites"]
        n = r["quarters"][0]["model"]["discordant_rows"]
        lines += [
            "",
            f"## {r['title'].capitalize()}, `{area}`",
            "",
            f"{r['variants']:,} variants, test set {r['test_rows']:,} rows, unseen genes {', '.join(r['unseen_genes'])}. "
            f"Model: {len(r['features']) - 1} scores plus log frequency. Files under `{r['folder']}/`. Run time about {r['seconds']} s.",
            "",
            "| hospital | population | final patients | final verdicts |",
            "|---|---|---|---|",
            *[f"| `{s['name']}` | {s['population']} | {s['final_patients']:,} | {s['final_verdicts']:,} |" for s in sites],
            "",
            "### What each hospital held",
            "",
            "| quarter | share | " + " | ".join(f"`{s['name']}` patients | verdicts (pathogenic)" for s in sites) + " | pooled verdicts |",
            "|---|---|" + "---|---|" * len(sites) + "---|",
        ]
        for q in r["quarters"]:
            cells = " | ".join(f"{q['per_site'][s['name']]['patients']:,} | {q['per_site'][s['name']]['verdicts']:,} "
                               f"({q['per_site'][s['name']]['pathogenic']:,})" for s in sites)
            lines.append(f"| {q['quarter']} | {q['share']:.0%} | {cells} | {q['model']['training_rows']:,} |")
        lines += [
            "",
            f"### The pooled model each quarter, false alarms on the {n} population-discordant benign test rows",
            "",
            "| quarter | " + " | ".join(SETTINGS) + " | frequency weight |",
            "|---|" + "---|" * (len(SETTINGS) + 1),
        ]
        for q in r["quarters"]:
            m = q["model"]
            lines.append(f"| {q['quarter']} | " + " | ".join(f"{m['false_alarms'][s]} / {n}" for s in SETTINGS)
                         + f" | {m['frequency_weight']['pooled']:+.2f} |")
        lines += ["", "AUC on all test rows, and sensitivity on the pathogenic test rows at the training threshold:", "",
                  "| quarter | " + " | ".join(f"AUC {s}" for s in SETTINGS) + " | " + " | ".join(f"sens. {s}" for s in SETTINGS) + " |",
                  "|---|" + "---|" * (2 * len(SETTINGS))]
        for q in r["quarters"]:
            m = q["model"]
            lines.append(f"| {q['quarter']} | " + " | ".join(f"{m['auc'][s]:.3f}" for s in SETTINGS) + " | "
                         + " | ".join(f"{m['sensitivity'][s]:.3f}" for s in SETTINGS) + " |")
        lines += ["", "Paired on the same rows, public reference to federated query: false alarms removed and introduced, with the exact p of step 3.", "",
                  "| quarter | removed | introduced | p |", "|---|---|---|---|"]
        for q in r["quarters"]:
            pair = q["model"]["paired"]["public_to_federated"]
            lines.append(f"| {q['quarter']} | {pair['removed']} | {pair['introduced']} | {pair['p']} |")
        lines += ["", "### The count query without a model", "",
                  "For a patient at each hospital, how many variants change call once the other two hospitals answer, "
                  "of the variants in the table, one row per name.", "",
                  "| quarter | " + " | ".join(f"patient at `{s['name']}`" for s in sites) + " | likely harmless once all answer |",
                  "|---|" + "---|" * (len(sites) + 1)]
        for q in r["quarters"]:
            lines.append(f"| {q['quarter']} | " + " | ".join(f"{q['query']['changed_calls_per_site'][s['name']]:,}" for s in sites)
                         + f" | {q['query']['calls_after_all_answer']['LIKELY HARMLESS']:,} |")
        c = r["q4_check"]
        files = "identical" if all(c["files_identical"].values()) else "NOT identical: " + ", ".join(k for k, v in c["files_identical"].items() if not v)
        lines += ["", "### Does Q4 land on the known numbers", ""]
        lines.append(f"- Q4 files against the build in `data/`: {files}.")
        if "false_alarms_step3" in c:
            same = "the same" if c["false_alarms_match"] else "NOT the same"
            lines.append(f"- False alarms of the pooled model, Q4 against `results_local.json`: {same}. "
                         f"Step 3 has {', '.join(f'{k} {v}' for k, v in c['false_alarms_step3'].items())}; "
                         f"Q4 has {', '.join(f'{k} {v}' for k, v in c['false_alarms_q4'].items())}. "
                         f"Frequency weight {c['frequency_weight_step3']} in step 3, {r['quarters'][-1]['model']['frequency_weight']['pooled']} here.")
        same = "the same" if c["changed_calls_match"] else "NOT the same"
        lines.append(f"- Calls changed by the query, Q4 against `hospital_query.overview()` on the build, missense rows: {same}. "
                     f"overview() gives {', '.join(f'{k} {v}' for k, v in c['changed_calls_overview_today'].items())}.")
    lines += [
        "",
        "## Limits",
        "",
        "- A simulation from start to end. Which hospital holds which verdict and every patient count are simulated in step 2; "
        "here they are additionally cut into quarters. No number describes a real hospital or a real quarter.",
        "- Counts among sick patients were generated from the verdict. They are demonstration only and never reach a model, "
        "in step 3 or here.",
        "- The federated model is stood in by the pooled fit. Step 4 showed FedAvg lands on the pooled weights for this model; "
        "NVFlare was not run for the quarters.",
        "- Cohorts and verdict piles grow uniformly, by a quarter of the final size each quarter, at all three hospitals at once. "
        "Real hospitals do not.",
        "- The sub-cohorts are drawn from the final counts, so the quarters are nested and Q4 is the build. An independent draw "
        "per quarter would give the same distribution but would let a carrier vanish between quarters and would not reproduce Q4.",
        "- The test set is locked and the same in every quarter, so the model is scored on a fixed yardstick while its training "
        "rows grow. The discordant benign rows are few, and step 3's caution about single-digit counts applies to every quarter.",
        "- The count query here covers the missense table only. `hospital_query.overview()` on the build also answers for "
        "`data/other_types/` when that is built; those rows were not replayed.",
    ]
    DOC_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay the simulated hospitals quarter by quarter and retrain each quarter")
    parser.add_argument("--areas", nargs="+", choices=list(AREAS), default=None,
                        help="which built areas to replay; default every area of %(default)s that is built")
    args = parser.parse_args(argv)

    areas = args.areas or [a for a, folder in AREAS.items() if (folder / "sites.json").exists()]
    missing = [a for a in areas if not (AREAS[a] / "sites.json").exists()]
    if missing:
        print(f"not built: {', '.join(missing)}. Run steps 1 and 2 for that area first.", file=sys.stderr)
        return 1

    step2 = load_script("02_simulate_hospitals.py")
    step3 = load_script("03_train_local.py")
    np.random.seed(step3.SEED)
    print(f"quarterly replay: build seed {step2.SEED} (step 2), quarter seed {QUARTER_SEED}, "
          f"quarters {', '.join(f'{q} {s:.0%}' for q, s in QUARTERS.items())}")

    result = {
        "note": {
            "what": "A simulation replayed quarter by quarter. The three simulated hospitals of step 2 grow to their final "
                    "cohorts and verdict piles in four equal steps; the step 3 model is retrained each quarter on the verdicts "
                    "held so far and scored on the locked test set with that quarter's counts as evidence. No number here "
                    "describes a real hospital.",
            "sick_counts": "Counts among affected patients were generated from the verdict in step 2. They are demonstration "
                           "only and are never fed to a model.",
            "federated": "The pooled fit stands in for the federated model, because step 4 showed FedAvg lands on the pooled "
                         "weights. NVFlare was not run for the quarters.",
            "quarters": "Q4 is step 2's build, replayed byte for byte. Earlier quarters are nested sub-cohorts of it and "
                        "prefixes of one fixed random order of each hospital's verdicts.",
            "seed": step2.SEED,
            "quarter_seed": QUARTER_SEED,
            "shares": QUARTERS,
            "evidence": {"none": "no frequency column", "public": "European-only public reference",
                         "own": "the patient's hospital alone, Oslo as in step 3",
                         "federated": "highest healthy-patient frequency any hospital reports",
                         "ceiling": "gnomAD's real per-population frequency, not allowed in real life"},
            "date": str(date.today()),
            "script": "scripts/05_quarterly_replay.py",
        },
        "areas": {},
    }
    for area in areas:
        result["areas"][area] = replay_area(step2, step3, area)

    write_doc(result)
    for area in result["areas"].values():
        del area["seconds"]
    DOC_JSON.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    size = DOC_JSON.stat().st_size / 1024
    print(f"\nwrote {DOC_JSON.relative_to(ROOT).as_posix()} ({size:.1f} KB), {DOC_MD.relative_to(ROOT).as_posix()}, "
          f"and {OUT_DIR.relative_to(ROOT).as_posix()}/<area>/q1..q4/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
