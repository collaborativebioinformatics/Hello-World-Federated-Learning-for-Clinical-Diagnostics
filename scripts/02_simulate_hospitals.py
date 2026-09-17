"""Step 2: lock a test set, then simulate three hospitals.

A real hospital keeps two things private. We imitate both:

    its verdicts   variants its lab has classified. Real ClinVar rows, dealt out
                   so that a variant common in one population mostly lands at
                   that population's hospital.
    its patients   who carries which variant. Simulated cohorts, drawn from the
                   real gnomAD frequency of each hospital's population.

What every hospital shares today is a public frequency reference. Ours covers
Europeans only. That is a deliberate pretence: it stands in for the many
populations that real references do not cover. gnomAD's real South Asian and
African columns play the truth that only the local hospital can see.

Reads   data/variants.csv
Writes  data/test/variants.csv          locked first, never trained on
        data/public_reference.csv       the frequency everyone already has
        data/site_*/verdicts.csv        that hospital's private labelled variants
        data/site_*/patient_counts.csv  that hospital's private carrier counts
        data/sites.json                 the settings and sizes of this run

Usage:
    uv run python scripts/02_simulate_hospitals.py
    uv run python scripts/02_simulate_hospitals.py --set-reference   # declare this build the team's reference

The seed is fixed, so everyone who starts from the same data/variants.csv gets
byte-identical files. The script says whether your build matches the team's
reference in config/reference_build.json. It can differ only if myvariant.info
has updated its data since the reference was made.

What every file and column means, and which columns may be trained on:
docs/data_contract.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REFERENCE_BUILD = ROOT / "config" / "reference_build.json"

SEED = 12  # same seed, same hospitals


@dataclass
class Site:
    population: str  # which gnomAD population its patients come from
    patients: int  # size of its sequenced cohort
    verdict_share: float  # its share of all classified variants


# Unequal on purpose: one big lab and two small ones.
SITES = {
    "site_oslo": Site(population="nfe", patients=20_000, verdict_share=0.70),
    "site_karachi": Site(population="sas", patients=4_000, verdict_share=0.15),
    "site_lagos": Site(population="afr", patients=4_000, verdict_share=0.15),
}
PUBLIC_REFERENCE_POPULATION = "nfe"

# Test set: a random fifth of the variants, plus every variant of a few whole
# genes, so we can also ask "does it work on a gene it has never seen?"
TEST_SHARE = 0.20
UNSEEN_GENES = 6
MIN_PER_VERDICT = 10  # an unseen gene needs this many pathogenic and benign rows

# Patient simulation. Illustrative settings, not estimates.
AFFECTED_SHARE = 0.15  # share of a cohort that are heart patients
DIAGNOSTIC_YIELD = 0.20  # share of heart patients explained by one pathogenic variant
PATHOGENIC_ENRICHMENT = 5  # how much more common a pathogenic variant is among the affected

# Columns that are truth from gnomAD. They stay out of the hospital files.
TRUTH_COLUMNS_PREFIXES = ("af_", "ac_", "an_")
TRUTH_COLUMNS_EXACT = ("in_gnomad", "pop_discordant")


def lock_test_set(table: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    per_gene = table.groupby("gene").label.agg(pathogenic="sum", rows="size")
    per_gene["benign"] = per_gene.rows - per_gene.pathogenic
    has_both = per_gene[(per_gene.pathogenic >= MIN_PER_VERDICT) & (per_gene.benign >= MIN_PER_VERDICT)]
    unseen_genes = sorted(rng.choice(sorted(has_both.index), size=UNSEEN_GENES, replace=False))

    in_unseen_gene = table.gene.isin(unseen_genes)
    rest = table[~in_unseen_gene]
    # Keep the mix of verdicts and of population-discordant variants the same in test and training.
    random_fifth = rest.groupby(["label", "pop_discordant"], group_keys=False).sample(frac=TEST_SHARE, random_state=SEED)

    test = pd.concat([
        table[in_unseen_gene].assign(test_kind="unseen_gene"),
        random_fifth.assign(test_kind="random"),
    ])
    training_pool = rest.drop(random_fifth.index)
    return test, training_pool, unseen_genes


def deal_verdicts(training_pool: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Give each variant to exactly one hospital.

    A variant seen mostly in one population goes mostly to that population's
    hospital. A variant gnomAD never saw goes anywhere, in proportion to lab size.
    """
    tiny = 1e-6
    frequencies = training_pool[[f"af_{site.population}" for site in SITES.values()]].to_numpy() + tiny
    share_of_carriers = frequencies / frequencies.sum(axis=1, keepdims=True)
    lab_sizes = np.array([site.verdict_share for site in SITES.values()])

    chance = share_of_carriers * lab_sizes
    chance = chance / chance.sum(axis=1, keepdims=True)

    draw = rng.random(len(training_pool))[:, None]
    chosen = (draw > chance.cumsum(axis=1)).sum(axis=1)
    return pd.Series(np.array(list(SITES))[chosen], index=training_pool.index)


def simulate_patient_counts(table: pd.DataFrame, site: Site, rng: np.random.Generator) -> pd.DataFrame:
    """Carrier counts for every variant in one hospital's cohort.

    Patients carry what they carry, whether or not anyone has classified it, so
    this covers the whole table, test variants included.

    Counts among UNAFFECTED patients depend only on the real population
    frequency. They know nothing about the verdict, so they are fair evidence.
    Counts among AFFECTED patients are generated using the verdict. They exist to
    demonstrate the query, and must never be fed to a model.
    """
    affected = round(site.patients * AFFECTED_SHARE)
    unaffected = site.patients - affected
    frequency = table[f"af_{site.population}"].to_numpy()
    is_pathogenic = table.label.to_numpy() == 1

    carriers_unaffected = rng.binomial(2 * unaffected, frequency)

    frequency_in_affected = np.where(is_pathogenic, np.minimum(frequency * PATHOGENIC_ENRICHMENT, 0.5), frequency)
    carriers_affected = rng.binomial(2 * affected, frequency_in_affected)

    # Heart patients whose disease is explained by one pathogenic variant from the table.
    solved_cases = rng.choice(np.flatnonzero(is_pathogenic), size=round(affected * DIAGNOSTIC_YIELD))
    carriers_affected = carriers_affected + np.bincount(solved_cases, minlength=len(table))

    return pd.DataFrame({
        "variant_id": table.variant_id.to_numpy(),
        "name": table.name.to_numpy(),
        "ac": carriers_affected + carriers_unaffected,
        "an": 2 * site.patients,
        "ac_affected": carriers_affected,
        "an_affected": 2 * affected,
        "ac_unaffected": carriers_unaffected,
        "an_unaffected": 2 * unaffected,
    })


def hospital_view(rows: pd.DataFrame, public_reference: pd.DataFrame, patient_counts: pd.DataFrame) -> pd.DataFrame:
    """What one hospital knows about its own classified variants: no gnomAD truth columns."""
    is_truth = [c.startswith(TRUTH_COLUMNS_PREFIXES) or c in TRUTH_COLUMNS_EXACT for c in rows.columns]
    view = rows.loc[:, [not t for t in is_truth]]

    local = patient_counts.assign(af_local=patient_counts.ac_unaffected / patient_counts.an_unaffected)
    view = view.merge(public_reference, on="variant_id").merge(local[["variant_id", "af_local"]], on="variant_id")

    front = ["name", "gene", "verdict", "label", "stars", "af_public", "af_local"]
    return view[front + [c for c in view.columns if c not in front]].round({"af_local": 6})


def save(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")  # same bytes on Windows, Mac and Linux


def fingerprint(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for file in sorted(files):
        digest.update(file.read_bytes())
    return digest.hexdigest()[:16]


def compare_with_reference(build: dict, set_reference: bool) -> None:
    if set_reference:
        REFERENCE_BUILD.write_text(json.dumps({"made_on": str(date.today()), **build}, indent=2) + "\n", encoding="utf-8")
        print(f"\nThis build is now the team's reference: {REFERENCE_BUILD.relative_to(ROOT)}. Commit that file.")
        return
    if not REFERENCE_BUILD.exists():
        print("\nNo reference build yet. Run once with --set-reference and commit config/reference_build.json.")
        return

    reference = json.loads(REFERENCE_BUILD.read_text(encoding="utf-8"))
    if all(build[key] == reference[key] for key in build):
        print(f"\nIDENTICAL to the team's reference build of {reference['made_on']}. You have exactly the same hospitals.")
    elif build["variant_table"] != reference["variant_table"]:
        print(
            f"\nNOT the team's reference build: your data/variants.csv differs "
            f"({build['variant_rows']} rows, reference {reference['variant_rows']}).\n"
            f"  myvariant.info has probably updated its data since {reference['made_on']}.\n"
            "  To match the team exactly, get a teammate's data/raw folder and rerun step 1 without --refresh.\n"
            "  Or agree on a new reference: rerun this with --set-reference and commit the file."
        )
    else:
        print("\nNOT the team's reference build, although the variant table matches. Check that the settings "
              "at the top of this script and uv.lock are unchanged.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Lock a test set and simulate three hospitals")
    parser.add_argument("--set-reference", action="store_true", help="declare this build the team's reference")
    args = parser.parse_args()

    table_file = DATA_DIR / "variants.csv"
    if not table_file.exists():
        print("data/variants.csv is missing. Run scripts/01_build_table.py first.", file=sys.stderr)
        return 1
    table = pd.read_csv(table_file)
    rng = np.random.default_rng(SEED)

    test, training_pool, unseen_genes = lock_test_set(table, rng)
    site_of_variant = deal_verdicts(training_pool, rng)

    public_reference = table[["variant_id"]].assign(af_public=table[f"af_{PUBLIC_REFERENCE_POPULATION}"])
    written = [DATA_DIR / "public_reference.csv", DATA_DIR / "test" / "variants.csv"]
    save(public_reference, written[0])
    save(test.merge(public_reference, on="variant_id"), written[1])

    run = {"seed": SEED, "unseen_genes": unseen_genes, "test_rows": len(test), "sites": {}}
    print(f"test set: {len(test)} rows  ({int((test.test_kind == 'random').sum())} random, "
          f"{int((test.test_kind == 'unseen_gene').sum())} from unseen genes: {', '.join(unseen_genes)})\n")
    print(f"{'hospital':<13} {'population':<11} {'patients':>8} {'verdicts':>9} {'pathogenic':>11} {'benign':>7} {'discordant':>11}")

    for site_name, site in SITES.items():
        folder = DATA_DIR / site_name
        patient_counts = simulate_patient_counts(table, site, rng)
        own_rows = training_pool[site_of_variant == site_name]
        save(patient_counts, folder / "patient_counts.csv")
        save(hospital_view(own_rows, public_reference, patient_counts), folder / "verdicts.csv")
        written += [folder / "patient_counts.csv", folder / "verdicts.csv"]

        pathogenic = int(own_rows.label.sum())
        print(f"{site_name:<13} {site.population:<11} {site.patients:>8} {len(own_rows):>9} "
              f"{pathogenic:>11} {len(own_rows) - pathogenic:>7} {int(own_rows.pop_discordant.sum()):>11}")
        run["sites"][site_name] = {
            "population": site.population,
            "patients": site.patients,
            "verdicts": len(own_rows),
            "pathogenic": pathogenic,
            "benign": len(own_rows) - pathogenic,
        }

    build = {
        "variant_rows": len(table),
        "variant_table": fingerprint([table_file]),
        "hospital_files": fingerprint(written),
    }
    run["build"] = build
    (DATA_DIR / "sites.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print_example(table, "DSP N1526K")
    print(f"\nwrote data/test/, data/public_reference.csv, data/sites.json and {len(SITES)} hospital folders")
    compare_with_reference(build, args.set_reference)
    return 0


def print_example(table: pd.DataFrame, name: str) -> None:
    """Show one variant across the hospitals, as the step 6 query will."""
    wanted = table.loc[table.name == name, "variant_id"]
    if wanted.empty:
        return
    print(f"\n{name} as each hospital's patients show it:")
    for site_name in SITES:
        counts = pd.read_csv(DATA_DIR / site_name / "patient_counts.csv")
        row = counts[counts.variant_id == wanted.iloc[0]].iloc[0]
        print(f"  {site_name:<13} carriers {row.ac:>5} of {row.an:>6}   "
              f"among affected {row.ac_affected:>4} of {row.an_affected:>5}   "
              f"among unaffected {row.ac_unaffected:>5} of {row.an_unaffected:>6}")


if __name__ == "__main__":
    raise SystemExit(main())
