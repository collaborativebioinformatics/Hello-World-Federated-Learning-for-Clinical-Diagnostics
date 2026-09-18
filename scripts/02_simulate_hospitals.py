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

Every run ends by checking itself: no test variant reaches a hospital, no
hospital lists a variant twice, every hospital has both verdicts and enough
positives to train on, and each hospital's af_local really comes from its own
population. --self-check proves those checks can still fail.

Usage:
    uv run python scripts/02_simulate_hospitals.py
    uv run python scripts/02_simulate_hospitals.py --panel cancer    # another disease area, from data/cancer/
    uv run python scripts/02_simulate_hospitals.py --set-reference   # declare this build the team's reference
    uv run python scripts/02_simulate_hospitals.py --self-check      # break the build on purpose, expect a stop

The seed is fixed, so everyone who starts from the same data/variants.csv gets
byte-identical files. The script says whether your build matches the team's
reference in config/reference_build.json. It can differ only if myvariant.info
has updated its data since the reference was made.

`cardiac` is the default panel and keeps the top of data/. Any other panel reads
data/<panel>/variants.csv, writes the same files into data/<panel>/ and keeps its
own reference in config/reference_build_<panel>.json, the rule step 1 follows.

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
DEFAULT_PANEL = "cardiac"

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

# How often a variant owned by one hospital is also classified by another. Real
# labs overlap. 0.0 is a clean partition, which keeps the federated-vs-pooled
# contrast sharpest; higher gives the small labs more rows to train on.
# Secondary pickup follows population alone, not lab size: a small lab still
# meets the variants common in its own population, it just classifies fewer
# variants overall.
SITE_OVERLAP = 0.5

# Test set: a random fifth of the variants, plus every variant of a few whole
# genes, so we can also ask "does it work on a gene it has never seen?"
# The random fifth is split by amino-acid POSITION, not by row: two DNA changes
# that both give ACTA2 M46I have near-identical scores and usually the same
# verdict, so splitting them across test and training flatters the model.
TEST_SHARE = 0.20
UNSEEN_GENES = 6
MIN_PER_VERDICT = 10  # an unseen gene needs this many pathogenic and benign rows

# A hospital with fewer positives than this cannot train; the run stops.
MIN_PATHOGENIC_PER_SITE = 100

# Patient simulation. Illustrative settings, not estimates.
AFFECTED_SHARE = 0.15  # share of a cohort that are heart patients
DIAGNOSTIC_YIELD = 0.20  # share of heart patients explained by one pathogenic variant
PATHOGENIC_ENRICHMENT = 5  # how much more common a pathogenic variant is among the affected

# Columns that are truth from gnomAD. They stay out of the hospital files.
TRUTH_COLUMNS_PREFIXES = ("af_", "ac_", "an_")
TRUTH_COLUMNS_EXACT = ("in_gnomad", "pop_discordant")


def amino_acid_position(table: pd.DataFrame) -> pd.Series:
    """`MYH7 R403Q` -> `MYH7:403`, the protein position several DNA changes can share.

    The digits are taken from the amino-acid change only. Gene names carry digits
    of their own, so `COL1A1 G272D` must not be read as position 1.
    """
    number = table.name.str.extract(r" [A-Za-z](\d+)[A-Za-z]$", expand=False)
    return (table.gene + ":" + number).fillna(table.variant_id)


def population_of(table: pd.DataFrame) -> pd.Series:
    """The site population a variant is most common in, or `none` if gnomAD never saw it.

    Same rule for every row. Two thirds of the table is `none`, so a per-population
    slice is a thin instrument; see docs/step2_review.md.
    """
    frequencies = table[[f"af_{site.population}" for site in SITES.values()]]
    highest = frequencies.idxmax(axis=1).str.removeprefix("af_")
    return highest.where(frequencies.max(axis=1) > 0, "none")


def lock_test_set(table: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    per_gene = table.groupby("gene").label.agg(pathogenic="sum", rows="size")
    per_gene["benign"] = per_gene.rows - per_gene.pathogenic
    has_both = per_gene[(per_gene.pathogenic >= MIN_PER_VERDICT) & (per_gene.benign >= MIN_PER_VERDICT)]
    unseen_genes = sorted(rng.choice(sorted(has_both.index), size=UNSEEN_GENES, replace=False))

    in_unseen_gene = table.gene.isin(unseen_genes)
    rest = table[~in_unseen_gene]
    # Split by amino-acid position, so every variant at one position lands on the
    # same side. Keep the mix of verdicts and of population-discordant variants
    # the same in test and training: a position counts as pathogenic, or as
    # discordant, if any of its variants is.
    position = amino_acid_position(rest)
    strata = rest[["label", "pop_discordant"]].groupby(position).max()
    chosen = strata.groupby(["label", "pop_discordant"], group_keys=False).sample(frac=TEST_SHARE, random_state=SEED)
    random_fifth = rest[position.isin(chosen.index)]

    test = pd.concat([
        table[in_unseen_gene].assign(test_kind="unseen_gene"),
        random_fifth.assign(test_kind="random"),
    ])
    training_pool = rest.drop(random_fifth.index)
    return test, training_pool, unseen_genes


def deal_verdicts(training_pool: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Decide which hospitals have classified which variant.

    Every variant gets one owner: a variant seen mostly in one population goes
    mostly to that population's hospital, and a variant gnomAD never saw goes
    anywhere, in proportion to lab size.

    With SITE_OVERLAP above zero another hospital can also hold it. That second
    draw follows the population alone, so a small lab still meets the variants
    common among its own patients.

    Returns a boolean frame, one column per hospital, one row per variant.
    """
    tiny = 1e-6
    frequencies = training_pool[[f"af_{site.population}" for site in SITES.values()]].to_numpy() + tiny
    share_of_carriers = frequencies / frequencies.sum(axis=1, keepdims=True)
    lab_sizes = np.array([site.verdict_share for site in SITES.values()])

    chance = share_of_carriers * lab_sizes
    chance = chance / chance.sum(axis=1, keepdims=True)

    draw = rng.random(len(training_pool))[:, None]
    owner = (draw > chance.cumsum(axis=1)).sum(axis=1)
    holds = np.zeros_like(chance, dtype=bool)
    holds[np.arange(len(owner)), owner] = True

    also = rng.random(chance.shape) < SITE_OVERLAP * share_of_carriers
    return pd.DataFrame(holds | also, index=training_pool.index, columns=list(SITES))


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


def check(test: pd.DataFrame, hospitals: dict[str, pd.DataFrame]) -> None:
    """The four ways this split can be silently wrong. Cheap to check, fatal to miss."""
    locked = set(test.variant_id)
    for name, rows in hospitals.items():
        # A test variant in a training file makes every later number meaningless.
        assert not locked & set(rows.variant_id), f"{name} holds test variants"
        # One row per variant per hospital: a duplicate would weight it twice.
        assert rows.variant_id.is_unique, f"{name} lists a variant twice"
        assert rows.label.nunique() == 2, f"{name} has only one verdict, nothing to train on"
        assert rows.label.sum() >= MIN_PATHOGENIC_PER_SITE, (
            f"{name} has {int(rows.label.sum())} pathogenic rows, under {MIN_PATHOGENIC_PER_SITE}"
        )

    # af_local must come from this hospital's own patients. Wiring the wrong
    # population in here would quietly delete the whole point of the project, and
    # the file would still look perfectly normal.
    truth = pd.read_csv(DATA_DIR / "variants.csv").set_index("variant_id")
    columns = [f"af_{s.population}" for s in SITES.values()]
    for name, site in SITES.items():
        rows = hospitals[name]
        seen = truth.loc[rows.variant_id, columns]
        # Only variants gnomAD actually saw. On the all-zero rows every population
        # ties, and a tie would let a miswired column pass.
        informative = seen.max(axis=1) > 0
        closest = seen[informative].sub(rows.af_local.to_numpy()[informative], axis=0).abs().sum().idxmin()
        assert closest == f"af_{site.population}", f"{name} af_local looks like {closest}, not its own population"


def self_check(panel: str) -> int:
    """Prove check() is not vacuous, by breaking the plumbing and expecting a stop.

    An assertion that cannot fail is worse than none: it reads like a guarantee.
    An earlier version of check() compared af_local against the same constant it
    was derived from and passed no matter what, which is why this exists.

    The broken run writes its files before check() stops it, so this always
    rebuilds afterwards. Leaving a deliberately wrong build on disk would be a
    nastier bug than the one being tested for.
    """
    global simulate_patient_counts
    honest = simulate_patient_counts
    simulate_patient_counts = lambda table, site, rng: honest(table, SITES["site_lagos"], rng)
    try:
        main(["--panel", panel])   # without --self-check, so this does not recurse
        passed = False
    except AssertionError as caught:
        print(f"\nself-check passed: the broken build was stopped with {caught}")
        passed = True
    finally:
        simulate_patient_counts = honest

    print("\nrebuilding the real hospitals, since the broken run overwrote them\n")
    main(["--panel", panel])
    if passed:
        return 0
    print("SELF-CHECK FAILED: every hospital was given Lagos's cohort and check() said nothing.", file=sys.stderr)
    return 1


def save(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")  # same bytes on Windows, Mac and Linux


def fingerprint(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for file in sorted(files):
        digest.update(file.read_bytes())
    return digest.hexdigest()[:16]


def data_dir(panel: str) -> Path:
    """The cardiac build keeps its place at the top of data/. Every other panel gets data/<panel>/, as in step 1."""
    return ROOT / "data" if panel == DEFAULT_PANEL else ROOT / "data" / panel


def reference_file(panel: str) -> Path:
    """One reference build per panel. The cardiac one keeps its old name."""
    return ROOT / "config" / ("reference_build.json" if panel == DEFAULT_PANEL else f"reference_build_{panel}.json")


def compare_with_reference(build: dict, set_reference: bool) -> None:
    if set_reference:
        REFERENCE_BUILD.write_text(json.dumps({"made_on": str(date.today()), **build}, indent=2) + "\n", encoding="utf-8")
        print(f"\nThis build is now the team's reference: {REFERENCE_BUILD.relative_to(ROOT)}. Commit that file.")
        return
    if not REFERENCE_BUILD.exists():
        print(f"\nNo reference build yet. Run once with --set-reference and commit {REFERENCE_BUILD.relative_to(ROOT).as_posix()}.")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lock a test set and simulate three hospitals")
    parser.add_argument("--panel", default=DEFAULT_PANEL, help="disease area: data/ for cardiac, data/<panel>/ for any other (default: cardiac)")
    parser.add_argument("--set-reference", action="store_true", help="declare this build the team's reference")
    parser.add_argument("--self-check", action="store_true", help="prove the safety checks can actually fail")
    args = parser.parse_args(argv)
    if args.self_check:
        return self_check(args.panel)

    global DATA_DIR, REFERENCE_BUILD
    DATA_DIR, REFERENCE_BUILD = data_dir(args.panel), reference_file(args.panel)
    table_file = DATA_DIR / "variants.csv"
    if not table_file.exists():
        print(f"{table_file.relative_to(ROOT).as_posix()} is missing. Run scripts/01_build_table.py first, with the same --panel.", file=sys.stderr)
        return 1
    table = pd.read_csv(table_file)
    rng = np.random.default_rng(SEED)

    test, training_pool, unseen_genes = lock_test_set(table, rng)
    held_by = deal_verdicts(training_pool, rng)

    public_reference = table[["variant_id"]].assign(af_public=table[f"af_{PUBLIC_REFERENCE_POPULATION}"])
    test = test.assign(pop=population_of(test))
    written = [DATA_DIR / "public_reference.csv", DATA_DIR / "test" / "variants.csv"]
    save(public_reference, written[0])
    save(test.merge(public_reference, on="variant_id"), written[1])

    run = {"seed": SEED, "site_overlap": SITE_OVERLAP, "unseen_genes": unseen_genes,
           "test_rows": len(test), "sites": {}}
    print(f"test set: {len(test)} rows  ({int((test.test_kind == 'random').sum())} random, "
          f"{int((test.test_kind == 'unseen_gene').sum())} from unseen genes: {', '.join(unseen_genes)})\n")
    print(f"{'hospital':<13} {'population':<11} {'patients':>8} {'verdicts':>9} {'pathogenic':>11} {'benign':>7} {'discordant':>11}")

    hospitals = {}
    for site_name, site in SITES.items():
        folder = DATA_DIR / site_name
        patient_counts = simulate_patient_counts(table, site, rng)
        own_rows = training_pool[held_by[site_name]]
        hospitals[site_name] = hospital_view(own_rows, public_reference, patient_counts)
        save(patient_counts, folder / "patient_counts.csv")
        save(hospitals[site_name], folder / "verdicts.csv")
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

    shared = int((held_by.sum(axis=1) > 1).sum())
    print(f"\noverlap {SITE_OVERLAP}: {shared} of {len(training_pool)} variants are held by more than one hospital")
    check(test, hospitals)
    print_population_slices(test)

    build = {
        "variant_rows": len(table),
        "variant_table": fingerprint([table_file]),
        "hospital_files": fingerprint(written),
    }
    run["build"] = build
    (DATA_DIR / "sites.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print_example(table, "DSP N1526K")
    print(f"\nwrote {DATA_DIR.relative_to(ROOT).as_posix()}/: test/, public_reference.csv, sites.json and {len(SITES)} hospital folders")
    compare_with_reference(build, args.set_reference)
    return 0


def print_population_slices(test: pd.DataFrame) -> None:
    """Per-population AUC needs positives. Print how few there are before anyone trusts one."""
    print("\ntest rows by population, the slices a per-population AUC would use:")
    counts = test.groupby("pop").label.agg(rows="size", pathogenic="sum")
    for population, row in counts.iterrows():
        warning = "  <- too few positives for an AUC" if 0 < row.pathogenic < 30 else ""
        print(f"  {population:<6} {row.rows:>5} rows {row.pathogenic:>5} pathogenic{warning}")
    print("  'none' means gnomAD never saw the variant in any of the three populations.")


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
