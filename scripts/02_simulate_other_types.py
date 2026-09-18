"""Step 2, for the other mutation types: the three hospitals' files for data/other_types/variants.csv.

Step 2 itself is not touched and not rerun. This script loads scripts/02_simulate_hospitals.py
and calls its own functions as they are: simulate_patient_counts(), deal_verdicts() and
hospital_view(), with its SITES and its PUBLIC_REFERENCE_POPULATION. So the cohorts have the
same sizes, the same share of heart patients and the same public reference covering Europeans
only. The random numbers come from a generator made here, so step 2's random stream and step 2's
files stay exactly as they were.

What differs from step 2:

    no test set        nothing trains on this table, so there is nothing to lock away.
    contested rows     a contested variant has no label. Step 2's function treats it like a
                       harmless one: its count among sick patients follows the population
                       frequency and nothing else. Those counts carry no signal, by construction.
    sick counts        step 2's function adds "solved cases" to the harmful variants of the table it
                       is given. Here that table is this one, so the counts among sick patients are
                       generated independently of the missense run. As everywhere, they are made from
                       the verdict: demo only, never evidence, never a model input.
    written_as         each hospital's lab writes insertions and deletions in its own convention.
                       `variant_id` stays the canonical spelling, which is what the query matches on.
                       `written_as` is what that lab would have typed into its own database:

                           site_oslo      leftmost, the way variant-calling tools write a VCF file
                           site_karachi   rightmost, the way the HGVS naming rules ask for
                           site_lagos     as the public database files it

                       A single-letter change can be written one way only, so there all three agree.
                       The query's --no-spelling-fix flag matches on `written_as` alone, to show what
                       goes wrong without a canonical spelling.

Reads   data/other_types/variants.csv
Writes  data/other_types/public_reference.csv       the frequency everyone already has (Europeans only)
        data/other_types/site_*/verdicts.csv        that hospital's classified variants, clean verdicts only
        data/other_types/site_*/patient_counts.csv  that hospital's carrier counts, every row of the table
        data/other_types/sites.json                 the settings and sizes of this run

Usage:
    uv run python scripts/02_simulate_other_types.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import variant_spelling

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "other_types"

SEED = 12  # the team's seed, in a generator of its own

# Which column of the table each hospital's lab copies its spelling from.
LAB_SPELLING = {"site_oslo": "leftmost", "site_karachi": "rightmost", "site_lagos": "as the public database files it"}

# Columns a hospital would not have: the other labs' spellings.
NOT_IN_A_HOSPITAL = ["rightmost_id", "database_id", "shift_room"]


def load_step2():
    """Import 02_simulate_hospitals.py, whose name starts with a digit, so that its own functions draw the patients."""
    spec = importlib.util.spec_from_file_location("step2", ROOT / "scripts" / "02_simulate_hospitals.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["step2"] = module  # a dataclass in the module looks itself up here while it is being made
    spec.loader.exec_module(module)
    return module


def written_by(site_name: str, table: pd.DataFrame) -> pd.Series:
    """The spelling this hospital's lab uses for every variant of the table."""
    if LAB_SPELLING[site_name] == "leftmost":
        return table.variant_id
    if LAB_SPELLING[site_name] == "rightmost":
        return table.rightmost_id
    # myvariant.info files a ClinVar duplication under the two positions the copy goes between, which no lab
    # would write. ClinVar itself shows a duplication the HGVS way, so that is what Lagos copies.
    return table.database_id.where(~table.database_id.str.endswith("dup"), table.rightmost_id)


def check(table: pd.DataFrame, step2, counts: dict[str, pd.DataFrame]) -> None:
    """The ways these files can be silently wrong."""
    columns = [f"af_{site.population}" for site in step2.SITES.values()]
    seen = table[columns].max(axis=1) > 0  # where gnomAD never saw the variant every population ties
    for site_name, site in step2.SITES.items():
        rows = counts[site_name]
        assert rows.variant_id.is_unique, f"{site_name} lists a variant twice"
        assert list(rows.variant_id) == list(table.variant_id), f"{site_name} does not cover the table row for row"

        # Healthy patients must come from this hospital's own population.
        healthy = (rows.ac_unaffected / rows.an_unaffected).to_numpy()
        closest = table.loc[seen, columns].sub(healthy[seen], axis=0).abs().sum().idxmin()
        assert closest == f"af_{site.population}", f"{site_name} healthy counts look like {closest}, not its own population"

        # Whatever the lab wrote has to be the same change as the canonical spelling.
        slides = rows[table.kind.isin(["insertion", "deletion"]).to_numpy()]
        wrong = [w for w, v in zip(slides.written_as, slides.variant_id, strict=True) if variant_spelling.canonical(w) != v]
        assert not wrong, f"{site_name} wrote {len(wrong)} variants in a way that is a different change, such as {wrong[:3]}"

    contested = (table.verdict == "contested").to_numpy()
    assert table.label[contested].isna().all(), "a contested variant has a label"


def main() -> int:
    table_file = OUT_DIR / "variants.csv"
    if not table_file.exists():
        print("data/other_types/variants.csv is missing. Run scripts/01_build_other_types.py first.", file=sys.stderr)
        return 1
    step2 = load_step2()
    table = pd.read_csv(table_file)
    rng = np.random.default_rng(SEED)

    judged = table[table.verdict != "contested"]  # only a clean verdict can sit in a hospital's verdict file
    held_by = step2.deal_verdicts(judged, rng)

    public_reference = table[["variant_id"]].assign(af_public=table[f"af_{step2.PUBLIC_REFERENCE_POPULATION}"])
    step2.save(public_reference, OUT_DIR / "public_reference.csv")

    run = {"seed": SEED, "rows": len(table), "contested_rows": int((table.verdict == "contested").sum()),
           "lab_spelling": LAB_SPELLING, "sites": {}}
    print(f"{'hospital':<13} {'population':<11} {'patients':>8} {'verdicts':>9} {'pathogenic':>11} {'benign':>7}  "
          f"insertions and deletions not written the canonical way")

    counts = {}
    for site_name, site in step2.SITES.items():
        folder = OUT_DIR / site_name
        counts[site_name] = step2.simulate_patient_counts(table, site, rng).assign(written_as=written_by(site_name, table).to_numpy())
        own_rows = judged[held_by[site_name]].assign(written_as=written_by(site_name, judged)[held_by[site_name]])
        view = step2.hospital_view(own_rows.drop(columns=NOT_IN_A_HOSPITAL), public_reference, counts[site_name].drop(columns="written_as"))
        step2.save(counts[site_name], folder / "patient_counts.csv")
        step2.save(view.astype({"label": int}), folder / "verdicts.csv")

        pathogenic = int(own_rows.label.sum())
        own_way = int((counts[site_name].written_as != counts[site_name].variant_id).sum())
        print(f"{site_name:<13} {site.population:<11} {site.patients:>8} {len(own_rows):>9} {pathogenic:>11} "
              f"{len(own_rows) - pathogenic:>7}  {own_way:>6}")
        run["sites"][site_name] = {
            "population": site.population, "patients": site.patients, "verdicts": len(own_rows),
            "pathogenic": pathogenic, "benign": len(own_rows) - pathogenic, "not_written_the_canonical_way": own_way,
        }

    check(table, step2, counts)
    (OUT_DIR / "sites.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"\nwrote data/other_types/public_reference.csv, data/other_types/sites.json and {len(step2.SITES)} hospital folders")
    print("Counts among sick patients are generated from the verdict: demo only, never evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
