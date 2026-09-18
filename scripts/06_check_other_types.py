"""Step 6, checked for the other mutation types: what does the count query buy beyond missense?

Every number here is computed at run time from data/other_types/, on rows with a clean
ClinVar verdict unless it says otherwise. Contested variants have no truth to score against.

    1. rows by mutation type and verdict
    2. how often the starting presumption from the mutation type is right
    3. the false alarms of the type rule: harmless variants that the type presumes harmful.
       How many does each source of frequency evidence clear, for a patient at each hospital?
    4. the cost: harmful variants that a source wrongly clears, by name
    5. population-discordant variants by verdict
    6. how many insertions and deletions have more than one valid spelling
    7. the MYBPC3 deletion story: the count each hospital gives, the zero without the
       spelling fix, the count with it, and ClinVar's contested verdicts

Counts among sick patients are generated from the verdict and are never used here as evidence.

Reads   data/other_types/, data/raw_other_types/MYBPC3.json, and the missense hospital files through hospital_query
Writes  data/other_types/results_checks.json

Usage:
    uv run python scripts/06_check_other_types.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

import variant_spelling
from hospital_query import (
    DEFAULT_MIN_COUNT, MISSENSE, PRESUMED_HARMFUL, PRESUMED_HARMLESS,
    list_sites, overview, query, _public_reference,
)

ROOT = Path(__file__).resolve().parents[1]
OTHER_TYPES_DIR = ROOT / "data" / "other_types"
MYBPC3_CACHE = ROOT / "data" / "raw_other_types" / "MYBPC3.json"

HARMFUL_TYPES = list(PRESUMED_HARMFUL)
HARMLESS_TYPES = list(PRESUMED_HARMLESS)


def load() -> pd.DataFrame:
    table_file = OTHER_TYPES_DIR / "variants.csv"
    if not table_file.exists():
        raise FileNotFoundError("data/other_types/variants.csv is missing. Run scripts/01_build_other_types.py first.")
    return pd.read_csv(table_file)


def calls_for(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """overview() at each hospital, restricted to this table's rows and joined to the verdict."""
    verdicts = table.set_index("variant_id")[["verdict", "mutation_type"]]
    by_site = {}
    for site in list_sites():
        calls = overview(site, DEFAULT_MIN_COUNT)
        calls = calls[calls.mutation_type != MISSENSE].join(verdicts.verdict, how="inner")
        by_site[site] = calls
    return by_site


def rows_by_type(table: pd.DataFrame) -> pd.DataFrame:
    counts = pd.crosstab(table.mutation_type, table.verdict)
    order = [t for t in ["protein_cutting", "frameshift", "splice_site", "start_or_stop_lost", "inframe_indel",
                         "missense_no_scores", "splice_region", "synonymous", "utr", "intronic", "other"] if t in counts.index]
    return counts.reindex(order).reindex(columns=["pathogenic", "benign", "contested"], fill_value=0)


def presumption_accuracy(clean: pd.DataFrame) -> dict:
    """Per type with a presumption: how often the presumption equals the verdict."""
    out = {}
    for mutation_type in HARMFUL_TYPES + HARMLESS_TYPES:
        rows = clean[clean.mutation_type == mutation_type]
        if rows.empty:
            continue
        presumed = "pathogenic" if mutation_type in HARMFUL_TYPES else "benign"
        right = int((rows.verdict == presumed).sum())
        out[mutation_type] = {"presumed": presumed, "rows": len(rows), "right": right, "share_right": round(right / len(rows), 4)}
    with_presumption = clean[clean.mutation_type.isin(HARMFUL_TYPES + HARMLESS_TYPES)]
    presumed = with_presumption.mutation_type.map(lambda t: "pathogenic" if t in HARMFUL_TYPES else "benign")
    out["all_types_with_a_presumption"] = {
        "rows": len(with_presumption), "right": int((presumed == with_presumption.verdict).sum()),
        "share_right": round(float((presumed == with_presumption.verdict).mean()), 4) if len(with_presumption) else None,
    }
    return out


def cleared_by_source(calls_by_site: dict[str, pd.DataFrame], verdict: str, types: list[str]) -> dict:
    """Among variants of `verdict` and these types: how many each source clears as LIKELY HARMLESS, per patient site.

    public                 the public reference alone, Europeans only
    own_and_public         the patient's own hospital plus the public reference (the query's `before`)
    all_hospitals          every hospital plus the public reference (the query's `after`)
    all_hospitals_rule_1   the same, by rule 1 alone: common among healthy patients somewhere. Rule 2
                           can take a clearance back, but it reads the counts among sick patients,
                           which are generated from the verdict. This line is free of them.

    Rule 1's line is the gene's own: 0.1% when one bad copy is enough to cause disease, 1% when both
    copies must be bad. overview() carries it in its `too_common` column.
    """
    public = pd.Series(_public_reference())
    out = {}
    for site, calls in calls_by_site.items():
        rows = calls[(calls.verdict == verdict) & (calls.mutation_type.isin(types))]
        common_in_public = public.reindex(rows.index).fillna(0.0) >= rows.too_common
        cleared_public = rows[common_in_public]
        cleared_own = rows[rows.before == "LIKELY HARMLESS"]
        cleared_all = rows[rows.after == "LIKELY HARMLESS"]
        cleared_rule_1 = rows[common_in_public | (rows.best_frequency >= rows.too_common)]
        out[site] = {
            "rows": len(rows),
            "public": len(cleared_public),
            "own_and_public": len(cleared_own),
            "all_hospitals": len(cleared_all),
            "all_hospitals_rule_1": len(cleared_rule_1),
            "names_public": sorted(cleared_public.name),
            "names_own_and_public": sorted(cleared_own.name),
            "names_all_hospitals": sorted(cleared_all.name),
            "names_all_hospitals_rule_1": sorted(cleared_rule_1.name),
        }
    return out


def discordant_by_verdict(table: pd.DataFrame) -> dict:
    discordant = table[table.pop_discordant == 1]
    return {
        "rows": len(discordant),
        **{verdict: int((discordant.verdict == verdict).sum()) for verdict in ("pathogenic", "benign", "contested")},
        "harmless_and_presumed_harmful": sorted(discordant[(discordant.verdict == "benign") & discordant.mutation_type.isin(HARMFUL_TYPES)].name),
    }


def spellings(table: pd.DataFrame) -> dict:
    slides = table[table.kind.isin(["insertion", "deletion"])]
    return {
        "insertions_and_deletions": len(slides),
        "with_more_than_one_spelling": int((slides.shift_room > 0).sum()),
        "share": round(float((slides.shift_room > 0).mean()), 4) if len(slides) else None,
        "most_room_to_slide": int(slides.shift_room.max()) if len(slides) else None,
        "database_id_is_the_canonical_one": int((slides.database_id == slides.variant_id).sum()),
        "database_id_is_the_rightmost_one": int((slides.database_id == slides.rightmost_id).sum()),
    }


def mybpc3_story(table: pd.DataFrame) -> dict:
    """One deletion, two ids: what each hospital answers with and without the spelling fix."""
    both = [variant_spelling.MYBPC3_AS_CLINVAR_AND_GNOMAD, variant_spelling.MYBPC3_AS_DBSNP]
    row = table[table.variant_id == both[0]]
    if row.empty:
        return {"missing": "the MYBPC3 deletion is not in the table"}
    row = row.iloc[0]
    story = {
        "name": row["name"], "canonical_id": row.variant_id, "rightmost_id": row.rightmost_id, "database_id": row.database_id,
        "verdict": row.verdict, "mutation_type": row.mutation_type, "shift_room": int(row.shift_room),
        "valid_spellings": len(variant_spelling.every_spelling(row.variant_id)),
        "gnomad": {pop: float(row[f"af_{pop}"]) for pop in ("nfe", "sas", "afr")},
        "asked": {},
    }
    for asked_as in both:
        for spelling_fix in (False, True):
            result = query(asked_as, "site_oslo", 0, spelling_fix=spelling_fix)
            story["asked"][f"{asked_as} | spelling_fix={spelling_fix}"] = {
                "call_all_hospitals": result.after.call,
                "healthy_carriers": {a.site: {"carriers": a.healthy.carriers, "of": a.healthy.total} for a in result.answers},
            }
    if MYBPC3_CACHE.exists():
        records = {r["_id"]: r for r in json.loads(MYBPC3_CACHE.read_text(encoding="utf-8"))}
        record = records.get(row.database_id, {})
        rcv = record.get("clinvar", {}).get("rcv", [])
        rcv = [rcv] if isinstance(rcv, dict) else rcv
        story["clinvar_records"] = [{"says": r.get("clinical_significance"), "review": r.get("review_status")} for r in rcv]
    return story


def main() -> int:
    table = load()
    clean = table[table.verdict != "contested"]
    calls_by_site = calls_for(table)

    by_type = rows_by_type(table)
    results = {
        "rows": len(table), "clean_rows": len(clean), "contested_rows": int((table.verdict == "contested").sum()),
        "rows_by_type_and_verdict": by_type.to_dict(orient="index"),
        "presumption_right": presumption_accuracy(clean),
        "false_alarms_of_the_type_rule": {
            "what": "harmless variants of a type presumed harmful, and how many each frequency source clears",
            "types": HARMFUL_TYPES, **cleared_by_source(calls_by_site, "benign", HARMFUL_TYPES),
        },
        "cost_harmful_variants_wrongly_cleared": {
            "what": "harmful variants of any type that a frequency source clears as LIKELY HARMLESS",
            **cleared_by_source(calls_by_site, "pathogenic", list(by_type.index)),
        },
        "population_discordant": discordant_by_verdict(table),
        "spellings": spellings(table),
        "mybpc3": mybpc3_story(table),
    }
    (OTHER_TYPES_DIR / "results_checks.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print_summary(results, by_type)
    return 0


def print_summary(results: dict, by_type: pd.DataFrame) -> None:
    print(f"{results['rows']} rows: {results['clean_rows']} with a clean verdict, {results['contested_rows']} contested\n")
    print("rows by mutation type and verdict:")
    print(by_type.to_string(), "\n")

    print("the starting presumption from the type, against the ClinVar verdict:")
    for mutation_type, numbers in results["presumption_right"].items():
        print(f"  {mutation_type:<32} {numbers['right']:>6} of {numbers['rows']:>6} right  ({numbers['share_right']:.1%})")

    print("\nfalse alarms of the type rule: harmless variants presumed harmful, and how many each source clears")
    print(f"  {'patient at':<14} {'of':>5} {'public':>8} {'own+public':>11} {'all hospitals':>14} {'rule 1 alone':>13}")
    for site, numbers in results["false_alarms_of_the_type_rule"].items():
        if isinstance(numbers, dict):
            print(f"  {site:<14} {numbers['rows']:>5} {numbers['public']:>8} {numbers['own_and_public']:>11} "
                  f"{numbers['all_hospitals']:>14} {numbers['all_hospitals_rule_1']:>13}")
    print("  cleared by rule 1 alone: " + ", ".join(results["false_alarms_of_the_type_rule"]["site_oslo"]["names_all_hospitals_rule_1"]))

    print("\nthe cost: harmful variants a source wrongly clears")
    for site, numbers in results["cost_harmful_variants_wrongly_cleared"].items():
        if isinstance(numbers, dict):
            print(f"  {site:<14} public {numbers['public']} ({', '.join(numbers['names_public']) or 'none'}), "
                  f"own+public {numbers['own_and_public']} ({', '.join(numbers['names_own_and_public']) or 'none'}), "
                  f"all hospitals {numbers['all_hospitals']} ({', '.join(numbers['names_all_hospitals']) or 'none'}), "
                  f"rule 1 alone {numbers['all_hospitals_rule_1']} ({', '.join(numbers['names_all_hospitals_rule_1']) or 'none'})")
    print("  rule 2 can take a clearance back, and it reads the counts among sick patients, which are generated: hence 'rule 1 alone'")

    discordant = results["population_discordant"]
    print(f"\npopulation-discordant: {discordant['rows']} rows, {discordant['pathogenic']} pathogenic, {discordant['benign']} benign, "
          f"{discordant['contested']} contested")
    print(f"  harmless, presumed harmful by type and lopsided across populations: {discordant['harmless_and_presumed_harmful'] or 'none'}")

    spelled = results["spellings"]
    print(f"\ninsertions and deletions: {spelled['insertions_and_deletions']}, with more than one valid spelling: "
          f"{spelled['with_more_than_one_spelling']} ({spelled['share']:.1%}), most room to slide: {spelled['most_room_to_slide']} letters")

    story = results["mybpc3"]
    if "missing" in story:
        print(f"\n{story['missing']}")
        return
    print(f"\n{story['name']}: {story['mutation_type']}, ClinVar {story['verdict']}, {story['valid_spellings']} valid spellings, "
          f"gnomAD nfe {story['gnomad']['nfe']:.4%}, sas {story['gnomad']['sas']:.2%}, afr {story['gnomad']['afr']:.4%}")
    for asked, answer in story["asked"].items():
        counts = ", ".join(f"{site.removeprefix('site_')} {c['carriers']} of {c['of']}" for site, c in answer["healthy_carriers"].items())
        print(f"  {asked:<58} {answer['call_all_hospitals']:<24} healthy carriers: {counts}")
    if "clinvar_records" in story:
        print("  ClinVar records: " + "; ".join(f"{r['says']} ({r['review']})" for r in story["clinvar_records"]))
    print("\nCounts among sick patients are generated from the verdict: demo only, never evidence.")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as problem:
        print(problem, file=sys.stderr)
        raise SystemExit(1)
