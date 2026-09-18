"""Export the patient query as one web page that runs anywhere: docs/demo/index.html.

Usage:
    uv run python scripts/export_web_demo.py                 # every built disease area
    uv run python scripts/export_web_demo.py --areas cardiac  # a smaller page with one area

The page is docs/demo/template.html with the data filled in. It is a single file with no
server, no framework and no request at run time, so it works opened from disk (file://),
from GitHub Pages, from raw.githack.com or from any static host. It draws the same screen
as scripts/06_query_tui.py, character for character, and applies the two rules of
scripts/hospital_query.py, written again in JavaScript; for the same variant, hospital and
minimum count it must give exactly the same call.

What goes in, per variant: id, name, mutation type, ClinVar verdict and stars, the public
European frequency, which hospitals hold a verdict, and for every hospital the raw carrier
counts among healthy and among sick patients. The counts are raw so that the page can hide
small counts itself and the F2 key can switch the hiding off. The page never sees a
prediction score: the score columns of data/variants.csv carry non-commercial terms and
are not exported. What the trained model says about a missense variant is exported as the
model's own output, one probability per setting of the hiding, never the scores it read.

Counts among sick patients were generated from the verdict in step 2. They are not evidence.
The page says so in its own help text, and so does this file.

Every variant of every built area is exported, because the screen shows how many there are.
The JSON travels gzip-compressed inside the page, and the page unpacks it with the browser's
own DecompressionStream (no library).

Several things come straight from hospital_query.py so that the page cannot disagree with
the terminal: the demo examples, rule 1's line per gene, the trained model's probabilities,
the same-class carrier sums, and the order of the "changed by asking" and "kept flagged"
lists, which pandas sorts with an unstable sort that JavaScript cannot reproduce.

Rerun this after any rebuild of data/. When an area's other_types/ folder exists (the same
file layout for every small mutation that is not missense, from scripts/01_build_other_types.py
and scripts/02_simulate_other_types.py) its variants are exported too, with their mutation type.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import time
from datetime import date
from pathlib import Path

import pandas as pd

import hospital_query
import query_drawing
from hospital_query import DEFAULT_MIN_COUNT, MISSENSE, PILE_UP_FOLD, PRESUMED_HARMFUL

query_drawing.USE_GLYPHS = True  # the page is drawn in a browser, which has every glyph

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "docs" / "demo"
TEMPLATE = DEMO_DIR / "template.html"
OUTPUT = DEMO_DIR / "index.html"

MAX_PLAIN_BYTES = 1_000_000  # larger than this, the data travels gzip-compressed
MIN_COUNTS = [DEFAULT_MIN_COUNT, 0]  # the two settings F2 switches between

POPULATIONS = {"nfe": "European", "sas": "South Asian", "afr": "African"}
COUNT_COLUMNS = ["variant_id", "name", "ac_unaffected", "an_unaffected", "ac_affected", "an_affected"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Write docs/demo/index.html from the built data")
    parser.add_argument("--areas", nargs="*", help="disease areas to include (default: every built area, the heart area first)")
    args = parser.parse_args()

    built = hospital_query.available_areas()
    areas = args.areas or built
    missing = [a for a in areas if a not in built]
    if missing:
        raise SystemExit(f"not built: {', '.join(missing)}. Built areas: {', '.join(built)}")

    data = {
        "built": date.today().isoformat(),
        "default_area": areas[0],
        "area_order": areas,
        "area_labels": {area: query_drawing.area_label(area) for area in areas},
        "help": {"keys": query_drawing.KEYS, "what_you_see": query_drawing.WHAT_YOU_SEE},
        "areas": {},
    }
    for area in areas:
        started = time.time()
        hospital_query.set_area(area)
        data["areas"][area] = export_area(area)
        exported = data["areas"][area]
        print(f"{area}: {len(exported['rows']):,} variants from {', '.join(exported['tables'])} in {time.time() - started:.0f} s; "
              f"examples: {', '.join(exported['examples'])}")

    plain = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    packed = len(plain.encode("utf-8")) > MAX_PLAIN_BYTES
    html = build_html(plain, packed)
    OUTPUT.write_text(html, encoding="utf-8", newline="\n")

    rows = sum(len(a["rows"]) for a in data["areas"].values())
    how = f"gzip-compressed from {len(plain.encode('utf-8')):,} bytes of JSON" if packed else "plain JSON"
    print(f"wrote {relative(OUTPUT)}: {OUTPUT.stat().st_size:,} bytes, {rows:,} variant rows embedded over {len(areas)} area(s) ({how})")


# ---------------------------------------------------------------------------
# One disease area
# ---------------------------------------------------------------------------
def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def table_dirs() -> list[Path]:
    """The area's missense build, with its other_types/ under it when that exists. Same rule as hospital_query."""
    found = [hospital_query.area_folder()]
    if (hospital_query.other_types_folder() / "public_reference.csv").exists():
        found.append(hospital_query.other_types_folder())
    return found


def read_sites() -> list[dict]:
    run = hospital_query._run()
    return [
        {
            "id": site,
            "short": site.removeprefix("site_").capitalize(),
            "population": settings["population"],
            "population_name": POPULATIONS.get(settings["population"], settings["population"]),
            "patients": int(settings["patients"]),
        }
        for site, settings in run["sites"].items()
    ]


def export_area(area: str) -> dict:
    sites = read_sites()
    table = read_everything(sites)
    chosen = primary_index(table, sites)
    genes = sorted(set(table.gene))
    types = [MISSENSE] + sorted(set(table.mutation_type) - {MISSENSE})
    return {
        "name": area,
        "title": hospital_query.area_title(area),
        "tables": [relative(d) for d in table_dirs()],
        "kind_labels": {kind: query_drawing.kind_label(kind) for kind in types},
        "kind_glyphs": {kind: query_drawing.kind_glyph(kind) for kind in types},
        "aliases": aliases(table),
        "rules": {"too_common": hospital_query.TOO_COMMON, "too_common_two_copies": hospital_query.TOO_COMMON_TWO_COPIES,
                  "pile_up_fold": PILE_UP_FOLD, "min_count": DEFAULT_MIN_COUNT},
        "disease_words": hospital_query.disease_words(),
        "min_counts": MIN_COUNTS,
        "sites": site_list(table, sites),
        "genes": {gene: list(hospital_query.inheritance_line(gene)) for gene in genes},
        "same_class": same_class_sums(sites),
        "model": model_settings(),
        "examples": hospital_query.examples(),
        "orders": list_orders(chosen, sites),
        "types": types,
        "verdicts": [""] + sorted(set(table.verdict) - {""}),
        "fields": ["variant_id", "name", "type", "verdict", "stars", "public_frequency", "judged_by",
                   "counts (healthy, sick, per hospital in order; absent when all zero)",
                   "totals (only when they differ from the hospital's; null when not needed)",
                   "what the trained model says (missense only; absent otherwise): probability as a word and 1 when it is above "
                   "the cut-off, for the first min count, then again for the second when they differ"],
        "sick_counts": "simulated from the verdict in step 2. Demonstration only, never evidence.",
        "rows": rows_of(table, sites),
    }


def read_everything(sites: list[dict]) -> pd.DataFrame:
    """One row per variant id, in the order the first hospital's file lists them, both tables stacked."""
    table = pd.concat([read_table(table_dir, sites) for table_dir in table_dirs()], ignore_index=True)
    if not table.variant_id.is_unique:
        duplicated = table.variant_id[table.variant_id.duplicated()].head(3).tolist()
        raise ValueError(f"a variant id appears more than once across the tables, for example {duplicated}")
    table["gene"] = table.name.map(hospital_query.gene_of)
    return table


def read_table(table_dir: Path, sites: list[dict]) -> pd.DataFrame:
    table = pd.read_csv(table_dir / sites[0]["id"] / "patient_counts.csv", usecols=["variant_id", "name"])

    # What the patient's hospital knows about the variant itself: type, verdict, stars.
    table = table.merge(read_variants(table_dir / "variants.csv"), on="variant_id", how="left")
    table["mutation_type"] = table.mutation_type.fillna(MISSENSE)

    public = pd.read_csv(table_dir / "public_reference.csv")
    table["public_frequency"] = table.variant_id.map(dict(zip(public.variant_id, public.af_public, strict=True))).fillna(0.0)

    # Every hospital's raw counts, and which hospitals hold a verdict for the variant.
    judged = pd.Series(0, index=table.index)
    for bit, site in enumerate(sites):
        counts = pd.read_csv(table_dir / site["id"] / "patient_counts.csv", usecols=COUNT_COLUMNS).set_index("variant_id")
        counts = counts.reindex(table.variant_id)
        if counts.ac_unaffected.isna().any():
            raise ValueError(f"{site['id']} lists different variants from {sites[0]['id']} in {relative(table_dir)}")
        for column in ["ac_unaffected", "an_unaffected", "ac_affected", "an_affected"]:
            table[f"{site['id']}:{column}"] = counts[column].to_numpy().astype(int)
        verdicts = pd.read_csv(table_dir / site["id"] / "verdicts.csv", usecols=["variant_id", "verdict", "stars"])
        held = table.variant_id.isin(verdicts.variant_id)
        judged = judged + held.astype(int) * (1 << bit)
        missing = table.verdict.isna() & held  # a variant only a hospital has judged still gets its verdict shown
        if missing.any():
            by_id = verdicts.drop_duplicates("variant_id").set_index("variant_id")
            table.loc[missing, "verdict"] = table.variant_id[missing].map(by_id.verdict)
            table.loc[missing, "stars"] = table.variant_id[missing].map(by_id.stars)
    table["judged_by"] = judged
    table["verdict"] = table.verdict.fillna("")
    table["stars"] = table.stars.fillna(0).astype(int)
    return table


def read_variants(table_file: Path) -> pd.DataFrame:
    """Only the readable columns of variants.csv. The score columns are never read here."""
    wanted = ["variant_id", "verdict", "stars", "mutation_type"]
    if not table_file.exists():
        return pd.DataFrame(columns=wanted)
    have = [c for c in wanted if c in pd.read_csv(table_file, nrows=0).columns]
    variants = pd.read_csv(table_file, usecols=have).drop_duplicates("variant_id")
    for column in wanted:
        if column not in variants:
            variants[column] = pd.NA
    return variants[wanted]


def site_list(table: pd.DataFrame, sites: list[dict]) -> list[dict]:
    """Totals are the same for every variant at a hospital, so they travel once per hospital."""
    out = []
    for site in sites:
        healthy_total = int(table[f"{site['id']}:an_unaffected"].mode().iloc[0])
        sick_total = int(table[f"{site['id']}:an_affected"].mode().iloc[0])
        out.append({**site, "healthy_total": healthy_total, "sick_total": sick_total})
    return out


def best_frequencies(table: pd.DataFrame, sites: list[dict], min_count: int) -> pd.Series:
    """The highest frequency among healthy patients at any hospital, with small counts hidden: what query() feeds the model."""
    columns = []
    for site in sites:
        carriers = table[f"{site['id']}:ac_unaffected"]
        hidden = (carriers > 0) & (carriers < min_count)
        columns.append(carriers.mask(hidden, 0) / table[f"{site['id']}:an_unaffected"])
    return pd.concat(columns, axis=1).max(axis=1)


def rows_of(table: pd.DataFrame, sites: list[dict]) -> list[list]:
    type_index = {t: i for i, t in enumerate([MISSENSE] + sorted(set(table.mutation_type) - {MISSENSE}))}
    verdict_index = {v: i for i, v in enumerate([""] + sorted(set(table.verdict) - {""}))}
    site_totals = site_list(table, sites)
    best = {m: best_frequencies(table, sites, m) for m in MIN_COUNTS}
    model = hospital_query._model()

    rows = []
    for index, record in enumerate(table.to_dict("records")):
        counts, totals, own_totals = [], [], False
        for site in site_totals:
            healthy, healthy_total = record[f"{site['id']}:ac_unaffected"], record[f"{site['id']}:an_unaffected"]
            sick, sick_total = record[f"{site['id']}:ac_affected"], record[f"{site['id']}:an_affected"]
            counts += [int(healthy), int(sick)]
            totals += [int(healthy_total), int(sick_total)]
            own_totals |= healthy_total != site["healthy_total"] or sick_total != site["sick_total"]
        row = [
            record["variant_id"],
            record["name"],
            type_index[record["mutation_type"]],
            verdict_index[record["verdict"]],
            int(record["stars"]),
            float(record["public_frequency"]),
            int(record["judged_by"]),
        ]
        said = None
        if record["mutation_type"] == MISSENSE and model is not None:
            verdicts = [hospital_query.model_verdict(record["variant_id"], float(best[m].iloc[index])) for m in MIN_COUNTS]
            if all(v.probability is not None for v in verdicts):
                # What the sentence shows, per setting of the hiding: the probability as a word, and its side of the cut-off.
                said = []
                for v in verdicts:
                    said += [hospital_query.percent_word(v.probability), int(v.probability >= model.threshold)]
                if said[:2] == said[2:]:
                    said = said[:2]
        if any(counts) or own_totals or said:
            row.append(counts)  # a row with no carrier anywhere stops here: the page reads missing counts as zero
        if own_totals or said:
            row.append(totals if own_totals else None)
        if said:
            row.append(said)
        rows.append(row)
    return rows


def primary_index(table: pd.DataFrame, sites: list[dict]) -> dict[str, int]:
    """For every name, the row the page shows: the DNA change patients carry most, the first listed on a tie."""
    carried = sum(table[f"{site['id']}:ac_unaffected"] + table[f"{site['id']}:ac_affected"] for site in sites)
    chosen: dict[str, int] = {}
    best: dict[str, int] = {}
    for index, (name, count) in enumerate(zip(table.name, carried, strict=True)):
        if name not in chosen or count > best[name]:
            chosen[name], best[name] = index, int(count)
    return chosen


def list_orders(chosen: dict[str, int], sites: list[dict]) -> dict:
    """The "changed by asking" and "kept flagged" lists in the order the terminal shows them, per setting.

    scripts/06_query_tui.py sorts them by best_frequency with pandas, whose sort is not stable, so the
    order of tied rows cannot be recomputed in JavaScript. The names are taken from hospital_query.overview()
    itself and stored as row indices.
    """
    orders = {}
    for site in sites:
        for min_count in MIN_COUNTS:
            overview = hospital_query.overview(site["id"], min_count)
            if set(overview.name) != set(chosen):
                raise ValueError("hospital_query.overview() lists different names from the exported table")
            changed = overview[overview.changed].sort_values("best_frequency", ascending=False).name
            flagged = overview[overview.after == "KEEP FLAGGED"].sort_values("best_frequency", ascending=False).name
            orders[f"{site['id']}|{min_count}"] = {
                "changed": [chosen[name] for name in changed],
                "flagged": [chosen[name] for name in flagged],
            }
    return orders


def same_class_sums(sites: list[dict]) -> dict:
    """Per gene and hospital: carriers of any protein-cutting, frameshift or splice-site change, healthy and sick."""
    out: dict[str, dict[str, list[int]]] = {}
    for site in sites:
        sums = hospital_query._same_class_sums(site["id"])
        for gene, row in sums.iterrows():
            out.setdefault(gene, {})[site["id"]] = [int(row.ac_unaffected), int(row.an_unaffected), int(row.ac_affected), int(row.an_affected)]
    return {"kinds": sorted(PRESUMED_HARMFUL), "genes": out}


def model_settings() -> dict | None:
    model = hospital_query._model()
    if model is None:
        return None
    sentence = hospital_query.model_sentence(model.threshold, model)
    return {"source": model.source, "threshold": model.threshold, "threshold_word": hospital_query.percent_word(model.threshold),
            "trained_words": sentence.split(" · ", 1)[1] if " · " in sentence else "",
            "not_scored": hospital_query.MODEL_NOT_SCORED, "missense_only": hospital_query.MODEL_IS_MISSENSE_ONLY}


def aliases(table: pd.DataFrame) -> dict[str, int]:
    """Other valid spellings of an insertion or deletion, as the tables record them: the public database's id and the
    rightmost spelling. The terminal turns any spelling into the canonical one; the page knows these two."""
    found: dict[str, int] = {}
    position = {variant_id: index for index, variant_id in enumerate(table.variant_id)}
    for table_dir in table_dirs():
        table_file = table_dir / "variants.csv"
        if not table_file.exists():
            continue
        columns = [c for c in ("rightmost_id", "database_id") if c in pd.read_csv(table_file, nrows=0).columns]
        if not columns:
            continue
        variants = pd.read_csv(table_file, usecols=["variant_id"] + columns)
        for column in columns:
            for variant_id, other in zip(variants.variant_id, variants[column], strict=True):
                if isinstance(other, str) and other != variant_id and other not in position and variant_id in position:
                    found.setdefault(other, position[variant_id])
    return found


# ---------------------------------------------------------------------------
# Writing the page
# ---------------------------------------------------------------------------
def build_html(plain: str, packed: bool) -> str:
    template = TEMPLATE.read_text(encoding="utf-8")
    for placeholder in ("__DEMO_DATA__", "__DEMO_DATA_TYPE__", "__BUILT__"):
        if placeholder not in template:
            raise ValueError(f"{relative(TEMPLATE)} has no {placeholder} placeholder")
    if packed:
        payload = base64.b64encode(gzip.compress(plain.encode("utf-8"), 9)).decode("ascii")
        kind = "application/gzip+base64"
    else:
        payload = plain.replace("</", "<\\/")  # never end the script block early
        kind = "application/json"
    return template.replace("__DEMO_DATA_TYPE__", kind).replace("__DEMO_DATA__", payload).replace("__BUILT__", date.today().isoformat())


if __name__ == "__main__":
    main()
