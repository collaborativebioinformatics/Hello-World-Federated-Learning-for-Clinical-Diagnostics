"""Step 1: build the variant table from myvariant.info.

One API record per variant already holds the three things we need:
    ClinVar  ->  the expert verdict           (our label)
    dbNSFP   ->  computer prediction scores   (our features)
    gnomAD   ->  frequency per population

The genes come from config/<panel>_gene_panel.txt, the list step 0 wrote for one
disease area. `cardiac` is the default panel and its table sits at the top of
data/. Any other panel gets a folder of its own, data/<panel>/, holding the same
two files with the same columns.

Genes that are not in the cache yet are asked for up to 50 at a time in one
query, and the answer is dealt out into the same per-gene cache files a
one-gene query would write, by the gene symbols ClinVar gives each record.
Everything after the download reads the per-gene files, so the table comes out
the same either way. --chunk 1 asks one gene at a time, as the script did first.

Writes into data/ (git-ignored):
    variants.csv    one row per variant, the readable columns first
    columns.json    which columns are features, frequencies and label
    raw/GENE.json   cached downloads, so reruns are instant. Shared by every panel

Usage:
    uv run python scripts/01_build_table.py
    uv run python scripts/01_build_table.py --panel cancer     # another disease area, into data/cancer/
    uv run python scripts/01_build_table.py --panel all        # every NHS signed-off panel, into data/all/
    uv run python scripts/01_build_table.py --refresh          # download again
    uv run python scripts/01_build_table.py --genes MYH7,TTR   # quick test
    uv run python scripts/01_build_table.py --chunk 1          # one gene per query, the slow way

What every column means: docs/table_columns.md
How to add a disease area: docs/disease_areas.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "raw"
DEFAULT_PANEL = "cardiac"

API_URL = "https://myvariant.info/v1/query"
GENOME_BUILD = "hg38"
DEFAULT_CHUNK = 50  # genes per download query

# Score columns: our column name -> where the value sits in the API record.
#
# These are dbNSFP "rank scores": each tool's output rescaled to 0..1, where
# higher always means more damaging. Same direction and same scale everywhere,
# so no hospital ever needs to share scaling statistics.
#
# Left out on purpose, because they were trained on ClinVar or HGMD labels and
# would leak the answer: REVEL, ClinPred, BayesDel, MetaLR, MetaSVM, MetaRNN,
# VEST4, M-CAP, MutPred, MVP, gMVP, VARITY, DEOGEN2, MutationTaster, FATHMM.
SCORES = {
    "alphamissense": "dbnsfp.alphamissense.rankscore",
    "cadd": "dbnsfp.cadd.raw_rankscore",
    "dann": "dbnsfp.dann.rankscore",
    "primateai": "dbnsfp.primateai.rankscore",
    "mpc": "dbnsfp.mpc.rankscore",
    "sift": "dbnsfp.sift.converted_rankscore",
    "sift4g": "dbnsfp.sift4g.converted_rankscore",
    "provean": "dbnsfp.provean.converted_rankscore",
    "phylop100": "dbnsfp.phylop.100way_vertebrate.rankscore",
    "phastcons100": "dbnsfp.phastcons.100way_vertebrate.rankscore",
    "siphy29": "dbnsfp.siphy_29way.logodds_rankscore",
    "bstatistic": "dbnsfp.bstatistic.converted_rankscore",
    "esm1b": "dbnsfp.esm1b.rankscore",
    "eve": "dbnsfp.eve.rankscore",
    "polyphen2_hdiv": "dbnsfp.polyphen2.hdiv.rankscore",
    "mutationassessor": "dbnsfp.mutationassessor.rankscore",
    "gerp91": "dbnsfp.gerp.91_mammals.rankscore",  # GERP++ has "++" in its API name, which the URL mangles
}

# A score column is only recommended to later steps if few values are missing.
# Gaps follow the gene (several tools skip huge genes like TTN) and the gene
# follows the label, so filling the gaps in would leak the answer.
MAX_MISSING_SHARE = 0.30

# gnomAD populations. The first three play our hospitals.
SITE_POPULATIONS = {"nfe": "site_oslo", "sas": "site_karachi", "afr": "site_lagos"}
OTHER_POPULATIONS = ["eas", "amr", "fin", "asj"]

# "Population-discordant": common in one site population, at least ten times
# rarer in another. Step 5 looks at these variants separately.
DISCORDANT_MIN_FREQUENCY = 1e-3
DISCORDANT_FOLD = 10
FREQUENCY_FLOOR = 1e-5

# ClinVar review status -> gold stars. Records with no stars are ignored.
REVIEW_STARS = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 1,
}
PATHOGENIC_TERMS = {"pathogenic", "likely pathogenic"}
BENIGN_TERMS = {"benign", "likely benign"}


def column_order() -> list[str]:
    """Left to right: what a person wants to read first, what the code needs last."""
    sites = list(SITE_POPULATIONS)
    return (
        ["name", "gene", "verdict", "label", "stars"]
        + [f"af_{pop}" for pop in sites]
        + ["pop_discordant"]
        + list(SCORES)
        + ["af_global"]
        + [f"af_{pop}" for pop in OTHER_POPULATIONS]
        + ["in_gnomad"]
        + [f"{count}_{pop}" for pop in sites for count in ("ac", "an")]
        + ["variant_id", "clinvar_id"]
    )


# ---------------------------------------------------------------------------
# The label: what did ClinVar's experts say?
# ---------------------------------------------------------------------------
def clinvar_verdict(records) -> tuple[str | None, int, str]:
    """Return (verdict, best_stars, why_dropped).

    A variant gets a verdict only when every starred ClinVar record agrees:
    all pathogenic / likely pathogenic, or all benign / likely benign.
    One "uncertain" or "conflicting" record is enough to drop it.
    """
    if isinstance(records, dict):
        records = [records]

    said_pathogenic = said_benign = said_unsure = False
    best_stars = 0
    for record in records or []:
        if not isinstance(record, dict):
            continue
        stars = REVIEW_STARS.get(str(record.get("review_status", "")).strip().lower(), 0)
        if stars == 0:
            continue
        best_stars = max(best_stars, stars)
        for term in re.split(r"[,/;]", str(record.get("clinical_significance", "")).lower()):
            term = term.strip()
            said_pathogenic |= term in PATHOGENIC_TERMS
            said_benign |= term in BENIGN_TERMS
            said_unsure |= "uncertain" in term or "conflicting" in term

    if best_stars == 0:
        return None, 0, "no record with review stars"
    if said_unsure:
        return None, best_stars, "uncertain or conflicting"
    if said_pathogenic and said_benign:
        return None, best_stars, "records disagree"
    if said_pathogenic:
        return "pathogenic", best_stars, ""
    if said_benign:
        return "benign", best_stars, ""
    return None, best_stars, "no pathogenic or benign call"


# ---------------------------------------------------------------------------
# One API record -> one table row
# ---------------------------------------------------------------------------
def build_row(record: dict, gene: str) -> tuple[dict | None, str]:
    verdict, stars, why_dropped = clinvar_verdict(get_nested(record, "clinvar.rcv"))
    if verdict is None:
        return None, why_dropped

    row = {
        "name": readable_name(record, gene),
        "gene": gene,
        "verdict": verdict,
        "label": int(verdict == "pathogenic"),
        "stars": stars,
        "variant_id": record["_id"],
        "clinvar_id": first_item(get_nested(record, "clinvar.variant_id")),
    }
    row.update(read_scores(record))
    row.update(read_frequencies(record))
    row["pop_discordant"] = int(is_population_discordant(row))
    return row, ""


def readable_name(record: dict, gene: str) -> str:
    """'MYH7 R403Q' = in gene MYH7, amino acid R at position 403 became Q."""
    before = first_item(get_nested(record, "dbnsfp.aa.ref"))
    position = first_item(get_nested(record, "dbnsfp.aa.pos"))
    after = first_item(get_nested(record, "dbnsfp.aa.alt"))
    if before and position and after:
        return f"{gene} {before}{position}{after}"
    return f"{gene} {record['_id']}"


def read_scores(record: dict) -> dict:
    return {column: to_number(get_nested(record, path)) for column, path in SCORES.items()}


def read_frequencies(record: dict) -> dict:
    def gnomad(path: str) -> float:
        value = to_number(get_nested(record, f"gnomad_exome.{path}"))
        return 0.0 if math.isnan(value) else value  # never seen in gnomAD = frequency zero

    row = {"in_gnomad": int("gnomad_exome" in record), "af_global": gnomad("af.af")}
    for pop in [*SITE_POPULATIONS, *OTHER_POPULATIONS]:
        row[f"af_{pop}"] = gnomad(f"af.af_{pop}")
    for pop in SITE_POPULATIONS:
        row[f"ac_{pop}"] = int(gnomad(f"ac.ac_{pop}"))  # copies of the variant seen
        row[f"an_{pop}"] = int(gnomad(f"an.an_{pop}"))  # gene copies looked at
    return row


def is_population_discordant(row: dict) -> bool:
    site_frequencies = [row[f"af_{pop}"] for pop in SITE_POPULATIONS]
    highest, lowest = max(site_frequencies), min(site_frequencies)
    return (
        highest >= DISCORDANT_MIN_FREQUENCY
        and highest >= DISCORDANT_FOLD * max(lowest, FREQUENCY_FLOOR)
    )


def get_nested(record, dotted_path: str):
    """get_nested(r, 'a.b.c') -> r['a']['b']['c'], or None. Lists: use the first item."""
    value = record
    for key in dotted_path.split("."):
        value = first_item(value)
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def first_item(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value


def as_list(value) -> list:
    """The API gives one item or a list. Always a list here, empty for nothing."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def to_number(value) -> float:
    """A float. For a list (one score per transcript) the highest. NaN if missing."""
    if isinstance(value, list):
        numbers = [n for n in map(to_number, value) if not math.isnan(n)]
        return max(numbers) if numbers else math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "team12-fl-clinical-diagnostics (hackathon)"


def fields_to_request() -> str:
    frequencies = ["gnomad_exome.af.af"]
    frequencies += [f"gnomad_exome.af.af_{pop}" for pop in [*SITE_POPULATIONS, *OTHER_POPULATIONS]]
    frequencies += [f"gnomad_exome.{c}.{c}_{pop}" for pop in SITE_POPULATIONS for c in ("ac", "an")]
    clinvar = ["clinvar.variant_id", "clinvar.rcv.clinical_significance", "clinvar.rcv.review_status"]
    amino_acids = ["dbnsfp.aa.ref", "dbnsfp.aa.pos", "dbnsfp.aa.alt"]
    return ",".join(clinvar + amino_acids + list(SCORES.values()) + frequencies)


def gene_query(genes: list[str]) -> str:
    """The query for one gene, or for several genes at once.

    Missense only: AlphaMissense exists only for missense variants.
    The pathogenic/benign filter just keeps the download small;
    the real decision is made locally in clinvar_verdict().
    """
    symbols = genes[0] if len(genes) == 1 else "(" + " OR ".join(genes) + ")"
    return (
        f"clinvar.gene.symbol:{symbols} AND _exists_:dbnsfp.alphamissense "
        "AND clinvar.rcv.clinical_significance:(pathogenic OR benign)"
    )


def fetch_records(query: str, fields: str) -> list[dict]:
    """Every record the query matches. A big answer arrives in pages, each asked for with the last page's id."""
    page = ask_api({"q": query, "fields": fields, "assembly": GENOME_BUILD, "fetch_all": "true"})
    expected = page.get("total", 0)
    records = list(page.get("hits", []))
    while page.get("_scroll_id") and len(records) < expected:
        time.sleep(0.2)
        page = ask_api({"scroll_id": page["_scroll_id"]})
        if not page.get("hits"):
            break
        records.extend(page["hits"])
    return records


def download_gene(gene: str, refresh: bool) -> list[dict]:
    cache_file = CACHE_DIR / f"{gene}.json"
    if cache_file.exists() and not refresh:
        return json.loads(cache_file.read_text(encoding="utf-8"))
    records = fetch_records(gene_query([gene]), fields_to_request())
    cache_file.write_text(json.dumps(records), encoding="utf-8")
    return records


def download_in_chunks(genes: list[str], chunk: int, refresh: bool) -> None:
    """Fill the cache for the genes that are not in it yet, several genes per query.

    One query for up to `chunk` genes returns the same records as one query per
    gene, and each record says which genes ClinVar assigns it to. So the answer
    is dealt out into the per-gene files download_gene() would have written, one
    per gene asked for, empty when nothing came back. A record that names two
    of the genes asked for goes into both files, as it would with two queries.
    """
    missing = [gene for gene in genes if refresh or not (CACHE_DIR / f"{gene}.json").exists()]
    if not missing:
        return
    fields = fields_to_request() + ",clinvar.gene.symbol"  # only to deal the records out
    batches = [missing[start : start + chunk] for start in range(0, len(missing), chunk)]
    print(f"downloading {len(missing)} genes in {len(batches)} queries of up to {chunk} genes each", flush=True)
    for number, batch in enumerate(batches, 1):
        started = time.time()
        records = fetch_records(gene_query(batch), fields)
        by_gene = split_by_gene(records, batch)
        for gene in batch:
            (CACHE_DIR / f"{gene}.json").write_text(json.dumps(by_gene[gene]), encoding="utf-8")
        print(f"  query {number:>3}/{len(batches)}  {len(batch):>3} genes  {len(records):>6} records  {time.time() - started:5.1f} s", flush=True)


def split_by_gene(records: list[dict], genes: list[str]) -> dict[str, list[dict]]:
    """Deal the records of a many-gene query out by gene, then drop the gene field that was only asked for to do this."""
    wanted = {gene.upper(): gene for gene in genes}
    by_gene: dict[str, list[dict]] = {gene: [] for gene in genes}
    for record in records:
        symbols = {symbol.upper() for symbol in clinvar_genes(record)}
        for entry in as_list(record.get("clinvar")):
            entry.pop("gene", None)
        for symbol in symbols & wanted.keys():
            by_gene[wanted[symbol]].append(record)
    return by_gene


def clinvar_genes(record: dict) -> list[str]:
    """The gene symbols ClinVar gives a record. `clinvar`, `clinvar.gene` and the symbol can each be one item or a list."""
    symbols = []
    for entry in as_list(record.get("clinvar")):
        for gene in as_list(entry.get("gene") if isinstance(entry, dict) else None):
            if isinstance(gene, dict):
                symbols += [str(symbol) for symbol in as_list(gene.get("symbol"))]
    return symbols


def ask_api(params: dict) -> dict:
    """GET with retries. 'No more pages' arrives as a 4xx with a JSON body: return it."""
    for attempt in range(6):
        try:
            response = SESSION.get(API_URL, params=params, timeout=90)
            if response.status_code in (200, 400, 404):
                return response.json()
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2**attempt)
    raise RuntimeError(f"myvariant.info kept failing for: {params}")


def read_gene_panel(path: Path) -> list[str]:
    lines = (line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines())
    return list(dict.fromkeys(line for line in lines if line))


def gene_panel_file(panel: str) -> Path:
    return CONFIG_DIR / f"{panel}_gene_panel.txt"


def table_folder(panel: str) -> Path:
    """The cardiac table keeps its place at the top of data/. Every other panel gets data/<panel>/."""
    return DATA_DIR if panel == DEFAULT_PANEL else DATA_DIR / panel


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Build the variant table of one disease area from myvariant.info")
    parser.add_argument("--panel", default=DEFAULT_PANEL, help="disease area: reads config/<panel>_gene_panel.txt (default: cardiac)")
    parser.add_argument("--refresh", action="store_true", help="ignore cached downloads")
    parser.add_argument("--genes", help="comma-separated genes instead of the whole panel")
    parser.add_argument("--out", type=Path, help="where to write the table (default: data/variants.csv, or data/<panel>/variants.csv)")
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK, help=f"genes per download query (default: {DEFAULT_CHUNK}). 1 asks one gene at a time")
    args = parser.parse_args()
    if args.chunk < 1:
        parser.error("--chunk must be at least 1")
    out = args.out or table_folder(args.panel) / "variants.csv"

    panel_file = gene_panel_file(args.panel)
    if not args.genes and not panel_file.exists():
        print(
            f"{panel_file.relative_to(ROOT)} is missing. Run step 0 first, with the same panel:\n"
            f"    uv run python scripts/00_fetch_gene_panel.py --panel {args.panel}",
            file=sys.stderr,
        )
        return 1
    genes = [gene.strip() for gene in (args.genes.split(",") if args.genes else read_gene_panel(panel_file))]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Many genes per query when the cache is short of them. With --chunk 1 the loop below downloads instead.
    if args.chunk > 1:
        download_in_chunks(genes, args.chunk, args.refresh)
    refresh_one_by_one = args.refresh and args.chunk == 1

    rows_by_id: dict[str, dict] = {}
    dropped = Counter()
    genes_with_nothing = []
    downloaded = 0

    for number, gene in enumerate(genes, 1):
        records = download_gene(gene, refresh_one_by_one)
        downloaded += len(records)
        rows_before = len(rows_by_id)
        for record in records:
            if record["_id"] in rows_by_id:
                dropped["already listed under another gene"] += 1
                continue
            row, why_dropped = build_row(record, gene)
            if row is None:
                dropped[why_dropped] += 1
            else:
                rows_by_id[record["_id"]] = row
        kept = len(rows_by_id) - rows_before
        print(f"[{number:>2}/{len(genes)}] {gene:<8} downloaded {len(records):>5}  kept {kept:>5}", flush=True)
        if not records:
            genes_with_nothing.append(gene)

    if not rows_by_id:
        print("No rows kept. Check the gene names and your connection.", file=sys.stderr)
        return 1

    table = tidy(pd.DataFrame(rows_by_id.values()))
    try:
        table.to_csv(out, index=False, lineterminator="\n")  # same bytes on Windows, Mac and Linux
    except PermissionError:
        print(f"\nCannot write {out}. It is open in another program, probably Excel. Close it and run again.", file=sys.stderr)
        return 1

    print_summary(table, downloaded, dropped)
    write_column_list(table, out.with_name("columns.json"))
    print(f"wrote {out}  ({len(table)} rows x {table.shape[1]} columns)")
    if genes_with_nothing:
        print(
            f"\nWARNING: nothing found for {', '.join(genes_with_nothing)}. myvariant.info lists "
            "ClinVar entries for these genes but has no dbNSFP scores for them in its hg38 index."
        )
    return 0


def tidy(table: pd.DataFrame) -> pd.DataFrame:
    table = table[column_order()].sort_values(["gene", "name", "variant_id"])
    frequency_columns = [c for c in table.columns if c.startswith("af_")]
    return table.round({**dict.fromkeys(SCORES, 4), **dict.fromkeys(frequency_columns, 6)})


def usable_scores(table: pd.DataFrame) -> tuple[list[str], list[str]]:
    missing_share = table[list(SCORES)].isna().mean()
    usable = [s for s in SCORES if missing_share[s] <= MAX_MISSING_SHARE]
    too_sparse = [s for s in SCORES if missing_share[s] > MAX_MISSING_SHARE]
    return usable, too_sparse


def write_column_list(table: pd.DataFrame, path: Path) -> None:
    """Later steps read this instead of hard-coding column names."""
    usable, too_sparse = usable_scores(table)
    column_list = {
        "id": "variant_id",
        "label": "label",
        "features": usable,
        "features_too_sparse": too_sparse,
        "af_global": "af_global",
        "af_by_site": {site: f"af_{pop}" for pop, site in SITE_POPULATIONS.items()},
        "evaluation_subset": "pop_discordant",
    }
    path.write_text(json.dumps(column_list, indent=2), encoding="utf-8")
    print(f"wrote {path}")


def print_summary(table: pd.DataFrame, downloaded: int, dropped: Counter) -> None:
    pathogenic = int(table.label.sum())
    print("\n================ SUMMARY ================")
    print(f"downloaded {downloaded} records, kept {len(table)}: {pathogenic} pathogenic, {len(table) - pathogenic} benign")
    for why, count in dropped.most_common():
        print(f"  dropped {count:>5}  {why}")

    per_gene = table.groupby("gene").label.agg(rows="size", pathogenic="sum")
    per_gene["benign"] = per_gene.rows - per_gene.pathogenic
    print("\nbiggest genes:")
    print(per_gene.sort_values("rows", ascending=False).head(10).to_string())

    usable, too_sparse = usable_scores(table)
    print(f"\nscores to use ({len(usable)}): {', '.join(usable)}")
    print(f"too many gaps, over {MAX_MISSING_SHARE:.0%} missing: {', '.join(too_sparse) or 'none'}")

    discordant = table[table.pop_discordant == 1]
    print(f"\nfound in gnomAD: {int(table.in_gnomad.sum())} of {len(table)}")
    print(
        f"population-discordant: {len(discordant)} "
        f"({int(discordant.label.sum())} pathogenic, {int((discordant.label == 0).sum())} benign)"
    )

    print("\na few rows:")
    preview = ["name", "verdict", "stars", "af_nfe", "af_sas", "af_afr", "alphamissense", "cadd"]
    sample = pd.concat([discordant.head(3), table[table.label == 1].head(3)])
    print(sample[preview].to_string(index=False))
    print()


if __name__ == "__main__":
    raise SystemExit(main())
