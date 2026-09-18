"""Step 1, for every other kind of small mutation: build the table the missense table leaves out.

scripts/01_build_table.py keeps single-letter changes that swap one amino acid, because
those are the ones the prediction scores exist for. This script collects the rest of
what ClinVar holds for the same genes: changes that cut the protein short, frameshifts,
splice-site changes, silent changes, changes inside introns, and insertions, deletions
and duplications up to 50 letters long. Nothing trains on this table. It exists so that
the hospital count query can answer for any small mutation.

Same source, same verdict rule and same gnomAD frequencies as step 1. Three differences:

    mutation_type   what the change does to the gene, from the snpEff annotation for the
                    panel gene, most severe effect first. There are no score columns.
    contested       a variant ClinVar's labs are unsure or disagree about is kept when it
                    reaches 0.1% in one of the three site populations. Those are the ones
                    a lab would ask other hospitals about. Verdict `contested`, no label.
    one spelling    an insertion or deletion inside a repeat can be written several valid
                    ways. `variant_id` is the canonical spelling from variant_spelling.py.
                    `database_id` keeps the id myvariant.info uses.

Reads   config/cardiac_gene_panel.txt, data/variants.csv (names and ids only, to stay clear of them)
Writes  data/other_types/variants.csv     one row per variant
        data/other_types/columns.json     what the columns are
        data/other_types/build.json       every count this prints
        data/raw_other_types/GENE.json    cached downloads, so a rerun carries on where it stopped
        data/raw_sequence/GENE.json       reference sequence, fetched by variant_spelling.py

Usage:
    uv run python scripts/01_build_other_types.py
    uv run python scripts/01_build_other_types.py --genes MYBPC3,TTR   # quick test, written to data/other_types/test_build/
    uv run python scripts/01_build_other_types.py --refresh            # download again

What every column means: docs/other_mutation_types.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

import variant_spelling

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "other_types"
CACHE_DIR = DATA_DIR / "raw_other_types"
PANEL_FILE = ROOT / "config" / "cardiac_gene_panel.txt"
MISSENSE_TABLE = DATA_DIR / "variants.csv"

GENOME_BUILD = "hg38"
LONGEST = 50  # letters deleted or inserted. Anything longer is a structural change and needs other tools
TOO_COMMON = 0.001  # a contested variant is kept when it is this common at one site. Same limit as the query's rule 1

SITE_POPULATIONS = ["nfe", "sas", "afr"]  # Oslo, Karachi, Lagos, as in step 1
OTHER_POPULATIONS = ["eas", "amr", "fin", "asj"]

CONTESTED_REASONS = {"uncertain or conflicting", "records disagree"}  # how clinvar_verdict() words them

# What a change does to the gene, most severe first. The words on the right are snpEff's.
MUTATION_TYPES = {
    "protein_cutting": ["stop_gained"],
    "frameshift": ["frameshift_variant"],
    # A deletion that takes out a whole small exon takes both of its splice sites with it.
    "splice_site": ["splice_acceptor_variant", "splice_donor_variant", "exon_loss_variant"],
    "start_or_stop_lost": ["start_lost", "stop_lost", "initiator_codon_variant"],
    "inframe_indel": ["conservative_inframe_insertion", "disruptive_inframe_insertion", "inframe_insertion",
                      "conservative_inframe_deletion", "disruptive_inframe_deletion", "inframe_deletion"],
    "missense_no_scores": ["missense_variant"],
    "splice_region": ["splice_region_variant"],
    "synonymous": ["synonymous_variant", "stop_retained_variant", "start_retained_variant"],
    "utr": ["5_prime_UTR_variant", "3_prime_UTR_variant", "5_prime_UTR_premature_start_codon_gain_variant"],
    "intronic": ["intron_variant"],
    "other": [],
}
SEVERITY = {mutation_type: rank for rank, mutation_type in enumerate(MUTATION_TYPES)}
TYPE_OF_EFFECT = {effect: mutation_type for mutation_type, effects in MUTATION_TYPES.items() for effect in effects}

AMINO_ACIDS = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H",
    "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V", "Sec": "U", "Ter": "*",
}  # fmt: skip


def load_step1():
    """Import 01_build_table.py, whose name starts with a digit, so the verdict and the frequencies are read its way."""
    spec = importlib.util.spec_from_file_location("step1", ROOT / "scripts" / "01_build_table.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["step1"] = module  # a dataclass in the module looks itself up here while it is being made
    spec.loader.exec_module(module)
    return module


step1 = load_step1()


def column_order() -> list[str]:
    """Left to right as in the missense table, with the type and the spelling columns where the scores were."""
    return (
        ["name", "gene", "verdict", "label", "stars", "mutation_type"]
        + [f"af_{pop}" for pop in SITE_POPULATIONS]
        + ["pop_discordant", "af_global"]
        + [f"af_{pop}" for pop in OTHER_POPULATIONS]
        + ["in_gnomad"]
        + [f"{count}_{pop}" for pop in SITE_POPULATIONS for count in ("ac", "an")]
        + ["kind", "length", "shift_room", "variant_id", "rightmost_id", "database_id", "clinvar_id"]
    )


# ---------------------------------------------------------------------------
# One API record -> one table row
# ---------------------------------------------------------------------------
def build_row(record: dict, gene: str) -> tuple[dict | None, str]:
    verdict, stars, why_dropped = step1.clinvar_verdict(step1.get_nested(record, "clinvar.rcv"))
    frequencies = step1.read_frequencies(record)
    if verdict is None:
        if why_dropped not in CONTESTED_REASONS:
            return None, why_dropped
        if max(frequencies[f"af_{pop}"] for pop in SITE_POPULATIONS) < TOO_COMMON:
            return None, f"{why_dropped}, and under {TOO_COMMON:.1%} at every site"
        verdict = "contested"

    spelling, why_dropped = spell(record)
    if spelling is None:
        return None, why_dropped
    if spelling.length > LONGEST:
        return None, f"longer than {LONGEST} letters"

    annotation, mutation_type = worst_annotation(record, gene)
    row = {
        "name": readable_name(record, gene, annotation, spelling.canonical),
        "gene": gene,
        "verdict": verdict,
        "label": {"pathogenic": 1, "benign": 0}.get(verdict),  # empty for a contested variant: it has no truth
        "stars": stars,
        "mutation_type": mutation_type,
        "type_from": "snpEff" if annotation else "ClinVar name",
        "kind": spelling.kind,
        "length": spelling.length,
        "shift_room": spelling.room,
        "variant_id": spelling.canonical,
        "rightmost_id": spelling.rightmost,
        "database_id": record["_id"],
        "clinvar_id": step1.first_item(step1.get_nested(record, "clinvar.variant_id")),
    }
    row.update(frequencies)
    row["pop_discordant"] = int(step1.is_population_discordant(row))
    return row, ""


def spell(record: dict) -> tuple[variant_spelling.Spelling | None, str]:
    """The canonical spelling of a record, or why it cannot have one.

    myvariant.info writes a ClinVar duplication as `g.START_ENDdup`, where START and END are
    the two positions the copy goes between. Read as HGVS that would be a different
    change, so a duplication is read from ClinVar's own before and after letters.
    """
    variant_id = record["_id"]
    if not variant_spelling.looks_like_an_id(variant_id):
        return None, "no exact DNA positions, mostly large deletions and duplications"
    try:
        if variant_id.endswith("dup"):
            chrom = variant_id.split(":")[0].removeprefix("chr")
            clinvar = step1.first_item(record.get("clinvar")) or {}
            position, before, after = step1.get_nested(clinvar, "hg38.start"), clinvar.get("ref"), clinvar.get("alt")
            if not (position and before and after):
                return None, "a duplication ClinVar gives no letters for"
            return variant_spelling.normalise_vcf(chrom, int(position), before, after), ""
        return variant_spelling.normalise(variant_id), ""
    except variant_spelling.NoSequence:
        return None, "outside the fetched reference sequence"
    except variant_spelling.SpellingError:
        return None, "letters do not match the reference sequence"


def worst_annotation(record: dict, gene: str) -> tuple[dict | None, str]:
    """The snpEff annotation of the panel gene with the most severe effect, and that effect as a mutation type.

    myvariant.info has no snpEff annotation for duplications. Those get their type from ClinVar's names.
    """
    annotations = step1.get_nested(record, "snpeff.ann") or []
    if isinstance(annotations, dict):
        annotations = [annotations]
    # A feature id with a colon in it is a protein-structure note, which repeats the transcript's own annotation.
    own = [a for a in annotations if a.get("genename") == gene and ":" not in str(a.get("feature_id"))]
    if not own:
        return None, type_from_clinvar_names(record)
    typed = [(type_of_effects(a.get("effect", "")), a) for a in own]
    mutation_type, annotation = min(typed, key=lambda pair: SEVERITY[pair[0]])  # min keeps the first of equals
    return annotation, mutation_type


def type_of_effects(effects: str) -> str:
    """'frameshift_variant&splice_region_variant' -> frameshift: the most severe of the effects named."""
    types = [TYPE_OF_EFFECT.get(effect, "other") for effect in effects.split("&")]
    return min(types, key=SEVERITY.get)


def type_from_clinvar_names(record: dict) -> str:
    """The same vocabulary read from ClinVar's protein and coding names, for records snpEff has not annotated."""
    protein, coding = clinvar_names(record)
    if protein:
        if "fs" in protein:
            return "frameshift"
        if re.search(r"\d+(Ter|\*)", protein) and "ext" not in protein:
            return "protein_cutting"
        if protein.endswith("=") or re.fullmatch(r"p\.([A-Z][a-z]{2})\d+\1", protein):
            return "synonymous"
        if re.match(r"p\.Met1[^0-9]", protein) or "ext" in protein:
            return "start_or_stop_lost"
        if re.search(r"(del|dup|ins)", protein):
            return "inframe_indel"
        if re.fullmatch(r"p\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2}", protein):
            return "missense_no_scores"
    if coding:
        into_intron = [int(offset) for offset in re.findall(r"\d[+-](\d+)", coding)]
        if into_intron:
            nearest = min(into_intron)
            return "splice_site" if nearest <= 2 else "splice_region" if nearest <= 8 else "intronic"
        if re.match(r"c\.[-*]\d", coding):
            return "utr"
    return "other"


def clinvar_names(record: dict) -> tuple[str, str]:
    """ClinVar's protein name and coding name for the variant, each the shortest of those given, without the transcript."""
    def shortest(names, prefix: str) -> str:
        names = [names] if isinstance(names, str) else names or []
        bare = [name.split(":")[-1] for name in names if name and name.split(":")[-1].startswith(prefix)]
        return min(bare, key=len, default="")

    clinvar = step1.first_item(record.get("clinvar")) or {}
    hgvs = clinvar.get("hgvs") or {}
    return shortest(hgvs.get("protein"), "p."), shortest(hgvs.get("coding"), "c.")


def readable_name(record: dict, gene: str, annotation: dict | None, variant_id: str) -> str:
    """'MYBPC3 R502*', 'ACTN2 P32fs', 'MYBPC3 c.3628-41_3628-17del': the protein change if there is one, else the coding change."""
    if annotation:
        protein, coding = annotation.get("hgvs_p", ""), annotation.get("hgvs_c", "")
    else:
        protein, coding = clinvar_names(record)
    if protein and re.search(r"[A-Za-z*=]", protein.removeprefix("p.")):  # snpEff writes a bare number for some silent changes
        return f"{gene} {short_protein_change(protein)}"
    return f"{gene} {coding or variant_id}"


def short_protein_change(protein: str) -> str:
    """p.Arg502* -> R502*, p.Gly115fs -> G115fs, p.Val849Val -> V849=, in the one-letter style of `MYH7 R403Q`."""
    short = re.sub(r"[A-Z][a-z]{2}", lambda found: AMINO_ACIDS.get(found.group(), found.group()), protein.removeprefix("p."))
    silent = re.fullmatch(r"([A-Z*])(\d+)\1", short)
    return f"{silent.group(1)}{silent.group(2)}=" if silent else short


def coding_change(record: dict, gene: str) -> str:
    annotation, _ = worst_annotation(record, gene)
    return (annotation or {}).get("hgvs_c") or clinvar_names(record)[1] or record["_id"]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def fields_to_request() -> str:
    frequencies = ["gnomad_exome.af.af"]
    frequencies += [f"gnomad_exome.af.af_{pop}" for pop in [*SITE_POPULATIONS, *OTHER_POPULATIONS]]
    frequencies += [f"gnomad_exome.{c}.{c}_{pop}" for pop in SITE_POPULATIONS for c in ("ac", "an")]
    clinvar = ["clinvar.variant_id", "clinvar.type", "clinvar.ref", "clinvar.alt", "clinvar.hg38.start",
               "clinvar.rcv.clinical_significance", "clinvar.rcv.review_status", "clinvar.hgvs.coding", "clinvar.hgvs.protein"]
    snpeff = [f"snpeff.ann.{part}" for part in ("effect", "genename", "feature_id", "hgvs_c", "hgvs_p")]
    return ",".join(clinvar + snpeff + frequencies)


def download_gene(gene: str, refresh: bool) -> tuple[list[dict], bool]:
    """Every ClinVar variant of the gene that the missense table cannot hold. Returns (records, was it downloaded now)."""
    cache_file = CACHE_DIR / f"{gene}.json"
    if cache_file.exists() and not refresh:
        return json.loads(cache_file.read_text(encoding="utf-8")), False

    # The filters only keep the download small: a verdict in words, or common at one site.
    # The real decisions are made locally, in build_row().
    common = " OR ".join(f"gnomad_exome.af.af_{pop}:[{TOO_COMMON} TO *]" for pop in SITE_POPULATIONS)
    query = (
        f"clinvar.gene.symbol:{gene} AND NOT _exists_:dbnsfp.alphamissense "
        f"AND (clinvar.rcv.clinical_significance:(pathogenic OR benign) OR {common})"
    )
    page = step1.ask_api({"q": query, "fields": fields_to_request(), "assembly": GENOME_BUILD, "fetch_all": "true"})
    expected = page.get("total", 0)
    records = list(page.get("hits", []))
    while page.get("_scroll_id") and len(records) < expected:
        time.sleep(0.2)
        page = step1.ask_api({"scroll_id": page["_scroll_id"]})
        if not page.get("hits"):
            break
        records.extend(page["hits"])
    if len(records) < expected:
        raise RuntimeError(f"{gene}: the API promised {expected} records and sent {len(records)}. Run again.")

    records = [only_this_gene(record, gene) for record in records]
    # Written under another name first, so a run stopped halfway never leaves half a file behind.
    unfinished = cache_file.with_suffix(".part")
    unfinished.write_text(json.dumps(records), encoding="utf-8")
    unfinished.replace(cache_file)
    return records, True


def only_this_gene(record: dict, gene: str) -> dict:
    """Drop snpEff's annotations for neighbouring genes. They are never read, and in places they are most of the record."""
    annotations = step1.get_nested(record, "snpeff.ann")
    if isinstance(annotations, list):
        record["snpeff"]["ann"] = [a for a in annotations if a.get("genename") == gene]
    record.pop("_score", None)
    return record


def span_of(records: list[dict]) -> tuple[str, int, int] | None:
    """(chromosome, lowest, highest position) of a gene's records, so the fetched sequence covers all of them."""
    places = [re.match(r"chr(\w+):g\.(\d+)", record["_id"]) for record in records]
    places = [(place.group(1), int(place.group(2))) for place in places if place]
    if not places:
        return None
    chrom = Counter(chrom for chrom, _ in places).most_common(1)[0][0]
    positions = [position for where, position in places if where == chrom]
    return chrom, min(positions), max(positions)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Build the table of every small mutation that is not in the missense table")
    parser.add_argument("--refresh", action="store_true", help="ignore cached downloads")
    parser.add_argument("--genes", help="comma-separated genes instead of the whole panel. Written to a test folder")
    args = parser.parse_args()

    genes = args.genes.split(",") if args.genes else step1.read_gene_panel(PANEL_FILE)
    out_dir = OUT_DIR / "test_build" if args.genes else OUT_DIR
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    missense = pd.read_csv(MISSENSE_TABLE, usecols=["name", "variant_id"]) if MISSENSE_TABLE.exists() else None
    in_missense_table = set(missense.variant_id) if missense is not None else set()

    rows_by_id: dict[str, dict] = {}  # by canonical spelling
    records_by_id: dict[str, tuple[dict, str]] = {}  # by the database's id: every record looked at, and its gene
    dropped = Counter()
    downloaded = seconds_downloading = 0

    for number, gene in enumerate(genes, 1):
        gene = gene.strip()
        started = time.time()
        records, fresh = download_gene(gene, args.refresh)
        seconds_downloading += (time.time() - started) if fresh else 0
        sequence = variant_spelling.fetch_gene(gene, must_cover=span_of(records))
        downloaded += len(records)
        rows_before = len(rows_by_id)

        for record in records:
            if record["_id"] in records_by_id:
                dropped["already listed under another gene"] += 1
                continue
            records_by_id[record["_id"]] = (record, gene)
            row, why_dropped = build_row(record, gene)
            if row is None:
                dropped[why_dropped] += 1
            elif row["variant_id"] in in_missense_table:
                dropped["already in the missense table under its simple spelling"] += 1
            elif row["variant_id"] in rows_by_id:
                dropped["same change filed twice under different spellings"] += 1
                if better(row, rows_by_id[row["variant_id"]]):
                    rows_by_id[row["variant_id"]] = row
            else:
                rows_by_id[row["variant_id"]] = row
        kept = len(rows_by_id) - rows_before
        print(f"[{number:>3}/{len(genes)}] {gene:<8} {'downloaded' if fresh else 'cached':<10} {len(records):>6}  "
              f"kept {kept:>6}  sequence {sequence}", flush=True)

    if not rows_by_id:
        print("No rows kept. Check the gene names and your connection.", file=sys.stderr)
        return 1

    table = pd.DataFrame(rows_by_id.values())
    table["name"] = unique_names(table, records_by_id, set(missense.name) if missense is not None else set())
    checks = check_reference_letters(table, records_by_id)
    type_from = table.type_from.value_counts().to_dict()
    table = tidy(table)
    try:
        table.to_csv(out_dir / "variants.csv", index=False, lineterminator="\n")  # same bytes on Windows, Mac and Linux
    except PermissionError:
        print(f"\nCannot write {out_dir / 'variants.csv'}. It is open in another program, probably Excel.", file=sys.stderr)
        return 1

    numbers = summarise(table, downloaded, dropped, checks, type_from, seconds_downloading)
    (out_dir / "build.json").write_text(json.dumps(numbers, indent=2), encoding="utf-8")
    write_column_list(out_dir / "columns.json")
    print(f"wrote {out_dir / 'variants.csv'}  ({len(table)} rows x {table.shape[1]} columns)")
    if checks["single_letter_changes_matching"] < checks["single_letter_changes"]:
        print("\nSTOP: some ids name a letter the reference sequence does not have there. The coordinates or the "
              f"genome build are off, so no spelling in this table can be trusted. First few: {checks['mismatches']}", file=sys.stderr)
        return 1
    return 0


def better(new: dict, old: dict) -> bool:
    """Of two records for one change keep the one that says more: a clean verdict, then gnomAD counts, then stars."""
    def worth(row: dict) -> tuple:
        return row["verdict"] != "contested", row["in_gnomad"], row["stars"]
    return worth(new) > worth(old)


def unique_names(table: pd.DataFrame, records_by_id: dict, missense_names: set[str]) -> pd.Series:
    """A name shared by several DNA changes, or already used in the missense table, gets its coding change added.

    `TTN R100*` can come from two different letter changes. In the missense table such pairs share
    a name. Here every name is unique, so that asking by name can never pick the wrong change.
    """
    names = table.name.copy()
    shared = names.duplicated(keep=False) | names.isin(missense_names)
    for index in names.index[shared]:
        coding = coding_change(*records_by_id[table.database_id[index]])
        if not names[index].endswith(coding):
            names[index] = f"{names[index]} {coding}"
    still_shared = names.duplicated(keep=False)
    names[still_shared] = names[still_shared] + " " + table.variant_id[still_shared].str.split(":g.").str[1]
    return names


def check_reference_letters(table: pd.DataFrame, records_by_id: dict) -> dict:
    """Are the coordinates and the genome build right? Every letter an id names must be the letter in the fetched sequence."""
    single = table[table.database_id.str.fullmatch(variant_spelling.SUBSTITUTION.pattern)]
    matching = 0
    mismatches = []
    for variant_id in single.database_id:
        chrom, position, before, _ = variant_spelling.SUBSTITUTION.fullmatch(variant_id).groups()
        found = variant_spelling.Sequence.covering(chrom, int(position)).letter(int(position))
        matching += found == before
        if found != before:
            mismatches.append(variant_id)

    # The same question for deletions: ClinVar gives the letters before the change, anchored one letter to the left.
    deletions = deletions_matching = 0
    for database_id in table.database_id[table.database_id.str.endswith("del")]:
        record, _ = records_by_id[database_id]
        clinvar = step1.first_item(record.get("clinvar")) or {}
        before, after = clinvar.get("ref"), clinvar.get("alt")
        if not (before and after and before.startswith(after)):
            continue
        change = variant_spelling.read_change(database_id)
        letters = variant_spelling.Sequence.covering(change.chrom, change.start).letters(change.start, change.end)
        deletions += 1
        deletions_matching += letters == before[len(after):]
    return {"single_letter_changes": len(single), "single_letter_changes_matching": matching, "mismatches": mismatches[:20],
            "deletions_with_letters": deletions, "deletions_matching": deletions_matching}


def tidy(table: pd.DataFrame) -> pd.DataFrame:
    table = table[column_order()].sort_values(["gene", "name", "variant_id"])
    table["label"] = table.label.astype("Int64")  # whole numbers with gaps, so 1 is written as 1 and contested stays empty
    frequency_columns = [c for c in table.columns if c.startswith("af_")]
    return table.round(dict.fromkeys(frequency_columns, 6))


def write_column_list(path: Path) -> None:
    """Later steps read this instead of hard-coding column names. No features list: nothing trains on this table."""
    column_list = {
        "id": "variant_id",
        "id_note": "canonical spelling: leftmost, a duplication written as an insertion. See scripts/variant_spelling.py",
        "other_spellings": {"as_the_database_writes_it": "database_id", "rightmost": "rightmost_id", "letters_it_can_slide": "shift_room"},
        "label": "label",
        "label_note": "1 pathogenic, 0 benign, empty for verdict 'contested'",
        "mutation_type": "mutation_type",
        "mutation_types_most_severe_first": list(MUTATION_TYPES),
        "af_global": "af_global",
        "af_by_site": dict(zip(["site_oslo", "site_karachi", "site_lagos"], [f"af_{pop}" for pop in SITE_POPULATIONS], strict=True)),
        "evaluation_subset": "pop_discordant",
    }
    path.write_text(json.dumps(column_list, indent=2), encoding="utf-8")
    print(f"wrote {path}")


def summarise(table: pd.DataFrame, downloaded: int, dropped: Counter, checks: dict, type_from: dict, seconds: float) -> dict:
    by_type = pd.crosstab(table.mutation_type, table.verdict).reindex(list(MUTATION_TYPES)).fillna(0).astype(int)
    by_type = by_type.reindex(columns=["pathogenic", "benign", "contested"], fill_value=0)
    slides = table[table.kind.isin(["insertion", "deletion"])]
    cache_bytes = sum(file.stat().st_size for file in CACHE_DIR.glob("*.json"))
    sequence_bytes = sum(file.stat().st_size for file in variant_spelling.SEQUENCE_DIR.glob("*.json"))

    print("\n================ SUMMARY ================")
    print(f"downloaded {downloaded} records, kept {len(table)}")
    for why, count in dropped.most_common():
        print(f"  dropped {count:>6}  {why}")
    print("\nrows by mutation type and verdict:")
    print(by_type.to_string())
    print(f"\nmutation type read from: {type_from}")
    print(f"found in gnomAD: {int(table.in_gnomad.sum())} of {len(table)}")
    print(f"population-discordant: {int(table.pop_discordant.sum())}")
    print(f"\nreference letter check: {checks['single_letter_changes_matching']} of {checks['single_letter_changes']} "
          f"single-letter changes name the letter found in the fetched sequence")
    print(f"                        {checks['deletions_matching']} of {checks['deletions_with_letters']} deletions remove the letters ClinVar says they remove")
    print(f"insertions and deletions: {len(slides)}, of which {int((slides.shift_room > 0).sum())} have more than one valid spelling")
    print(f"download cache: {cache_bytes / 1e6:.1f} MB, reference sequence: {sequence_bytes / 1e6:.1f} MB, "
          f"time spent downloading in this run: {seconds:.0f} s\n")

    return {
        "downloaded": downloaded, "kept": len(table), "dropped": dict(dropped.most_common()),
        "rows_by_type_and_verdict": by_type.to_dict(orient="index"), "mutation_type_read_from": type_from,
        "in_gnomad": int(table.in_gnomad.sum()), "pop_discordant": int(table.pop_discordant.sum()),
        "reference_letter_check": checks,
        "insertions_and_deletions": len(slides), "with_more_than_one_spelling": int((slides.shift_room > 0).sum()),
        "download_cache_mb": round(cache_bytes / 1e6, 1), "reference_sequence_mb": round(sequence_bytes / 1e6, 1),
        "seconds_downloading_this_run": round(seconds),
    }


if __name__ == "__main__":
    raise SystemExit(main())
