"""One DNA change, one spelling.

An insertion or a deletion that sits in a repeat can be written several valid ways.
Take the letters `GAGAGGG GAGAGGG`: deleting the first copy or the second leaves the
same DNA, yet the two deletions get different positions and so different ids. If one
hospital files the variant under one id and another hospital asks for the other, the
count comes back zero and nobody notices.

The cure is to agree on one spelling. This module slides every insertion and deletion
as far LEFT as the reference sequence allows and writes it there:

    substitution   chr14:g.23429278C>T              cannot slide, stays as it is
    deletion       chr11:g.47332275_47332299del     leftmost position
    insertion      chr11:g.47350563_47350564insC    leftmost position. A duplication is written
                                                    as the insertion it is, so the `dup` and
                                                    `ins` spellings of one change collapse too

It also reports the rightmost spelling, which is the one the HGVS naming rules ask
for, and how many letters of room the change has to slide. Room above zero means the
variant has more than one valid spelling.

The reference sequence of every panel gene is downloaded once from Ensembl (GRCh38)
into data/raw_sequence/. After that everything here runs offline.

Usage:
    uv run python scripts/variant_spelling.py --fetch          # download the sequences that are missing
    uv run python scripts/variant_spelling.py --self-check     # small made-up cases, plus the real MYBPC3 pair
    uv run python scripts/variant_spelling.py "chr11:g.47332282_47332306del"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SEQUENCE_DIR = DATA_DIR / "raw_sequence"
REGIONS_FILE = SEQUENCE_DIR / "regions.json"
PANEL_FILE = ROOT / "config" / "cardiac_gene_panel.txt"

ENSEMBL = "https://rest.ensembl.org"
ASSEMBLY = "GRCh38"  # the genome build that myvariant.info calls hg38
PADDING = 5_000  # letters fetched on each side of a gene
LETTERS_PER_REQUEST = 4_000_000  # Ensembl allows 10 million. DMD, the longest gene here, takes one request
FURTHEST_VARIANT = 200_000  # a variant filed under a gene but further from it than this is not fetched for

SUBSTITUTION = re.compile(r"chr(\w+):g\.(\d+)([ACGT])>([ACGT])")
CHANGE = re.compile(r"chr(\w+):g\.(\d+)(?:_(\d+))?(delins|del|dup|ins)([ACGTN]*)", re.IGNORECASE)

# The heart example: one 25-letter deletion in MYBPC3, common in South Asians, filed under two ids.
MYBPC3_AS_CLINVAR_AND_GNOMAD = "chr11:g.47332275_47332299del"
MYBPC3_AS_DBSNP = "chr11:g.47332282_47332306del"


class SpellingError(ValueError):
    """The text is not a DNA change this module can read."""


class NoSequence(LookupError):
    """The change lies outside every reference sequence in data/raw_sequence/."""


@dataclass(frozen=True)
class Change:
    """An id as it was written, before any sliding."""

    chrom: str
    start: int  # first position named in the id
    end: int  # last position named in the id
    kind: str  # del, dup, ins or delins
    letters: str  # the letters put in, for ins and delins


@dataclass(frozen=True)
class Spelling:
    canonical: str  # the one spelling everybody uses: leftmost, a duplication written as an insertion
    rightmost: str  # the same change as far right as it goes, a duplication written as `dup`
    room: int  # letters it can slide. 0 means there is one position only
    kind: str  # substitution, deletion, insertion or delins
    length: int  # letters deleted or inserted, whichever is more

    @property
    def has_other_spellings(self) -> bool:
        return self.room > 0


# ---------------------------------------------------------------------------
# Reading an id, sliding the change, writing it again
# ---------------------------------------------------------------------------
def normalise(variant_id: str) -> Spelling:
    """The canonical spelling of any id the API uses: `>`, del, dup, ins and delins."""
    text = variant_id.strip()
    if SUBSTITUTION.fullmatch(text):
        return Spelling(text, text, 0, "substitution", 1)  # cannot slide, and needs no sequence
    change = read_change(text)
    return slide(change, Sequence.covering(change.chrom, change.start))


def normalise_vcf(chrom: str, position: int, before: str, after: str) -> Spelling:
    """The same for a change written the way VCF files write it: a position, the letters there, the letters instead."""
    change = Change(chrom, position, position + len(before) - 1, "delins", after.upper())
    sequence = Sequence.covering(chrom, position)
    if sequence.letters(change.start, change.end) != before.upper():
        raise SpellingError(f"chr{chrom}:{position} does not read {before} in the reference sequence")
    return slide(change, sequence)


def canonical(variant_id: str) -> str:
    return normalise(variant_id).canonical


def looks_like_an_id(text: str) -> bool:
    return bool(SUBSTITUTION.fullmatch(text.strip()) or CHANGE.fullmatch(text.strip()))


def read_change(text: str) -> Change:
    found = CHANGE.fullmatch(text)
    if not found:
        raise SpellingError(f"'{text}' is not a DNA change such as {MYBPC3_AS_CLINVAR_AND_GNOMAD}")
    chrom, start, end, kind, letters = found.groups()
    change = Change(chrom, int(start), int(end or start), kind.lower(), letters.upper())

    if change.end < change.start:
        raise SpellingError(f"'{text}' ends before it starts")
    if change.kind == "ins" and (change.end != change.start + 1 or not change.letters):
        raise SpellingError(f"'{text}': an insertion names the two positions it goes between, then the letters")
    if change.kind == "delins" and not change.letters:
        raise SpellingError(f"'{text}': a delins says which letters go in")
    return change


def slide(change: Change, sequence: Sequence) -> Spelling:
    """Reduce the change to the letters that really differ, then find how far it slides each way."""
    chrom = change.chrom
    # (position of the first letter touched, letters taken out, letters put in).
    # An insertion takes nothing out, and its position is the letter it goes in front of.
    if change.kind == "ins":
        start, taken_out, put_in = change.end, "", change.letters
    elif change.kind == "dup":  # a copy of the named letters, put in right after them
        start, taken_out, put_in = change.end + 1, "", sequence.letters(change.start, change.end)
    else:  # del or delins. Letters repeated after `del` add nothing and are ignored
        start, taken_out = change.start, sequence.letters(change.start, change.end)
        put_in = change.letters if change.kind == "delins" else ""

    # Drop what both sides share, from the right first so that what remains sits leftmost.
    while taken_out and put_in and taken_out[-1] == put_in[-1]:
        taken_out, put_in = taken_out[:-1], put_in[:-1]
    while taken_out and put_in and taken_out[0] == put_in[0]:
        taken_out, put_in, start = taken_out[1:], put_in[1:], start + 1
    if not taken_out and not put_in:
        raise SpellingError("the change puts back exactly what it takes out")

    if taken_out and put_in:  # letters swapped for other letters: nothing to slide
        if len(taken_out) == len(put_in) == 1:
            spelled = f"chr{chrom}:g.{start}{taken_out}>{put_in}"
            return Spelling(spelled, spelled, 0, "substitution", 1)
        spelled = f"chr{chrom}:g.{span(start, len(taken_out))}delins{put_in}"
        return Spelling(spelled, spelled, 0, "delins", max(len(taken_out), len(put_in)))
    if taken_out:
        return slide_deletion(chrom, start, len(taken_out), sequence)
    return slide_insertion(chrom, start, put_in, sequence)


def slide_deletion(chrom: str, start: int, size: int, sequence: Sequence) -> Spelling:
    """Deleting letters 5 to 7 equals deleting 4 to 6 whenever letter 4 equals letter 7."""
    left = right = start
    while sequence.letter(left - 1) and sequence.letter(left - 1) == sequence.letter(left + size - 1):
        left -= 1
    while sequence.letter(right + size) and sequence.letter(right + size) == sequence.letter(right):
        right += 1
    leftmost, rightmost = f"chr{chrom}:g.{span(left, size)}del", f"chr{chrom}:g.{span(right, size)}del"
    return Spelling(leftmost, rightmost, right - left, "deletion", size)


def slide_insertion(chrom: str, start: int, put_in: str, sequence: Sequence) -> Spelling:
    """Inserting `CAG` in front of a `G` equals inserting `GCA` one place to the left: the letters rotate."""
    left, at_left = start, put_in
    while sequence.letter(left - 1) == at_left[-1]:
        left, at_left = left - 1, at_left[-1] + at_left[:-1]
    right, at_right = start, put_in
    while sequence.letter(right) == at_right[0]:
        right, at_right = right + 1, at_right[1:] + at_right[0]

    size = len(put_in)
    if right - size >= sequence.start and sequence.letters(right - size, right - 1) == at_right:
        rightmost = f"chr{chrom}:g.{span(right - size, size)}dup"  # a copy of the letters just before it
    else:
        rightmost = f"chr{chrom}:g.{right - 1}_{right}ins{at_right}"
    return Spelling(f"chr{chrom}:g.{left - 1}_{left}ins{at_left}", rightmost, right - left, "insertion", size)


def every_spelling(variant_id: str) -> list[str]:
    """Every valid way to write the change, left to right. A duplication is listed as `ins` and as `dup`."""
    spelling = normalise(variant_id)
    if spelling.kind in ("substitution", "delins"):
        return [spelling.canonical]
    change = read_change(spelling.canonical)
    sequence = Sequence.covering(change.chrom, change.start)

    spelled = []
    for step in range(spelling.room + 1):
        if spelling.kind == "deletion":
            spelled.append(f"chr{change.chrom}:g.{span(change.start + step, spelling.length)}del")
            continue
        start = change.end + step
        put_in = change.letters[step % spelling.length:] + change.letters[:step % spelling.length]
        spelled.append(f"chr{change.chrom}:g.{start - 1}_{start}ins{put_in}")
        if start - spelling.length >= sequence.start and sequence.letters(start - spelling.length, start - 1) == put_in:
            spelled.append(f"chr{change.chrom}:g.{span(start - spelling.length, spelling.length)}dup")
    return spelled


def span(start: int, size: int) -> str:
    return str(start) if size == 1 else f"{start}_{start + size - 1}"


# ---------------------------------------------------------------------------
# The reference sequence, read from the cache in data/raw_sequence/
# ---------------------------------------------------------------------------
class Sequence:
    """The reference letters of one stretch of a chromosome. Positions count from 1, as in the ids."""

    def __init__(self, chrom: str, start: int, letters: str) -> None:
        self.chrom, self.start, self.text = chrom, start, letters
        self.end = start + len(letters) - 1

    def letter(self, position: int) -> str:
        """One letter, or '' beyond the edge of what was fetched, which stops any sliding there."""
        return self.text[position - self.start] if self.start <= position <= self.end else ""

    def letters(self, first: int, last: int) -> str:
        if last < first:
            return ""
        if first < self.start or last > self.end:
            raise NoSequence(f"chr{self.chrom}:{first}-{last} runs past the fetched sequence")
        return self.text[first - self.start:last - self.start + 1]

    @staticmethod
    def covering(chrom: str, position: int) -> Sequence:
        for gene, (region_chrom, start, end) in regions().items():
            if region_chrom == chrom and start <= position <= end:
                return gene_sequence(gene)
        raise NoSequence(f"chr{chrom}:{position} is outside the panel genes, so its spelling cannot be checked")


@cache
def regions() -> dict[str, tuple[str, int, int]]:
    """gene -> (chromosome, first position, last position) of every cached sequence."""
    return {gene: (chrom, start, end) for gene, (chrom, start, end, _, _) in read_index().items()}


def read_index() -> dict[str, list]:
    """data/raw_sequence/regions.json: gene -> [chromosome, first and last position fetched, where the gene itself starts and ends]."""
    return json.loads(REGIONS_FILE.read_text(encoding="utf-8")) if REGIONS_FILE.exists() else {}


@cache
def gene_sequence(gene: str) -> Sequence:
    saved = json.loads((SEQUENCE_DIR / f"{gene}.json").read_text(encoding="utf-8"))
    return Sequence(saved["chrom"], saved["start"], saved["sequence"])


# ---------------------------------------------------------------------------
# Download, once per gene
# ---------------------------------------------------------------------------
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "team12-fl-clinical-diagnostics (hackathon)"
SESSION.headers["Content-Type"] = "application/json"


def fetch_gene(gene: str, must_cover: tuple[str, int, int] | None = None, refresh: bool = False) -> str:
    """Cache the sequence of one gene, padded, and wide enough for `must_cover` = (chromosome, lowest, highest).

    Returns a word for the log: cached, fetched, or not found. A cached gene needs no network.
    """
    known = read_index().get(gene)
    if known and not refresh:
        chrom, _, _, gene_start, gene_end = known
    else:
        found = ask_ensembl(f"/lookup/symbol/homo_sapiens/{gene}")
        if found and found.get("assembly_name") == ASSEMBLY:
            chrom, gene_start, gene_end = str(found["seq_region_name"]), int(found["start"]), int(found["end"])
        elif must_cover:  # Ensembl does not know the symbol: fetch around the variants instead
            chrom, gene_start, gene_end = must_cover
        else:
            return "not found"

    start, end = gene_start, gene_end
    if must_cover and must_cover[0] == chrom:  # variants filed under the gene but lying just outside it
        start = min(start, max(must_cover[1], gene_start - FURTHEST_VARIANT))
        end = max(end, min(must_cover[2], gene_end + FURTHEST_VARIANT))
    start, end = max(1, start - PADDING), end + PADDING
    if known and not refresh and known[1] <= start and known[2] >= end:
        return "cached"

    pieces = []
    for first in range(start, end + 1, LETTERS_PER_REQUEST):
        last = min(first + LETTERS_PER_REQUEST - 1, end)
        piece = ask_ensembl(f"/sequence/region/human/{chrom}:{first}..{last}:1", {"coord_system_version": ASSEMBLY})
        if not piece or len(piece.get("seq", "")) != last - first + 1:
            raise RuntimeError(f"Ensembl did not return chr{chrom}:{first}-{last} for {gene}")
        pieces.append(piece["seq"].upper())
        time.sleep(0.2)

    SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)
    saved = {"gene": gene, "assembly": ASSEMBLY, "chrom": chrom, "start": start, "end": end,
             "source": f"{ENSEMBL}/sequence/region/human", "sequence": "".join(pieces)}
    (SEQUENCE_DIR / f"{gene}.json").write_text(json.dumps(saved), encoding="utf-8")

    # The index is rewritten after every gene, so a rerun carries on where this one stopped.
    index = {**read_index(), gene: [chrom, start, end, gene_start, gene_end]}
    REGIONS_FILE.write_text(json.dumps(index, indent=1), encoding="utf-8")
    regions.cache_clear()
    gene_sequence.cache_clear()
    return "fetched"


def ask_ensembl(path: str, params: dict | None = None) -> dict | None:
    """GET with retries. An unknown gene symbol arrives as a 400 or a 404: return None."""
    for attempt in range(6):
        try:
            response = SESSION.get(ENSEMBL + path, params=params, timeout=120)
            if response.status_code == 200:
                return response.json()
            if response.status_code in (400, 404):
                return None
            if response.status_code == 429:  # asked too fast: Ensembl says how long to wait
                time.sleep(float(response.headers.get("Retry-After", 1)))
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2**attempt)
    raise RuntimeError(f"Ensembl kept failing for: {path}")


def read_gene_panel(path: Path) -> list[str]:
    lines = (line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines())
    return list(dict.fromkeys(line for line in lines if line))


def fetch_panel(refresh: bool = False) -> int:
    genes = read_gene_panel(PANEL_FILE)
    missing = []
    for number, gene in enumerate(genes, 1):
        outcome = fetch_gene(gene, refresh=refresh)
        chrom, start, end = regions().get(gene, ("?", 0, -1))
        print(f"[{number:>3}/{len(genes)}] {gene:<8} {outcome:<9} chr{chrom}:{start:,}-{end:,}", flush=True)
        if outcome == "not found":
            missing.append(gene)
    letters = sum(end - start + 1 for _, start, end in regions().values())
    print(f"\n{len(regions())} gene sequences in {SEQUENCE_DIR.relative_to(ROOT)}, {letters:,} letters")
    if missing:
        print(f"Ensembl does not know: {', '.join(missing)}")
    return 1 if missing else 0


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def self_check() -> int:
    """Made-up sequences where the right answer is plain to see, then the real MYBPC3 pair."""
    failures = 0

    def expect(what: str, got, wanted) -> None:
        nonlocal failures
        ok = got == wanted
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {what}: {got}" + ("" if ok else f"   wanted {wanted}"))

    def on(letters: str, text: str) -> Spelling:
        return slide(read_change(text), Sequence("T", 1, letters))

    print("a run of one letter, GGC AAAAA TCC: losing any one A is the same change")
    spelled = {on("GGCAAAAATCC", f"chrT:g.{position}del") for position in range(4, 9)}
    expect("five ids, one spelling", len(spelled), 1)
    expect("spelling", spelled.pop(), Spelling("chrT:g.4del", "chrT:g.8del", 4, "deletion", 1))

    print("a two-letter repeat, TT CACACA GG: losing one CA, or one AC, is the same change")
    spelled = {on("TTCACACAGG", f"chrT:g.{position}_{position + 1}del") for position in range(3, 8)}
    expect("five ids, one spelling", len(spelled), 1)
    expect("spelling", spelled.pop(), Spelling("chrT:g.3_4del", "chrT:g.7_8del", 4, "deletion", 2))

    print("dup against ins, AA GCTGCT TA: a third GCT, written four ways")
    spelled = {on("AAGCTGCTTA", text) for text in ("chrT:g.6_8dup", "chrT:g.5_6insGCT", "chrT:g.3_5dup", "chrT:g.4_5insTGC")}
    expect("four ids, one spelling", len(spelled), 1)
    expect("spelling", spelled.pop(), Spelling("chrT:g.2_3insGCT", "chrT:g.6_8dup", 6, "insertion", 3))

    print("nothing to slide along, ACGTACGT")
    expect("deletion stays put", on("ACGTACGT", "chrT:g.3del"), Spelling("chrT:g.3del", "chrT:g.3del", 0, "deletion", 1))
    expect("insertion stays put", on("ACGTACGT", "chrT:g.2_3insTT").room, 0)
    expect("a different insertion is a different change", on("AAGCTGCTTA", "chrT:g.5_6insGCA").canonical, "chrT:g.5_6insGCA")

    print("a delins that is a simpler change in disguise")
    expect("one letter swapped", on("ACGTACGT", "chrT:g.2_3delinsCA").canonical, "chrT:g.3G>A")
    expect("two letters lost from a run", on("GGCAAAAATCC", "chrT:g.5_7delinsA").canonical, "chrT:g.4_5del")
    expect("a real delins is left alone", on("ACGTACGT", "chrT:g.2_3delinsTTT").canonical, "chrT:g.2_3delinsTTT")

    print("the edge of the fetched sequence stops the sliding")
    expect("run of A reaching the edge", on("AAAAC", "chrT:g.3del"), Spelling("chrT:g.1del", "chrT:g.4del", 3, "deletion", 1))

    print("the real pair: the MYBPC3 25-letter deletion, as ClinVar and gnomAD file it and as dbSNP files it")
    try:
        first, second = normalise(MYBPC3_AS_CLINVAR_AND_GNOMAD), normalise(MYBPC3_AS_DBSNP)
        expect("two ids, one spelling", first, second)
        expect("canonical", first.canonical, MYBPC3_AS_CLINVAR_AND_GNOMAD)
        expect("rightmost", first.rightmost, MYBPC3_AS_DBSNP)
        expect("every spelling", len(every_spelling(MYBPC3_AS_DBSNP)), first.room + 1)
    except NoSequence as problem:
        failures += 1
        print(f"  FAIL {problem}\n       Fetch the sequences first: uv run python scripts/variant_spelling.py --fetch")

    print("\nself-check passed" if not failures else f"\nSELF-CHECK FAILED: {failures} wrong")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Give a DNA change its one canonical spelling")
    parser.add_argument("variant", nargs="?", help="an id such as chr11:g.47332282_47332306del")
    parser.add_argument("--fetch", action="store_true", help="download the reference sequence of every panel gene")
    parser.add_argument("--refresh", action="store_true", help="with --fetch: download again")
    parser.add_argument("--self-check", action="store_true", help="prove the sliding on cases with a known answer")
    args = parser.parse_args()

    if args.fetch:
        return fetch_panel(args.refresh)
    if args.self_check:
        return self_check()
    if not args.variant:
        parser.print_help()
        return 1
    try:
        spelling = normalise(args.variant)
        ways = every_spelling(args.variant)
    except (SpellingError, NoSequence) as problem:
        print(problem, file=sys.stderr)
        return 1
    print(f"canonical  {spelling.canonical}\nrightmost  {spelling.rightmost}")
    print(f"{spelling.kind} of {spelling.length} letter{'' if spelling.length == 1 else 's'}, room to slide: {spelling.room}")
    print(f"{len(ways)} valid spelling{'' if len(ways) == 1 else 's'}:")
    for way in ways:
        print(f"  {way}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
