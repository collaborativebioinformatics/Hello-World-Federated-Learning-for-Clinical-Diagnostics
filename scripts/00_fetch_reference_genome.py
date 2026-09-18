"""Step 0: fetch the GRCh38 human reference genome, so later steps can read DNA offline.

Variant ids in this project name a place on the genome, for example
chr11:g.47332275_47332299del. Turning that into letters needs the reference
genome. Asking Ensembl for one gene region at a time works, but it needs the
network every run and gets slow across hundreds of genes. This script fetches
the whole genome once (about 1 GB packed, 3.2 GB unpacked), checks it against
the MD5 the publisher lists, unpacks it, and writes a small index so any slice
can be read with one file seek.

Source: UCSC hg38.fa.gz, which is GRCh38 (GenBank GCA_000001405.15) with
`chr1`, `chrX` style names and repeats in lowercase (soft-masked). The Broad
GATK copy on Google Cloud Storage was tried first, but that bucket refused
anonymous downloads (HTTP 403 on 2026-09-18), so UCSC it is.

Writes, under data/reference_genome/ (git-ignored):
    hg38.fa.gz    the download, kept so the checksum can be rechecked
    hg38.fa       the unpacked genome
    hg38.fa.fai   its index: name, length, byte offset, bases per line, bytes per line
    md5sum.txt    the checksums UCSC publishes for that folder

Usage:
    uv run python scripts/00_fetch_reference_genome.py                                 # download, verify, unpack, index
    uv run python scripts/00_fetch_reference_genome.py --read chr11:47332275-47332299  # print one slice, 1-based inclusive

A stopped download resumes where it left off: just rerun. From another script,
load this file the way the other numbered steps load each other
(importlib.util.spec_from_file_location) and call read("chr11", 47332275, 47332299).
"""

from __future__ import annotations

import argparse
import functools
import gzip
import hashlib
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
GENOME_DIR = ROOT / "data" / "reference_genome"
FASTA_GZ = GENOME_DIR / "hg38.fa.gz"
FASTA = GENOME_DIR / "hg38.fa"
INDEX = GENOME_DIR / "hg38.fa.fai"
MD5SUMS = GENOME_DIR / "md5sum.txt"

BASE_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/"
RETRIES = 10
RETRY_DELAY = 15  # seconds between retries, like curl --retry-delay 15
CHUNK = 1 << 20  # 1 MB per network read and per hash update
REPORT_EVERY = 100 * CHUNK  # print progress about once per 100 MB


def remote_size(url: str) -> int:
    """Ask the server how big the file is, so a partial download can resume and a finished one can be checked."""
    response = requests.head(url, allow_redirects=True, timeout=60)
    response.raise_for_status()
    return int(response.headers["Content-Length"])


def fetch_rest(url: str, path: Path, expected_size: int) -> None:
    """One download pass. If part of the file is already on disk, ask the server only for the bytes after it."""
    have = path.stat().st_size if path.exists() else 0
    if have > expected_size:  # cannot be a partial copy of this file, so start over
        path.unlink()
        have = 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=120) as response:
        response.raise_for_status()
        resuming = response.status_code == 206  # 206 means the server honoured the range, 200 means it starts from zero
        got, started, next_report = 0, time.monotonic(), REPORT_EVERY
        with open(path, "ab" if resuming else "wb") as out:
            for chunk in response.iter_content(CHUNK):
                out.write(chunk)
                got += len(chunk)
                if got >= next_report:
                    print(f"  {(have if resuming else 0) + got:>13,} bytes  {got / (time.monotonic() - started) / 1e6:5.1f} MB/s")
                    next_report += REPORT_EVERY


def complete(path: Path, expected_size: int) -> bool:
    return path.exists() and path.stat().st_size == expected_size


def download(url: str, path: Path, expected_size: int) -> None:
    """Fetch url to path, resuming after network trouble until the file has the expected size."""
    attempt = 0
    while not complete(path, expected_size):
        if attempt > RETRIES:
            raise SystemExit(f"Could not finish downloading {url} after {RETRIES} retries. Rerun to resume.")
        attempt += 1
        try:
            fetch_rest(url, path, expected_size)
        except (requests.RequestException, OSError) as error:
            print(f"  attempt {attempt} stopped: {error}. Waiting {RETRY_DELAY} s, then resuming.")
            time.sleep(RETRY_DELAY)


def md5_of(path: Path) -> str:
    """Streaming MD5, so a file of a few GB never has to fit in memory."""
    digest = hashlib.md5()
    with open(path, "rb") as source:
        while chunk := source.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def published_md5(filename: str) -> str:
    """UCSC lists every file in the folder in md5sum.txt, one 'hash  name' line each."""
    for line in MD5SUMS.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        if name.strip() == filename:
            return digest
    raise SystemExit(f"{MD5SUMS.relative_to(ROOT)} has no line for {filename}")


def verify(path: Path, expected_size: int, expected_md5: str) -> None:
    size = path.stat().st_size
    digest = md5_of(path)
    print(f"  size {size:,} bytes  (published {expected_size:,})")
    print(f"  md5  {digest}  (published {expected_md5})")
    if size != expected_size or digest != expected_md5:
        raise SystemExit(f"{path.relative_to(ROOT)} does not match what UCSC published. Delete it and rerun.")
    print("  size and md5 match")


class Contig:
    """One sequence in the FASTA, described the way a samtools .fai line describes it.

    A plain class rather than a dataclass, because the other steps load this file with
    importlib without registering it, and a dataclass cannot resolve its annotations then.
    """

    def __init__(self, name: str, offset: int, length: int = 0, linebases: int = 0, linewidth: int = 0) -> None:
        self.name = name
        self.offset = offset  # byte offset of the first base
        self.length = length
        self.linebases = linebases  # bases on a full line
        self.linewidth = linewidth  # bytes on a full line, newline included
        self.ended = False  # a short line has been seen, so no more sequence lines may follow

    def fai_line(self) -> str:
        return f"{self.name}\t{self.length}\t{self.offset}\t{self.linebases}\t{self.linewidth}\n"


def unpack_and_index(gz_path: Path, fa_path: Path, fai_path: Path) -> list[Contig]:
    """Unpack the gzip to a plain FASTA and write a .fai index beside it, in one pass.

    The index lets read() turn a position into a byte offset with arithmetic, which only works
    when every line of a sequence except the last has the same width. UCSC files do, and this
    stops with a message if one does not.
    """
    contigs: list[Contig] = []
    position = 0  # bytes written so far, which is the offset of the next line
    with gzip.open(gz_path, "rb") as source, open(fa_path, "wb") as out:
        for line in source:
            out.write(line)
            if line.startswith(b">"):
                contigs.append(Contig(name=line[1:].split()[0].decode("ascii"), offset=position + len(line)))
            elif contigs:
                contig = contigs[-1]
                bases = len(line.rstrip(b"\r\n"))
                if contig.linebases == 0:  # the first sequence line sets the line shape
                    contig.linebases, contig.linewidth = bases, len(line)
                elif contig.ended or bases > contig.linebases:
                    raise SystemExit(f"{contig.name} has uneven line widths, which a .fai index cannot describe")
                contig.ended = bases < contig.linebases
                contig.length += bases
            position += len(line)
    fai_path.write_text("".join(contig.fai_line() for contig in contigs), encoding="utf-8", newline="\n")
    return contigs


@functools.lru_cache(maxsize=1)
def load_index() -> dict[str, Contig]:
    if not INDEX.exists():
        raise SystemExit(f"No reference genome yet. Run: uv run python {Path(__file__).relative_to(ROOT).as_posix()}")
    index = {}
    for line in INDEX.read_text(encoding="utf-8").splitlines():
        name, length, offset, linebases, linewidth = line.split("\t")
        index[name] = Contig(name, int(offset), int(length), int(linebases), int(linewidth))
    return index


def read(chrom: str, start: int, end: int) -> str:
    """Return the reference letters at chrom:start-end, 1-based and inclusive like HGVS.

    Repeats come back in lowercase because UCSC soft-masks them. Upper-case the result when
    only the letters matter. Accepts Ensembl-style names too ("11", "X", "MT").
    """
    index = load_index()
    if chrom not in index:
        chrom = {"MT": "chrM"}.get(chrom, f"chr{chrom}")
    if chrom not in index:
        raise KeyError(f"{chrom} is not in the reference genome")
    contig = index[chrom]
    if not 1 <= start <= end <= contig.length:
        raise ValueError(f"{chrom}:{start}-{end} is outside 1..{contig.length}")

    def byte_of(base: int) -> int:
        # base is 0-based. Every full line before it adds one newline on top of its bases.
        return contig.offset + (base // contig.linebases) * contig.linewidth + base % contig.linebases

    first, last = byte_of(start - 1), byte_of(end - 1)
    with open(FASTA, "rb") as source:
        source.seek(first)
        raw = source.read(last - first + 1)
    return raw.replace(b"\n", b"").replace(b"\r", b"").decode("ascii")


def fetch() -> None:
    GENOME_DIR.mkdir(parents=True, exist_ok=True)
    if FASTA.exists() and INDEX.exists():
        print(f"{FASTA.relative_to(ROOT)} and its index are already there. Delete them to rebuild.")
        return

    print("fetching the checksums UCSC publishes")
    response = requests.get(BASE_URL + "md5sum.txt", timeout=60)
    response.raise_for_status()
    MD5SUMS.write_bytes(response.content)

    print(f"downloading {FASTA_GZ.name}")
    started = time.monotonic()
    size = remote_size(BASE_URL + FASTA_GZ.name)
    download(BASE_URL + FASTA_GZ.name, FASTA_GZ, size)
    print(f"  done in {time.monotonic() - started:,.0f} s")

    print(f"verifying {FASTA_GZ.name}")
    verify(FASTA_GZ, size, published_md5(FASTA_GZ.name))

    print(f"unpacking to {FASTA.name} and writing {INDEX.name}")
    started = time.monotonic()
    contigs = unpack_and_index(FASTA_GZ, FASTA, INDEX)
    main_chromosomes = [contig for contig in contigs if "_" not in contig.name]
    print(f"  {len(contigs)} sequences, {len(main_chromosomes)} of them whole chromosomes, in {time.monotonic() - started:,.0f} s")
    print(f"  {FASTA.stat().st_size:,} bytes in {FASTA.relative_to(ROOT)}")


def parse_region(text: str) -> tuple[str, int, int]:
    """'chr11:47332275-47332299' -> ('chr11', 47332275, 47332299). Commas in the numbers are allowed."""
    chrom, _, span = text.partition(":")
    start, _, end = span.partition("-")
    if not (chrom and start and end):
        raise SystemExit("A region looks like chr11:47332275-47332299")
    return chrom, int(start.replace(",", "")), int(end.replace(",", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the GRCh38 reference genome from UCSC and index it for offline reads")
    parser.add_argument("--read", metavar="CHROM:START-END", help="print the reference letters at this 1-based inclusive region instead")
    args = parser.parse_args()
    if args.read:
        print(read(*parse_region(args.read)))
    else:
        fetch()


if __name__ == "__main__":
    main()
