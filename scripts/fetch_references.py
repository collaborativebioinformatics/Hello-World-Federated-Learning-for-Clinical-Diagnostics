"""Fetch the manuscript's references from PubMed and arXiv, so no author list is typed by hand.

Hand-typed references pick up wrong authors and page numbers. Here a reference is
one line: a short key and a DOI or arXiv id. The record comes back from the
source, and the reference list in Manuscript.md is pasted from what this prints.
DOIs are looked up in PubMed, or in Europe PMC, which mirrors it, when PubMed
cannot be reached.

Writes  docs/references.ris   every record, for Zotero or EndNote
Prints  the numbered reference list, in the order below

To add a reference: add a line to REFERENCES where it is first cited, rerun, and
paste the printed list over the References section of Manuscript.md.

Usage:
    uv run python scripts/fetch_references.py
"""

from __future__ import annotations

import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RIS_FILE = ROOT / "docs" / "references.ris"

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EUROPE_PMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
ARXIV = "http://export.arxiv.org/api/query"
ATOM = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
PAUSE = 0.4  # seconds between calls; PubMed allows three a second without a key
AUTHORS_SHOWN = 6  # Vancouver style: the first six, then "et al."
COMPOUND_SURNAMES = ["Agüera y Arcas"]  # arXiv does not mark where a surname starts

# In order of first citation in Manuscript.md. The number a reference gets is its place here.
REFERENCES = [
    ("richards2015", "10.1038/gim.2015.30"),
    ("manrai2016", "10.1056/NEJMsa1507092"),
    ("popejoy2016", "10.1038/538161a"),
    ("karczewski2020", "10.1038/s41586-020-2308-7"),
    ("rieke2020", "10.1038/s41746-020-00323-1"),
    ("mcmahan2017", "arXiv:1602.05629"),
    ("roth2022", "arXiv:2210.13291"),
    ("rambla2022", "10.1002/humu.24369"),
    ("montalvo2025", "10.1093/bioinformatics/btaf523"),
    ("martin2019", "10.1038/s41588-019-0528-2"),
    ("xin2016", "10.1186/s13059-016-0953-9"),
    ("landrum2018", "10.1093/nar/gkx1153"),
    ("liu2020", "10.1186/s13073-020-00803-9"),
    ("grimm2015", "10.1002/humu.22768"),
    ("whiffin2017", "10.1038/gim.2017.26"),
    ("jacobson1997", "10.1056/NEJM199702133360703"),
    ("cheng2023", "10.1126/science.adg7492"),
    ("shringarpure2015", "10.1016/j.ajhg.2015.09.010"),
    ("gargano2024", "10.1093/nar/gkad1005"),
]


@dataclass
class Reference:
    key: str
    authors: list[str]
    title: str
    year: str
    journal: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    doi: str = ""
    arxiv_id: str = ""
    pmid: str = ""
    full_authors: list[str] = field(default_factory=list)  # "Surname, Given", for the RIS file

    def vancouver(self) -> str:
        names = ", ".join(self.authors[:AUTHORS_SHOWN]) + (", et al" if len(self.authors) > AUTHORS_SHOWN else "")
        title = self.title.rstrip(".")
        if self.arxiv_id:
            return f"{names}. {title}. arXiv:{self.arxiv_id} [preprint]. {self.year}."
        where = f"{self.year};{self.volume}" + (f"({self.issue})" if self.issue else "") + (f":{self.pages}" if self.pages else "")
        return f"{names}. {title}. {self.journal}. {where}. doi:{self.doi}"

    def ris(self) -> str:
        lines = [("TY", "GEN" if self.arxiv_id else "JOUR"), ("ID", self.key)]
        lines += [("AU", name) for name in self.full_authors]
        lines += [("TI", self.title), ("PY", self.year)]
        if self.arxiv_id:
            lines += [("PB", "arXiv"), ("UR", f"https://arxiv.org/abs/{self.arxiv_id}")]
        else:
            lines += [("JO", self.journal), ("VL", self.volume), ("IS", self.issue), ("SP", self.pages),
                      ("DO", self.doi), ("AN", self.pmid)]
        return "\n".join(f"{tag}  - {value}" for tag, value in lines if value) + "\nER  - \n"


def get(url: str, **params: str) -> requests.Response:
    """GET with a pause before every call and a few retries, since both services drop requests now and then."""
    for attempt in range(5):
        time.sleep(PAUSE * 2**attempt)
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.ok:
                return response
        except requests.RequestException:
            pass
    raise ConnectionError(f"no answer from {url} for {params}")


def from_pubmed(key: str, doi: str) -> Reference:
    """DOI to PubMed id, then the full record. Fails loudly if PubMed does not know the DOI."""
    found = get(f"{EUTILS}/esearch.fcgi", db="pubmed", term=f"{doi}[AID]", retmode="json").json()
    ids = found["esearchresult"]["idlist"]
    if len(ids) != 1:
        raise LookupError(f"{key}: PubMed returned {len(ids)} records for DOI {doi}")

    article = ET.fromstring(get(f"{EUTILS}/efetch.fcgi", db="pubmed", id=ids[0], retmode="xml").content)

    def text(path: str) -> str:
        found = article.find(path)
        return "".join(found.itertext()).strip() if found is not None else ""

    authors, full_authors = [], []
    for author in article.iterfind(".//AuthorList/Author"):
        surname, initials, given = (author.findtext(tag, "") for tag in ("LastName", "Initials", "ForeName"))
        if surname:
            authors.append(f"{surname} {initials}".strip())
            full_authors.append(f"{surname}, {given}".rstrip(", "))
        elif author.findtext("CollectiveName"):
            authors.append(author.findtext("CollectiveName"))
            full_authors.append(author.findtext("CollectiveName"))

    return Reference(
        key=key, authors=authors, full_authors=full_authors,
        title=text(".//ArticleTitle"),
        year=text(".//JournalIssue/PubDate/Year") or text(".//JournalIssue/PubDate/MedlineDate")[:4],
        journal=text(".//Journal/ISOAbbreviation").replace(".", ""),
        volume=text(".//JournalIssue/Volume"), issue=text(".//JournalIssue/Issue"),
        pages=text(".//Pagination/MedlinePgn"), doi=doi, pmid=ids[0],
    )


def from_europe_pmc(key: str, doi: str) -> Reference:
    """The same record from Europe PMC, which mirrors PubMed. Used when PubMed cannot be reached."""
    found = get(EUROPE_PMC, query=f'DOI:"{doi}"', format="json", resultType="core").json()
    records = [r for r in found["resultList"]["result"] if r.get("doi", "").lower() == doi.lower()]
    if len(records) != 1:
        raise LookupError(f"{key}: Europe PMC returned {len(records)} records for DOI {doi}")
    record, journal = records[0], records[0].get("journalInfo", {})

    authors, full_authors = [], []
    for author in record.get("authorList", {}).get("author", []):
        if author.get("lastName"):
            authors.append(f"{author['lastName']} {author.get('initials', '')}".strip())
            full_authors.append(f"{author['lastName']}, {author.get('firstName', '')}".rstrip(", "))
        elif author.get("collectiveName"):
            authors.append(author["collectiveName"])
            full_authors.append(author["collectiveName"])

    return Reference(
        key=key, authors=authors, full_authors=full_authors, title=record["title"],
        year=str(journal.get("yearOfPublication", record.get("pubYear", ""))),
        journal=journal.get("journal", {}).get("isoabbreviation", "").replace(".", ""),
        volume=journal.get("volume", ""), issue=journal.get("issue", ""),
        pages=record.get("pageInfo", ""), doi=doi, pmid=record.get("pmid", ""),
    )


pubmed_reachable = True


def from_doi(key: str, doi: str) -> Reference:
    global pubmed_reachable
    if pubmed_reachable:
        try:
            return from_pubmed(key, doi)
        except ConnectionError:
            pubmed_reachable = False  # do not wait for it again on every reference
            print("  PubMed cannot be reached, using Europe PMC instead", file=sys.stderr)
    return from_europe_pmc(key, doi)


def split_name(name: str) -> tuple[str, str]:
    """arXiv gives one string per author. The surname is the last word, unless it is a known compound."""
    for surname in COMPOUND_SURNAMES:
        if name.endswith(surname):
            return name.removesuffix(surname).strip(), surname
    given, _, surname = name.rpartition(" ")
    return given, surname


def from_arxiv(key: str, arxiv_id: str) -> Reference:
    feed = ET.fromstring(get(ARXIV, id_list=arxiv_id).content)
    entry = feed.find("atom:entry", ATOM)
    if entry is None or entry.find("atom:title", ATOM) is None:
        raise LookupError(f"{key}: arXiv does not know {arxiv_id}")

    authors, full_authors = [], []
    for author in entry.iterfind("atom:author/atom:name", ATOM):
        given, surname = split_name(author.text)
        initials = "".join(part[0] for part in given.replace("-", " ").split())
        authors.append(f"{surname} {initials}")
        full_authors.append(f"{surname}, {given}")

    return Reference(
        key=key, authors=authors, full_authors=full_authors,
        title=" ".join(entry.findtext("atom:title", "", ATOM).split()),
        year=entry.findtext("atom:published", "", ATOM)[:4], arxiv_id=arxiv_id,
    )


def main() -> int:
    references = []
    for key, source in REFERENCES:
        is_arxiv = source.startswith("arXiv:")
        references.append(from_arxiv(key, source.removeprefix("arXiv:")) if is_arxiv else from_doi(key, source))
        print(f"  fetched {key}", file=sys.stderr)

    RIS_FILE.write_text("\n".join(reference.ris() for reference in references), encoding="utf-8", newline="\n")
    for number, reference in enumerate(references, start=1):
        print(f"{number}. {reference.vancouver()}")
    print(f"\nwrote {RIS_FILE.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
