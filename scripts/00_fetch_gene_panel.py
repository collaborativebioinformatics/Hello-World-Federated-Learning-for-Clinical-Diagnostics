"""Step 0: fetch a gene list from Genomics England PanelApp.

PanelApp holds the gene panels NHS labs use to decide which genes to read for a
patient. Every gene on a panel is rated, and "green" means the evidence is strong
enough to use the gene for diagnosis. We take the green genes from one named set
of panels signed off by the NHS Genomic Medicine Service.

The sets live in config/panel_sets.json, one per disease area: `cardiac`, which
is the default, and `cancer`. Every panel in a set is pinned to a version, so the
list can be cited and rebuilt exactly. To update, change a version number there
and rerun.

Writes config/<NAME>_gene_panel.txt. Open API, no login.

Usage:
    uv run python scripts/00_fetch_gene_panel.py                  # the cardiac set
    uv run python scripts/00_fetch_gene_panel.py --panel cancer   # another set from config/panel_sets.json
    uv run python scripts/00_fetch_gene_panel.py --list cancer    # signed-off panels with "cancer" in the name

How to add a disease area: docs/disease_areas.md

Cite: Martin AR et al. PanelApp crowdsources expert knowledge to establish
consensus diagnostic gene panels. Nature Genetics 51, 1560-1565 (2019).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
PANEL_SETS = CONFIG_DIR / "panel_sets.json"

API_URL = "https://panelapp.genomicsengland.co.uk/api/v1/panels/{panel_id}/"
SIGNED_OFF_URL = "https://panelapp.genomicsengland.co.uk/api/v1/panels/signedoff/"
GREEN = "3"  # PanelApp's code for diagnostic-grade evidence
PAUSE = 1  # seconds between requests. PanelApp refuses callers that ask too fast

# Explains the short inheritance words. Written into the header of every gene list.
MODE_LEGEND = [
    "#   MONOALLELIC           one bad copy of the gene causes disease, also called dominant",
    "#   BIALLELIC             both copies must be bad, also called recessive. Healthy carriers exist",
    "#   BOTH                  PanelApp records both of the patterns above for this gene",
    "#   X-LINKED-MONOALLELIC  on the X chromosome. One bad copy in males, and one can be enough in females",
    "#   X-LINKED-BIALLELIC    on the X chromosome. One bad copy in males, two in females",
    "#   MITOCHONDRIAL         inherited through the mother's mitochondria",
    "#   UNKNOWN               PanelApp gives no clear mode",
]


def read_panel_set(name: str) -> dict:
    panel_sets = json.loads(PANEL_SETS.read_text(encoding="utf-8"))
    if name not in panel_sets:
        raise SystemExit(f"No panel set named '{name}' in {PANEL_SETS.relative_to(ROOT)}. It has: {', '.join(panel_sets)}.")
    return panel_sets[name]


def short_mode(text: str | None) -> str:
    """PanelApp spells inheritance out as a long sentence. Keep the word that says how many bad copies it takes."""
    text = (text or "").upper()
    if text.startswith("X-LINKED"):
        # Males have one X, so one bad copy is enough. The rest of the sentence says what it takes in females.
        return "X-LINKED-MONOALLELIC" if "MONOALLELIC" in text else "X-LINKED-BIALLELIC"
    for word in ("BOTH", "MONOALLELIC", "BIALLELIC", "MITOCHONDRIAL"):
        if text.startswith(word):
            return word
    return "UNKNOWN"


def ask_panelapp(url: str, params: dict | None = None) -> dict:
    """GET with retries. PanelApp answers 429 when asked too fast: wait as long as it says, then ask again."""
    for attempt in range(6):
        response = requests.get(url, params=params, timeout=60)
        if response.status_code != 429:
            response.raise_for_status()
            time.sleep(PAUSE)
            return response.json()
        asked_to_wait = response.headers.get("Retry-After", "")
        time.sleep(int(asked_to_wait) if asked_to_wait.isdigit() else 10 * 2**attempt)
    raise RuntimeError(f"PanelApp kept answering 'too many requests' for: {url}")


def fetch_green_genes(panel_id: int, version: str) -> tuple[str, dict[str, set[str]]]:
    """Return the panel's name, and for every green gene its modes of inheritance."""
    panel = ask_panelapp(API_URL.format(panel_id=panel_id), {"version": version})
    # A panel can list one gene twice (once per inheritance pattern), so a gene can collect several modes.
    modes_of_gene: dict[str, set[str]] = {}
    for entry in panel["genes"]:
        if str(entry["confidence_level"]) == GREEN:
            gene = entry["gene_data"]["gene_symbol"]
            modes_of_gene.setdefault(gene, set()).add(short_mode(entry.get("mode_of_inheritance")))
    return panel["name"], modes_of_gene


def write_gene_panel(name: str) -> None:
    panel_set = read_panel_set(name)
    panels_of_gene: dict[str, list[int]] = {}
    modes_of_gene: dict[str, set[str]] = {}
    summary_lines = []
    for panel in panel_set["panels"]:
        panel_id, version = panel["id"], panel["version"]
        panel_name, green = fetch_green_genes(panel_id, version)
        for gene, modes in green.items():
            panels_of_gene.setdefault(gene, []).append(panel_id)
            modes_of_gene.setdefault(gene, set()).update(modes)
        summary_lines.append(f"#   {panel_id:>4}  v{version:<6} {len(green):>3} green genes  {panel_name}")
        print(summary_lines[-1].lstrip("# "))

    header = [
        f"# {panel_set['title']} gene panel: green (diagnostic-grade) genes from Genomics England PanelApp,",
        "# taken from the panels below, which are signed off by the NHS Genomic Medicine Service.",
        f"# Written by scripts/00_fetch_gene_panel.py --panel {name} on {date.today()}. Rerunning overwrites this file.",
        "#",
        *summary_lines,
        "#",
        f"# {len(panels_of_gene)} genes. Each gene line reads:  GENE  # panels that list it | how the gene is inherited",
        "# Inheritance is PanelApp's mode_of_inheritance, shortened to one word. A gene with more than one mode",
        "# keeps them all, separated by '; '.",
        *MODE_LEGEND,
        "# Cite: Martin AR et al., Nature Genetics 51, 1560-1565 (2019). https://panelapp.genomicsengland.co.uk",
        "",
    ]
    gene_lines = [
        f"{gene:<10}# {', '.join(map(str, panel_ids))} | {'; '.join(sorted(modes_of_gene[gene]))}"
        for gene, panel_ids in sorted(panels_of_gene.items())
    ]
    gene_panel = CONFIG_DIR / f"{name}_gene_panel.txt"
    gene_panel.write_text("\n".join(header + gene_lines) + "\n", encoding="utf-8")
    print(f"\nwrote {gene_panel}  ({len(panels_of_gene)} genes)")


def list_signed_off(keyword: str) -> None:
    """Print the signed-off panels whose name contains the keyword, to pick panels for a new disease area."""
    matches = []
    url = SIGNED_OFF_URL
    while url:  # the list comes in pages, and each page gives the address of the next
        page = ask_panelapp(url)
        matches += [panel for panel in page["results"] if keyword.lower() in panel["name"].lower()]
        url = page.get("next")

    print(f"{'id':>5}  {'version':<8} {'genes':>5}  name")
    for panel in sorted(matches, key=lambda panel: panel["name"]):
        print(f"{panel['id']:>5}  v{panel['version']:<7} {panel['stats']['number_of_genes']:>5}  {panel['name']}")
    print(f"\n{len(matches)} signed-off panels have '{keyword}' in their name.")
    print("'version' is the signed-off one. 'genes' counts every rating, and step 0 keeps only the green ones.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a gene list from Genomics England PanelApp")
    parser.add_argument("--panel", default="cardiac", help="which set in config/panel_sets.json to fetch (default: cardiac)")
    parser.add_argument("--list", dest="keyword", help="only list the signed-off panels whose name contains KEYWORD")
    args = parser.parse_args()
    if args.keyword is not None:
        list_signed_off(args.keyword)
    else:
        write_gene_panel(args.panel)


if __name__ == "__main__":
    main()
