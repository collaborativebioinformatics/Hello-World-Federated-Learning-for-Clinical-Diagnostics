"""Step 0: fetch the gene list from Genomics England PanelApp.

PanelApp holds the gene panels NHS labs use to decide which genes to read for a
patient. Every gene on a panel is rated, and "green" means the evidence is strong
enough to use the gene for diagnosis. We take the green genes from the
heart-related panels signed off by the NHS Genomic Medicine Service.

Writes config/cardiac_gene_panel.txt. Open API, no login.

Usage:
    uv run python scripts/00_fetch_gene_panel.py

Cite: Martin AR et al. PanelApp crowdsources expert knowledge to establish
consensus diagnostic gene panels. Nature Genetics 51, 1560-1565 (2019).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
GENE_PANEL = ROOT / "config" / "cardiac_gene_panel.txt"

API_URL = "https://panelapp.genomicsengland.co.uk/api/v1/panels/{panel_id}/"
GREEN = "3"  # PanelApp's code for diagnostic-grade evidence

# Panel id -> the version we pinned, so the list can be cited and rebuilt exactly.
# To update, change a version number here and rerun.
PANELS = {
    49: "6.4",  # Hypertrophic cardiomyopathy
    652: "4.11",  # Dilated and arrhythmogenic cardiomyopathy
    842: "14.33",  # Cardiac arrhythmias (long QT, Brugada, CPVT, conduction disease)
    700: "5.8",  # Thoracic aortic aneurysm or dissection
    772: "2.7",  # Familial hypercholesterolaemia
    502: "1.31",  # Hereditary systemic amyloidosis (this is where TTR comes from)
}


def fetch_green_genes(panel_id: int, version: str) -> tuple[str, list[str]]:
    response = requests.get(API_URL.format(panel_id=panel_id), params={"version": version}, timeout=60)
    response.raise_for_status()
    panel = response.json()
    # A set, because a panel can list one gene twice (once per inheritance pattern).
    green = {g["gene_data"]["gene_symbol"] for g in panel["genes"] if str(g["confidence_level"]) == GREEN}
    return panel["name"], sorted(green)


def main() -> None:
    panels_of_gene: dict[str, list[int]] = {}
    summary_lines = []
    for panel_id, version in PANELS.items():
        name, genes = fetch_green_genes(panel_id, version)
        for gene in genes:
            panels_of_gene.setdefault(gene, []).append(panel_id)
        summary_lines.append(f"#   {panel_id:>4}  v{version:<6} {len(genes):>3} green genes  {name}")
        print(summary_lines[-1].lstrip("# "))

    header = [
        "# Cardiac gene panel: green (diagnostic-grade) genes from Genomics England PanelApp,",
        "# taken from the heart-related panels signed off by the NHS Genomic Medicine Service.",
        f"# Written by scripts/00_fetch_gene_panel.py on {date.today()}. Rerunning overwrites this file.",
        "#",
        *summary_lines,
        "#",
        f"# {len(panels_of_gene)} genes. The numbers after each gene are the panels that list it.",
        "# Cite: Martin AR et al., Nature Genetics 51, 1560-1565 (2019). https://panelapp.genomicsengland.co.uk",
        "",
    ]
    gene_lines = [
        f"{gene:<10}# {', '.join(map(str, panel_ids))}" for gene, panel_ids in sorted(panels_of_gene.items())
    ]
    GENE_PANEL.write_text("\n".join(header + gene_lines) + "\n", encoding="utf-8")
    print(f"\nwrote {GENE_PANEL}  ({len(panels_of_gene)} genes)")


if __name__ == "__main__":
    main()
