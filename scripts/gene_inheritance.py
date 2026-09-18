"""How many bad copies a gene needs, read from the panel file.

PanelApp records a mode of inheritance per gene, and step 0 keeps it in the
comment of each gene's line:

    MUTYH     # 504 | BIALLELIC
    BRCA1     # 635, 143, 524, 1223 | BOTH; MONOALLELIC

This matters to the model because the frequency rule is not one rule:

    one copy   a common variant is almost certainly harmless -> strong veto
    two copies carriers of one bad copy are HEALTHY, so a harmful variant can be
               common in the population -> the veto is wrong here

`HFE C259Y` is pathogenic and carried by 5.7% of Europeans; `SERPINA1 E288V` by
3.7%. Both are two-copy genes. Of the 127 pathogenic variants in the all-panels
table that are common somewhere, 92 are in two-copy genes and only 2 in one-copy
genes. A single frequency weight has to compromise between two opposite rules.

This is public metadata: every hospital has the same PanelApp file, so a feature
read from it costs nothing in privacy and needs no federation.

Usage:
    from gene_inheritance import inheritance_class, needs_two_copies
    klass = inheritance_class("all")          # {"MUTYH": "two_copies", ...}
    flag = needs_two_copies(table.gene, "all")  # 1.0 / 0.0 per row
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ONE_COPY, TWO_COPIES, BOTH, X_LINKED, UNKNOWN = (
    "one_copy", "two_copies", "both", "x_linked", "unknown",
)


@cache
def inheritance_class(panel: str = "cardiac") -> dict[str, str]:
    """Gene -> one of the five classes above, from config/<panel>_gene_panel.txt."""
    path = ROOT / "config" / f"{panel}_gene_panel.txt"
    classes: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        gene = line.split()[0]
        words = line.split("|", 1)[1].upper() if "|" in line else ""
        # X-linked first: PanelApp spells it X-LINKED-BIALLELIC and the like, so
        # testing for BIALLELIC before this would swallow every X-linked gene.
        if "X-LINKED" in words or "X LINKED" in words:
            classes[gene] = X_LINKED
        elif "MITOCHONDRIAL" in words:
            classes[gene] = UNKNOWN  # never in gnomAD's exome frequencies anyway
        elif "BOTH" in words or ("MONOALLELIC" in words and "BIALLELIC" in words):
            classes[gene] = BOTH
        elif "BIALLELIC" in words:
            classes[gene] = TWO_COPIES
        elif "MONOALLELIC" in words:
            classes[gene] = ONE_COPY
        else:
            classes[gene] = UNKNOWN
    return classes


def needs_two_copies(genes: pd.Series, panel: str = "cardiac") -> np.ndarray:
    """1.0 where one bad copy is NOT enough, so a harmful variant may be common.

    `both` counts as 1.0: if either pattern applies, the veto cannot be trusted.
    An unknown gene gets 0.0, the conservative value, because it leaves the
    strong veto in place rather than switching it off on a guess.
    """
    classes = inheritance_class(panel)
    return np.isin(genes.map(classes).to_numpy(), [TWO_COPIES, BOTH, X_LINKED]).astype(float)


def demo() -> None:
    """The claim this module exists for, checked against the panel files."""
    for panel in ("cardiac", "all"):
        if not (ROOT / "config" / f"{panel}_gene_panel.txt").exists():
            continue
        classes = inheritance_class(panel)
        counts = pd.Series(list(classes.values())).value_counts()
        print(f"{panel}: {len(classes)} genes  " + "  ".join(f"{k} {v}" for k, v in counts.items()))

    known = inheritance_class("all")
    # Spot checks: the textbook common-and-harmful variants all sit in genes
    # where one bad copy is not enough, which is exactly why they can be common.
    for gene, expected in [("MUTYH", TWO_COPIES), ("HFE", TWO_COPIES), ("SERPINA1", TWO_COPIES)]:
        assert known.get(gene) == expected, f"{gene} is {known.get(gene)}, expected {expected}"
    assert known.get("G6PD") == X_LINKED, f"G6PD is {known.get('G6PD')}"
    # Across 296 panels most well-known genes carry more than one mode, and that
    # is biology rather than a data error: one bad BRCA2 copy gives cancer risk,
    # two give Fanconi anaemia. So the flag is taken from the file, never guessed.
    one_copy_gene = next(g for g, k in known.items() if k == ONE_COPY)
    frame = pd.DataFrame({"gene": ["MUTYH", one_copy_gene, "HFE", "BRCA2"]})
    flags = needs_two_copies(frame.gene, "all")
    assert list(flags) == [1.0, 0.0, 1.0, 1.0], dict(zip(frame.gene, flags))
    # An unseen gene must not silently become "veto off".
    assert needs_two_copies(pd.Series(["NOT_A_GENE"]), "all")[0] == 0.0
    print(f"self-check passed: the common-and-harmful genes are two-copy or X-linked, "
          f"and {one_copy_gene} is one-copy.")


if __name__ == "__main__":
    demo()
