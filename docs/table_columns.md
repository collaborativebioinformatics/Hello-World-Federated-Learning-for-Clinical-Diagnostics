# What is in `data/variants.csv`

One row is one DNA variant: a single-letter change that swaps one amino acid in a protein, in one of 104 heart-related genes. The gene list is the diagnostic-grade genes from six NHS-signed-off Genomics England PanelApp panels, see `config/cardiac_gene_panel.txt`. Build the file with `uv run python scripts/01_build_table.py`. It opens in Excel.

Columns run left to right from what a person wants to read to what the code needs. Rows are sorted by gene, then name.

## Who is it, and what did the experts say

| column | meaning |
|---|---|
| `name` | Gene plus the amino-acid swap. `MYH7 R403Q` means: in gene MYH7, position 403, R became Q. |
| `gene` | The gene symbol. |
| `verdict` | ClinVar's expert verdict in words: `pathogenic` (causes disease) or `benign` (harmless). |
| `label` | The same verdict as a number, for the model: 1 = pathogenic, 0 = benign. **This is the answer column.** |
| `stars` | How well reviewed the verdict is. 1 = one lab with stated criteria, 2 = several labs agree, 3 = expert panel, 4 = practice guideline. |

## How common is it

Frequencies come from gnomAD, a count of variants in about 125,000 mostly healthy people. `0.012` means 1.2% of gene copies carry it. A variant gnomAD never saw is recorded as 0.

| column | meaning |
|---|---|
| `af_nfe` | Frequency in Northern Europeans. This is what **site_oslo** sees. |
| `af_sas` | Frequency in South Asians. This is what **site_karachi** sees. |
| `af_afr` | Frequency in African-ancestry people. This is what **site_lagos** sees. |
| `pop_discordant` | 1 if the variant is reasonably common (at least 0.1%) at one site and at least ten times rarer at another. These are the variants where knowing the patient's population changes the call. |

## The computer guesses (the model's inputs)

Each is one published tool's opinion of how damaging the variant is, rescaled to run from 0 to 1. **Higher always means more damaging.** Because every column has the same scale and direction, hospitals never need to share scaling information.

| column | what the tool looks at |
|---|---|
| `alphamissense` | DeepMind's model: does this amino-acid swap break the protein's structure or function? |
| `cadd` | Blends dozens of annotations into one "how harmful" score. |
| `dann` | A neural-network variant of CADD. |
| `primateai` | Learned from variants that are common in other primates, which are presumably harmless. |
| `mpc` | Is this stretch of the gene unusually free of amino-acid changes in healthy people? |
| `sift`, `sift4g` | Do related proteins in other species tolerate this amino acid at this position? |
| `provean` | Same idea as SIFT, scored differently. |
| `phylop100`, `phastcons100` | Has this DNA letter stayed the same across 100 vertebrate species? |
| `siphy29` | The same question across 29 mammals. |
| `bstatistic` | Is the surrounding region under evolutionary pressure to stay unchanged? |
| `polyphen2_hdiv` | Combines protein structure and conservation to judge the amino-acid swap. |
| `esm1b`, `eve`, `mutationassessor`, `gerp91` | Four more tools. **Not recommended as inputs:** more than 30% of rows have no value. |

`data/columns.json` lists the thirteen recommended score columns so later steps don't hard-code them.

Tools that were trained on ClinVar or a similar disease database are deliberately absent (REVEL, ClinPred, BayesDel and others). They have effectively seen the answer column, so including them would make every comparison look the same.

## Extra detail and bookkeeping

| column | meaning |
|---|---|
| `af_global` | Frequency across all of gnomAD. |
| `af_eas`, `af_amr`, `af_fin`, `af_asj` | East Asian, Latino, Finnish, Ashkenazi Jewish. Not used as sites. |
| `in_gnomad` | 1 if gnomAD has seen the variant at all. |
| `ac_nfe`, `an_nfe`, and the same for `sas`, `afr` | Allele count and allele number: copies of the variant seen, out of gene copies looked at. Frequency is `ac / an`. Used later to simulate realistic hospital counts. |
| `variant_id` | The exact DNA change, e.g. `chr14:g.23429278C>T`, on genome build hg38. **The true unique key.** |
| `clinvar_id` | ClinVar's id, to look the variant up at `ncbi.nlm.nih.gov/clinvar/variation/<id>`. |

## Things that look odd but are right

- **About 100 rows share a name with another row.** Different DNA letter changes can produce the same amino-acid swap. `variant_id` tells them apart.
- **Most frequencies are 0.** Disease-causing variants are usually too rare to show up even once in gnomAD.
- **Position numbers can differ from papers.** Names use the numbering in dbNSFP. The well-known `TTR V122I` appears here as `TTR V142I`.
- **Labels follow genes.** DMD, APOB and TTN rows are almost all benign; FBN1, LDLR and MYH7 almost all pathogenic. Keep that in mind when reading any accuracy number.
- **Missing scores are blank cells**, not zeros.
