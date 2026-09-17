# Disease areas: how to add one, and what the second one looks like

The pipeline was built on heart genes. A disease area here is a named set of PanelApp panels, and everything after the gene list is the same code. This note says how to add an area, gives the numbers of the second one we built, inherited cancer, and lists what does not scale yet.

## How to add a disease area

1. **Find the panels.** Search the panels signed off by the NHS Genomic Medicine Service by a word in their name:

   ```
   uv run python scripts/00_fetch_gene_panel.py --list cancer
   ```

   It prints each panel's id, its signed-off version, its number of genes and its name. The gene count covers every rating and step 0 keeps only the green genes, so expect fewer. The search looks at the panel name only, so try more than one word. `--list cancer` finds 14 panels and misses three of the nine we used: the Lynch syndrome, familial melanoma and endocrine neoplasia panels have no "cancer" in their names.

2. **Add a set to `config/panel_sets.json`.** Give it a short name, a title for people, and each panel's id with the version to pin. Keep the version in quotes, because `"1.30"` and `1.3` are different versions. The `note` is for readers and the code ignores it.

   ```json
   "cancer": {
     "title": "Inherited cancer",
     "panels": [
       {"id": 635, "version": "3.2", "note": "Inherited breast cancer and ovarian cancer"}
     ]
   }
   ```

3. **Run steps 0 and 1 with the set's name.**

   ```
   uv run python scripts/00_fetch_gene_panel.py --panel cancer   # writes config/cancer_gene_panel.txt
   uv run python scripts/01_build_table.py --panel cancer        # writes data/cancer/variants.csv and columns.json
   ```

With no `--panel`, both steps build the heart set, named `cardiac`, into the same paths as before. The heart table was rebuilt from the cache with the changed scripts and came out with the same bytes as the team's reference.

## What the gene list now records

Each gene line in `config/<panel>_gene_panel.txt` reads `GENE  # panels that list it | how the gene is inherited`:

```
MUTYH     # 504 | BIALLELIC
BRCA1     # 635, 143, 524, 1223 | BOTH; MONOALLELIC
```

The inheritance word is PanelApp's `mode_of_inheritance`, shortened. `MONOALLELIC` means one bad copy of the gene causes disease. `BIALLELIC` means both copies must be bad, so healthy carriers exist. The header of each file explains every word. A gene that PanelApp lists with more than one mode keeps them all, separated by `; `.

Step 1 ignores everything after `#`, so it reads the same gene list as before: the heart list is the same 104 genes in the same order. No script reads the inheritance yet. It is recorded for the step that decides whether a variant is too common to cause disease, which needs to know whether one bad copy or two are required.

## The inherited cancer build

Nine panels, pinned on 17 September 2026. Each pinned version is the one the signed-off list reported that day: 635 v3.2, 143 v5.2, 504 v4.2, 503 v1.15, 524 v3.4, 1223 v1.7, 521 v1.30, 522 v2.15 and 648 v3.8. Together they give 41 green genes.

| | heart, `cardiac` | inherited cancer, `cancer` |
|---|---|---|
| panels | 6 | 9 |
| green genes | 104 | 41 |
| genes with nothing downloaded | 7 | 0 |
| records downloaded | 16,747 | 12,911 |
| rows kept | 8,790 | 4,107 |
| pathogenic | 4,960 | 1,721 |
| benign | 3,830 | 2,386 |
| share pathogenic | 56.4% | 41.9% |
| dropped as uncertain or conflicting | 7,238 | 8,697 |
| score columns recommended | 13 of 17 | 16 of 17 |
| rows found in gnomAD | 3,680 | 1,288 |
| population-discordant rows | 622 | 267 |
| of those, pathogenic and benign | 2 and 620 | 2 and 265 |
| of those, highest in AFR, SAS, NFE | 371, 177, 74 | 161, 59, 47 |
| genes with at least 10 rows | 79 | 37 |
| of those, genes with 90% or more of their rows under one verdict | 29 | 13 |
| genes with at least ten of each verdict | 30 | 17 |

The cancer table has the same 40 columns in the same order, the same rounding and the same line endings as the heart table. The only score column with too many gaps is `eve`, missing for 34.2% of rows. `esm1b`, `mutationassessor` and `gerp91` are recommended here and were too sparse in the heart table. There, `esm1b` and `mutationassessor` have no value at all for whole genes such as FBN1, APOB and COL1A1. `gerp91` is close to the line in both tables: 31.6% missing for heart and 28.1% for cancer, so a data update could move it either way.

The ten biggest genes:

| gene | rows | pathogenic | benign | inheritance |
|---|---|---|---|---|
| BRCA1 | 564 | 192 | 372 | BOTH; MONOALLELIC |
| BRCA2 | 492 | 90 | 402 | MONOALLELIC |
| MSH2 | 407 | 134 | 273 | MONOALLELIC |
| MLH1 | 222 | 176 | 46 | MONOALLELIC |
| PTEN | 211 | 207 | 4 | MONOALLELIC |
| MSH6 | 209 | 58 | 151 | MONOALLELIC |
| ATM | 199 | 89 | 110 | MONOALLELIC |
| FH | 162 | 156 | 6 | MONOALLELIC |
| MEN1 | 150 | 130 | 20 | MONOALLELIC |
| RET | 136 | 66 | 70 | MONOALLELIC |

Labels follow genes here too. PTEN, BRCA1 and MLH1 supply 575 of the 1,721 pathogenic rows, 33.4%. BRCA2, BRCA1, MSH2, MSH6 and ATM supply 1,308 of the 2,386 benign rows, 54.8%.

### The two pathogenic variants that are common somewhere

Both are in MUTYH, which is `BIALLELIC`: a person needs two bad copies to develop polyposis, so carriers of one copy are healthy and the variant can be fairly common. Both are most common in Europeans. They are also the only two pathogenic rows in the table at or over 0.1% in any of the three site populations.

| name in the table | usual name | NFE | SAS | AFR | stars | ClinVar id |
|---|---|---|---|---|---|---|
| `MUTYH G63D` | G396D | 0.4917% | 0.0098% | 0.0821% | 2 | 5294 |
| `MUTYH Y151C` | Y179C | 0.2541% | 0.0033% | 0.0493% | 2 | 5293 |

The table names a variant by the first protein position dbNSFP lists for it, which is why these two well-known variants appear under unfamiliar numbers. For each one, the usual position is among the positions dbNSFP gives.

### Candidate demo variants

Benign, population-discordant, and under 0.1% in Europeans, so a hospital that only sees the European reference cannot clear them. 208 of the 265 benign discordant rows are of this kind. The five with the biggest gap between the highest and lowest site frequency:

| name | NFE | SAS | AFR | stars |
|---|---|---|---|---|
| `POLD1 S173N` | 0.0247% | 0.0065% | 10.2628% | 2 |
| `PMS2 T511M` | 0.0238% | 0.0947% | 8.1278% | 3 |
| `MLH1 H718Y` | 0.0203% | 0.0098% | 7.8821% | 3 |
| `MSH2 N127S` | 0.0344% | 0.0131% | 7.8154% | 3 |
| `BAP1 S596G` | 0.0475% | 0.0065% | 7.4013% | 2 |

The biggest gap of all belongs to `ATM D126E`, at 20.1698% in AFR. It sits at 0.1259% in Europeans, just over the 0.1% line, so it makes a weaker demo. All five above are most common in African-ancestry samples. 59 benign discordant rows are most common in South Asians, and the best of those for a demo is `RNF43 R657P`: 0.0088% NFE, 2.1689% SAS, 0.0125% AFR, two stars.

## A first look downstream, unofficial

Steps 2 and 3 belong to the ML side and were left untouched. As a trial they were loaded from a throwaway script with `DATA_DIR` pointed at `data/cancer/`, and for step 2 with the reference file pointed away from `config/reference_build.json`. Both ran to the end with no other change. The `DSP N1526K` example in step 2 prints nothing on this table, because the function returns quietly when the variant is absent. Step 4 was not tried.

**Step 2.** Every self-check passed.

| | verdicts | pathogenic | benign | population-discordant |
|---|---|---|---|---|
| `site_oslo` | 1,560 | 647 | 913 | 43 |
| `site_karachi` | 667 | 250 | 417 | 38 |
| `site_lagos` | 647 | 231 | 416 | 98 |

The test set is 1,829 rows: 582 random and 1,247 from the six unseen genes ATM, BRCA1, MLH1, RET, SMAD4 and VHL. That is 44.5% of the table, against 27.0% in the heart build, and it leaves 2,278 training variants, 550 of them held by more than one hospital.

**Step 3.** Logistic regression on 15 scores plus log frequency, 17 weights with the intercept. The heart model has 14.

| trained on | AUC, public frequency | AUC, federated count query | AUC, unseen genes only, federated query |
|---|---|---|---|
| `site_oslo` | 0.9747 | 0.9770 | 0.9743 |
| `site_karachi` | 0.9753 | 0.9778 | 0.9750 |
| `site_lagos` | 0.9734 | 0.9757 | 0.9717 |
| pooled | 0.9754 | 0.9778 | 0.9748 |

False alarms of the pooled model on the 100 population-discordant benign test rows, by the source of frequency evidence, next to the heart build's 183 rows:

| frequency evidence | cancer, of 100 | heart, of 183 |
|---|---|---|
| no frequency at all, scores only | 12 | 15 |
| public reference, Europeans only | 8 | 7 |
| own hospital, Oslo | 9 | 11 |
| federated count query | 3 | 2 |
| ceiling, real gnomAD frequencies | 3 | 2 |

Going from the public reference to the federated query removed 5 false alarms and introduced none, exact p = 0.0625. From Oslo's own patients to the federated query it removed 6 and introduced none, exact p = 0.0312. The direction matches the heart result and the counts are small, so this is a first look and should be read that way. Sensitivity of the pooled model on the 851 pathogenic test rows was 0.912 with the public reference and 0.911 with the federated query.

The test set holds 101 discordant rows and one of them is pathogenic: `MUTYH Y151C`. The pooled model still called it pathogenic under every source of frequency evidence, with a probability of 0.545 against a cut of 0.411 under the federated query.

## What does not scale yet

Based on what these two builds showed.

- **PanelApp limits how fast it can be asked.** It answered "too many requests" after about 18 calls within a few minutes while this was being built, and it does not say what its limit is. Step 0 now waits one second between calls, and when it is refused it waits and asks again. `--list` reads all 296 signed-off panels in three calls every time it runs.
- **Picking panels still needs a person who knows the disease area.** `--list` matches on the panel name only, and a third of the cancer set would have been missed by the obvious keyword.
- **Download time was not a limit at this size, and larger sets are untested.** The 40 cancer genes that were not yet cached, 12,911 records, downloaded in 23 seconds, and BRCA2 was the largest gene at 2,608 records. Genes are fetched one after another. A set with many hundreds of genes has not been tried.
- **Recessive genes and the fixed 0.1% rule.** The "too common to cause disease" rule in the step 6 query is one fixed line at 0.1%, whatever the gene, and `pop_discordant` in step 1 uses the same line to pick the rows where frequency should settle the call. For a recessive gene that line is too strict, because healthy carriers are expected. The cancer set has 3 `BIALLELIC` genes, MBD4, MUTYH and NTHL1, with 118 rows, and the two MUTYH variants above sit at 0.49% and 0.25% in Europeans. The heart set has 17 `BIALLELIC` genes. The inheritance is now recorded in the gene list and no rule reads it yet. The step 6 query was not run on the cancer data.
- **The recommended score columns differ between disease areas**, 13 for heart and 16 for cancer, because the 30% missing rule is applied per table. A model trained on one area cannot score the other until a shared list is agreed. The 13 heart columns are all among the cancer 16, so the heart list would work for both.
- **Genes without scores cannot be known in advance.** Seven of the 104 heart genes returned nothing from myvariant.info. All 41 cancer genes returned rows.
- **Label balance moves with the disease area.** The heart table is 56.4% pathogenic and the cancer table 41.9%. Within genes it is more uneven still: PTEN is 207 pathogenic to 4 benign, FH 156 to 6, BRCA2 90 to 402.
- **A smaller table makes every slice thinner.** Only 17 cancer genes have at least ten of each verdict, step 2 holds six of them out whole, and this time the draw included BRCA1, the largest gene. The test set came to 44.5% of the table. The African slice of the test set has 10 pathogenic rows and the South Asian slice 13, both under the 30 that step 2 asks for before it trusts an AUC.
- **Missense variants only.** In genes such as BRCA1, BRCA2 and APC, many disease-causing variants are of other kinds, for example ones that cut the protein short. Those are outside this table, so it covers only part of what a cancer genetics lab classifies.
- **More uncertain verdicts.** 8,697 of the 12,911 downloaded cancer records, 67.4%, were dropped because ClinVar's reviewed records were uncertain or in conflict.
- **Heart wording is hard-coded downstream.** The step 2 settings speak of heart patients and print the `DSP N1526K` example, step 3 prints "12 scores" in its ablation rows whatever the number of scores, and the step 6 rule text says "rare heart disease". None of it stops a run. All of it would mislead a reader of cancer output.
- **Variant names follow dbNSFP's first listed protein position.** Well-known cancer variants can appear under unfamiliar numbers, as the two MUTYH variants do. `variant_id` and `clinvar_id` are the safe keys.
