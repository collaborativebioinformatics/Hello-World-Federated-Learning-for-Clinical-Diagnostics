# Disease areas: how to add one, and what the second one looks like

The pipeline was built on heart genes. A disease area here is a named set of PanelApp panels, and everything after the gene list is the same code. This note says how to add an area, gives the numbers of the second one we built, inherited cancer, then of the build over every NHS signed-off panel, and lists what does not scale yet.

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

## All NHS signed-off panels

The third build takes every panel on PanelApp's signed-off list at once. The judges asked whether the pipeline scales past a hand-picked set of panels. This section says what it took and what came out, on 18 September 2026.

### How to build it

```
uv run python scripts/00_fetch_gene_panel.py --panel all   # writes config/all_gene_panel.txt and config/all_panel_versions.json
uv run python scripts/01_build_table.py --panel all        # writes data/all/variants.csv and data/all/columns.json
```

The set `all` in `config/panel_sets.json` has no hand-written panel list. It carries the flag `from_signed_off_list`, and step 0 reads PanelApp's signed-off list, three pages, and takes every panel on it at the version the list reports that day. The panel ids, versions and green counts go into the header of `config/all_gene_panel.txt` and into `config/all_panel_versions.json`, so the claim "every NHS signed-off panel as of 18 September 2026" names 296 panels with one version each and can be checked and rebuilt. Each panel answer is kept in `data/raw_panelapp/<id>_v<version>.json`. A panel version does not change once published, so a rerun reads from there, and a run that PanelApp cuts off with "too many requests" carries on from where it stopped.

Step 1 no longer asks myvariant.info one gene at a time. Genes missing from the cache are asked for in groups of up to 50 in one query, `clinvar.gene.symbol:(A OR B OR ...)` with the same filters and fields as before, and the answer is dealt out into the same per-gene files `data/raw/GENE.json` by the gene symbols ClinVar gives each record. A record that names two of the genes asked for goes into both files, as it would with two queries, and a gene that returns nothing gets an empty file so a rerun skips it. Everything after the download reads the per-gene files as before, and `--chunk 1` restores one query per gene. Before the build, the two paths were compared on nine heart genes already in the cache, TTN, MYH7, DMD, KCNQ1, LMNA, ACTA2, NKX2-5, APOA2 and GLA, 2,925 records in all: the same record ids, the same contents, in the same order, and the same as the files already on disk. Rebuilt from the cache with the changed script, the heart table and its `columns.json` have the same bytes as before.

Step 0 took 376 seconds for the 296 panels, at one call per second. Step 1 took 342 seconds. 4,062 of the 4,206 genes were not yet cached, the 144 heart and cancer genes were, and the 82 grouped queries downloaded 167,317 records in 298 seconds, the largest query returning 6,856 records in 14 seconds. Building the rows from the cache and writing the table took the remaining 44 seconds. A second run from the full cache took 10 seconds and wrote the same bytes.

That first build left 356 genes with nothing. The set `all` now also carries `resolve_symbols`, and with that flag step 1 asks a second time for every gene whose own symbol found no record. It asks HGNC for the gene's current symbol and its previous symbols, keeps HGNC's answer in `data/raw_hgnc/<GENE>.json`, and queries myvariant.info under all of those names at once, in ClinVar's gene field and in dbNSFP's: `(clinvar.gene.symbol:(A OR B) OR dbnsfp.genename:(A OR B))`, with the same filters as the first query. The answer goes to `data/raw_other_names/<GENE>.json`, apart from `data/raw/`, so the cardiac and cancer sets, which do not carry the flag, read the same cache files as before, and their tables were rebuilt from the cache with the same bytes to check that. The gene keeps the panel's symbol in the table, so the inheritance recorded in `config/all_gene_panel.txt` still applies to it. The second try for the 356 genes took 143 seconds in eight queries of 50 genes, most of it the HGNC lookups at five a second, and returned 6,504 records for 228 genes. A rerun from the caches took 12 seconds. The cache is 251 MiB in `data/raw/`, 8.5 MiB in `data/raw_other_names/` and 0.1 MiB in `data/raw_hgnc/`, and the table is 28.2 MiB.

### What came out

296 panels give 4,206 green genes: 1,161 listed only as one-copy, 2,246 only as two-copy, 537 with both patterns, 225 X-linked, 36 mitochondrial, and GNAS, whose only mode is `UNKNOWN`. A gene sits on 6 panels at the median, 615 genes sit on one panel only, and POLG sits on 26. The count was checked against an independent count of the same list made the day before with a throwaway script: the same 4,206 genes, the same split, and the same version for every panel.

| | heart, `cardiac` | inherited cancer, `cancer` | every signed-off panel, `all` |
|---|---|---|---|
| panels | 6 | 9 | 296 |
| green genes | 104 | 41 | 4,206 |
| genes with nothing downloaded | 7 | 0 | 128 |
| genes found only under another name | flag off | flag off | 228 |
| records downloaded | 16,747 | 12,911 | 203,384 |
| rows kept | 8,790 | 4,107 | 127,618 |
| pathogenic | 4,960 | 1,721 | 49,994 |
| benign | 3,830 | 2,386 | 77,624 |
| share pathogenic | 56.4% | 41.9% | 39.2% |
| dropped as uncertain or conflicting | 7,238 | 8,697 | 62,531 |
| score columns recommended | 13 of 17 | 16 of 17 | 16 of 17 |
| rows found in gnomAD | 3,680 | 1,288 | 75,554 |
| population-discordant rows | 622 | 267 | 16,982 |
| of those, pathogenic and benign | 2 and 620 | 2 and 265 | 136 and 16,846 |
| of those, highest in AFR, SAS, NFE | 371, 177, 74 | 161, 59, 47 | 10,252, 4,366, 2,364 |
| genes with at least 10 rows | 79 | 37 | 2,436 |
| genes with at least ten of each verdict | 30 | 17 | 547 |

A further 12,871 records were dropped for having no reviewed ClinVar record, 246 because another gene on the list had already brought the same variant, 111 because reviewed records disagreed and 7 for carrying no pathogenic or benign call. 89 genes returned records but kept no row, so 3,989 genes have rows. The table has the same 40 columns in the same order as the other two. The 16 recommended score columns are the cancer 16; `eve` is missing for 43.5% of rows, and `gerp91` at 28.4% is again close to the 30% line.

The ten biggest genes hold 8,446 rows, 6.6% of the table, so no gene dominates the way MYH7 and TTN do in the heart table:

| gene | rows | pathogenic | benign | inheritance | panels |
|---|---|---|---|---|---|
| FBN1 | 1,271 | 1,200 | 71 | BOTH; MONOALLELIC | 10 |
| DNAH11 | 1,072 | 19 | 1,053 | BIALLELIC | 3 |
| KMT2D | 1,066 | 75 | 991 | MONOALLELIC | 15 |
| SCN1A | 986 | 915 | 71 | MONOALLELIC | 15 |
| LDLR | 750 | 674 | 76 | MONOALLELIC | 6 |
| COL4A5 | 730 | 598 | 132 | X-LINKED-MONOALLELIC | 5 |
| ADGRV1 | 687 | 15 | 672 | BIALLELIC | 3 |
| OBSCN | 651 | 0 | 651 | BIALLELIC | 4 |
| ABCA4 | 640 | 614 | 26 | BIALLELIC | 1 |
| NEB | 593 | 5 | 588 | BIALLELIC; BOTH | 8 |

Labels still follow genes. FBN1, SCN1A and ABCA4 are almost all pathogenic; DNAH11, ADGRV1, OBSCN and NEB are almost all benign.

### Split by inheritance

Each gene's class is read from the inheritance words in `config/all_gene_panel.txt`. "One copy only" is a gene whose only word is `MONOALLELIC`, "two copies only" a gene whose only word is `BIALLELIC`, "both" a gene with `BOTH` or with both words, "X-linked" any gene with an X-linked word, and "mitochondrial" the genes on the mitochondrial genome. Where `UNKNOWN` sits next to another word, the other word decides.

| | one copy only | two copies only | both | X-linked | mitochondrial | GNAS |
|---|---|---|---|---|---|---|
| genes | 1,161 | 2,246 | 537 | 225 | 36 | 1 |
| genes with nothing downloaded | 50 | 47 | 3 | 5 | 23 | 0 |
| genes found only under another name | 52 | 142 | 24 | 10 | 0 | 0 |
| records downloaded | 68,005 | 66,932 | 52,729 | 14,761 | 816 | 141 |
| rows kept | 44,896 | 43,746 | 27,669 | 10,449 | 768 | 90 |
| pathogenic | 15,931 | 14,708 | 14,245 | 5,004 | 43 | 63 |
| benign | 28,965 | 29,038 | 13,424 | 5,445 | 725 | 27 |
| share pathogenic | 35.5% | 33.6% | 51.5% | 47.9% | 5.6% | 70.0% |
| rows found in gnomAD | 24,459 | 32,606 | 13,479 | 4,984 | 0 | 26 |
| population-discordant rows | 4,197 | 9,307 | 2,830 | 638 | 0 | 10 |
| of those, pathogenic | 2 | 96 | 32 | 6 | 0 | 0 |
| pathogenic rows at or over 0.1% in a site population | 3 | 117 | 37 | 6 | 0 | 0 |
| genes with at least ten of each verdict | 207 | 156 | 122 | 61 | 0 | 1 |

The split says where the fixed 0.1% line breaks. Of the 136 pathogenic rows that are common somewhere, 96 are in two-copy genes and 32 more in genes with both patterns; the one-copy genes contribute 2. Of the 163 pathogenic rows at or over 0.1% in one of the three site populations, 117 are in two-copy genes. The `pop_discordant` flag and the step 6 rule were left as they are, because changing them would alter the heart table; this table measures the cost instead. The 36 mitochondrial genes never appear in gnomAD's exome frequencies, so frequency evidence cannot help there, and 23 of them have no scored missense record: 22 are transfer RNA genes with no missense variants at all, and MT-CYB has 322 ClinVar records but no dbNSFP score in the index.

### The pathogenic variants that are common somewhere

The 20 pathogenic population-discordant rows with the highest frequency in any site population. All but one are in genes where a person needs two bad copies, or where PanelApp records both patterns, and healthy carriers are expected. The one-copy exception is `TTR V142I`, the amyloidosis variant carried by about 1.6% of African-ancestry samples; the only other one-copy pathogenic row that is common somewhere, `TSHZ3 S58G` at 0.14% in Europeans with one star, sits further down the list. The two HBB rows are new: HBB came in through the second try, because ClinVar's index files its variants under the neighbouring locus LOC106099062.

| name in the table | NFE | SAS | AFR | stars | ClinVar id | inheritance |
|---|---|---|---|---|---|---|
| `HFE C259Y` | 5.7410% | 0.2221% | 1.0704% | 2 | 9 | BIALLELIC |
| `HBB E7V` | 0.0053% | 0.0621% | 4.6998% | 2 | 15333 | BOTH; MONOALLELIC |
| `SERPINA1 E288V` | 3.6534% | 0.0000% | 0.7936% | 2 | 17969 | BIALLELIC |
| `WNT10A F228I` | 2.1253% | 0.1668% | 0.3721% | 2 | 4462 | BIALLELIC |
| `ABCA4 R899H` | 0.0070% | 0.0033% | 1.9754% | 2 | 99448 | BIALLELIC |
| `TTR V142I` | 0.0035% | 0.0065% | 1.5686% | 2 | 13426 | MONOALLELIC |
| `ABCA4 G753E` | 0.3482% | 1.3784% | 0.0492% | 2 | 7888 | BIALLELIC |
| `GNE V696M` | 0.0035% | 1.3392% | 0.0000% | 2 | 6028 | BIALLELIC; BOTH |
| `HBB E7K` | 0.0026% | 0.0000% | 1.3351% | 2 | 15126 | BOTH; MONOALLELIC |
| `GJB2 M34T` | 1.2417% | 0.0000% | 0.2399% | 3 | 17000 | BOTH; MONOALLELIC |
| `SLC4A1 E238V` | 0.0063% | 1.1782% | 0.0130% | 1 | 1343088 | BOTH; MONOALLELIC |
| `G6PD E317K` | 0.0061% | 1.1377% | 0.0000% | 2 | 10401 | X-LINKED-BIALLELIC; X-LINKED-MONOALLELIC |
| `ABCA4 G991R` | 0.0062% | 0.0098% | 0.7566% | 2 | 99182 | BIALLELIC |
| `ACADS W177R` | 0.0009% | 0.0000% | 0.6645% | 2 | 3828 | BIALLELIC |
| `SMN1 A2G` | 0.0000% | 0.0000% | 0.6593% | 1 | 9168 | BIALLELIC |
| `TNFRSF13B A181E` | 0.6501% | 0.0000% | 0.0800% | 2 | 5303 | BIALLELIC |
| `ACADM K293E` | 0.6297% | 0.0294% | 0.1417% | 2 | 3586 | BIALLELIC |
| `SERPINC1 T147A` | 0.0018% | 0.0033% | 0.5721% | 3 | 1170692 | BOTH |
| `G6PD L323P` | 0.0000% | 0.0000% | 0.5626% | 2 | 10388 | X-LINKED-BIALLELIC; X-LINKED-MONOALLELIC |
| `TNFRSF13B C104R` | 0.5441% | 0.0261% | 0.1661% | 2 | 5302 | BIALLELIC |

As with the two MUTYH variants in the cancer table, the position in the name is the first one dbNSFP lists, so `HFE C259Y` is the variant usually called C282Y, `SERPINA1 E288V` is the Z allele usually called E342K, `HBB E7V` is the sickle cell variant usually written E6V and `HBB E7K` is haemoglobin C, usually E6K. `variant_id` and `clinvar_id` are the safe keys.

### Steps 2 and 3 on this table, unofficial

The same trial as for cancer, rerun on the table with the second try: steps 2 and 3 were loaded unchanged from a throwaway script with `DATA_DIR` pointed at `data/all/`, and for step 2 with the reference file pointed away from `config/reference_build.json`. Both ran to the end. Step 2 took 3.9 seconds and every self-check passed. Step 3 took 36 seconds. The `DSP N1526K` example in step 2 prints here, because DSP is on the list, and step 3 still labels its ablation rows "12 scores" while using 15. The random draw of step 2 changed with the table, so the six unseen genes are not the same as in the first build.

**Step 2.**

| | verdicts | pathogenic | benign | population-discordant |
|---|---|---|---|---|
| `site_oslo` | 65,113 | 29,662 | 35,451 | 2,845 |
| `site_karachi` | 27,743 | 11,180 | 16,563 | 3,336 |
| `site_lagos` | 31,101 | 10,709 | 20,392 | 8,184 |

The test set is 26,023 rows: 25,347 random and 676 from the six unseen genes ARID1A, COL10A1, CYP11A1, MYH9, TPO and TUBA1A. 20,877 of the 101,595 training variants are held by more than one hospital. The per-population slices of the test set clear the 30 positives step 2 asks for: 229 pathogenic rows most common in AFR, 411 in SAS, 1,094 in NFE, and 8,466 in the 12,827 rows gnomAD never saw in any of the three.

**Step 3.** Logistic regression on 15 scores plus log frequency, 17 weights with the intercept. The largest weights of the pooled model are log frequency at -3.86, AlphaMissense at 3.15 and CADD at 2.38.

| trained on | AUC, public frequency | AUC, federated count query | AUC, unseen genes only, federated query |
|---|---|---|---|
| `site_oslo` | 0.9659 | 0.9700 | 0.9257 |
| `site_karachi` | 0.9660 | 0.9701 | 0.9306 |
| `site_lagos` | 0.9659 | 0.9700 | 0.9292 |
| pooled | 0.9661 | 0.9703 | 0.9278 |

The four rows are nearly the same, which is what a linear model with 17 weights trained on 27,743 or more rows would be expected to give. False alarms of the pooled model on the 3,396 population-discordant benign test rows, by the source of frequency evidence, next to the two smaller builds:

| frequency evidence | all, of 3,396 | cancer, of 100 | heart, of 183 |
|---|---|---|---|
| no frequency at all, scores only | 566 | 12 | 15 |
| public reference, Europeans only | 375 | 8 | 7 |
| own hospital, Oslo | 420 | 9 | 11 |
| federated count query | 110 | 3 | 2 |
| ceiling, real gnomAD frequencies | 110 | 3 | 2 |

Going from the public reference to the federated query removed 266 false alarms and introduced one, exact p = 2.3e-78. From Oslo's own patients to the federated query it removed 310 and introduced none, exact p = 9.6e-94. Sensitivity of the pooled model on the 10,200 pathogenic test rows was 0.948 with the public reference and 0.947 with the federated query. On every benign test row, for scale, the public reference gives 0.155 false alarms and the federated query 0.131; there the query removed 498 calls and introduced 115. The per-population AUC of the pooled model moves from 0.960 to 0.971 on the AFR slice and from 0.929 to 0.941 on the SAS slice, and goes from 0.950 to 0.948 on the NFE slice.

The same false-alarm table split by the inheritance class of the gene, computed in the throwaway script from the same pooled model and cut:

| frequency evidence | one copy only, of 834 | two copies only, of 1,872 | both, of 549 | X-linked, of 138 |
|---|---|---|---|---|
| no frequency at all, scores only | 115 | 324 | 106 | 21 |
| public reference, Europeans only | 73 | 225 | 61 | 16 |
| own hospital, Oslo | 87 | 246 | 68 | 19 |
| federated count query | 31 | 58 | 14 | 7 |
| ceiling, real gnomAD frequencies | 30 | 59 | 14 | 7 |

Public reference to federated query, by class: 43 removed and one introduced in one-copy genes, p = 5.1e-12; 167 and none in two-copy genes, p = 1.1e-50; 47 and none in genes with both patterns, p = 1.4e-14; 9 and none in X-linked genes, p = 0.0039. So the effect is there in every class and is largest, in count, in the two-copy genes, where it is also least safe: sensitivity on pathogenic test rows is 0.962 with the public reference and 0.961 with the federated query in one-copy genes, and 0.923 and 0.921 in two-copy genes. The test set holds 27 pathogenic population-discordant rows; the pooled model called 11 of them pathogenic with the public reference and the same 11 with the federated query, so no true call was lost in this draw. In the first build's draw, three were, among them `CRB1 P767T` and `SRD5A2 R246Q`, both in two-copy genes. The cancer test set held one such row, `MUTYH Y151C`, and it kept its call.

### Limits seen in this build

- **128 genes returned nothing, 3.0% of the list, after the second try under other names.** The first build had 356. myvariant.info knew 228 of them by another name. For 62 of those, ClinVar's index uses a symbol HGNC approved later than the panel's: AARS1 for AARS, MMUT for MUT, GSDME for DFNA5. For 136, ClinVar's index names a neighbouring locus and only dbNSFP names the gene itself: LOC106099062 for HBB, RPL36A-HNRNPH2 for GLA, FPGT-TNNI3K for TNNI3K. The other 30 have records of both kinds. The second try recovered 6,504 records and 4,164 rows for 202 genes: 1,801 pathogenic and 2,363 benign, 703 of them population-discordant, 9 of those pathogenic, the sickle cell variant among them. The biggest gains were FLG with 311 rows, COL4A3 with 294, GLA with 258 and MUT with 155. By inheritance the gained rows are 1,118 in one-copy genes, 1,771 in two-copy genes, 780 in genes with both patterns and 495 in X-linked genes. The `gene` column keeps the panel's symbol, so the rows read `MUT` and `AARS`. A name is skipped when it would fetch another gene's variants: HGNC's aliases are never used, because an alias can be another gene's approved symbol, as SMAD1 is an alias of GARS1; a previous symbol that is now another gene's approved symbol is dropped, VARS2 for VARS; and so is any symbol of another gene on the list, QARS for EPRS. A variant that two genes on the list both bring is kept once, under the gene that comes first: the second try runs after every gene's own answer, so ClinVar's own assignment wins over a match through dbNSFP's gene name, and 246 records were set aside that way. The heart set's seven genes with nothing would gain 314 rows the same way, 259 of them in GLA, but the cardiac set does not carry the flag, so its table is unchanged. Of the 128 genes still empty, 46 are not protein-coding genes: 22 mitochondrial transfer RNA genes, 10 small nuclear RNA, 5 long non-coding RNA, 3 microRNA, 2 small nucleolar RNA, 1 immunoglobulin, 1 T cell receptor and 1 other RNA gene, plus AL117258.1, which HGNC does not know. The 82 protein-coding genes were checked one by one with count queries. 65 have missense records with an AlphaMissense score but no pathogenic or benign verdict among them. 14 have ClinVar records with dbNSFP scores but no AlphaMissense score, NOTCH1 with 3,975 ClinVar records among them, with JAK1, DSPP, BSND, ALX4 and CSF2RA; AlphaMissense is the filter every query carries, so these are the cost of that choice. MT-CYB and DHRSX have ClinVar records with no dbNSFP score at all, and RYBP has no ClinVar record under any name.
- **The 0.1% line is wrong for half the list.** 2,246 of the 4,206 genes are two-copy only and 537 more carry both patterns, against 17 two-copy genes in the heart set. The numbers above show what that costs and where. A rule that reads the inheritance word, now recorded for every gene, is the obvious next step and has not been written.
- **The unseen-gene test shrinks to nothing.** Step 2 holds out six whole genes whatever the size of the list, so the "honest" AUC rests on 676 of 26,023 test rows, 2.6%, from six genes out of 3,989 with rows. For heart it rested on 777 of 2,373 rows.
- **Sizes.** The hospital files are 11.0 MiB for Oslo's verdicts and 6.7 MiB for its patient counts, and the run is 78 MiB in all. Nothing in steps 2 and 3 slowed down in a way that matters: 4 seconds and 36 seconds on a laptop.
- **Time.** PanelApp is the slow part, at one call per second, and a full rebuild from nothing is about 15 minutes for the two steps, the second try under other names included. A rebuild from the caches needs no network.
- **Every gene is scored the same way whatever its panel.** A variant in a gene that sits on 15 panels is one row, attributed to the gene, and the table does not record which disease area asked for it. The panel ids are in the gene list header and per gene line, so that can be recovered.

## What does not scale yet

Based on what these two builds showed, and the build over every signed-off panel above.

- **PanelApp limits how fast it can be asked.** It answered "too many requests" after about 18 calls within a few minutes while this was being built, and it does not say what its limit is. Step 0 now waits one second between calls, and when it is refused it waits and asks again. `--list` reads all 296 signed-off panels in three calls every time it runs.
- **Picking panels still needs a person who knows the disease area.** `--list` matches on the panel name only, and a third of the cancer set would have been missed by the obvious keyword. `--panel all` sidesteps the choice by taking every signed-off panel.
- **Download time was not a limit at this size.** The 40 cancer genes that were not yet cached, 12,911 records, downloaded in 23 seconds one gene at a time, and BRCA2 was the largest gene at 2,608 records. With the grouped query, the 4,062 uncached genes of the full list downloaded in 298 seconds.
- **Recessive genes and the fixed 0.1% rule.** The "too common to cause disease" rule in the step 6 query is one fixed line at 0.1%, whatever the gene, and `pop_discordant` in step 1 uses the same line to pick the rows where frequency should settle the call. For a recessive gene that line is too strict, because healthy carriers are expected. The cancer set has 3 `BIALLELIC` genes, MBD4, MUTYH and NTHL1, with 118 rows, and the two MUTYH variants above sit at 0.49% and 0.25% in Europeans. The heart set has 17 `BIALLELIC` genes. The inheritance is now recorded in the gene list and no rule reads it yet. The step 6 query was not run on the cancer data.
- **The recommended score columns differ between disease areas**, 13 for heart and 16 for cancer, because the 30% missing rule is applied per table. A model trained on one area cannot score the other until a shared list is agreed. The 13 heart columns are all among the cancer 16, so the heart list would work for both.
- **Genes without scores cannot be known in advance.** Seven of the 104 heart genes returned nothing from myvariant.info. All 41 cancer genes returned rows.
- **Label balance moves with the disease area.** The heart table is 56.4% pathogenic and the cancer table 41.9%. Within genes it is more uneven still: PTEN is 207 pathogenic to 4 benign, FH 156 to 6, BRCA2 90 to 402.
- **A smaller table makes every slice thinner.** Only 17 cancer genes have at least ten of each verdict, step 2 holds six of them out whole, and this time the draw included BRCA1, the largest gene. The test set came to 44.5% of the table. The African slice of the test set has 10 pathogenic rows and the South Asian slice 13, both under the 30 that step 2 asks for before it trusts an AUC.
- **Missense variants only.** In genes such as BRCA1, BRCA2 and APC, many disease-causing variants are of other kinds, for example ones that cut the protein short. Those are outside this table, so it covers only part of what a cancer genetics lab classifies.
- **More uncertain verdicts.** 8,697 of the 12,911 downloaded cancer records, 67.4%, were dropped because ClinVar's reviewed records were uncertain or in conflict.
- **Heart wording is hard-coded downstream.** The step 2 settings speak of heart patients and print the `DSP N1526K` example, step 3 prints "12 scores" in its ablation rows whatever the number of scores, and the step 6 rule text says "rare heart disease". None of it stops a run. All of it would mislead a reader of cancer output.
- **Variant names follow dbNSFP's first listed protein position.** Well-known cancer variants can appear under unfamiliar numbers, as the two MUTYH variants do. `variant_id` and `clinvar_id` are the safe keys.
