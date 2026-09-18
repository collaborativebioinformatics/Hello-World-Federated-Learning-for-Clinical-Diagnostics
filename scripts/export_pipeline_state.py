"""Export the state of the whole pipeline into one live page: docs/pipeline_live/index.html.

The page is a map of the system in the spirit of Nextflow's pipeline view. Every box shows
whether its step is built on this machine, every number on it comes from the state this
script computes from the repository, and three animations show what moves: data from the
public sources into the tables and the hospitals, a quarter ticking over with a federated
training job, and a doctor's question going out to the hospitals with the counts coming back.

    uv run python scripts/export_pipeline_state.py            # writes docs/pipeline_live/index.html
    uv run python scripts/export_pipeline_state.py --scan-all # also scan the large all-panels table

What it reads, all read-only:

    config/panel_sets.json, config/<area>_gene_panel.txt, config/all_panel_versions.json
    data/<area>/sites.json, columns.json, results_local.json, results_federated.json if present,
    the hospital files and the test set (presence only), variants.csv for the smaller areas
    data/other_types/build.json and sites.json, data/reference_genome/hg38.fa (presence and size)
    scripts/hospital_query.py, imported, for the three demo stories at every hospital
    docs/quarterly_replay.json when it exists; otherwise a clearly labelled placeholder is embedded

The template is docs/pipeline_live/template.html. The state goes in as one JSON block, so the
page needs no network and no server: it runs from file://, from GitHub Pages and from raw.githack.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "config"
DOCS_DIR = ROOT / "docs"
OUT_DIR = DOCS_DIR / "pipeline_live"
TEMPLATE = OUT_DIR / "template.html"
OUT = OUT_DIR / "index.html"
REPLAY_FILE = DOCS_DIR / "quarterly_replay.json"
GENOME = DATA_DIR / "reference_genome" / "hg38.fa"
STEP4 = ROOT / "scripts" / "04_federated_train.py"

DEFAULT_AREA = "cardiac"
OTHER_TYPES = "other_types"
# The all-panels table is 30 MB and is rebuilt often. Its row count comes from sites.json and its
# gene count from the panel file; the table itself is only scanned with --scan-all.
LIGHT_AREAS = {"all"}
AREA_LABELS = {"cardiac": "heart", "cancer": "inherited cancer", "all": "all NHS panels"}
SITE_NAMES = {"site_oslo": "Oslo", "site_karachi": "Karachi", "site_lagos": "Lagos"}
POPULATION_WORDS = {"nfe": "European", "sas": "South Asian", "afr": "African"}
TWO_COPY_MODES = {"BIALLELIC", "X-LINKED-BIALLELIC"}
MIN_COUNT = 5
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]

# The three stories, all heart variants. The MYBPC3 deletion is asked for by its ClinVar and gnomAD id,
# the leftmost spelling, which is the one that Karachi's lab does not use: that is the spelling demo.
STORIES = [
    {"key": "dsp", "label": "DSP N1526K", "ask": "DSP N1526K",
     "blurb": "looks ultra-rare in the European database; 15% of healthy patients at Lagos carry it"},
    {"key": "ttr", "label": "TTR V142I", "ask": "TTR V142I",
     "blurb": "common at Lagos, yet piles up among the sick, so it stays flagged"},
    {"key": "mybpc3", "label": "MYBPC3 25-letter deletion", "ask": "chr11:g.47332275_47332299del",
     "blurb": "3.2% at Karachi, written two ways; found only with the spelling fix"},
]

# results_local.json keys -> the five evidence settings the page draws
EVIDENCE = {"scores_only": "none", "public": "public", "own_hospital": "own", "federated_query": "federated", "ceiling": "ceiling"}
# The rows the table box shows, as in docs/pipeline_flowchart_built.html: the area's demo variants
EXAMPLE_ROWS = {"cardiac": ["MYH7 R403Q", "DSP N1526K", "TTR V142I", "TNNT2 R92W"], "cancer": ["POLD1 S173N", "PMS2 T511M", "MUTYH G63D", "RNF43 R657P"]}
# The three of the model's numbers the picture shows: AlphaMissense, CADD, log frequency
SHOWN_WEIGHTS = ["alphamissense", "cadd", "log_frequency"]


def _num(value):
    """A float for JSON, or None for a missing score."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def area_folder(area: str) -> Path:
    return DATA_DIR if area == DEFAULT_AREA else DATA_DIR / area


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def file_stamp(path: Path) -> str | None:
    return dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if path.exists() else None


# ---------------------------------------------------------------------------
# Step 0: the gene lists
# ---------------------------------------------------------------------------
def read_panel_file(area: str) -> dict:
    """Genes, inheritance split and the pinned panels from config/<area>_gene_panel.txt."""
    path = CONFIG_DIR / f"{area}_gene_panel.txt"
    if not path.exists():
        return {"built": False, "file": path.relative_to(ROOT).as_posix()}
    genes, two_copy, panels, written = 0, 0, [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            found = re.match(r"^#\s+(\d+)\s+v([\d.]+)\s+(\d+) green genes\s+(.*)$", line)
            if found:
                panels.append({"id": int(found.group(1)), "version": found.group(2), "green_genes": int(found.group(3)), "name": found.group(4).strip()})
            date = re.search(r"on (\d{4}-\d{2}-\d{2})", line)
            if date and written is None:
                written = date.group(1)
            continue
        gene, _, comment = line.partition("#")
        if not gene.strip():
            continue
        genes += 1
        _, _, inheritance = comment.partition("|")
        modes = [m.strip() for m in inheritance.split(";") if m.strip()]
        if modes and all(m in TWO_COPY_MODES for m in modes):
            two_copy += 1
    return {"built": True, "file": path.relative_to(ROOT).as_posix(), "genes": genes, "two_copy_genes": two_copy,
            "one_copy_genes": genes - two_copy, "panels": panels, "written": written}


def read_panel_versions() -> dict | None:
    versions = read_json(CONFIG_DIR / "all_panel_versions.json")
    if not versions:
        return None
    return {"date": versions.get("date"), "panel_count": versions.get("panel_count"),
            "unique_green_genes": versions.get("unique_green_genes"), "source": versions.get("source")}


# ---------------------------------------------------------------------------
# Steps 1 and 2: the tables and the hospitals
# ---------------------------------------------------------------------------
def read_table(area: str, scan: bool) -> dict:
    folder = area_folder(area)
    table, columns = folder / "variants.csv", folder / "columns.json"
    out = {"built": table.exists() and columns.exists(), "file": table.relative_to(ROOT).as_posix(),
           "rows": None, "scores": None, "pathogenic": None, "benign": None, "genes_with_rows": None, "discordant": None,
           "scanned": False, "written": file_stamp(table)}
    if not out["built"]:
        return out
    cols = read_json(columns)
    out["scores"] = len(cols.get("features", []))
    sites = read_json(folder / "sites.json")
    if sites:
        out["rows"] = sites.get("build", {}).get("variant_rows")
    if scan:
        import pandas as pd

        frame = pd.read_csv(table, usecols=["name", "gene", "label", "pop_discordant", "cadd", "alphamissense", "af_nfe", "af_sas", "af_afr"])
        out.update(rows=int(len(frame)), pathogenic=int(frame.label.sum()), benign=int((frame.label == 0).sum()),
                   genes_with_rows=int(frame.gene.nunique()), discordant=int(frame.pop_discordant.sum()), scanned=True)
        # A few real rows for the picture, the way docs/pipeline_flowchart_built.html shows them: the area's demo variants
        wanted = EXAMPLE_ROWS.get(area, [])
        shown = frame[frame.name.isin(wanted)].drop_duplicates("name").set_index("name")
        out["example_rows"] = [
            {"name": name, "cadd": _num(shown.loc[name, "cadd"]), "alphamissense": _num(shown.loc[name, "alphamissense"]),
             "af_nfe": _num(shown.loc[name, "af_nfe"]), "af_sas": _num(shown.loc[name, "af_sas"]), "af_afr": _num(shown.loc[name, "af_afr"]),
             "label": int(shown.loc[name, "label"])}
            for name in wanted if name in shown.index
        ]
    elif out["rows"] is None:
        with table.open("rb") as handle:
            out["rows"] = sum(1 for _ in handle) - 1
    return out


def read_hospitals(area: str, scan: bool) -> dict:
    folder = area_folder(area)
    sites = read_json(folder / "sites.json")
    if not sites:
        return {"built": False, "sites": {}, "test": {"built": False}}
    hospitals = {}
    for site, info in sites["sites"].items():
        files = [folder / site / "verdicts.csv", folder / site / "patient_counts.csv"]
        hospitals[site] = {
            "name": SITE_NAMES.get(site, site), "population": info["population"],
            "population_word": POPULATION_WORDS.get(info["population"], info["population"]),
            "patients": info["patients"], "verdicts": info["verdicts"], "pathogenic": info["pathogenic"], "benign": info["benign"],
            "built": all(f.exists() for f in files),
        }
    test = folder / "test" / "variants.csv"
    kinds = None
    if scan and test.exists():
        import pandas as pd

        kinds = {k: int(v) for k, v in pd.read_csv(test, usecols=["test_kind"]).test_kind.value_counts().items()}
    return {"built": bool(hospitals) and all(h["built"] for h in hospitals.values()), "sites": hospitals,
            "seed": sites.get("seed"), "site_overlap": sites.get("site_overlap"),
            "test": {"built": test.exists(), "rows": sites.get("test_rows"), "unseen_genes": sites.get("unseen_genes", []), "kinds": kinds},
            "written": file_stamp(folder / "sites.json")}


def read_other_types(area: str) -> dict | None:
    folder = area_folder(area) / OTHER_TYPES
    build, sites = read_json(folder / "build.json"), read_json(folder / "sites.json")
    if not (folder / "variants.csv").exists() or not build or not sites:
        return None
    return {"built": True, "rows": build.get("kept"), "harmful": sum(v.get("pathogenic", 0) for v in build.get("rows_by_type_and_verdict", {}).values()),
            "harmless": sum(v.get("benign", 0) for v in build.get("rows_by_type_and_verdict", {}).values()),
            "contested": sites.get("contested_rows"), "insertions_and_deletions": build.get("insertions_and_deletions"),
            "with_more_than_one_spelling": build.get("with_more_than_one_spelling"),
            "lab_spelling": sites.get("lab_spelling", {}), "sites": sites.get("sites", {}),
            "reference_letter_check": build.get("reference_letter_check", {}), "written": file_stamp(folder / "variants.csv")}


# ---------------------------------------------------------------------------
# Steps 3 and 4: the models
# ---------------------------------------------------------------------------
def read_results(area: str) -> dict:
    folder = area_folder(area)
    path = folder / "results_local.json"
    saved = read_json(path)
    if not saved:
        return {"built": False, "file": path.relative_to(ROOT).as_posix()}
    features = list(saved.get("features", []))
    pooled = saved.get("training", {}).get("pooled", {})
    evidence = saved.get("evidence", {})
    counts = saved.get("false_alarm_counts", {})
    false_alarms = {EVIDENCE[k]: counts[k] for k in EVIDENCE if k in counts}

    def pooled_row(block: str) -> dict | None:
        rows = evidence.get(block, {}).get("pooled")
        if not rows:
            return None
        return {EVIDENCE.get(k, k): v for k, v in rows.items()}

    auc_unseen = next((v for k, v in evidence.items() if k.startswith("AUC, unseen")), {})
    return {
        "built": True, "file": path.relative_to(ROOT).as_posix(), "trained_on": file_stamp(path),
        "scores": len(features) - 1, "weights": len(features) + 1, "features": features,
        "pooled": {"rows": pooled.get("rows"), "threshold": pooled.get("threshold"),
                   "log_frequency_weight": pooled.get("weights", {}).get("log_frequency"),
                   "shown": [pooled.get("weights", {}).get(w) for w in SHOWN_WEIGHTS]},
        "shown_weights": SHOWN_WEIGHTS,
        "per_site": {site: {"rows": t.get("rows"), "threshold": t.get("threshold"), "log_frequency_weight": t.get("weights", {}).get("log_frequency"),
                            "shown": [t.get("weights", {}).get(w) for w in SHOWN_WEIGHTS]}
                     for site, t in saved.get("training", {}).items() if site != "pooled"},
        "auc_all": pooled_row("AUC, all test rows"),
        "auc_unseen": {EVIDENCE.get(k, k): v for k, v in auc_unseen.get("pooled", {}).items()} or None,
        "sensitivity": pooled_row("sensitivity"),
        "false_alarms": false_alarms,
        "discordant_benign_rows": saved.get("discordant_benign_rows"),
        "test_rows": saved.get("test_rows"), "pathogenic_test_rows": saved.get("pathogenic_test_rows"),
        "paired": saved.get("paired", {}).get("discordant benign|public"),
        "ablation_auc": {EVIDENCE.get(k, k): v.get("auc") for k, v in saved.get("ablation", {}).items()},
    }


def read_federated(area: str) -> dict:
    path = area_folder(area) / "results_federated.json"
    saved = read_json(path)
    if not saved:
        return {"present": False, "file": path.relative_to(ROOT).as_posix()}
    runs = saved.get("per_run", {}).get("federated", [])
    usable = bool(runs) and "weights" in runs[0] and "threshold" in runs[0]
    return {"present": True, "file": path.relative_to(ROOT).as_posix(), "runs": len(runs), "usable": usable, "written": file_stamp(path)}


def read_sources() -> dict:
    """The names, versions and sites of the data sources, as the repository records them.

    ClinVar, dbNSFP and gnomAD versions come from the still picture docs/pipeline_flowchart_built.html, with
    docs/build_notes.md as the fallback; NVFlare's version from uv.lock; the URLs from the scripts that call them.
    """
    def grab(path: Path, pattern: str) -> str | None:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        found = re.search(pattern, text)
        return found.group(1) if found else None

    picture, notes, lock = DOCS_DIR / "pipeline_flowchart_built.html", DOCS_DIR / "build_notes.md", ROOT / "uv.lock"
    clinvar = grab(picture, r"ClinVar</p><p class=\"sub\"[^>]*>expert verdicts · ([\d-]+)") or grab(notes, r"ClinVar[^\n]*?(20\d\d-\d\d)")
    dbnsfp = grab(picture, r"dbNSFP</p><p class=\"sub\"[^>]*>prediction scores · ([\w.]+)") or grab(notes, r"dbNSFP[^\n]*?version ([\w.]+)")
    gnomad = grab(picture, r"gnomAD</p><p class=\"sub\"[^>]*>frequency per population · (v[\d.]+)") or grab(notes, r"gnomAD[^\n]*?exomes (v[\d.]+)")
    nvflare = grab(lock, r"name = \"nvflare\"\nversion = \"([\d.]+)\"")
    panel_url = grab(ROOT / "scripts" / "00_fetch_gene_panel.py", r"\"(https://panelapp\.genomicsengland\.co\.uk)")
    ucsc_url = grab(ROOT / "scripts" / "00_fetch_reference_genome.py", r"BASE_URL = \"(https://[^\"]+)\"")
    ensembl_url = grab(ROOT / "scripts" / "variant_spelling.py", r"ENSEMBL = \"(https://[^\"]+)\"")
    build = grab(ROOT / "scripts" / "01_build_table.py", r"GENOME_BUILD = \"(\w+)\"")
    versions = read_panel_versions() or {}
    return {
        "panelapp": {"name": "Genomics England PanelApp", "url": panel_url or "https://panelapp.genomicsengland.co.uk",
                     "version": f"{versions.get('panel_count')} signed-off panels · {versions.get('date')}" if versions.get("panel_count") else None},
        "myvariant": {"name": "myvariant.info", "url": "https://myvariant.info", "version": f"{build} index" if build else None},
        "clinvar": {"name": "ClinVar", "url": "https://www.ncbi.nlm.nih.gov/clinvar/", "version": clinvar},
        "dbnsfp": {"name": "dbNSFP", "url": "https://dbnsfp.org", "version": dbnsfp},
        "gnomad": {"name": "gnomAD", "url": "https://gnomad.broadinstitute.org", "version": f"exomes {gnomad}" if gnomad else None},
        "ucsc": {"name": "UCSC", "url": ucsc_url or "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/", "version": "hg38" if GENOME.exists() else None},
        "ensembl": {"name": "Ensembl", "url": ensembl_url or "https://rest.ensembl.org", "version": "REST, per-gene cache"},
        "nvflare": {"name": "NVIDIA FLARE", "url": "https://nvidia.github.io/NVFlare/", "version": nvflare},
    }


def read_step4_settings() -> dict:
    """ROUNDS, LOCAL_EPOCHS and RUNS as scripts/04_federated_train.py declares them. Read as text: nvflare is not imported."""
    out = {"rounds": None, "local_steps": None, "runs": None, "file": STEP4.relative_to(ROOT).as_posix()}
    if not STEP4.exists():
        return out
    text = STEP4.read_text(encoding="utf-8")
    for key, name in [("rounds", "ROUNDS"), ("local_steps", "LOCAL_EPOCHS"), ("runs", "RUNS")]:
        found = re.search(rf"^{name}\s*=\s*(\d+)", text, re.M)
        if found:
            out[key] = int(found.group(1))
    return out


def model_version(results: dict, federated: dict) -> dict:
    """What the query uses: the federated model when step 4 saved a usable file, else step 3's pooled model."""
    if not results.get("built"):
        return {"built": False, "source": None, "label": "no model yet"}
    source = "federated" if federated.get("usable") else "pooled"
    trained = results["trained_on"][:10]
    return {"built": True, "source": source, "trained_on": results["trained_on"], "label": f"v{trained} {source}",
            "threshold": results["pooled"]["threshold"], "weights": results["weights"], "scores": results["scores"]}


# ---------------------------------------------------------------------------
# Step 6: the three stories, through scripts/hospital_query.py
# ---------------------------------------------------------------------------
def load_query_module():
    sys.path.insert(0, str(ROOT / "scripts"))
    import hospital_query  # noqa: PLC0415

    return hospital_query


def count_dict(count) -> dict:
    return {"carriers": count.carriers, "total": count.total, "hidden_below": count.hidden_below,
            "hidden": count.hidden, "frequency": count.frequency, "text": count.text()}


def result_dict(result) -> dict:
    raw = asdict(result)
    return {
        "name": result.name, "variant_id": result.variant_id, "gene": result.gene, "mutation_type": result.mutation_type,
        "patient_at": result.patient_at, "asked_as": result.asked_as, "spelling_fix": result.spelling_fix,
        "public_frequency": result.public_frequency, "public_text": f"{result.public_frequency:.2%}",
        "answers": [{"site": a.site, "healthy": count_dict(a.healthy), "sick": count_dict(a.sick), "verdict": a.verdict, "stars": a.stars}
                    for a in result.answers],
        "before": raw["before"], "after": raw["after"],
        "best_frequency": result.best_frequency, "best_site": result.best_site,
        "model": result.model, "model_probability": result.model_probability, "model_threshold": result.model_threshold,
        "model_call": result.model_call, "model_source": result.model_source,
        "too_common": result.too_common, "too_common_text": f"{result.too_common:.1%}" if result.too_common < 0.01 else f"{result.too_common:.0%}",
        "line_reason": result.line_reason, "presumption": result.presumption,
    }


def run_stories(hq) -> tuple[list[dict], str | None]:
    hq.set_area(DEFAULT_AREA)
    try:
        sites = hq.list_sites()
    except FileNotFoundError as missing:
        return [], str(missing).splitlines()[0]
    stories = []
    for story in STORIES:
        entry = {**story, "area": DEFAULT_AREA, "per_site": {}, "no_fix": None, "error": None}
        try:
            for site in sites:
                entry["per_site"][site] = result_dict(hq.query(story["ask"], patient_at=site, min_count=MIN_COUNT))
            if story["key"] == "mybpc3":
                entry["no_fix"] = {site: result_dict(hq.query(story["ask"], patient_at=site, min_count=MIN_COUNT, spelling_fix=False)) for site in sites}
        except (LookupError, FileNotFoundError) as problem:
            entry["error"] = str(problem).splitlines()[0]
        stories.append(entry)
    return stories, None


def changed_calls(hq, area: str) -> dict | None:
    """For a patient at each hospital: how many variant names change call once the other hospitals answer."""
    try:
        hq.set_area(area)
        sites = hq.list_sites()
        out = {}
        for site in sites:
            table = hq.overview(site, MIN_COUNT)
            out[site] = {"changed": int(table.changed.sum()), "names": int(len(table)),
                         "cleared": int((table.changed & (table.after == "LIKELY HARMLESS")).sum()),
                         "flagged": int((table.changed & (table.after == "KEEP FLAGGED")).sum())}
        return out
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# The quarterly cycle: the replay another script writes, or a labelled placeholder
# ---------------------------------------------------------------------------
def placeholder_replay(area: dict) -> dict | None:
    hospitals, results = area["hospitals"], area["results"]
    if not hospitals.get("built"):
        return None
    quarters = []
    for i, name in enumerate(QUARTERS):
        share = (i + 1) / len(QUARTERS)
        last = i == len(QUARTERS) - 1
        per_site = {}
        for site, h in hospitals["sites"].items():
            pathogenic = round(h["pathogenic"] * share)
            verdicts = round(h["verdicts"] * share)
            per_site[site] = {"patients": round(h["patients"] * share), "verdicts": verdicts, "pathogenic": pathogenic, "benign": verdicts - pathogenic}
        model = None
        if last and results.get("built"):
            model = {"auc": results["auc_unseen"], "false_alarms": results["false_alarms"], "discordant_rows": results["discordant_benign_rows"],
                     "sensitivity": results["sensitivity"], "frequency_weight": {"pooled": results["pooled"]["log_frequency_weight"]}}
        query = {"changed_calls_per_site": {s: v["changed"] for s, v in area["query"].items()}} if last and area.get("query") else None
        quarters.append({"quarter": name, "share": share, "per_site": per_site, "model": model, "query": query, "measured": last})
    return {"placeholder": True, "sites": list(hospitals["sites"]), "quarters": quarters,
            "note": ("Placeholder written by scripts/export_pipeline_state.py because docs/quarterly_replay.json was not found. "
                     "Q4 is the build on this machine. Q1 to Q3 scale the hospital sizes by a quarter each and carry no model numbers. "
                     "Rerun the exporter once the replay exists and it is embedded instead.")}


def note_text(note) -> str | None:
    """The replay's note as one string. The replay writes it as a dict of short paragraphs."""
    if note is None:
        return None
    if isinstance(note, dict):
        return " ".join(v.strip() for v in note.values() if isinstance(v, str) and v.strip())
    return str(note)


def load_replay(areas: dict) -> dict:
    saved = read_json(REPLAY_FILE)
    out = {"file": REPLAY_FILE.relative_to(ROOT).as_posix(), "present": bool(saved), "note": None, "areas": {}}
    per_area = {}
    if saved:
        out["note"] = note_text(saved.get("note"))
        per_area = saved["areas"] if isinstance(saved.get("areas"), dict) else {DEFAULT_AREA: saved}
    for area, info in areas.items():
        found = per_area.get(area)
        if found and found.get("quarters"):
            quarters = [{**q, "measured": True} for q in found["quarters"]]
            sites = found.get("sites")
            out["areas"][area] = {"placeholder": False, "sites": list(sites) if sites else list(info["hospitals"].get("sites", {})),
                                  "quarters": quarters, "note": note_text(found.get("note")) or out["note"]}
        else:
            replay = placeholder_replay(info)
            if replay:
                out["areas"][area] = replay
    return out


# ---------------------------------------------------------------------------
# Putting the state together
# ---------------------------------------------------------------------------
def collect(scan_all: bool) -> dict:
    panel_sets = read_json(CONFIG_DIR / "panel_sets.json") or {}
    hq = load_query_module()
    step4 = read_step4_settings()
    areas = {}
    for area, spec in panel_sets.items():
        scan = area not in LIGHT_AREAS or scan_all
        info = {
            "title": spec.get("title", area), "label": AREA_LABELS.get(area, area), "light": not scan,
            "panel_list": read_panel_file(area), "table": read_table(area, scan), "hospitals": read_hospitals(area, scan),
            "other_types": read_other_types(area), "results": read_results(area), "federated": read_federated(area),
        }
        info["model"] = model_version(info["results"], info["federated"])
        info["query"] = changed_calls(hq, area) if scan and info["hospitals"].get("built") else None
        areas[area] = info
    stories, story_problem = run_stories(hq)
    hq.set_area(DEFAULT_AREA)
    genome = {"built": GENOME.exists(), "file": GENOME.relative_to(ROOT).as_posix(),
              "size_gb": round(GENOME.stat().st_size / 1e9, 2) if GENOME.exists() else None,
              "fetched": file_stamp(GENOME)}
    cache = DATA_DIR / "raw"
    state = {
        "built_on": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "exporter": "scripts/export_pipeline_state.py",
        "rebuild": "uv run python scripts/export_pipeline_state.py",
        "repo": "https://github.com/collaborativebioinformatics/Confidently-Flirting-With-Federated-Learning-for-Clinical-Diagnostics",
        "demo_url": "https://collaborativebioinformatics.github.io/Confidently-Flirting-With-Federated-Learning-for-Clinical-Diagnostics/demo/",
        "default_area": DEFAULT_AREA,
        "areas": areas,
        "panelapp": read_panel_versions(),
        "myvariant": {"built": cache.exists(), "genes_cached": len(list(cache.glob("*.json"))) if cache.exists() else 0,
                      "other_names_cached": len(list((DATA_DIR / "raw_other_names").glob("*.json"))) if (DATA_DIR / "raw_other_names").exists() else 0,
                      "sources": [{"name": "ClinVar", "gives": "expert verdicts"}, {"name": "dbNSFP", "gives": "prediction scores"},
                                  {"name": "gnomAD", "gives": "frequency per population"}]},
        "genome": genome,
        "sources": read_sources(),
        "federated_settings": step4,
        "rules": {"min_count": MIN_COUNT, "too_common": hq.TOO_COMMON, "too_common_two_copies": hq.TOO_COMMON_TWO_COPIES, "pile_up_fold": hq.PILE_UP_FOLD},
        "stories": stories, "story_problem": story_problem,
    }
    state["replay"] = load_replay(areas)
    return state


def render(state: dict) -> str:
    template = TEMPLATE.read_text(encoding="utf-8")
    payload = json.dumps(state, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    for marker in ("__STATE_JSON__", "__BUILT__"):
        if marker not in template:
            raise SystemExit(f"{TEMPLATE.relative_to(ROOT).as_posix()} has no {marker} marker")
    return template.replace("__STATE_JSON__", payload).replace("__BUILT__", state["built_on"])


def describe(state: dict) -> None:
    print(f"Embedded state, built {state['built_on']}")
    if state["panelapp"]:
        p = state["panelapp"]
        print(f"  PanelApp signed-off list: {p['panel_count']} panels, {p['unique_green_genes']:,} genes, {p['date']}")
    g = state["genome"]
    print(f"  reference genome: {'built, ' + str(g['size_gb']) + ' GB' if g['built'] else 'not built'}")
    print(f"  myvariant.info cache: {state['myvariant']['genes_cached']:,} gene files")
    print("  sources: " + ", ".join(f"{s['name']} {s['version'] or '(no version found)'}" for s in state["sources"].values()))
    for area, a in state["areas"].items():
        t, h, r = a["table"], a["hospitals"], a["results"]
        rows = f"{t['rows']:,}" if t.get("rows") else "no"
        genes = a["panel_list"].get("genes")
        print(f"  {area}: {genes} genes on the list, {rows} table rows{' (light read)' if a['light'] else ''}, "
              f"hospitals {'built' if h['built'] else 'not built'}, results {'built' if r['built'] else 'not built'}")
        if h["built"]:
            for site, s in h["sites"].items():
                print(f"      {site}: {s['population_word']}, {s['patients']:,} patients, {s['verdicts']:,} verdicts")
        if r["built"]:
            fa = r["false_alarms"]
            print(f"      false alarms of {r['discordant_benign_rows']}: " + ", ".join(f"{k} {v}" for k, v in fa.items())
                  + f"; model {a['model']['label']}, cut-off {a['model']['threshold']:.0%}")
        if a.get("other_types"):
            o = a["other_types"]
            print(f"      other mutation types: {o['rows']:,} rows, {o['insertions_and_deletions']:,} insertions and deletions, "
                  f"{o['with_more_than_one_spelling']:,} with more than one spelling")
        if a.get("query"):
            print("      calls changed once the other hospitals answer: " + ", ".join(f"{s} {v['changed']:,} of {v['names']:,}" for s, v in a["query"].items()))
    for story in state["stories"]:
        if story["error"]:
            print(f"  story {story['label']}: {story['error']}")
            continue
        r = story["per_site"]["site_oslo"]
        counts = ", ".join(f"{SITE_NAMES[a['site']]} {a['healthy']['text']}" for a in r["answers"])
        print(f"  story {story['label']}: patient at Oslo, {r['before']['call']} -> {r['after']['call']}; healthy carriers {counts}")
        if story["no_fix"]:
            n = story["no_fix"]["site_oslo"]
            print(f"      without the spelling fix: {n['after']['call']}, " + ", ".join(f"{SITE_NAMES[a['site']]} {a['healthy']['text']}" for a in n["answers"]))
    if state["story_problem"]:
        print(f"  stories: {state['story_problem']}")
    rp = state["replay"]
    for area, r in rp["areas"].items():
        kind = "placeholder" if r["placeholder"] else f"from {rp['file']}"
        print(f"  quarterly replay for {area}: {kind}, {len(r['quarters'])} quarters")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scan-all", action="store_true", help="also scan the large all-panels table for gene and verdict counts")
    args = parser.parse_args()
    state = collect(args.scan_all)
    html = render(state)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8", newline="\n")
    describe(state)
    size = OUT.stat().st_size
    print(f"Wrote {OUT.relative_to(ROOT).as_posix()}: {size / 1024:.0f} KB")
    if size > 1_000_000:
        print("WARNING: the page is over 1 MB")


if __name__ == "__main__":
    main()
