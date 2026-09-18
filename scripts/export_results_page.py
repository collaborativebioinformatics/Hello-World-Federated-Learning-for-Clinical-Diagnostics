"""Export the results page: docs/results/index.html.

One page for clinicians and judges: what was found, to read before trying the patient query. Every
number on it is read from a file in the repository when this script runs, never typed here:

    data/results_local.json, data/cancer/results_local.json, data/all/results_local.json
        false-alarm counts, the paired comparison public -> hospitals' counts, sensitivity,
        the number of population-discordant benign test rows
    data/<area>/sites.json            variant rows, the hospitals' sizes and populations
    config/panel_sets.json            the areas, in order, and their pinned panels
    config/all_panel_versions.json    the signed-off list: how many panels, how many genes
    config/<area>_gene_panel.txt      genes per area
    docs/quarterly_replay.json        the heart area quarter by quarter
    docs/disease_areas_table.md       the twelve specialties, a Markdown table, parsed
    docs/step4_results.md, docs/cancer_results.md
                                      the NVIDIA FLARE summary row, mean (sd) over the runs
    scripts/04_federated_train.py     RUNS, how many federated runs an official area had
    README.md                         the sentences reused word for word and the four AUC numbers
                                      of "What we found"

The three pictures are referenced as files, not embedded: ../disease_areas_figure.png,
../manuscript_figure3.png and ../manuscript_figure4.png.

    uv run python scripts/export_results_page.py   # writes docs/results/index.html, prints what went in
"""

from __future__ import annotations

import datetime as dt
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "config"
DOCS_DIR = ROOT / "docs"
OUT_DIR = DOCS_DIR / "results"
OUT = OUT_DIR / "index.html"
README = ROOT / "README.md"
REPLAY_FILE = DOCS_DIR / "quarterly_replay.json"
SPECIALTY_TABLE = DOCS_DIR / "disease_areas_table.md"
STEP4 = ROOT / "scripts" / "04_federated_train.py"

REPO = "https://github.com/collaborativebioinformatics/REFLECT-Respectfully-Exchanging-Federated-Learning-Evidence-across-Clinics-Together"
BLOB = REPO + "/blob/main/docs/"

DEFAULT_AREA = "cardiac"
SITE_NAMES = {"site_oslo": "Oslo", "site_karachi": "Karachi", "site_lagos": "Lagos"}
POPULATION_WORDS = {"nfe": "European", "sas": "South Asian", "afr": "African"}
# Column header, and the phrase used in running text, per area
AREA_WORDS = {"cardiac": ("Heart", "in the heart area"), "cancer": ("Inherited cancer", "in inherited cancer"), "all": ("Every panel", "across every panel")}
# The five sources of frequency evidence, top to bottom, in the README's words, and how each bar is drawn
EVIDENCE_ROWS = [
    ("scores_only", "Nothing", "grey"),
    ("public", "The public database, Europeans only", "grey"),
    ("own_hospital", "The patient's own hospital, Oslo", "blue"),
    ("federated_query", "Counts from all three hospitals", "green"),
    ("ceiling", "The true frequency in every population, which no hospital has", "outline"),
]
# The pages the NVIDIA FLARE summary row is read from, per official area
NVFLARE_PAGES = {"cardiac": DOCS_DIR / "step4_results.md", "cancer": DOCS_DIR / "cancer_results.md"}
FULL_DOCS = ["step3_results.md", "step4_results.md", "cancer_results.md", "disease_areas.md", "disease_areas_table.md", "quarterly_replay.md"]

embedded: list[tuple[str, str, str]] = []  # (what, value, source), printed at the end


def note(what: str, value, source: str):
    """Record a number that goes on the page, and hand it back."""
    embedded.append((what, str(value), source))
    return value


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_json(path: Path):
    if not path.exists():
        raise SystemExit(f"missing {rel(path)}")
    return json.loads(path.read_text(encoding="utf-8"))


def esc(text) -> str:
    return html.escape(str(text), quote=True)


def n(value) -> str:
    """An integer with thousands separators."""
    return f"{int(round(float(value))):,}"


def fix(value, places: int) -> str:
    return f"{float(value):.{places}f}"


def p_text(p: float) -> str:
    """An exact p as the README writes it: 0.0625, or 2.3 × 10^−78 for the very small ones."""
    if p >= 0.0001:
        return f"{p:.4g}"
    mantissa, exponent = f"{p:.1e}".split("e")
    return f"{mantissa} × 10<sup>{int(exponent)}</sup>".replace("-", "−")


def area_folder(area: str) -> Path:
    return DATA_DIR if area == DEFAULT_AREA else DATA_DIR / area


# ---------------------------------------------------------------------------
# The sources
# ---------------------------------------------------------------------------
def read_gene_count(area: str) -> int:
    """Gene lines in config/<area>_gene_panel.txt, checked against the '# N genes' line when there is one."""
    path = CONFIG_DIR / f"{area}_gene_panel.txt"
    text = path.read_text(encoding="utf-8")
    genes = sum(1 for line in text.splitlines() if line.strip() and not line.startswith("#"))
    stated = re.search(r"^#\s*([\d,]+) genes\.", text, re.M)
    if stated and int(stated.group(1).replace(",", "")) != genes:
        print(f"WARNING: {rel(path)} says {stated.group(1)} genes, {genes} gene lines counted")
    return note(f"{area}: genes", genes, rel(path))


def read_areas() -> dict:
    panel_sets = read_json(CONFIG_DIR / "panel_sets.json")
    versions = read_json(CONFIG_DIR / "all_panel_versions.json")
    areas = {}
    for area, spec in panel_sets.items():
        folder = area_folder(area)
        results = read_json(folder / "results_local.json")
        sites = read_json(folder / "sites.json")
        src = rel(folder / "results_local.json")
        if spec.get("from_signed_off_list"):
            panels = note(f"{area}: panels", versions["panel_count"], rel(CONFIG_DIR / "all_panel_versions.json"))
            genes = note(f"{area}: genes", versions["unique_green_genes"], rel(CONFIG_DIR / "all_panel_versions.json"))
            listed = read_gene_count(area)
            if listed != genes:
                print(f"WARNING: {area}: {genes} genes on the signed-off list, {listed} in the panel file")
        else:
            panels = note(f"{area}: panels", len(spec["panels"]), rel(CONFIG_DIR / "panel_sets.json"))
            genes = read_gene_count(area)
        counts = results["false_alarm_counts"]
        areas[area] = {
            "title": spec.get("title", area),
            "header": AREA_WORDS[area][0], "phrase": AREA_WORDS[area][1],
            "panels": panels, "genes": genes,
            "variants": note(f"{area}: variant rows", sites["build"]["variant_rows"], rel(folder / "sites.json")),
            "hospitals": {site: {"name": SITE_NAMES.get(site, site), "population": POPULATION_WORDS.get(info["population"], info["population"]),
                                 "patients": note(f"{area}: {site} patients", info["patients"], rel(folder / "sites.json"))}
                          for site, info in sites["sites"].items()},
            "discordant": note(f"{area}: discordant benign test rows", results["discordant_benign_rows"], src),
            "false_alarms": {key: note(f"{area}: false alarms, {key}", counts[key], src) for key, _, _ in EVIDENCE_ROWS},
            "paired": {k: note(f"{area}: public -> hospitals' counts, {k}", v, src) for k, v in results["paired"]["discordant benign|public"].items()},
            "sensitivity": {k: note(f"{area}: sensitivity, pooled model, {k}", round(v, 3), src)
                            for k, v in results["evidence"]["sensitivity"]["pooled"].items() if k in ("public", "federated_query")},
        }
    return areas


def read_runs() -> int:
    found = re.search(r"^RUNS\s*=\s*(\d+)", STEP4.read_text(encoding="utf-8"), re.M)
    if not found:
        raise SystemExit(f"no RUNS line in {rel(STEP4)}")
    return note("federated runs per official area", int(found.group(1)), rel(STEP4))


def read_nvflare_row(area: str, discordant: int) -> dict:
    """The 'Federated, NVFlare FedAvg' row of the summary table: mean (sd) over the runs."""
    path = NVFLARE_PAGES[area]
    text = path.read_text(encoding="utf-8")
    header = re.search(r"^\|\s*\|\s*AUC, public frequency\s*\|\s*AUC, federated count query\s*\|\s*false alarms / (\d+), public\s*\|\s*false alarms / (\d+), federated counts\s*\|", text, re.M)
    row = re.search(r"^\|\s*\*\*Federated, NVFlare FedAvg\*\*\s*\|(.+)$", text, re.M)
    if not header or not row:
        raise SystemExit(f"no NVFlare summary table in {rel(path)}")
    if int(header.group(2)) != discordant:
        print(f"WARNING: {rel(path)} counts false alarms of {header.group(2)}, results_local.json has {discordant} discordant rows")
    cells = [c.strip().strip("*").strip() for c in row.group(1).strip().strip("|").split("|")]
    pairs = []
    for cell in cells:
        found = re.match(r"([\d.]+)\s*\(([\d.]+)\)", cell)
        if not found:
            raise SystemExit(f"cannot read '{cell}' in {rel(path)}")
        pairs.append((float(found.group(1)), float(found.group(2))))
    fa_mean, fa_sd = pairs[3]
    return {"file": rel(path), "false_alarms_mean": note(f"{area}: NVFlare federated model, false alarms mean", fa_mean, rel(path)),
            "false_alarms_sd": note(f"{area}: NVFlare federated model, false alarms sd", fa_sd, rel(path))}


def read_specialties() -> list[dict]:
    """docs/disease_areas_table.md, one dict per specialty."""
    lines = [l for l in SPECIALTY_TABLE.read_text(encoding="utf-8").splitlines() if l.startswith("|")]
    header = [c.strip() for c in lines[0].strip().strip("|").split("|")]
    rows = []
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        row = dict(zip(header, cells))
        rows.append(row)
    wanted = ["Disease area", "Genes", "Discordant benign", "False alarms, Oslo alone", "False alarms, federated", "False alarms, pooled",
              "AUC, Oslo alone", "AUC, federated", "AUC, pooled"]
    missing = [w for w in wanted if w not in header]
    if missing:
        raise SystemExit(f"{rel(SPECIALTY_TABLE)} has no column(s) {missing}")
    out = []
    for row in rows:
        entry = {"specialty": row["Disease area"], "genes": int(row["Genes"].replace(",", "")),
                 "discordant": int(row["Discordant benign"].replace(",", ""))}
        for key, col in [("fa_oslo", "False alarms, Oslo alone"), ("fa_fed", "False alarms, federated"), ("fa_pooled", "False alarms, pooled")]:
            entry[key] = float(row[col])
        for key, col in [("auc_oslo", "AUC, Oslo alone"), ("auc_fed", "AUC, federated"), ("auc_pooled", "AUC, pooled")]:
            entry[key] = float(row[col].split("(")[0])  # "0.9634 (0.0005)": the mean; the sd is in the source table
        for key in ("genes", "discordant", "fa_oslo", "fa_fed", "fa_pooled", "auc_oslo", "auc_fed", "auc_pooled"):
            note(f"specialty {entry['specialty']}: {key}", entry[key], rel(SPECIALTY_TABLE))
        out.append(entry)
    return out


def read_replay() -> dict:
    saved = read_json(REPLAY_FILE)
    area = saved["areas"][DEFAULT_AREA]
    src = rel(REPLAY_FILE)
    quarters = []
    for q in area["quarters"]:
        quarters.append({
            "quarter": q["quarter"],
            "patients": {site: note(f"replay {q['quarter']}: {site} patients", v["patients"], src) for site, v in q["per_site"].items()},
            "fa_public": note(f"replay {q['quarter']}: false alarms, public", q["model"]["false_alarms"]["public"], src),
            "fa_federated": note(f"replay {q['quarter']}: false alarms, hospitals' counts", q["model"]["false_alarms"]["federated"], src),
            "changed_oslo": note(f"replay {q['quarter']}: readings changed at Oslo", q["query"]["changed_calls_per_site"]["site_oslo"], src),
            "query_variants": note(f"replay {q['quarter']}: query variants", q["query"]["variants"], src),
            "discordant": q["model"]["discordant_rows"],
        })
    what = saved["note"]["what"].split(". ")[0].rstrip(".") + "."
    q4 = saved["note"]["quarters"].split(". ")[0].rstrip(".") + "."
    return {"quarters": quarters, "sites": [s["name"] for s in area["sites"]], "what": what, "q4": q4, "file": src}


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=\.)\s+(?=[A-Z])", text.strip()) if s.strip()]


def read_readme() -> dict:
    """The sentences and the four AUC numbers reused from README.md, found by pattern; the build stops if one is missing."""
    text = README.read_text(encoding="utf-8")
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # keep link words, drop targets

    def find(pattern: str, flags=0) -> re.Match:
        found = re.search(pattern, text, flags)
        if not found:
            raise SystemExit(f"README.md no longer has: {pattern[:60]}")
        return found

    auc = find(r"AUC, stayed between ([\d.]+) and ([\d.]+)")
    alone = find(r"AlphaMissense alone reaches an AUC of ([\d.]+), the gene name alone ([\d.]+), and within single genes the scores still reach about ([\d.]+) to ([\d.]+)")
    matter = find(r"(In each area the test variants that matter are [^:]+): ([^.]+)\.")
    table_line = find(r"(The table counts how many of them one model, with one fixed threshold, wrongly flags)\. (The only thing that changes is what the model is told about frequency)\.")
    block = find(r"### The same across twelve specialties\s+(.+?)\n\n", re.S)
    setup = sentences(block.group(1))[:2]
    caveat = find(r"(The numbers per specialty are in docs/disease_areas_table\.md, averaged over[^.]+\.)\s+(Specialties share genes, so only [^.]+\.)")
    src = rel(README)
    return {
        "auc_lo": note("AUC range, low", auc.group(1), src), "auc_hi": note("AUC range, high", auc.group(2), src),
        "alphamissense": note("AlphaMissense alone, AUC", alone.group(1), src), "gene_name": note("gene name alone, AUC", alone.group(2), src),
        "within_lo": note("within single genes, low", alone.group(3), src), "within_hi": note("within single genes, high", alone.group(4), src),
        "matter": matter.group(1), "matter_counts": matter.group(2),
        "table_line": f"{table_line.group(1)}, and the {table_line.group(2)[4:]}.",
        "setup": setup, "caveat_shared": caveat.group(2), "caveat_pass": caveat.group(1),
    }


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------
CSS = """
  :root{--bg:#1e1e1e;--panel:#262626;--ink:#e6e6e6;--ink2:#c6c6c6;--g58:#8a8a8a;--g39:#626262;--g23:#303030;--edge:#4a4a4a;
        --blue:#5fafff;--green:#7ed321;--orange:#f5a623;--red:#e55353;
        --sans:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;--mono:ui-monospace,Menlo,Consolas,"Cascadia Mono",monospace}
  *{box-sizing:border-box}
  html{background:var(--bg)}
  body{margin:0;background:var(--bg);color:var(--ink2);font:14px/1.55 var(--sans);padding:22px 16px 48px}
  .wrap{max-width:56rem;margin:0 auto}
  .top{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 14px;margin:0 0 18px}
  .wm{font-size:14px;font-weight:600;color:#fff;letter-spacing:.01em;text-decoration:none}
  .wm:hover{color:var(--blue);text-decoration:underline}
  .wm b{font-weight:600}
  .ver{font-family:var(--mono);font-size:10.5px;color:var(--g58)}
  h1{font:600 14px/1.4 var(--sans);color:#fff;margin:0;letter-spacing:.01em}
  h1::before{content:"\\00b7";color:var(--g58);margin:0 8px 0 0}
  .lead{max-width:48rem;margin:0 0 6px;color:var(--ink)}
  .lead+.lead{color:var(--ink2)}
  h2{font:600 15px/1.4 var(--sans);color:#fff;margin:34px 0 8px;padding-top:18px;border-top:1px solid var(--g23)}
  p{max-width:48rem;margin:0 0 10px}
  ul{max-width:48rem;margin:0 0 10px;padding-left:1.2em}
  li{margin:0 0 6px}
  b{color:var(--ink);font-weight:600}
  a{color:var(--blue)}
  sup{font-size:.8em;line-height:0}
  .scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:10px 0 12px;max-width:100%}
  table{border-collapse:collapse;font-size:13px}
  th{font-weight:600;color:var(--ink);text-align:left;border-bottom:1px solid var(--edge);padding:7px 12px 7px 0;white-space:nowrap;vertical-align:bottom;line-height:1.3}
  td{padding:6px 12px 6px 0;border-bottom:1px solid var(--g23);vertical-align:middle;white-space:nowrap;color:var(--ink2)}
  th:last-child,td:last-child{padding-right:0}
  td.label{white-space:normal;min-width:7rem;max-width:22rem;color:var(--ink)}
  .evidence td.label{min-width:13rem}
  .num,th.num{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right}
  td.cell{text-align:left}
  td.cell .n{display:inline-block;min-width:3.2em;text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums;color:var(--ink)}
  .bar{display:inline-block;width:112px;height:9px;vertical-align:middle;margin-left:10px}
  .bar i{display:block;height:100%;border-radius:1px}
  .bar.grey i{background:var(--g58)}
  .bar.blue i{background:var(--blue)}
  .bar.green i{background:var(--green)}
  .bar.outline i{background:transparent;border:1px solid var(--ink2)}
  td.worse{background:rgba(245,166,35,.11);color:var(--ink)}
  figure{margin:10px 0 12px;max-width:56rem}
  figure img{display:block;width:100%;height:auto;border:1px solid var(--g23);border-radius:4px}
  figcaption{font-size:12px;color:var(--g58);margin-top:6px}
  .small{font-size:12.5px;color:var(--g58)}
  .small a{color:var(--g58)}
  .small a:hover{color:var(--ink)}
  .try{display:grid;gap:12px;max-width:48rem;margin:0 0 14px}
  a.big{display:block;padding:18px 20px;border:1px solid var(--edge);border-radius:6px;background:var(--panel);color:var(--blue);text-decoration:none;font-size:1.2rem;font-weight:600}
  a.big:hover,a.big:focus-visible{border-color:var(--blue)}
  a.big small{display:block;margin-top:5px;color:var(--ink2);font-size:.85rem;font-weight:400}
  .foot{margin:36px 0 0;padding-top:14px;border-top:1px solid var(--g23);font-size:12px;color:var(--g58);max-width:56rem}
  .foot a{color:var(--g58)}
  @media (min-width:640px){.try{grid-template-columns:1fr 1fr}}
  @media (max-width:480px){.bar{width:80px}td.label{min-width:11rem}}
"""


def bar_cell(value: int, column_max: int, style: str) -> str:
    width = 0 if column_max == 0 else round(100 * value / column_max, 1)
    return f'<td class="cell"><span class="n">{n(value)}</span><span class="bar {style}" aria-hidden="true"><i style="width:{width}%"></i></span></td>'


def headline_table(areas: dict) -> str:
    order = list(areas)
    maxima = {a: max(areas[a]["false_alarms"].values()) for a in order}
    head = "".join(f'<th class="num">{esc(areas[a]["header"])}, of {n(areas[a]["discordant"])}</th>' for a in order)
    rows = []
    for key, label, style in EVIDENCE_ROWS:
        cells = "".join(bar_cell(areas[a]["false_alarms"][key], maxima[a], style) for a in order)
        rows.append(f'<tr><td class="label">{esc(label)}</td>{cells}</tr>')
    return ('<div class="scroll"><table class="evidence"><thead><tr><th>What the model is told about frequency</th>' + head + "</tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div>")


def area_lines(areas: dict, runs: int, nvflare: dict) -> str:
    items = []
    for area, a in areas.items():
        removed, introduced, p = a["paired"]["removed"], a["paired"]["introduced"], a["paired"]["p"]
        intro = "none" if introduced == 0 else n(introduced)
        line = (f"<b>{esc(a['header'])}.</b> Going from the public database to the counts from all three hospitals removed "
                f"{n(removed)} false alarm{'' if removed == 1 else 's'} and introduced {intro}, p = {p_text(p)}. ")
        if area in nvflare:
            f = nvflare[area]
            if f["false_alarms_sd"] == 0:
                every = f"left {n(f['false_alarms_mean'])} false alarms of {n(a['discordant'])} in every run"
            else:
                every = f"left {f['false_alarms_mean']:g} false alarms of {n(a['discordant'])} on average, standard deviation {f['false_alarms_sd']:g}"
            line += f"Official: {n(runs)} NVIDIA FLARE runs, and the federated model {every}."
        else:
            line += "A first look from the same scripts, without a federated run."
        items.append(f"<li>{line}</li>")
    return "<ul>" + "".join(items) + "</ul>"


def sensitivity_line(areas: dict) -> str:
    parts = [f"{fix(a['sensitivity']['public'], 3)} and {fix(a['sensitivity']['federated_query'], 3)} {esc(a['phrase'])}" for a in areas.values()]
    return ("<p>The share of harmful test variants still caught, with the public database first and the hospitals' counts second, was "
            + ", ".join(parts[:-1]) + " and " + parts[-1] + ".</p>")


def specialty_table(rows: list[dict]) -> str:
    head = ("<tr><th>Specialty</th><th class=\"num\">Genes</th><th class=\"num\">Discordant<br>benign variants</th>"
            "<th class=\"num\">False alarms,<br>Oslo alone</th><th class=\"num\">federated</th><th class=\"num\">pooled</th>"
            "<th class=\"num\">AUC,<br>Oslo alone</th><th class=\"num\">federated</th><th class=\"num\">pooled</th></tr>")
    body = []
    for r in rows:
        worse = ' worse' if r["fa_oslo"] > r["fa_fed"] else ""
        body.append(
            f'<tr><td class="label">{esc(r["specialty"])}</td><td class="num">{n(r["genes"])}</td><td class="num">{n(r["discordant"])}</td>'
            f'<td class="num{worse}">{r["fa_oslo"]:g}</td><td class="num">{r["fa_fed"]:g}</td><td class="num">{r["fa_pooled"]:g}</td>'
            f'<td class="num">{fix(r["auc_oslo"], 4)}</td><td class="num">{fix(r["auc_fed"], 4)}</td><td class="num">{fix(r["auc_pooled"], 4)}</td></tr>')
    return '<div class="scroll"><table><thead>' + head + "</thead><tbody>" + "".join(body) + "</tbody></table></div>"


def replay_table(replay: dict) -> str:
    sites = replay["sites"]
    names = " / ".join(SITE_NAMES.get(s, s) for s in sites)
    of = replay["quarters"][0]["discordant"]
    variants = replay["quarters"][0]["query_variants"]
    head = (f"<tr><th>Quarter</th><th class=\"num\">Patients at<br>{esc(names)}</th><th class=\"num\">False alarms of {n(of)},<br>public database</th>"
            f"<th class=\"num\">False alarms of {n(of)},<br>hospitals' counts</th><th class=\"num\">Readings changed<br>at Oslo, of {n(variants)}</th></tr>")
    body = "".join(
        f'<tr><td class="label">{esc(q["quarter"])}</td><td class="num">{" / ".join(n(q["patients"][s]) for s in sites)}</td>'
        f'<td class="num">{n(q["fa_public"])}</td><td class="num">{n(q["fa_federated"])}</td><td class="num">{n(q["changed_oslo"])}</td></tr>'
        for q in replay["quarters"])
    return '<div class="scroll"><table><thead>' + head + "</thead><tbody>" + body + "</tbody></table></div>"


def render(areas: dict, runs: int, nvflare: dict, specialties: list[dict], replay: dict, readme: dict, built: str) -> str:
    order = list(areas)
    heart = areas[DEFAULT_AREA]
    hospitals = ", ".join(f"{h['name']} with {n(h['patients'])} patients of {h['population']} ancestry" for h in heart["hospitals"].values())
    hospitals = " and ".join(hospitals.rsplit(", ", 1))
    built_from = "; ".join(f"{a['header'].lower() if a['header'] != 'Every panel' else 'every signed-off panel'}, {n(a['panels'])} panels, "
                           f"{n(a['genes'])} genes and {n(a['variants'])} variants" for a in areas.values())
    counts = ", ".join(f"{n(areas[a]['discordant'])} {areas[a]['phrase']}" for a in order)
    counts = " and ".join(counts.rsplit(", ", 1))
    what_tested = (f"Three simulated hospitals, {hospitals}, judged missense variants in three disease areas built from gene panels signed off by the NHS: "
                   f"{built_from}.")
    real_simulated = ("Every expert verdict, prediction score and population frequency is real, from ClinVar, dbNSFP and gnomAD; every patient, "
                      "every carrier count and which hospital has classified which variant is simulated, and so is the public database that covers Europeans only.")
    sources = ", ".join(sorted({s for _, _, s in embedded}))
    caveat_pass = re.sub(r"docs/disease_areas_table\.md", f'<a href="{BLOB}disease_areas_table.md">docs/disease_areas_table.md</a>', esc(readme["caveat_pass"]))
    doc_links = ", ".join(f'<a href="{BLOB}{d}">{d}</a>' for d in FULL_DOCS)
    parts = [
        "<!DOCTYPE html>",
        "<!--",
        f"  Built by scripts/export_results_page.py on {built}. Do not edit index.html by hand: rerun",
        "    uv run python scripts/export_results_page.py",
        f"  Every number on this page was read from: {sources}.",
        "  The three pictures are files next to this folder: ../disease_areas_figure.png, ../manuscript_figure3.png, ../manuscript_figure4.png.",
        "-->",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">',
        '<meta name="color-scheme" content="dark">',
        "<title>Results</title>",
        '<meta name="description" content="What Team 12 found: false alarms across three disease areas and twelve clinical specialties when three simulated hospitals share counts and model weights instead of patient records.">',
        f"<style>{CSS}</style></head><body><div class=\"wrap\">",
        f'<div class="top"><a class="wm" href="{REPO}#readme">REFLECT</a><h1>Results</h1><span class="ver">built {esc(built)} by scripts/export_results_page.py</span></div>',
        f'<p class="lead">{esc(what_tested)}</p>',
        f'<p class="lead">{esc(real_simulated)}</p>',

        "<h2>False alarms across three disease areas</h2>",
        f"<p>{esc(readme['matter'])}: {esc(counts)}. {esc(readme['table_line'])}</p>",
        headline_table(areas),
        area_lines(areas, runs, nvflare),
        sensitivity_line(areas),
        '<figure><img src="../manuscript_figure3.png" alt="Three bar charts, one per disease area, of false alarms under the five sources of frequency evidence. '
        'The bar for the federated query is the shortest in every area and equals the ceiling.">'
        "<figcaption>The same counts as the manuscript draws them.</figcaption></figure>",

        "<h2>Twelve clinical specialties</h2>",
        f"<p>{esc(' '.join(readme['setup']))}</p>",
        '<figure><img src="../disease_areas_figure.png" alt="Two panels over twelve clinical specialties. Panel A, AUC on variants of South Asian or African ancestry patients seen at Oslo, '
        'has the three lines on top of each other in every specialty. Panel B, false alarms among population-discordant benign variants, separates them: Oslo training alone is above '
        'federated and pooled in most specialties."></figure>',
        specialty_table(specialties),
        f'<p class="small">Orange cells: Oslo alone worse than federated. {esc(readme["caveat_shared"])} {caveat_pass}</p>',

        "<h2>Over four quarters</h2>",
        f"<p>The three hospitals grow to their final size in four equal steps and the model is retrained each quarter on the verdicts held so far; the heart area is shown. "
        f"{esc(replay['what'])} {esc(replay['q4'])}</p>",
        '<figure><img src="../manuscript_figure4.png" alt="Two line charts over four quarters for the heart area. Left, false alarms with the public database stay near 7 while '
        'the federated query falls to 2 and meets the ceiling. Right, readings changed by the count query rise for a patient at each hospital, Oslo highest."></figure>',
        replay_table(replay),

        "<h2>What the accuracy score cannot see</h2>",
        f"<p>The usual summary score, AUC, stayed between {esc(readme['auc_lo'])} and {esc(readme['auc_hi'])} whatever the model was told and whichever hospital trained it. "
        f"On the heart test set AlphaMissense alone reaches an AUC of {esc(readme['alphamissense'])}, the gene name alone {esc(readme['gene_name'])}, and within single genes "
        f"the scores still reach about {esc(readme['within_lo'])} to {esc(readme['within_hi'])}. "
        "So the AUC reports how the expert verdicts were made, and the false alarms above are the measurement that needs the hospitals' counts.</p>",

        "<h2>Try it</h2>",
        '<div class="try">',
        '<a class="big" href="../demo/">The patient query<small>Pick a variant, every hospital answers with counts, and the screen gives the call twice: the patient\'s hospital alone, then all three.</small></a>',
        '<a class="big" href="../pipeline_live/">The pipeline in motion<small>Data comes in, hospitals train together, a quarter passes, a doctor asks.</small></a>',
        "</div>",
        f'<p class="small">The full write-ups on GitHub: {doc_links}.</p>',

        f'<p class="foot">A research prototype from a hackathon. The hospitals and their patients are simulated. It must not be used for patient care. '
        f'Code and write-ups: <a href="{REPO}">the repository on GitHub</a>.</p>',
        "</div></body></html>",
    ]
    return "\n".join(parts) + "\n"


def describe() -> None:
    by_source: dict[str, list[tuple[str, str]]] = {}
    for what, value, source in embedded:
        by_source.setdefault(source, []).append((what, value))
    print("Embedded, by source file:")
    for source, items in by_source.items():
        print(f"  {source}")
        for what, value in items:
            print(f"      {what}: {value}")


def main() -> None:
    areas = read_areas()
    runs = read_runs()
    nvflare = {area: read_nvflare_row(area, areas[area]["discordant"]) for area in NVFLARE_PAGES if area in areas}
    specialties = read_specialties()
    replay = read_replay()
    readme = read_readme()
    # The README's counts of the variants that matter must be the JSON's, or the page would say two things
    stated = [int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", readme["matter_counts"])]
    actual = [areas[a]["discordant"] for a in areas]
    if stated != actual:
        print(f"WARNING: README says {stated} discordant benign rows, results_local.json says {actual}")
    built = dt.datetime.now().strftime("%Y-%m-%d")
    page = render(areas, runs, nvflare, specialties, replay, readme, built)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8", newline="\n")
    describe()
    print(f"Wrote {rel(OUT)}: {OUT.stat().st_size / 1024:.0f} KB, {len(embedded)} numbers embedded")


if __name__ == "__main__":
    main()
