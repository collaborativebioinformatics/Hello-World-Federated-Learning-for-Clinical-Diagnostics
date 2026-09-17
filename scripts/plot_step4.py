"""Draw the step 4 figure from data/results_federated.json.

Three arms, two metrics, error bars over the five runs. Same house style and the
same WeasyPrint route as the step 3 figure; text-anchor is set as an SVG
attribute because WeasyPrint ignores the CSS property.

The error bars on the AUC panel are invisible at this axis. That is the finding,
not a drawing mistake, and the caption says so: an axis stretched until they
showed would manufacture a difference that is not there.

Usage:
    uv run python scripts/plot_step4.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURE = ROOT / "docs" / "step4_figure.html"

MUTED, FULL, INK, MUTED_INK, SURFACE = "#8CA86A", "#3B6D11", "#2c2c2a", "#5f5e5a", "#faf9f5"

ARMS = [
    ("oslo_only", "Oslo only", "one hospital, its own rows"),
    ("federated", "Federated, NVFlare FedAvg", "only weights travel"),
    ("pooled", "Everything pooled", "not allowed in real life"),
]


def bar(x: float, y: float, width: float, height: float, radius: float = 4) -> str:
    radius = min(radius, max(width, 0.1))
    return (f'M{x},{y} H{x + width - radius} A{radius},{radius} 0 0 1 {x + width},{y + radius} '
            f'V{y + height - radius} A{radius},{radius} 0 0 1 {x + width - radius},{y + height} H{x} Z')


def panel(title: str, note: str, stats: dict[str, tuple[float, float]], top: float,
          axis_max: float, fmt: str, tick_fmt: str) -> str:
    label_width, plot_left, plot_width, row_height, bar_height = 214.0, 224.0, 286.0, 40.0, 20.0
    out = [f'<text x="0" y="{top}" class="panel">{title}</text>',
           f'<text x="0" y="{top + 17}" class="note">{note}</text>']
    baseline = top + 32
    for step in range(5):
        x = plot_left + plot_width * step / 4
        out.append(f'<line x1="{x}" y1="{baseline}" x2="{x}" y2="{baseline + row_height * len(ARMS)}" class="grid"/>')
        out.append(f'<text x="{x}" y="{baseline + row_height * len(ARMS) + 13}" '
                   f'text-anchor="middle" class="tick">{axis_max * step / 4:{tick_fmt}}</text>')

    for row, (key, label, sub) in enumerate(ARMS):
        mean, std = stats[key]
        y = baseline + row * row_height + (row_height - bar_height) / 2
        width = max(1.5, plot_width * mean / axis_max)
        is_pooled, is_method = key == "pooled", key == "federated"
        fill = "url(#hatch)" if is_pooled else (FULL if is_method else MUTED)
        edge = f' stroke="{FULL}" stroke-width="1.5"' if is_pooled else ""
        out.append(f'<g><title>{label}: {mean:{fmt}} (sd {std:{fmt}})</title>'
                   f'<path d="{bar(plot_left, y, width, bar_height)}" fill="{fill}"{edge}/></g>')
        if std > 0:
            lo = plot_left + plot_width * max(0.0, mean - std) / axis_max
            hi = plot_left + plot_width * min(axis_max, mean + std) / axis_max
            mid = y + bar_height / 2
            out.append(f'<line x1="{lo}" y1="{mid}" x2="{hi}" y2="{mid}" class="err"/>'
                       f'<line x1="{lo}" y1="{mid - 4}" x2="{lo}" y2="{mid + 4}" class="err"/>'
                       f'<line x1="{hi}" y1="{mid - 4}" x2="{hi}" y2="{mid + 4}" class="err"/>')
        out.append(f'<text x="{label_width}" y="{y + 8}" text-anchor="end" '
                   f'class="cat{" strong" if is_method else ""}">{label}</text>')
        out.append(f'<text x="{label_width}" y="{y + 19}" text-anchor="end" class="sub">{sub}</text>')
        out.append(f'<text x="{max(plot_left + width, hi if std > 0 else 0) + 8}" y="{y + 14}" '
                   f'class="value">{mean:{fmt}} ({std:{fmt}})</text>')
    return "\n".join(out)


def forest(title: str, note: str, rows: list[tuple[str, str, float, float]], top: float,
           span: float) -> str:
    """Paired differences against zero. A centred axis, because the sign is the question."""
    label_width, plot_left, plot_width, row_height = 214.0, 224.0, 286.0, 38.0
    zero = plot_left + plot_width / 2
    out = [f'<text x="0" y="{top}" class="panel">{title}</text>',
           f'<text x="0" y="{top + 17}" class="note">{note}</text>']
    baseline = top + 32
    for step in range(5):
        x = plot_left + plot_width * step / 4
        value = -span + 2 * span * step / 4
        out.append(f'<line x1="{x}" y1="{baseline}" x2="{x}" y2="{baseline + row_height * len(rows)}" '
                   f'class="{"zero" if step == 2 else "grid"}"/>')
        out.append(f'<text x="{x}" y="{baseline + row_height * len(rows) + 13}" '
                   f'text-anchor="middle" class="tick">{value:+.3f}</text>')

    for row, (label, sub, mean, std) in enumerate(rows):
        y = baseline + row * row_height + row_height / 2
        centre = zero + plot_width / 2 * mean / span
        lo, hi = zero + plot_width / 2 * (mean - std) / span, zero + plot_width / 2 * (mean + std) / span
        out.append(f'<g><title>{label}: {mean:+.4f} (sd {std:.4f})</title>'
                   f'<line x1="{lo}" y1="{y}" x2="{hi}" y2="{y}" class="err"/>'
                   f'<line x1="{lo}" y1="{y - 4}" x2="{lo}" y2="{y + 4}" class="err"/>'
                   f'<line x1="{hi}" y1="{y - 4}" x2="{hi}" y2="{y + 4}" class="err"/>'
                   f'<circle cx="{centre}" cy="{y}" r="4.5" fill="{FULL}"/></g>')
        out.append(f'<text x="{label_width}" y="{y - 1}" text-anchor="end" class="cat">{label}</text>')
        out.append(f'<text x="{label_width}" y="{y + 10}" text-anchor="end" class="sub">{sub}</text>')
        out.append(f'<text x="{plot_left + plot_width + 8}" y="{y + 4}" class="value">{mean:+.4f}</text>')
    return "\n".join(out)


def main() -> int:
    results = json.loads((ROOT / "data" / "results_federated.json").read_text())
    summary = results["summary"]
    auc = {arm: tuple(summary[arm]["auc_federated_query"]) for arm, _, _ in ARMS}
    alarms = {arm: tuple(summary[arm]["false_alarms_federated_query"]) for arm, _, _ in ARMS}
    home = results["home_site_comparison"]
    rows_by_pop = results["population_rows"]
    readable = {"nfe": "European (NFE)", "sas": "South Asian (SAS)", "afr": "African (AFR)"}
    difference = [
        (f'{readable[p]} vs {home[p]["own_site"].replace("site_", "").capitalize()}',
         f'{rows_by_pop[p][0]} rows, {rows_by_pop[p][1]} pathogenic',
         home[p]["difference"][0], home[p]["difference"][1])
        for p in ("nfe", "sas", "afr")
    ]

    figure = f"""<!DOCTYPE html>
<!--
  Step 4 figure. Generated by scripts/plot_step4.py from data/results_federated.json.
  Do not edit the numbers here; rerun the script, then render with
  scripts/render_docs_png.py.
-->
<html lang="en">
<head>
<meta charset="utf-8">
<title>Step 4: federated learning against its baselines</title>
<style>
  body{{margin:0;background:{SURFACE};color:{INK};font-family:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
  .page{{width:680px;margin:0 auto;padding:24px 12px}}
  h1{{font-size:18px;font-weight:500;margin:0 0 3px}}
  .lede{{font-size:12px;color:{MUTED_INK};margin:0 0 4px;line-height:1.5}}
  .panel{{font-size:13px;font-weight:500;fill:{INK}}}
  .note{{font-size:10.5px;fill:{MUTED_INK}}}
  .cat{{font-size:11.5px;fill:{INK}}} .cat.strong{{font-weight:500}}
  .sub{{font-size:9.5px;fill:{MUTED_INK}}}
  .value{{font-size:11px;fill:{INK};font-family:ui-monospace,Menlo,Consolas,monospace}}
  .tick{{font-size:9.5px;fill:{MUTED_INK};font-family:ui-monospace,Menlo,Consolas,monospace}}
  .grid{{stroke:#d9d7cf;stroke-width:1}}
  .err{{stroke:{INK};stroke-width:1.5}}
  .zero{{stroke:{INK};stroke-width:1.5}}
  .caption{{font-size:11px;color:{MUTED_INK};margin:10px 0 0;line-height:1.55}}
  .caption b{{color:{INK};font-weight:500}}
</style>
</head>
<body>
<div class="page">
<h1>Federated learning matches pooling, without moving a row</h1>
<p class="lede">NVFlare {results["nvflare_version"]} FedAvg, {results["rounds"]} aggregation rounds,
three simulated hospitals as separate processes. {results["runs"]} runs; what changes between
runs is which hospital classified which variant. Mean (standard deviation) over the runs,
scored on the locked test set with the federated count query as frequency evidence.</p>

<svg width="680" height="592" role="img"
     aria-label="Two panels, three arms each. AUC is 0.9781 for Oslo only, 0.9783 for federated and 0.9784 for pooled, with standard deviations of 0.0002 or less. False alarms among the 183 discordant benign variants are 2.2 for Oslo only with a standard deviation of 0.4, and exactly 2.0 for both federated and pooled in every run.">
  <defs>
    <pattern id="hatch" width="6" height="6" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
      <rect width="6" height="6" fill="{SURFACE}"/>
      <line x1="0" y1="0" x2="0" y2="6" stroke="{FULL}" stroke-width="3"/>
    </pattern>
  </defs>
{panel("AUC on the locked test set", "higher is better &middot; error bars are there, and smaller than the line width", auc, 18, 1.0, ".4f", ".2f")}
{panel("False alarms among the 183 population-discordant benign variants",
       "lower is better &middot; each arm uses a threshold it could actually have computed", alarms, 218, 8, ".1f", ".0f")}
{forest("Does the global model serve a population worse than its own hospital's model?",
        "federated minus that population's own hospital &middot; right of zero means federating helped",
        difference, 415, 0.008)}
</svg>

<p class="caption"><b>Federated is indistinguishable from pooled on every metric</b>, and the gap
to it is smaller than the run-to-run spread. Pooling is hatched because it is the arm no
privacy law allows: it is an upper bound, not an option. The claim federation earns here is
not that it wins, it is that <b>it costs nothing</b>.</p>
<p class="caption">Oslo alone is slightly behind and, more tellingly, the only arm that
<b>varies</b>: its false alarms move with which variants it happened to be dealt, while
federated and pooled returned exactly the same count in all five runs. A single hospital's
result depends on its luck of the draw; federating removes that dependence.</p>
<p class="caption"><b>No population is served worse by the global model than by its own
hospital's.</b> The one population that gains is African-ancestry, the smallest and the one
gnomAD covers worst, and it is also where a single site is least stable: Lagos alone varies
by 0.0027 between deals against 0.0008 federated. Every interval here crosses or touches
zero, so the honest claim is that federating <b>does not hurt anyone</b>, not that it
measurably helps AFR &mdash; that slice holds 12 pathogenic variants and the difference is
about 1.3 standard deviations.</p>
<p class="caption">The original hypothesis, that a single site collapses on populations it
does not serve, is <b>refuted</b>: Oslo alone scores 0.9346, 0.9275 and 0.9539 on the
European, South Asian and African slices. The prediction scores carry the ranking and they
do not depend on ancestry. What depends on ancestry is the frequency evidence at scoring
time, and that moves the decision rather than the ranking.</p>
<p class="caption">The federated arm never sees another site's rows, including for its
threshold: each site takes the quantile on its own rows under the global model and the
server averages them by row count, so one number per site crosses the wire alongside the
14 weights. Using the pooled threshold instead changes none of these figures.</p>
</div>
</body>
</html>
"""
    FIGURE.write_text(figure, encoding="utf-8")
    print(f"wrote {FIGURE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
