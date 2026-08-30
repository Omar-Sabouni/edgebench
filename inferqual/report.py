"""Write the compact JSON and HTML benchmark result."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

_REPORT_CSS = """
:root{color-scheme:light;--ink:#141414;--muted:#5c5c5c;--rule:#141414;--hair:#d6d6d6;--gold:#c9a227;--paper:#fff}
*{box-sizing:border-box}
html{background:#efefef}
body{margin:0;padding:40px 18px 56px;background:#efefef;color:var(--ink);font:15px/1.5 "Helvetica Neue",Helvetica,Arial,sans-serif;font-variant-numeric:tabular-nums}
main{width:min(100%,980px);margin:auto;padding:36px 40px 40px;background:var(--paper);box-shadow:0 1px 3px rgb(0 0 0/15%)}
header{padding-bottom:14px;border-bottom:4px solid var(--rule)}
.event{margin:0 0 4px;color:var(--muted);font-size:14px}
h1{margin:0;font-size:38px;line-height:1.05;font-weight:800;letter-spacing:-.01em}
.lede{max-width:640px;margin:8px 0 0;color:var(--muted)}
h2{margin:0 0 6px;font-size:15px}
.winner{display:grid;grid-template-columns:auto 1fr;gap:4px 18px;align-items:center;margin:26px 0 30px}
.winner .medal{grid-row:span 2;display:grid;width:64px;height:64px;place-items:center;border-radius:50%;background:var(--gold);color:var(--ink);font-size:30px;font-weight:800;box-shadow:inset 0 0 0 4px #e4c65a}
.winner strong{font-size:24px;line-height:1.2}
.winner span{color:var(--muted)}
.table-region{overflow-x:auto}
.table-region:focus-visible{outline:2px solid var(--ink);outline-offset:3px}
table{width:100%;min-width:680px;border-collapse:collapse}
caption{padding-bottom:8px;color:var(--muted);font-size:13px;text-align:left}
th,td{padding:10px 12px 10px 0;border-bottom:1px solid var(--hair);text-align:right;vertical-align:baseline}
thead th{border-bottom:2px solid var(--rule);font-size:13px;font-weight:700;white-space:nowrap}
th:first-child,td:first-child{width:56px;text-align:left}
tbody th{text-align:left;font-weight:600}
.place{font-size:22px;font-weight:800}
.time{font-size:18px;font-weight:700}
thead th:nth-child(2){text-align:left}
tbody tr.selected{background:#fbf3d6}
tbody tr:last-child>*{border-bottom:2px solid var(--rule)}
.caveat{margin:22px 0 0;color:var(--muted);font-size:13px}
@media(max-width:720px){body{padding:0}main{padding:24px 16px 32px;box-shadow:none}h1{font-size:30px}}
@media print{html,body{background:#fff}body{padding:0}main{box-shadow:none;padding:0}}
"""
_REPORT_CSS = "".join(line.strip() for line in _REPORT_CSS.splitlines())


def _text(value: Any) -> str:
    return html.escape(str(value))


def render_html(report: dict[str, Any]) -> str:
    """Render the benchmark result."""
    results = sorted(report["results"], key=lambda item: item["latency"]["p50_ms"])
    selected = next(item for item in results if item["id"] == report["selected"])
    comparison = report.get("comparison", {})
    speedup = comparison.get("speedup_vs_pytorch_cpu")
    speedup_note = (
        f", {_text(speedup)}× faster than PyTorch on CPU"
        if isinstance(speedup, (int, float))
        else ""
    )
    selected_place = results.index(selected) + 1
    rows = []
    for place, result in enumerate(results, start=1):
        latency = result["latency"]
        is_selected = result["id"] == report["selected"]
        class_name = ' class="selected"' if is_selected else ""
        rows.append(
            f'<tr{class_name}><td class="place">{place}</td>'
            f'<th scope="row">{_text(result["label"])}</th>'
            f"<td>{_text(result['device'])}, {_text(result['precision'])}</td>"
            f'<td class="time">{_text(latency["p50_ms"])} ms</td>'
            f"<td>{_text(latency['p95_ms'])} ms</td>"
            f"<td>{_text(latency['throughput_per_second'])}</td>"
            f"<td>{_text(result['cosine_similarity'])}</td></tr>"
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light">
<title>InferQual result</title><style>{_REPORT_CSS}</style></head><body><main>
<header><p class="event">InferQual</p>
<h1>Benchmark result</h1>
<p class="lede">PyTorch and ONNX Runtime, timed on the built-in vision model.</p>
</header>
<section aria-labelledby="result-title">
<div class="winner"><span class="medal" aria-hidden="true">{selected_place}</span>
<strong id="result-title">{_text(selected["label"])}</strong>
<span>{_text(selected["latency"]["p50_ms"])} ms median{speedup_note}</span></div>
<div class="table-region" role="region" aria-label="Benchmark result table" tabindex="0">
<table><caption>Placed by median time. Lower is better.</caption><thead><tr>
<th scope="col">Place</th><th scope="col">Runtime</th><th scope="col">Device</th>
<th scope="col">Median</th><th scope="col">p95</th><th scope="col">Samples/s</th>
<th scope="col">Output match</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div></section>
<p class="caveat">Times depend on the machine. Accuracy isn't measured here.</p>
</main></body></html>"""


def write_bundle(report: dict[str, Any], output: Path) -> tuple[Path, Path]:
    json_path = output / "report.json"
    html_path = output / "report.html"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    html_path.write_text(render_html(report), encoding="utf-8")
    return json_path, html_path
