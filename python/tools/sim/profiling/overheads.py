"""
HTML report on where a solver's production step time goes beyond kernel work.

Combines sweep anatomies (``profiling/anatomy.py``), Nsight Compute kernel summaries
(``profiling/ncu.py``), production step timings (``simulate.py --profile-out``) and a findings file
into one page: the per-step time budget (sweep and boundary kernels and gaps), the subproblem
boundary by phase, the steady sweep as a timeline, a kernel comparison, the kernel's stall hotspots by
Python function (from ``ncu.py --lineinfo`` summaries), and before/after timings.

Usage (from the repo root):

    python -m python.tools.sim.profiling.overheads \\
        --anatomy VBD=out/vbd.anatomy.json --anatomy AAAVBD=out/aaavbd.anatomy.json \\
        --ncu VBD=out/vbd.ncu.json --ncu AAAVBD=out/aaavbd.ncu.json \\
        --production "AAAVBD=out/aaavbd_prod.json" --findings findings.txt -o overheads.html
"""

import argparse
import json
import math

from . import report

_esc = report._esc
_bar_path = report._bar_path

# Budget segments: (key, label, color token). Kernel work in blue/aqua, idle gaps in orange/yellow.
_BUDGET = [
    ("sweep_kernels_ms", "Sweep kernels", "var(--series-1)"),
    ("sweep_gaps_ms", "Sweep gaps", "var(--series-4)"),
    ("boundary_kernels_ms", "Boundary kernels", "var(--series-3)"),
    ("boundary_gaps_ms", "Boundary gaps", "var(--series-2)"),
    ("other_ms", "Other", "var(--other)"),
]
_PHASE_COLORS = {"sweep": "var(--series-1)", "contact.dual": "var(--series-5)"}


def _load(pairs: list[str]) -> list[tuple[str, dict]]:
    out = []
    for pair in pairs or []:
        label, path = pair.split("=", 1)
        with open(path, encoding="utf-8") as f:
            out.append((label, json.load(f)))
    return out


def _legend(items: list[tuple[str, str]]) -> str:
    return '<div class="legend">' + "".join(
        f'<span class="key"><span class="swatch" style="background:{c}"></span>{_esc(l)}</span>'
        for l, c in items
    ) + "</div>"


def _ticks(vmax: float, n: int = 6) -> float:
    step = 10 ** math.floor(math.log10(vmax))
    return next(t * step for t in (1, 2, 5, 10) if vmax / (t * step) <= n)


def _budget_chart(anatomies: list[tuple[str, dict]]) -> str:
    """Stacked bars of each anatomy's per-step budget (sweep and boundary kernels and gaps, other)."""
    width, bar_h, gap, label_w, right = 760, 28, 18, 170, 80
    totals = [sum(a["budget"][k] for k, _, _ in _BUDGET) for _, a in anatomies]
    vmax = max(totals) or 1.0
    tick = _ticks(vmax)
    plot_w = width - label_w - right
    height = len(anatomies) * (bar_h + gap) + 30
    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
           f'aria-label="Production step time budget">']
    t = 0.0
    while t <= vmax * 1.0001:
        x = label_w + plot_w * t / vmax
        out.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="0" y2="{height - 22}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{height - 6}" text-anchor="middle">{t:g} ms</text>')
        t += tick
    for i, ((label, a), total) in enumerate(zip(anatomies, totals)):
        y = i * (bar_h + gap) + 4
        out.append(f'<text class="label" x="{label_w - 10}" y="{y + bar_h / 2 + 4}" '
                   f'text-anchor="end">{_esc(label)}</text>')
        x = float(label_w)
        segments = [(k, l, c) for k, l, c in _BUDGET if a["budget"][k] > 0.0]
        for j, (key, seg_label, color) in enumerate(segments):
            ms = a["budget"][key]
            w = plot_w * ms / vmax
            last = j == len(segments) - 1
            tip = f"{label} · {seg_label}: {ms:.2f} ms/step ({100 * ms / total:.1f}%)"
            out.append(f'<path class="mark" data-tip="{_esc(tip)}" fill="{color}" '
                       f'd="{_bar_path(x, y, max(w - (0 if last else 2), 0.5), bar_h, 4 if last else 0)}"/>')
            x += w
        out.append(f'<text class="value" x="{x + 6:.1f}" y="{y + bar_h / 2 + 4}">{total:.1f} ms</text>')
    out.append("</svg>")
    return "".join(out)


def _boundary_table(anatomies: list[tuple[str, dict]]) -> str:
    """Kernel time and preceding gaps per phase in one subproblem-boundary window, per anatomy."""
    vmax = max(
        (p["kernel_us"] + p["gap_before_us"] for _, a in anatomies for p in a["boundary_windows"]["phases"]),
        default=1.0,
    ) or 1.0
    rows = []
    for label, a in anatomies:
        bw = a["boundary_windows"]
        rows.append(
            f'<tr class="group"><td colspan="4">{_esc(label)}: {bw["per_step"]:.0f} boundaries per step, '
            f'{bw["mean_span_us"]:.0f} µs each</td></tr>'
        )
        for p in bw["phases"]:
            k, g = p["kernel_us"], p["gap_before_us"]
            bar = (f'<span class="hbar"><span style="width:{100 * k / vmax:.2f}%;background:var(--series-3)"></span>'
                   f'<span style="width:{100 * g / vmax:.2f}%;background:var(--series-2)"></span></span>')
            rows.append(f"<tr><td>{_esc(p['phase'])}</td><td>{k:.1f}</td><td>{g:.1f}</td><td class='barcell'>"
                        f"{bar}</td></tr>")
    return ("<table><thead><tr><th>Phase</th><th>Kernels (µs)</th><th>Gaps before (µs)</th>"
            "<th class='barcell'></th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def _node_color(node: dict) -> str:
    phase = node["phase"]
    if phase.endswith("sweep"):
        return _PHASE_COLORS["sweep"]
    if node["name"].startswith(("memcpy", "memset")):
        return "var(--other)"
    return _PHASE_COLORS.get(phase, "var(--series-7)")


def _sweep_timeline(anatomies: list[tuple[str, dict]]) -> str:
    """Lane timeline of each anatomy's steady sweep (concurrent nodes in separate lanes)."""
    width, lane_h, label_w, right = 760, 14, 170, 20
    plot_w = width - label_w - right
    vmax = max(max(n["start_us"] + n["duration_us"] for n in a["sweep"]["nodes"]) for _, a in anatomies)
    tick = _ticks(vmax, 8)
    blocks, y = [], 0
    for label, a in anatomies:
        lanes: list[float] = []
        placed = []
        for n in sorted(a["sweep"]["nodes"], key=lambda n: n["start_us"]):
            lane = next((i for i, end in enumerate(lanes) if end <= n["start_us"] + 1e-6), len(lanes))
            if lane == len(lanes):
                lanes.append(0.0)
            lanes[lane] = n["start_us"] + n["duration_us"]
            placed.append((lane, n))
        n_lanes = max(len(lanes), 1)
        blocks.append((label, a, placed, n_lanes, y))
        y += n_lanes * (lane_h + 2) + 26
    height = y + 22
    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Steady sweep timeline">']
    t = 0.0
    while t <= vmax * 1.0001:
        x = label_w + plot_w * t / vmax
        out.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="0" y2="{height - 22}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{height - 6}" text-anchor="middle">{t:g} µs</text>')
        t += tick
    for label, a, placed, n_lanes, y0 in blocks:
        sweep = a["sweep"]
        out.append(f'<text class="label" x="{label_w - 10}" y="{y0 + 12}" text-anchor="end">{_esc(label)}</text>')
        out.append(f'<text class="tick" x="{label_w - 10}" y="{y0 + 26}" text-anchor="end">'
                   f'{sweep["span_us"]:.0f} µs, gaps {sweep["gap_us"]:.1f} µs</text>')
        for lane, n in placed:
            x = label_w + plot_w * n["start_us"] / vmax
            w = max(plot_w * n["duration_us"] / vmax - 1, 1.0)
            yy = y0 + lane * (lane_h + 2)
            tip = (f"{label} · {n['name']}: starts at {n['start_us']:.2f} µs, runs {n['duration_us']:.2f} µs, "
                   f"idle {n['gap_before_us']:.2f} µs before")
            out.append(f'<rect class="mark" data-tip="{_esc(tip)}" x="{x:.2f}" y="{yy}" width="{w:.2f}" '
                       f'height="{lane_h}" rx="2" fill="{_node_color(n)}"/>')
    out.append("</svg>")
    return "".join(out)


_NCU_ROWS = [
    ("duration_us", "Duration (µs, base clocks)", "{:.2f}"),
    ("instructions", "Executed instructions", "{:,.0f}"),
    ("dram_throughput_pct", "DRAM throughput (%)", "{:.1f}"),
    ("memory_throughput_pct", "Memory throughput (%)", "{:.1f}"),
    ("sm_throughput_pct", "SM throughput (%)", "{:.1f}"),
    ("ipc_active", "IPC (active)", "{:.2f}"),
    ("warp_cycles_per_instruction", "Warp cycles per instruction", "{:.1f}"),
    ("no_eligible_pct", "Cycles with no eligible warp (%)", "{:.1f}"),
    ("registers", "Registers per thread", "{:.0f}"),
    ("grid", "Blocks per launch (32 threads each)", "{:.0f}"),
    ("waves_per_sm", "Waves per SM", "{:.2f}"),
    ("theoretical_occupancy_pct", "Theoretical occupancy (%)", "{:.1f}"),
    ("achieved_occupancy_pct", "Achieved occupancy (%)", "{:.1f}"),
    ("l1_hit_pct", "L1 hit rate (%)", "{:.1f}"),
    ("l2_hit_pct", "L2 hit rate (%)", "{:.1f}"),
]


def _ncu_section(kernels: list[tuple[str, dict]]) -> str:
    """Nsight Compute metrics table (with the change vs the first kernel when there are two) and warp
    stall shares."""
    if not kernels:
        return ""
    head = "".join(f"<th>{_esc(l)}</th>" for l, _ in kernels)
    delta = len(kernels) == 2
    rows = []
    for key, label, fmt in _NCU_ROWS:
        values = [k["metrics"].get(key) for _, k in kernels]
        if all(v is None for v in values):
            continue
        cells = "".join(f"<td>{'' if v is None else fmt.format(v)}</td>" for v in values)
        if delta and None not in values and values[0]:
            cells += f"<td>{100 * (values[1] - values[0]) / values[0]:+.1f}%</td>"
        rows.append(f"<tr><td>{_esc(label)}</td>{cells}</tr>")
    table = ("<table><thead><tr><th>Metric</th>" + head + ("<th>Change</th>" if delta else "")
             + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")
    reasons = []
    for _, k in kernels:
        for r in list(k["stall_share"])[:5]:
            if r not in reasons:
                reasons.append(r)
    colors = [f"var(--series-{i + 1})" for i in range(len(reasons))]
    bars = []
    for label, k in kernels:
        segs = "".join(
            f'<span data-tip="{_esc(f"{label} · {r}: {100 * k["stall_share"].get(r, 0):.1f}% of samples")}" '
            f'style="width:{100 * k["stall_share"].get(r, 0):.2f}%;background:{c}"></span>'
            for r, c in zip(reasons, colors)
        )
        bars.append(f'<div class="stallrow"><span class="stalllabel">{_esc(label)}</span>'
                    f'<span class="hbar wide">{segs}</span></div>')
    return (table + "<h3>Warp stall reasons (share of PC samples)</h3>"
            + _legend([(r.replace("_", " "), c) for r, c in zip(reasons, colors)]) + "".join(bars))


def _hotspot_section(kernels: list[tuple[str, dict]], top: int = 10) -> str:
    """Stall-sample share by Python function, one bar column per kernel."""
    kernels = [(l, k) for l, k in kernels if "hotspots" in k]
    if not kernels:
        return ""
    shares = {
        label: {f["function"]: f for f in k["hotspots"]["functions"]} for label, k in kernels
    }
    names = sorted(
        {n for per in shares.values() for n in per},
        key=lambda n: -max(per.get(n, {}).get("sample_share", 0.0) for per in shares.values()),
    )[:top]
    vmax = max((per[n]["sample_share"] for per in shares.values() for n in names if n in per), default=1.0)
    head = "".join(f"<th class='barcell'>{_esc(l)}</th>" for l, _ in kernels)
    rows = []
    for name in names:
        cells = []
        for label, _ in kernels:
            f = shares[label].get(name)
            if f is None:
                cells.append("<td class='barcell'></td>")
                continue
            tip = (f"{label} · {name}: {100 * f['sample_share']:.1f}% of stall samples, "
                   f"{100 * f['instruction_share']:.1f}% of instructions")
            cells.append(
                f"<td class='barcell'><span class='hbar' data-tip='{_esc(tip)}'><span style='width:"
                f"{100 * f['sample_share'] / vmax:.2f}%;background:var(--series-1)'></span></span>"
                f"<span class='pct'>{100 * f['sample_share']:.1f}%</span></td>"
            )
        rows.append(f"<tr><td class='kname'>{_esc(name)}</td>{''.join(cells)}</tr>")
    return ("<table><thead><tr><th>Function (file::name)</th>" + head + "</tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def _production_table(production: list[tuple[str, dict]]) -> str:
    """Production step times, with each 'X + change' row compared with row 'X'."""
    if not production:
        return ""
    means = {l: sum(s["step_timeline_ms"] for s in r["steps"]) / len(r["steps"]) for l, r in production}
    rows = []
    for label, r in production:
        # "X + change" rows compare with row "X" when it exists, so changes can be chained.
        base_label = label.rsplit(" + ", 1)[0]
        change = ""
        if base_label != label and base_label in means:
            change = f"{100 * (means[label] - means[base_label]) / means[base_label]:+.1f}% vs {_esc(base_label)}"
        rows.append(f"<tr><td>{_esc(label)}</td><td>{len(r['steps'])}</td><td>{means[label]:.2f}</td>"
                    f"<td>{change}</td></tr>")
    return ("<table><thead><tr><th>Configuration</th><th>Steps</th><th>Production step (ms)</th>"
            "<th>Change</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def _findings(path: str | None) -> str:
    """The findings file's '- ' lines as an ordered list."""
    if not path:
        return ""
    with open(path, encoding="utf-8") as f:
        items = [line[2:].strip() for line in f if line.startswith("- ")]
    return "<ol class='findings'>" + "".join(f"<li>{_esc(i)}</li>" for i in items) + "</ol>"


_EXTRA_STYLE = """<style>
h3 {{ font-size:15px; margin:20px 0 8px; color:var(--ink-2); }}
.hbar {{ display:flex; height:12px; width:100%; min-width:120px; border-radius:3px; overflow:hidden;
  background:var(--grid); gap:2px; }}
.hbar.wide {{ height:16px; }}
.hbar span {{ display:block; height:100%; }}
.barcell {{ width:40%; }}
tr.group td {{ font-weight:600; color:var(--ink); padding-top:14px; }}
.stallrow {{ display:grid; grid-template-columns:110px 1fr; gap:10px; align-items:center; margin:6px 0; }}
.stalllabel {{ font-size:13px; color:var(--ink-2); }}
.findings {{ margin:0; padding-left:20px; display:grid; gap:8px; }}
.findings li {{ max-width:75ch; }}
</style>
"""


def render(anatomies, kernels, production, findings_path, title, note) -> str:
    """Render the overheads page from labeled anatomies, ncu summaries, production results and
    findings."""
    meta = anatomies[0][1]["metadata"] if anatomies else {}
    scene = meta.get("scene", "")
    body = f"""<main>
<h1>{_esc(title)}</h1>
<p class="sub">{_esc(scene)}</p>{f'<p class="sub">{_esc(note)}</p>' if note else ''}
<h2>Findings</h2>
<div class="card">{_findings(findings_path) or '<p class="note">No findings file given.</p>'}</div>
<h2>Where a production step goes</h2>
<div class="card">{_legend([(l, c) for _, l, c in _BUDGET])}{_budget_chart(anatomies)}
<p class="note">From Nsight Systems traces of CUDA graph replays with per-node tracing. Kernels are the union of node
execution intervals; gaps are idle time between nodes. Boundaries are the windows between subproblems
(full dual update, contact detection and filtering, constraint-set update).</p></div>
<h2>One subproblem boundary, by phase</h2>
<div class="card">{_legend([("Kernels", "var(--series-3)"), ("Gaps before the phase's nodes", "var(--series-2)")])}
{_boundary_table(anatomies)}</div>
<h2>Steady sweep timeline</h2>
<div class="card">{_legend([("Vertex solve (one launch per color)", "var(--series-1)"),
                            ("Dual update", "var(--series-5)"), ("Copies and fills", "var(--other)")])}
{_sweep_timeline([(l, a) for l, a in anatomies if a["sweep"]["nodes"]])}</div>
<h2>Vertex-solve kernel</h2>
<div class="card">{_ncu_section(kernels)}
<p class="note">Nsight Compute, mean over the launches of one sweep. It locks clocks to base and flushes caches,
so durations are longer than in normal runs; compare the columns with each other.</p></div>
<h2>Where the vertex solve stalls, by function</h2>
<div class="card">{_hotspot_section(kernels) or '<p class="note">No line-info profile given.</p>'}
<p class="note">Share of Nsight Compute warp stall samples per Python function, from a profile compiled with line
information. Instructions of inlined Warp builtins are folded into the nearest preceding simulator source line,
their likely caller; this attribution is a heuristic.</p></div>
<h2>Production step timings</h2>
<div class="card">{_production_table(production)}
<p class="note">CUDA events around each graph replay (simulate.py --profile-out with profiling off).</p></div>
</main>
"""
    page = (report._HEAD + _EXTRA_STYLE).format(
        title=_esc(title),
        slot_vars_light="".join(f"--series-{i + 1}:{l};" for i, (l, _) in enumerate(report._SLOTS)),
        slot_vars_dark="".join(f"--series-{i + 1}:{d};" for i, (_, d) in enumerate(report._SLOTS)),
    )
    return page + body + report._TOOLTIP.format()


def main():
    """Command line for the overheads report."""
    parser = argparse.ArgumentParser(description="Report on production step overheads")
    parser.add_argument("--anatomy", action="append", help="LABEL=file.anatomy.json (repeatable).")
    parser.add_argument("--ncu", action="append", help="LABEL=file.ncu.json (repeatable, baseline first).")
    parser.add_argument("--production", action="append", help="LABEL=production.json (repeatable).")
    parser.add_argument("--findings", default=None, help="Text file; lines starting with '- ' are findings.")
    parser.add_argument("--title", default="Solver overheads")
    parser.add_argument("--note", default=None)
    parser.add_argument("-o", "--out", required=True, dest="out")
    args = parser.parse_args()
    page = render(
        _load(args.anatomy),
        _load(args.ncu),
        _load(args.production),
        args.findings,
        args.title,
        args.note,
    )
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
