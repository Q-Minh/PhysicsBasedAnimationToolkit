"""
HTML report comparing per-phase GPU cost across solvers.

Takes one profiling result per solver (``simulate.py --profile kernels --profile-out`` or
``profiling/nsys.py``), plus optional production timings, and writes a self-contained HTML page: a
summary table (busy and union time, production step time, iterations, cost per iteration), stacked
bars of mean GPU time per step by phase, one part-to-whole donut per solver, the distribution of step
and phase times, the steps grouped by percentiles of their GPU time and of their peak node speed (phase
donuts, early exits, iterations and residual ratios per group), the phases over the run, a scope tree
per solver with total (inclusive) and self times, a phase-by-solver table and the top GPU activities.

Phase times are unions over streams (work running concurrently on several streams counts once), or busy
time for results recorded without unions. Solve outcomes need the statuses ``simulate.py`` records, and
peak speeds need ``profiling/trajectory.py``; sections without their data are left out.

Usage (from the repo root):

    python -m python.tools.sim.profiling.report out/vbd.json out/aaavbd.json ... -o report.html [--slices 10,90]
"""

import argparse
import html
import math

import numpy as np

from . import results, step_stats

# Categorical slots (light, dark), assigned to phases in fixed order of total cost; the rest fold
# into "Other". Values from the dataviz reference palette, validated for adjacent pairs.
_SLOTS = [
    ("#2a78d6", "#3987e5"),
    ("#eb6834", "#d95926"),
    ("#1baf7a", "#199e70"),
    ("#eda100", "#c98500"),
    ("#e87ba4", "#d55181"),
    ("#008300", "#008300"),
    ("#6250d6", "#9085e9"),
]
_OTHER = "Other"
_DONUT_SEGMENTS = 6
# Scopes whose entries count one solver iteration, for the per-iteration normalization.
_ITERATION_SCOPES = step_stats.ITERATION_SCOPES
_TIMESERIES_BINS = 300
# Per-step quantities the steps are grouped by: key in step_stats.per_step, title, unit.
_SLICE_BY = (
    ("time_ms", "GPU time per step", "ms"),
    ("peak_speed", "peak node speed", "m/s"),
)


def _esc(x) -> str:
    return html.escape(str(x))


def _mean_calls(result: dict, leaf_name: str) -> float:
    total = 0
    for step in result["steps"]:
        total += sum(n for p, n in step.get("calls", {}).items() if results.leaf(p) == leaf_name)
    return total / max(len(result["steps"]), 1)


def _iterations_per_step(result: dict) -> tuple[float, str]:
    """Mean solver iterations per step (VBD sweeps or Newton iterations) and their unit."""
    for name in _ITERATION_SCOPES:
        n = _mean_calls(result, name)
        if n > 0:
            return n, "Newton iteration" if name.startswith("newton") else "VBD sweep"
    return 0.0, ""


def _mean(result: dict, key: str) -> float | None:
    values = [step[key] for step in result["steps"] if key in step]
    return sum(values) / len(values) if values else None


def _phase_order(per_solver: dict[str, dict[str, float]]) -> list[str]:
    totals: dict[str, float] = {}
    for phases in per_solver.values():
        for phase, ms in phases.items():
            totals[phase] = totals.get(phase, 0.0) + ms
    return sorted(totals, key=lambda p: -totals[p])


def _fold(phases: dict[str, float], keep: list[str]) -> dict[str, float]:
    out = {p: phases.get(p, 0.0) for p in keep}
    other = sum(ms for p, ms in phases.items() if p not in keep)
    if other > 0.0:
        out[_OTHER] = other
    return out


def _color(phase: str, colored: list[str]) -> str:
    return f"var(--series-{colored.index(phase) + 1})" if phase in colored else "var(--other)"


def _stacked_bars(solvers, folded, colored) -> str:
    """Horizontal stacked bars of mean busy ms/step by phase, one bar per solver."""
    width, bar_h, gap, label_w, right = 760, 28, 18, 150, 70
    vmax = max(sum(f.values()) for f in folded.values()) or 1.0
    step = 10 ** math.floor(math.log10(vmax))
    tick = next(t * step for t in (1, 2, 5, 10) if vmax / (t * step) <= 6)
    plot_w = width - label_w - right
    height = len(solvers) * (bar_h + gap) + 30
    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
           f'aria-label="Mean GPU busy time per step by phase, per solver">']
    t = 0.0
    while t <= vmax * 1.0001:
        x = label_w + plot_w * t / vmax
        out.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="0" y2="{height - 22}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{height - 6}" text-anchor="middle">'
                   f'{t:g}</text>')
        t += tick
    for i, solver in enumerate(solvers):
        y = i * (bar_h + gap) + 4
        out.append(f'<text class="label" x="{label_w - 10}" y="{y + bar_h / 2 + 4}" '
                   f'text-anchor="end">{_esc(solver)}</text>')
        x = float(label_w)
        total = sum(folded[solver].values())
        segments = [(p, ms) for p, ms in folded[solver].items() if ms > 0.0]
        for j, (phase, ms) in enumerate(segments):
            w = plot_w * ms / vmax
            last = j == len(segments) - 1
            drawn = max(w - (0 if last else 2), 0.5)
            r = 4 if last else 0
            tip = f"{solver} · {phase}: {ms:.3f} ms/step ({100 * ms / total:.1f}%)"
            out.append(f'<path class="mark" data-tip="{_esc(tip)}" fill="{_color(phase, colored)}" '
                       f'd="{_bar_path(x, y, drawn, bar_h, r)}"/>')
            x += w
        out.append(f'<text class="value" x="{x + 6:.1f}" y="{y + bar_h / 2 + 4}">'
                   f'{total:.2f} ms</text>')
    out.append("</svg>")
    return "".join(out)


def _bar_path(x, y, w, h, r) -> str:
    r = min(r, w / 2, h / 2)
    return (f"M{x:.2f},{y:.2f}h{w - r:.2f}a{r},{r} 0 0 1 {r},{r}v{h - 2 * r:.2f}"
            f"a{r},{r} 0 0 1 {-r},{r}h{-(w - r):.2f}z")


def _donut(solver, phases, colored) -> str:
    """Part-to-whole donut of one solver's phases."""
    size, r_out, r_in = 180, 84, 52
    c = size / 2
    total = sum(phases.values()) or 1.0
    out = [f'<svg class="donut" viewBox="0 0 {size} {size}" role="img" '
           f'aria-label="{_esc(solver)} phase shares">']
    angle = -math.pi / 2
    for phase, ms in phases.items():
        if ms <= 0.0:
            continue
        sweep = 2 * math.pi * ms / total
        tip = f"{solver} · {phase}: {ms:.3f} ms/step ({100 * ms / total:.1f}%)"
        out.append(f'<path class="mark" data-tip="{_esc(tip)}" fill="{_color(phase, colored)}" '
                   f'd="{_arc(c, r_out, r_in, angle, angle + sweep)}"/>')
        angle += sweep
    out.append(f'<text class="donut-total" x="{c}" y="{c + 2}" text-anchor="middle">'
               f'{total:.1f}</text>')
    out.append(f'<text class="tick" x="{c}" y="{c + 18}" text-anchor="middle">ms/step</text>')
    out.append("</svg>")
    return "".join(out)


def _arc(c, r1, r0, a0, a1) -> str:
    if a1 - a0 >= 2 * math.pi - 1e-6:
        a1 = a0 + 2 * math.pi - 1e-4
    large = 1 if a1 - a0 > math.pi else 0
    p = lambda r, a: (c + r * math.cos(a), c + r * math.sin(a))
    (x0, y0), (x1, y1) = p(r1, a0), p(r1, a1)
    (x2, y2), (x3, y3) = p(r0, a1), p(r0, a0)
    return (f"M{x0:.2f},{y0:.2f}A{r1},{r1} 0 {large} 1 {x1:.2f},{y1:.2f}"
            f"L{x2:.2f},{y2:.2f}A{r0},{r0} 0 {large} 0 {x3:.2f},{y3:.2f}Z")


def _legend(phases, colored) -> str:
    items = "".join(
        f'<span class="key"><span class="swatch" style="background:{_color(p, colored)}"></span>'
        f"{_esc(p)}</span>"
        for p in phases
    )
    return f'<div class="legend">{items}</div>'


def _scene_name(scene: str) -> str:
    """``.../trampoline.Dump.h5:fem`` -> ``Trampoline``."""
    stem = scene.rsplit(":", 1)[0].replace("\\", "/").rsplit("/", 1)[-1]
    name = stem.split(".", 1)[0] or "scene"
    return name[0].upper() + name[1:]


def _scope_tree_section(solvers: list[str], by_solver: dict[str, dict]) -> str:
    """One collapsible scope tree per solver, with profiler-style total (inclusive) and self columns."""
    sections = []
    for n, s in enumerate(solvers):
        rows = results.scope_tree(by_solver[s])
        step_total = sum(r["total_ms"] for r in rows if r["depth"] == 0) or 1.0
        body = []
        for r in rows:
            union = "" if r["union_incl_ms"] is None else f"{r['union_incl_ms']:.3f}"
            share = 100 * r["total_ms"] / step_total
            body.append(
                f"<tr><td class='scope' style='padding-left:{10 + 18 * r['depth']}px'>{_esc(r['name'])}</td>"
                f"<td>{r['total_ms']:.3f}<span class='pct'>{share:.1f}%</span></td>"
                f"<td>{r['self_ms']:.3f}</td><td>{union}</td>"
                f"<td>{r['total_launches']:.0f}</td><td>{r['self_launches']:.0f}</td>"
                f"<td>{r['calls']:.0f}</td></tr>"
            )
        body = "".join(body)
        sections.append(
            f"<details{' open' if n == 0 else ''}><summary>{_esc(s)}</summary><table><thead><tr>"
            "<th>Scope</th><th>Total ms/step</th><th>Self ms/step</th><th>Total union ms/step</th>"
            "<th>Total launches</th><th>Self launches</th><th>Calls/step</th></tr></thead>"
            f"<tbody>{body}</tbody></table></details>"
        )
    return "".join(sections)


def _fmt(x, digits: int = 3) -> str:
    return "" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{digits}g}"


def _distribution_section(solvers, tables, production) -> str:
    """Quantiles of the step time per solver (union, span, production timeline) and, collapsed, of each
    phase's time."""
    head = "".join(f"<th>{n}</th>" for n in ("Steps", "Mean", *step_stats.QUANTILE_NAMES))
    rows = []
    for s in solvers:
        series = [("GPU union", tables[s]["time_ms"]), ("Span (eager)", tables[s]["span_ms"])]
        if s in production:
            series.append(("Production", step_stats.per_step(production[s])["time_ms"]))
        for name, values in series:
            d = step_stats.describe(values)
            if d is None:
                continue
            cells = "".join(f"<td>{d[k]:.3f}</td>" for k in ("mean", *step_stats.QUANTILE_NAMES))
            rows.append(f"<tr><td>{_esc(s)} · {_esc(name)}</td><td>{d['n']}</td>{cells}</tr>")
    steps = f"<table><thead><tr><th>Step time (ms)</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    phase_sections = []
    for s in solvers:
        phases = tables[s]["phases"]
        prow = []
        for p in sorted(phases, key=lambda p: -phases[p].mean()):
            d = step_stats.describe(phases[p])
            cells = "".join(f"<td>{d[k]:.3f}</td>" for k in ("mean", *step_stats.QUANTILE_NAMES))
            prow.append(f"<tr><td>{_esc(p)}</td><td>{d['n']}</td>{cells}</tr>")
        phase_sections.append(
            f"<details><summary>{_esc(s)}: phase times per step</summary><table><thead><tr><th>Phase (ms)</th>"
            f"{head}</tr></thead><tbody>{''.join(prow)}</tbody></table></details>"
        )
    return steps + "".join(phase_sections)


def _slices_section(solvers, tables, keep, colored, edges, key, unit) -> str:
    """Per solver: one phase donut per percentile slice of ``key``, and a table of the slices' solve
    outcomes. Empty if no solver has the quantity."""
    blocks = []
    for s in solvers:
        table = tables[s]
        slices = step_stats.percentile_slices(table[key], edges)
        if not slices:
            continue
        donuts, rows = [], []
        for sl in slices:
            summary = step_stats.summarize_slice(table, sl["mask"])
            vrange = (f"{_fmt(sl['vmin'])}–{_fmt(sl['vmax'])} {unit}" if sl["vmin"] is not None else "empty")
            if summary["steps"]:
                name = f"{s} {sl['label']}"
                donuts.append(
                    f"<figure>{_donut(name, _fold(summary['phases'], keep[s]), colored)}"
                    f"<figcaption>{_esc(sl['label'])}<br>{_esc(vrange)}</figcaption></figure>"
                )
            early = summary["early_exit"]
            rows.append(
                f"<tr><td>{_esc(sl['label'])}</td><td>{_esc(vrange)}</td><td>{summary['steps']}</td>"
                f"<td>{_fmt(summary['time_ms'])}</td><td>{_fmt(summary['iterations'])}</td>"
                f"<td>{_fmt(summary['outer'])}</td>"
                f"<td>{'' if early is None else f'{100 * early:.0f}%'}"
                f"<span class='pct'>{summary['solves']} solves</span></td>"
                f"<td>{_fmt(summary['gradient_ratio_median'])}</td><td>{_fmt(summary['gradient_ratio_max'])}</td>"
                f"<td>{_fmt(summary['feasibility_ratio_max'])}</td><td>{_fmt(summary['peak_speed'])}</td></tr>"
            )
        unit_iter = table["unit"] or "iterations"
        blocks.append(
            f"<h3>{_esc(s)}</h3><div class='donuts'>{''.join(donuts)}</div>"
            "<table><thead><tr><th>Slice</th><th>Range</th><th>Steps</th><th>Median step (ms)</th>"
            f"<th>Median {_esc(unit_iter)}</th><th>Median outer</th><th>Early exits</th>"
            "<th>Median gradient ratio</th><th>Worst gradient ratio</th><th>Worst feasibility ratio</th>"
            f"<th>Median peak speed (m/s)</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
        )
    return "".join(blocks)


def _timeseries(solver, table, keep, colored) -> str:
    """Stacked areas of the phase times over the run, averaged over at most :data:`_TIMESERIES_BINS`
    bins of consecutive steps."""
    t = table["t"]
    n = t.shape[0]
    if n < 2:
        return ""
    names = list(keep) + [_OTHER]
    others = [p for p in table["phases"] if p not in keep]
    series = {p: table["phases"][p] for p in keep if p in table["phases"]}
    series[_OTHER] = sum((table["phases"][p] for p in others), np.zeros(n))
    edges = np.linspace(0, n, min(n, _TIMESERIES_BINS) + 1).astype(int)
    binned = {p: np.array([v[a:b].mean() for a, b in zip(edges[:-1], edges[1:])]) for p, v in series.items()}
    xs = np.array([t[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
    width, height, left, bottom, top = 760, 200, 50, 22, 18
    plot_w, plot_h = width - left - 10, height - bottom - top
    total = sum(binned.values())
    vmax = float(total.max()) or 1.0
    x = lambda v: left + plot_w * (v - xs[0]) / max(xs[-1] - xs[0], 1e-9)
    y = lambda v: top + plot_h * (1 - v / vmax)
    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
           f'aria-label="{_esc(solver)} phase times over the run">']
    tick = 10 ** math.floor(math.log10(vmax))
    tick = next(k * tick for k in (1, 2, 5, 10) if vmax / (k * tick) <= 5)
    v = 0.0
    while v <= vmax * 1.0001:
        out.append(f'<line class="grid" x1="{left}" x2="{width - 10}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                   f'<text class="tick" x="{left - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:g}</text>')
        v += tick
    for frac, anchor in ((0.0, "start"), (0.5, "middle"), (1.0, "end")):
        tv = xs[0] + frac * (xs[-1] - xs[0])
        out.append(f'<text class="tick" x="{x(tv):.1f}" y="{height - 6}" text-anchor="{anchor}">step {tv:.0f}</text>')
    out.append(f'<text class="tick" x="{left - 6}" y="{top + 2}" text-anchor="end" dy="-0.2em">ms</text>')
    base = np.zeros_like(xs)
    for p in names:
        if p not in binned or not binned[p].any():
            continue
        upper = base + binned[p]
        pts = [f"{x(a):.1f},{y(b):.1f}" for a, b in zip(xs, upper)]
        pts += [f"{x(a):.1f},{y(b):.1f}" for a, b in zip(xs[::-1], base[::-1])]
        tip = f"{solver} · {p}: mean {binned[p].mean():.3f} ms/step"
        out.append(f'<polygon class="mark" data-tip="{_esc(tip)}" fill="{_color(p, colored)}" '
                   f'points="{" ".join(pts)}"/>')
        base = upper
    out.append("</svg>")
    return "".join(out)


def render(result_list: list[dict], note: str | None = None, edges: tuple[float, ...] = (10.0, 90.0)) -> str:
    """Render the comparison page for one phase-timed result per solver, plus optional production
    results (``source: "production"``, graph replay timings) for the same solvers. ``edges`` are the
    percentiles that group steps into slices."""
    production = {
        r["metadata"].get("solver"): r for r in result_list if r["source"] == "production"
    }
    result_list = [r for r in result_list if r["source"] != "production"]
    solvers = [r["metadata"].get("solver", f"run {i}") for i, r in enumerate(result_list)]
    by_solver = {s: r for s, r in zip(solvers, result_list)}
    tables = {s: step_stats.per_step(r) for s, r in by_solver.items()}
    unions = all(step_stats.has_unions(r) for r in result_list)
    measure = "GPU time (union over streams)" if unions else "GPU busy time"
    per_solver = {s: {p: float(v.mean()) for p, v in tables[s]["phases"].items()} for s in solvers}
    order = _phase_order(per_solver)
    colored = [p for p in order if p != results.UNSCOPED][: len(_SLOTS)]
    folded = {s: _fold(per_solver[s], colored) for s in solvers}
    legend_phases = colored + ([_OTHER] if any(_OTHER in f for f in folded.values()) else [])
    legend_html = _legend(legend_phases, colored)

    meta = result_list[0]["metadata"]
    sources = sorted({r["source"] for r in result_list})
    rows = []
    for s in solvers:
        r = by_solver[s]
        total = sum(results.mean_busy_ms_per_step(r).values())
        iters, unit = _iterations_per_step(r)
        union = _mean(r, "step_union_ms")
        per_iter = union if union is not None else total
        launches = sum(sum(st["launches"].values()) for st in r["steps"]) / max(len(r["steps"]), 1)
        prod = _mean(production[s], "step_timeline_ms") if s in production else None
        prod_mode = production[s]["metadata"].get("execution", "") if s in production else ""
        rows.append(
            f"<tr><td>{_esc(s)}</td><td>{len(r['steps'])}</td><td>{total:.3f}</td>"
            f"<td>{'' if union is None else f'{union:.3f}'}</td>"
            f"<td>{'' if prod is None else f'{prod:.3f}'}"
            f"{f'<span class=pct>{_esc(prod_mode)}</span>' if prod_mode else ''}</td>"
            f"<td>{launches:.0f}</td>"
            f"<td>{f'{iters:.0f} {_esc(unit)}s' if iters else ''}</td>"
            f"<td>{'' if not iters else f'{1000 * per_iter / iters:.1f}'}</td></tr>"
        )
    summary = (
        "<table><thead><tr><th>Solver</th><th>Steps</th><th>GPU busy (ms/step)</th>"
        "<th>GPU union (ms/step)</th><th>Production step (ms)</th><th>Activities/step</th>"
        "<th>Iterations/step</th>"
        f"<th>µs per iteration ({'union' if unions else 'busy'})</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )
    phase_rows = []
    for p in order:
        cells = "".join(
            f"<td>{per_solver[s].get(p, 0.0):.3f}"
            f"<span class='pct'>{100 * per_solver[s].get(p, 0.0) / (sum(per_solver[s].values()) or 1):.1f}%"
            f"</span></td>"
            for s in solvers
        )
        phase_rows.append(f"<tr><td>{_esc(p)}</td>{cells}</tr>")
    phase_table = (
        "<table><thead><tr><th>Phase (innermost scope)</th>"
        + "".join(f"<th>{_esc(s)}</th>" for s in solvers)
        + "</tr></thead><tbody>" + "".join(phase_rows) + "</tbody></table>"
    )
    donuts, keep = [], {}
    for s in solvers:
        top = sorted(per_solver[s], key=lambda p: -per_solver[s][p])
        keep[s] = [p for p in top if p in colored][: _DONUT_SEGMENTS - 1]
        donuts.append(f'<figure>{_donut(s, _fold(per_solver[s], keep[s]), colored)}'
                      f"<figcaption>{_esc(s)}</figcaption></figure>")
    slices = []
    for key, title, unit in _SLICE_BY:
        body = _slices_section(solvers, tables, keep, colored, edges, key, unit)
        if body:
            slices.append(f"<h2>Steps by percentile of {_esc(title)}</h2><div class='card'>{legend_html}{body}</div>")
    timeseries = "".join(
        f"<h3>{_esc(s)}</h3>{_timeseries(s, tables[s], keep[s], colored)}" for s in solvers
    )
    overlaps = [step_stats.union_overlap(tables[s]) for s in solvers]
    overlap = max((o for o in overlaps if o is not None), default=None)
    kernel_sections = []
    for s in solvers:
        r = by_solver[s]
        n = max(len(r["steps"]), 1)
        krows = "".join(
            f"<tr><td>{_esc(results.leaf(k['path']))}</td><td class='kname'>{_esc(k['name'][:90])}</td>"
            f"<td>{k['busy_ms'] / n:.3f}</td><td>{k['count'] / n:.0f}</td></tr>"
            for k in r["kernels"][:12]
        )
        kernel_sections.append(
            f"<details><summary>{_esc(s)}: top GPU activities</summary><table><thead><tr>"
            "<th>Phase</th><th>Activity</th><th>ms/step</th><th>count/step</th></tr></thead>"
            f"<tbody>{krows}</tbody></table></details>"
        )
    meta_items = {
        "Scene": meta.get("scene"),
        "GPU": f"{meta.get('gpu')} (sm_{meta.get('gpu_arch')})",
        "Driver": f"{meta.get('display_driver')} (CUDA {meta.get('cuda_driver')})",
        "Warp": meta.get("warp"),
        "Contact detection": meta.get("cd"),
        "Time step": f"{meta.get('dt')} s × {meta.get('substeps')} substep(s), BDF{meta.get('bdf_order')}",
        "Mesh": f"{meta.get('n_nodes')} nodes, {meta.get('n_elements')} tetrahedra",
        "Timing source": ", ".join(sources),
        "Commit": (meta.get("git_commit") or "")[:10],
    }
    meta_html = "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>" for k, v in meta_items.items())
    source_note = (
        "Busy time sums the GPU duration of every activity attributed to a phase (the innermost "
        "profiling scope open at launch). Steps ran eagerly; host launch overhead and idle gaps "
        "are excluded. "
        + ("Phase times in the charts, distributions and slices are unions over streams: the time a "
           "phase's activities keep the GPU busy, with work running concurrently on several streams "
           "counted once. They exclude graph or kernel launch overhead and the gaps between kernels, "
           "which production runs (graph replay) include. "
           if unions else "")
        + (f"Phase unions add up to {100 * overlap:.2g}% more than the step unions (phases overlapping "
           "in time). " if overlap else "")
        + "Slices group each solver's own steps by percentiles of a per-step quantity; ties at a "
        "percentile fall into the lower slice. Early exits are the solves (one per substep) that passed "
        "their convergence tests within the budget; ratios are the final residuals to their tolerances "
        "(at most 1 passes), worst over a step's substeps. Peak node speed is the largest node "
        "displacement between saved frames over their time interval (profiling/trajectory.py). "
        + ("Nsight Systems times every kernel, including CUB and cupy ones; the union column "
           "does not double-count work running concurrently on several streams. "
           if "nsys" in sources else "")
        + ("The kernels backend times every kernel in-process with CUPTI; the union column does "
           "not double-count work running concurrently on several streams. "
           if "kernels" in sources else "")
        + ("Production step is the GPU time of one step as the simulator normally runs it (CUDA "
           "events around a CUDA graph replay; Newton runs eagerly), so it includes the scheduling "
           "gaps between the graph's many small kernels. "
           if production else "")
    )
    return _PAGE.format(
        title=f"{_scene_name(meta.get('scene', ''))} solver costs",
        note_html=f'<p class="sub">{_esc(note)}</p>' if note else "",
        scene=_esc(meta.get("scene", "")),
        slot_vars_light="".join(f"--series-{i + 1}:{l};" for i, (l, _) in enumerate(_SLOTS)),
        slot_vars_dark="".join(f"--series-{i + 1}:{d};" for i, (_, d) in enumerate(_SLOTS)),
        summary=summary,
        measure=_esc(measure),
        legend=legend_html,
        bars=_stacked_bars(solvers, folded, colored),
        donuts="".join(donuts),
        distribution=_distribution_section(solvers, tables, production),
        slices="".join(slices),
        timeseries=timeseries,
        phase_table=phase_table,
        scope_tree=_scope_tree_section(solvers, by_solver),
        kernels="".join(kernel_sections),
        meta=meta_html,
        note=_esc(source_note),
    )


# Page pieces shared with profiling/overheads.py, as str.format templates. The page body has no document
# skeleton: it opens directly in a browser, and claude.ai artifact publishing adds the skeleton.
_HEAD = """<title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="claude:viewport-fit-sides" content="cover">
<style>
:root {{ color-scheme: light; --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --border:rgba(11,11,11,0.10); --other:#b9b8b1; {slot_vars_light} }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme: dark;
  --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink-2:#c3c2b7; --muted:#898781; --grid:#2c2c2a;
  --border:rgba(255,255,255,0.10); --other:#5c5b57; {slot_vars_dark} }} }}
:root[data-theme="dark"] {{ color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec;
  --ink-2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --border:rgba(255,255,255,0.10); --other:#5c5b57;
  {slot_vars_dark} }}
body {{ margin:0; background:var(--page); color:var(--ink);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:960px; margin:0 auto; padding-block:32px 64px;
  padding-left:max(16px, env(safe-area-inset-left, 0px)); padding-right:max(16px, env(safe-area-inset-right, 0px)); }}
h1 {{ font-size:28px; margin:0 0 4px; }} h2 {{ font-size:18px; margin:36px 0 12px; }}
h3 {{ font-size:15px; margin:20px 0 8px; color:var(--ink-2); }}
.sub {{ color:var(--ink-2); margin:0 0 24px; overflow-wrap:anywhere; }}
.card {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:16px;
  overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; font-size:14px; }}
th, td {{ text-align:right; padding:6px 10px; border-bottom:1px solid var(--grid); white-space:nowrap; }}
th:first-child, td:first-child {{ text-align:left; }} th {{ color:var(--ink-2); font-weight:600; }}
.pct {{ color:var(--muted); margin-left:6px; font-size:12px; }}
.kname {{ text-align:left; white-space:normal; overflow-wrap:anywhere; }}
.legend {{ display:flex; flex-wrap:wrap; gap:6px 16px; margin:0 0 12px; font-size:13px; color:var(--ink-2); }}
.key {{ display:inline-flex; align-items:center; gap:6px; }}
.swatch {{ width:10px; height:10px; border-radius:3px; display:inline-block; }}
svg {{ display:block; width:100%; height:auto; }}
.grid {{ stroke:var(--grid); stroke-width:1; }}
.tick {{ fill:var(--muted); font-size:11px; }}
.label {{ fill:var(--ink); font-size:13px; }} .value {{ fill:var(--ink-2); font-size:12px; }}
.donut-total {{ fill:var(--ink); font-size:20px; font-weight:600; }}
.donut path {{ stroke:var(--surface); stroke-width:2; }}
.mark:hover {{ opacity:0.8; }}
.donuts {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(160px,1fr)); gap:16px; }}
figure {{ margin:0; }} figcaption {{ text-align:center; color:var(--ink-2); font-size:13px; }}
details {{ margin:8px 0; }} summary {{ cursor:pointer; color:var(--ink-2); }}
dl {{ display:grid; grid-template-columns:max-content 1fr; gap:4px 16px; margin:0; font-size:13px; }}
dt {{ color:var(--muted); }} dd {{ margin:0; overflow-wrap:anywhere; }}
.note {{ color:var(--ink-2); font-size:13px; }}
#tip {{ position:fixed; pointer-events:none; background:var(--surface); color:var(--ink);
  border:1px solid var(--border); border-radius:8px; padding:6px 10px; font-size:13px;
  box-shadow:0 4px 16px rgba(0,0,0,0.15); display:none; max-width:320px; }}
</style>
"""

_TOOLTIP = """<div id="tip"></div>
<script>
const tip = document.getElementById("tip");
document.querySelectorAll("[data-tip]").forEach(el => {{
  el.addEventListener("mousemove", e => {{
    tip.textContent = el.dataset.tip; tip.style.display = "block";
    const x = Math.min(e.clientX + 12, window.innerWidth - tip.offsetWidth - 8);
    tip.style.left = x + "px"; tip.style.top = (e.clientY + 12) + "px";
  }});
  el.addEventListener("mouseleave", () => {{ tip.style.display = "none"; }});
}});
</script>
"""

_PAGE = (
    _HEAD
    + """<main>
<h1>{title}</h1>
<p class="sub">{scene}</p>{note_html}
<div class="card">{summary}</div>
<h2>{measure} per step, by phase</h2>
<div class="card">{legend}{bars}</div>
<h2>Share of each phase</h2>
<div class="card">{legend}<div class="donuts">{donuts}</div></div>
<h2>Distribution of step times</h2>
<div class="card">{distribution}</div>
{slices}
<h2>Phases over the run</h2>
<div class="card">{legend}{timeseries}</div>
<h2>Scope tree</h2>
<div class="card">{scope_tree}
<p class="note">Total includes nested scopes; self is the scope's own work. Busy times sum activity durations, so
work on concurrent streams counts once per stream; the total union counts overlapping work once.</p></div>
<h2>Phases by solver (ms/step)</h2>
<div class="card">{phase_table}</div>
<h2>Top GPU activities</h2>
<div class="card">{kernels}</div>
<h2>Run</h2>
<div class="card"><dl>{meta}</dl><p class="note">{note}</p></div>
</main>"""
    + _TOOLTIP
)


def main():
    """Command line for the per-solver cost report."""
    parser = argparse.ArgumentParser(description="Per-phase GPU cost report across solvers")
    parser.add_argument("results", nargs="+", help="Profiling result .json files, one per solver.")
    parser.add_argument("-o", "--out", required=True, help="Output .html.", dest="out")
    parser.add_argument("--note", default=None, help="Context shown under the title.")
    parser.add_argument(
        "--slices",
        default="10,90",
        help="Percentiles that group the steps into slices, comma-separated (default: 10,90, i.e. the lowest "
        "10%%, the middle 80%% and the highest 10%%).",
    )
    args = parser.parse_args()
    edges = tuple(float(e) for e in args.slices.split(","))
    page = render([results.load(p) for p in args.results], args.note, edges)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
