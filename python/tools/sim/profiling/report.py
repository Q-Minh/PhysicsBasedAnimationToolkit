"""
HTML report comparing per-phase GPU cost across solvers.

Takes one profiling result per solver (``simulate.py --profile kernels --profile-out`` or
``profiling/nsys.py``) and writes a self-contained HTML page: stacked bars of mean GPU busy time per
step by phase, one part-to-whole donut per solver, a phase-by-solver table and the top kernels.

Usage (from the repo root):

    python -m python.tools.sim.profiling.report out/vbd.json out/aaavbd.json ... -o report.html
"""

import argparse
import html
import math

from . import results

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
_ITERATION_SCOPES = ("vbd.sweep", "aaavbd.sweep", "newton.linear_solve")


def _esc(x) -> str:
    return html.escape(str(x))


def _mean_calls(result: dict, leaf_name: str) -> float:
    total = 0
    for step in result["steps"]:
        total += sum(n for p, n in step.get("calls", {}).items() if results.leaf(p) == leaf_name)
    return total / max(len(result["steps"]), 1)


def _iterations_per_step(result: dict) -> tuple[float, str]:
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


def render(result_list: list[dict], note: str | None = None) -> str:
    """Render the comparison page for one phase-timed result per solver, plus optional production
    results (``source: "production"``, graph replay timings) for the same solvers."""
    production = {
        r["metadata"].get("solver"): r for r in result_list if r["source"] == "production"
    }
    result_list = [r for r in result_list if r["source"] != "production"]
    solvers = [r["metadata"].get("solver", f"run {i}") for i, r in enumerate(result_list)]
    by_solver = {s: r for s, r in zip(solvers, result_list)}
    per_solver = {s: results.mean_busy_ms_per_step(r, by_leaf=True) for s, r in by_solver.items()}
    order = _phase_order(per_solver)
    colored = [p for p in order if p != results.UNSCOPED][: len(_SLOTS)]
    folded = {s: _fold(per_solver[s], colored) for s in solvers}
    legend_phases = colored + ([_OTHER] if any(_OTHER in f for f in folded.values()) else [])

    meta = result_list[0]["metadata"]
    sources = sorted({r["source"] for r in result_list})
    rows = []
    for s in solvers:
        r = by_solver[s]
        total = sum(per_solver[s].values())
        iters, unit = _iterations_per_step(r)
        union = _mean(r, "step_union_ms")
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
            f"<td>{'' if not iters else f'{1000 * total / iters:.1f}'}</td></tr>"
        )
    summary = (
        "<table><thead><tr><th>Solver</th><th>Steps</th><th>GPU busy (ms/step)</th>"
        "<th>GPU union (ms/step)</th><th>Production step (ms)</th><th>Activities/step</th>"
        "<th>Iterations/step</th>"
        "<th>µs per iteration</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
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
    donuts = []
    for s in solvers:
        top = sorted(per_solver[s], key=lambda p: -per_solver[s][p])
        keep = [p for p in top if p in colored][: _DONUT_SEGMENTS - 1]
        donuts.append(f'<figure>{_donut(s, _fold(per_solver[s], keep), colored)}'
                      f"<figcaption>{_esc(s)}</figcaption></figure>")
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
        legend=_legend(legend_phases, colored),
        bars=_stacked_bars(solvers, folded, colored),
        donuts="".join(donuts),
        phase_table=phase_table,
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
<h2>GPU busy time per step, by phase</h2>
<div class="card">{legend}{bars}</div>
<h2>Share of each phase</h2>
<div class="card">{legend}<div class="donuts">{donuts}</div></div>
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
    parser = argparse.ArgumentParser(description="Per-phase GPU cost report across solvers")
    parser.add_argument("results", nargs="+", help="Profiling result .json files, one per solver.")
    parser.add_argument("-o", "--out", required=True, help="Output .html.", dest="out")
    parser.add_argument("--note", default=None, help="Context shown under the title.")
    args = parser.parse_args()
    page = render([results.load(p) for p in args.results], args.note)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
