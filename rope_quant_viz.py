#!/usr/bin/env python3
"""
rope_quant_viz.py - visualise integer-cycle quantisation of RoPE frequencies for ONE position axis.

Setup
-----
One axis has `n` positions (e.g. n=32 patches along h) and `planes` rotation planes
(head dim per axis = 2 * planes, e.g. 32 dims -> 16 planes). Plane j has frequency

    omega_j = base ** (-j / planes)          (rad per token)

and makes  cycles_j = omega_j * n / (2*pi)  turns across the edge.

Per-plane rule (same semantics as the `quantize` / `positions` helpers):
    round        k = round(cycles)         -> omega' = 2*pi*k/n   (k=0: plane is blind to position)
    fundamental  like round but k >= 1
    keep         omega unchanged (not n-periodic unless cycles happens to be an integer)
    cos / sin    omega unchanged, but the plane sees position bent onto a circle:
                 X(p) = (cos a + 1) n/2   or   Y(p) = (sin a + 1) n/2,  a = 2*pi*p/n
                 (n-periodic again, but NOT translation invariant)
    <integer>    that many cycles

Position-only kernel (amplitudes A_j = 1, content phases = 0):

    K(m, p) = mean_j cos( phi_j(m) - phi_j(p) ),   phi_j(p) = omega'_j * P_j(p)

2D map for a query patch (r, c):  v[h, w] = ( K(r, h) + K(c, w) ) / 2
(assumes the w axis uses the same rules as the h axis).

Examples
--------
    python rope_quant_viz.py                                  # n=32, base=10000, everything rounded
    python rope_quant_viz.py --n 64 --base 2000 --keep 7-15   # keep planes 7..15 at their original frequency
    python rope_quant_viz.py --keep-zero                      # keep the still-"round" planes that would round to 0 cycles
                                                              # (explicit --keep/--cos/--sin/--rules choices win over it)
    python rope_quant_viz.py --keep-zero --cos 8,10,12 --sin 9,11,13   # low planes bent onto a circle
    python rope_quant_viz.py --rules round,round,keep,...     # one rule per plane, comma separated
    python rope_quant_viz.py --interactive                    # tick boxes to choose keep / round, click maps to move the query

Requires: numpy, matplotlib.
"""
from __future__ import annotations

import argparse
import math
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

TWO_PI = 2 * math.pi
CORAL, TEAL, GRAY, PURPLE = "#D85A30", "#1D9E75", "#888780", "#7F77DD"
CMAP = LinearSegmentedColormap.from_list("rope", ["#26215C", "#D85A30", "#FAEEDA"])
CMAP_DIV = LinearSegmentedColormap.from_list("rope_div", ["#26215C", "#F5F5F0", "#D85A30"])
NAMED_RULES = ("round", "fundamental", "keep", "cos", "sin")
SPECIAL = ("cos", "sin")


# ----------------------------------------------------------------------------- core maths
def make_omega(base: float, planes: int) -> np.ndarray:
    return base ** (-np.arange(planes) / planes)


def quantize(omega: np.ndarray, n: int, rules: list[str]) -> np.ndarray:
    """Move each frequency onto one whose rotation repeats after n tokens (n*omega' = 2*pi*k)."""
    cycles = omega * n / TWO_PI
    out = []
    for c, r in zip(cycles, rules):
        if r == "round":
            k = math.floor(c + 0.5)
        elif r == "fundamental":
            k = max(math.floor(c + 0.5), 1)
        elif r in ("keep", "cos", "sin"):
            k = c
        else:
            k = float(int(r))
        out.append(k)
    return np.array(out) * TWO_PI / n


def positions(n: int, rules: list[str]) -> np.ndarray:
    """[n, planes]: what each plane multiplies its frequency with."""
    p = np.arange(n, dtype=float)
    a = p * TWO_PI / n
    cols = []
    for r in rules:
        if r == "cos":
            cols.append((np.cos(a) + 1) * n / 2)
        elif r == "sin":
            cols.append((np.sin(a) + 1) * n / 2)
        else:
            cols.append(p)
    return np.stack(cols, axis=-1)


def kernel_matrix(omega_q: np.ndarray, n: int, rules: list[str]) -> np.ndarray:
    """K[m, p] = mean over planes of cos(phi_j(m) - phi_j(p))."""
    phi = positions(n, rules) * omega_q
    return np.cos(phi[:, None, :] - phi[None, :, :]).mean(-1)


def kernel_delta(omega_q: np.ndarray, n: int):
    """Translation-invariant case: K as a function of the offset d = m - p, d in [-(n-1), n-1]."""
    d = np.arange(-(n - 1), n)
    return d, np.cos(np.outer(d, omega_q)).mean(1)


def residual(K: np.ndarray) -> np.ndarray:
    """K minus its average along each diagonal (m - p = const). Zero iff K is translation invariant."""
    n = K.shape[0]
    R = np.zeros_like(K)
    for k in range(-(n - 1), n):
        m = np.arange(max(0, k), min(n, n + k))
        p = m - k
        R[m, p] = K[m, p] - K[m, p].mean()
    return R


def map_2d(K: np.ndarray, r: int, c: int) -> np.ndarray:
    return (K[r, :][:, None] + K[c, :][None, :]) / 2


# ----------------------------------------------------------------------------- rule selection
def parse_idx(spec: str, planes: int) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    bad = [i for i in out if not 0 <= i < planes]
    if bad:
        raise SystemExit(f"plane indices out of range 0..{planes - 1}: {bad}")
    return out


def build_rules(args, omega: np.ndarray) -> list[str]:
    planes = len(omega)
    rules = ["round"] * planes
    if args.rules:
        rules = [r.strip() for r in args.rules.split(",")]
        if len(rules) != planes:
            raise SystemExit(f"--rules has {len(rules)} entries, expected {planes}")
    for name in ("fundamental", "keep", "cos", "sin"):
        spec = getattr(args, name)
        if spec:
            for i in parse_idx(spec, planes):
                rules[i] = name
    if args.keep_zero:
        cyc = omega * args.n / TWO_PI
        for j, c in enumerate(cyc):
            if math.floor(c + 0.5) == 0 and rules[j] == "round":
                rules[j] = "keep"
    for r in rules:
        if r not in NAMED_RULES:
            int(r)  # raises ValueError for anything that is neither a rule name nor an integer
    return rules


# ----------------------------------------------------------------------------- drawing
def draw_freq(ax, omega, omega_q, n, rules):
    ax.clear()
    j = np.arange(len(omega))
    c = omega * n / TWO_PI
    cq = omega_q * n / TWO_PI
    floor = c.min() / 4
    ax.set_yscale("log")
    for jj in j:
        r = rules[jj]
        kept = r in ("keep", "cos", "sin")
        y = cq[jj] if cq[jj] > 1e-9 else floor
        if not kept:
            ax.plot([jj, jj], [c[jj], y], color=CORAL, alpha=0.45, lw=1.5, zorder=1)
        color = TEAL if r == "keep" else PURPLE if r in SPECIAL else CORAL
        marker = "v" if cq[jj] <= 1e-9 else "o"
        ax.scatter([jj], [y], s=28 if kept else 55, color=color, marker=marker, zorder=3)
    ax.scatter(j, c, s=55, facecolors="none", edgecolors=GRAY, linewidths=1.5, zorder=2, label="original")
    ticks = [t for t in (1e-3, 1e-2, 1e-1, 1, 10, 100) if floor < t < c.max() * 2]
    ax.set_yticks([floor] + ticks)
    ax.set_yticklabels(["0"] + [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(floor / 1.8, c.max() * 2)
    err = [abs(omega_q[i] - omega[i]) * (n - 1) if rules[i] not in SPECIAL else float("nan") for i in j]
    ax.set_xticks(j)
    ax.set_xticklabels([f"{i}\n{e:.1f}" if e == e else f"{i}\n-" for i, e in zip(j, err)], fontsize=7)
    ax.set_xlabel("plane j  /  phase error at max offset (rad)", fontsize=8)
    ax.set_title(f"cycles across the edge (n={n})   hollow = original, filled = after", fontsize=9)
    ax.grid(axis="y", alpha=0.25)


def draw_kernel(ax, omega, omega_q, n, rules, Ko, Kq, r):
    ax.clear()
    if not any(x in SPECIAL for x in rules):
        d, ko = kernel_delta(omega, n)
        _, kq = kernel_delta(omega_q, n)
        ax.plot(d, ko, color=GRAY, lw=2, label="original K(offset)")
        ax.plot(d, kq, color=CORAL, lw=2.2, label="current K(offset)")
        for dd in (n - 1, -1):
            ax.scatter([dd], [kq[dd + n - 1]], s=60, facecolors="none", edgecolors=CORAL, linewidths=2, zorder=3)
        ax.set_xlabel("offset  m - p  (circles: offsets n-1 and -1)", fontsize=8)
    else:
        x = np.arange(n)
        ax.plot(x, Ko[r], color=GRAY, lw=2, label=f"original K({r}, p)")
        ax.plot(x, Kq[r], color=CORAL, lw=2.2, label=f"current K({r}, p)")
        ax.axvline(r, color=GRAY, lw=1, ls="--")
        ax.set_xlabel(f"key position p   (query at m = {r}; kernel is no longer a function of m - p alone)", fontsize=8)
    ax.set_title("position-only kernel", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)


def draw_maps(ax0, ax1, Ko, Kq, r, c, n):
    vo, vq = map_2d(Ko, r, c), map_2d(Kq, r, c)
    vmin = min(vo.min(), vq.min())
    for ax, v, title in ((ax0, vo, "original"), (ax1, vq, "current")):
        ax.clear()
        ax.imshow(v, cmap=CMAP, vmin=vmin, vmax=1, extent=(-0.5, n - 0.5, n - 0.5, -0.5), interpolation="nearest")
        ax.plot([c], [r], "s", mfc="none", mec="white", ms=9, mew=1.6)
        ax.set_title(f"{title}: position-only map, query (r={r}, c={c})", fontsize=9)
        ax.set_xlabel("w")
        ax.set_ylabel("h")


def draw_resid(ax, Kq):
    ax.clear()
    R = residual(Kq)
    mx = float(np.abs(R).max())
    if mx < 1e-9:
        ax.text(0.5, 0.5, "translation invariant\nresidual = 0", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.imshow(R, cmap=CMAP_DIV, vmin=-mx, vmax=mx, interpolation="nearest")
        ax.set_xlabel("p")
        ax.set_ylabel("m")
    ax.set_title("shift-invariance residual R[m, p]", fontsize=9)
    return float(np.sqrt((R ** 2).mean())), mx


def draw_text(ax, omega, omega_q, n, base, rules, Kq, rms, mx):
    ax.clear()
    ax.axis("off")
    cq = omega_q * n / TWO_PI
    c = omega * n / TWO_PI
    zero = [j for j, r in enumerate(rules) if r not in ("keep",) + SPECIAL and cq[j] < 0.5]
    ks = sorted({int(round(cq[j])) for j, r in enumerate(rules) if r not in ("keep",) + SPECIAL and cq[j] >= 0.5}, reverse=True)
    nonint = [j for j, r in enumerate(rules) if r == "keep" and abs(c[j] - round(c[j])) > 1e-9]
    if not any(x in SPECIAL for x in rules):
        _, kq = kernel_delta(omega_q, n)
        wrap = f"K({n - 1}) / K(-1) = {kq[-1]:.3f} / {kq[n - 2]:.3f}"
    else:
        wrap = f"K(0, {n - 1}) / K(0, 1) = {Kq[0, n - 1]:.3f} / {Kq[0, 1]:.3f}"
    tags = [f"{j}:{r[:3]}" for j, r in enumerate(rules)]
    rule_rows = ["  " + " ".join(tags[i:i + 6]) for i in range(0, len(tags), 6)]
    lines = [
        f"n = {n},  base = {base:g},  planes = {len(omega)}",
        "rules (j:rule):",
        *rule_rows,
        "",
        f"rounded to 0 (blind to position): {len(zero)} planes",
        f"non-zero whole-cycle counts: {{{', '.join(map(str, ks))}}}",
        ("strictly n-periodic: yes" if not nonint else f"strictly n-periodic: NO ({len(nonint)} kept plane(s) not whole cycles)"),
        wrap,
        ("shift-invariance residual: 0" if mx < 1e-9 else f"shift-invariance residual: rms {rms:.4f}, max {mx:.3f}"),
    ]
    body = "\n".join(textwrap.fill(l, 46, subsequent_indent="    ") if l.strip() and not l.startswith("  ") else l for l in lines)
    ax.text(0.0, 1.0, body, va="top", ha="left", fontsize=8.5, family="monospace", transform=ax.transAxes)


def render(axes, omega, n, base, rules, query):
    omega_q = quantize(omega, n, rules)
    Ko = kernel_matrix(omega, n, ["keep"] * len(omega))
    Kq = kernel_matrix(omega_q, n, rules)
    r, c = query
    draw_freq(axes["freq"], omega, omega_q, n, rules)
    draw_kernel(axes["kern"], omega, omega_q, n, rules, Ko, Kq, r)
    draw_maps(axes["m0"], axes["m1"], Ko, Kq, r, c, n)
    rms, mx = draw_resid(axes["res"], Kq)
    draw_text(axes["text"], omega, omega_q, n, base, rules, Kq, rms, mx)


def build_figure(left: float):
    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 3, left=left, right=0.98, top=0.92, bottom=0.08, wspace=0.3, hspace=0.45)
    axes = {
        "freq": fig.add_subplot(gs[0, 0]),
        "kern": fig.add_subplot(gs[0, 1]),
        "text": fig.add_subplot(gs[0, 2]),
        "m0": fig.add_subplot(gs[1, 0]),
        "m1": fig.add_subplot(gs[1, 1]),
        "res": fig.add_subplot(gs[1, 2]),
    }
    return fig, axes


# ----------------------------------------------------------------------------- interactive mode
def run_interactive(args, omega, rules, query):
    from matplotlib.widgets import CheckButtons

    n = args.n
    fig, axes = build_figure(left=0.17)
    cyc = omega * n / TWO_PI
    labels = [f"j={j}   c={c:.2f}" for j, c in enumerate(cyc)]
    rax = fig.add_axes([0.01, 0.08, 0.13, 0.84])
    rax.set_title("checked = keep\nunchecked = round", fontsize=9)
    chk = CheckButtons(rax, labels, [r in ("keep",) + SPECIAL for r in rules])
    for lbl in chk.labels:
        lbl.set_fontsize(8)
    state = {"query": list(query)}

    def redraw(_=None):
        cur = ["keep" if s else "round" for s in chk.get_status()]
        render(axes, omega, n, args.base, cur, tuple(state["query"]))
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes in (axes["m0"], axes["m1"]) and event.xdata is not None:
            state["query"] = [int(min(n - 1, max(0, round(event.ydata)))), int(min(n - 1, max(0, round(event.xdata))))]
            redraw()

    chk.on_clicked(redraw)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.suptitle("RoPE integer-cycle quantisation (interactive: tick planes to keep, click a map to move the query)", fontsize=11)
    redraw()
    plt.show()


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Visualise integer-cycle quantisation of RoPE frequencies for one axis.")
    ap.add_argument("--n", type=int, default=32, help="edge length in tokens (default 32)")
    ap.add_argument("--base", type=float, default=10000, help="RoPE base theta (default 10000)")
    ap.add_argument("--planes", type=int, default=16, help="rotation planes per axis = dims/2 (default 16)")
    ap.add_argument("--rules", help="one rule per plane, comma separated (round|fundamental|keep|cos|sin|<int>)")
    ap.add_argument("--keep", help="planes to keep at original frequency, e.g. '5-15,2'")
    ap.add_argument("--fundamental", help="planes to round with k >= 1")
    ap.add_argument("--cos", help="planes using the cos circle coordinate")
    ap.add_argument("--sin", help="planes using the sin circle coordinate")
    ap.add_argument("--keep-zero", action="store_true", help="keep every plane that would round to 0 cycles")
    ap.add_argument("--query", help="'row,col' of the query patch in the 2D maps (default n/16,n/16)")
    ap.add_argument("--out", default="rope_quant.png", help="output image path (default rope_quant.png)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--show", action="store_true", help="also open a window")
    ap.add_argument("--interactive", action="store_true", help="tick boxes to choose keep/round (needs a GUI backend)")
    args = ap.parse_args(argv)

    omega = make_omega(args.base, args.planes)
    rules = build_rules(args, omega)
    if args.query:
        r, c = (int(x) for x in args.query.split(","))
    else:
        r = c = max(0, round(args.n / 16))
    query = (min(args.n - 1, max(0, r)), min(args.n - 1, max(0, c)))

    if args.interactive:
        run_interactive(args, omega, rules, query)
        return

    fig, axes = build_figure(left=0.06)
    render(axes, omega, args.n, args.base, rules, query)
    fig.suptitle(f"RoPE integer-cycle quantisation, one axis (n={args.n}, base={args.base:g})", fontsize=12)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"saved {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
