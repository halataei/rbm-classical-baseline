"""Figures for the Matters Arising and its Supplementary Information.
Inputs: the result files of the runs (paths below). Outputs: fig1_timing.pdf, fig2_accuracy.pdf,
figS1_seeds.pdf, figS2_budget.pdf (and .png previews) in figs_out/."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")                      # put the result files listed below here
NSL_Q = os.path.join(RES, "summary_quantile.json")
NSL_BENCH = os.path.join(RES, "nsl_benchmark.json")
CSE_GPU = {1: os.path.join(RES, "gpu_cse_k1_20261120_073100.json"),
           10: os.path.join(RES, "gpu_cse_k10_20261120_073100.json")}
CSE_BENCH = os.path.join(RES, "cse_benchmark.json")
OUT = os.path.join(HERE, "..", "figures")
os.makedirs(OUT, exist_ok=True)

# palette (reference instance, light mode): slots 1-3 validated all-pairs
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
REF = "#9c9a94"          # neutral grey for the published values of ref. 1

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round", "pdf.fonttype": 42,
})

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# ------------------------------------------------------------------ Fig. 1: timing
def fig_timing():
    rows = ["Ref. 1, CPU, 1 core", "Ref. 1, CPU, 128 threads", "Ref. 1, D-Wave QPU (10 samples)",
            "This work, CPU, 1 chain", "This work, CPU, 10 chains", "This work, CPU, 128 chains"]
    nsl = [70, 35, 16, 0.136, 0.198, 0.795]
    cse = [1100, 710, 17, 1.754, 3.660, 22.564]
    cols = [REF, REF, ORANGE, BLUE, BLUE, BLUE]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.35), sharey=True)
    for ax, vals, title in [(axes[0], nsl, "NSL-KDD model (85 + 30 units, $k$ = 10)"),
                            (axes[1], cse, "CSE-CIC-IDS2018 model (156 + 90 units, $k$ = 100)")]:
        y = np.arange(len(rows))[::-1]
        ax.barh(y, [v - 0.05 for v in vals], height=0.56, color=cols, left=0.05)
        ax.set_xscale("log"); ax.set_xlim(0.05, 8000)
        ax.set_xticks([0.1, 1, 10, 100, 1000])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
        ax.grid(axis="x", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        for yi, v in zip(y, vals):
            lab = f"{v:.0f} ms" if v >= 10 else f"{v:.2g} ms" if v >= 1 else f"{v:.2f} ms"
            ax.text(v * 1.2, yi, lab, va="center", ha="left", fontsize=7, color=INK2)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("Time per negative phase (ms, log scale)")
        ax.axhline(2.5, color=MUTED, linewidth=0.5)
    axes[0].set_yticks(np.arange(len(rows))[::-1]); axes[0].set_yticklabels(rows, color=INK)
    save(fig, "fig1_timing")

# ------------------------------------------------------------------ helpers for curves
def nsl_curves():
    d = json.load(open(NSL_Q)); out = {}
    for k in (1, 10):
        runs = [r for r in d if r["k"] == k]
        ep = sorted(map(int, runs[0]["curve"]))
        acc = np.array([[r["curve"][str(e)]["acc"] for e in ep] for r in runs])
        vote = np.array([[r["curve"][str(e)]["acc_vote"] for e in ep] for r in runs])
        out[k] = (np.array(ep), acc, vote)
    return out

def cse_curves():
    out = {}
    for k in (1, 10):
        R = json.load(open(CSE_GPU[k]))["runs"][f"k={k}"]; ep = sorted(map(int, R["curves"][0]))
        acc = np.array([[c[str(e)]["acc"] for e in ep] for c in R["curves"]])
        vote = np.array([[c[str(e)]["acc_vote100"] for e in ep] for c in R["curves"]])
        out[k] = (np.array(ep), acc, vote)
    return out

def best_classifier_nsl():
    return max(m["acc"] for m in json.load(open(NSL_BENCH))["models"])

def best_classifier_cse():
    best = 0
    for m in json.load(open(CSE_BENCH))["results"]:
        if m["kind"] != "classifier": continue
        prec, npv, acc = m["TP"], m["TN"], m["acc"]
        q = (npv - acc) / (npv - prec); p = q * prec + (1 - q) * m["FN"]
        best = max(best, (q * prec / p + (1 - q) * npv / (1 - p)) / 2)
    return best

# ------------------------------------------------------------------ Fig. 2: accuracy
def fig_accuracy():
    nsl, cse = nsl_curves(), cse_curves()
    panels = [("NSL-KDD (official test set)", nsl, (0.906, 0.938), 0.937, best_classifier_nsl(), 2000, "10 steps"),
              ("CSE-CIC-IDS2018 (balanced test set)", cse, (0.800, 0.924), 0.854, best_classifier_cse(), 2000, "100 steps")]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=True)
    for ax, (title, C, band, dwave, bestc, emax, vsteps) in zip(axes, panels):
        ax.axhspan(*band, color=REF, alpha=0.22, linewidth=0)
        if "NSL" in title:
            ax.text(1.05, band[1] + 0.004, "Ref. 1, classical RBM ($k$ = 1–1000)", fontsize=6.8, color=INK2, va="bottom")
        else:
            ax.text(emax * 0.98, band[1] - 0.004, "Ref. 1, classical RBM ($k$ = 1–1000)", fontsize=6.8, color=INK2,
                    va="top", ha="right")
        ax.axhline(dwave, color=INK2, linewidth=1.0, linestyle=(0, (4, 2)))
        ax.text(emax * 0.98, dwave + 0.004, "Ref. 1, D-Wave", fontsize=6.8, color=INK2, va="bottom", ha="right")
        ax.axhline(bestc, color=MUTED, linewidth=1.0, linestyle=(0, (1, 1.5)))
        ax.text(emax * 0.98, bestc + 0.004, "best standard classifier (this work)", fontsize=6.8, color=INK2,
                va="bottom", ha="right")
        ep, acc10, vote10 = C[10]; ep1, acc1, _ = C[1]
        ax.fill_between(ep, acc10.min(0), acc10.max(0), color=BLUE, alpha=0.12, linewidth=0)
        ax.plot(ep, acc10.mean(0), color=BLUE, linewidth=1.6, label="CD-10, exact read-out (mean; band: range over 5 seeds)")
        ax.plot(ep1, acc1.mean(0), color=AQUA, linewidth=1.6, label="CD-1, exact read-out (mean)")
        ax.plot(ep, vote10.mean(0), color=ORANGE, linewidth=1.6, label="CD-10, read-out of ref. 1 (mean)")
        ax.set_xscale("log"); ax.set_xlim(1, emax); ax.set_ylim(0.45, 1.0)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
        ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
        ax.set_title(title, loc="left", color=INK); ax.set_xlabel("Epoch (one pass over the training set; log scale)")
        ax.text(0.98, 0.03, f"vote after {vsteps}", transform=ax.transAxes, fontsize=6.5, color=MUTED, ha="right")
    axes[0].set_ylabel("Test accuracy")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02),
               title="This work (RBM as specified in ref. 1)", title_fontsize=7)
    save(fig, "figS1_accuracy")

# ------------------------------------------------------------------ Fig. S1: individual seeds
def fig_seeds():
    nsl, cse = nsl_curves(), cse_curves()
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.6), sharey=True)
    shades = ["#1d5fae", "#2a78d6", "#5c97e0", "#8db6ea", "#b7d0f1"]
    for j, (name, C, emax) in enumerate([("NSL-KDD", nsl, 2000), ("CSE-CIC-IDS2018", cse, 2000)]):
        for i, k in enumerate((1, 10)):
            ax = axes[i, j]; ep, acc, _ = C[k]
            for s in range(acc.shape[0]):
                ax.plot(ep, acc[s], color=shades[s], linewidth=1.0, label=f"seed {s}")
            ax.set_xscale("log"); ax.set_xlim(1, emax); ax.set_ylim(0.4, 1.0)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
            ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
            ax.set_title(f"{name}, CD-{k}, exact read-out", loc="left", color=INK)
            if i == 1: ax.set_xlabel("Epoch (log scale)")
            if j == 0: ax.set_ylabel("Test accuracy")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "figS2_seeds")

# ------------------------------------------------------------------ Fig. S2: budget
def fig_budget():
    day = 86400.0
    tr = 0.8 * 3040074
    items = [("NSL-KDD, D-Wave, 1 run", 117260 / 10 * 0.016 * 2000 / day, ORANGE),
             ("CSE-CIC-IDS2018, D-Wave, 1 run", tr / 10 * 0.017 * 2000 / day, ORANGE),
             ("CSE-CIC-IDS2018, D-Wave, 5 runs", 5 * tr / 10 * 0.017 * 2000 / day, ORANGE),
             ("CSE-CIC-IDS2018, CPU, CD-100, 1 run", tr / 128 * 0.71 * 2000 / day, REF),
             ("CSE-CIC-IDS2018, CPU, CD-1000, 1 run", tr / 128 * 7.1 * 2000 / day, REF),
             ("CSE-CIC-IDS2018, CPU, Fig. 5 of ref. 1\n(CD-100 and CD-1000, 10 runs each)",
              10 * tr / 128 * (0.71 + 7.1) * 2000 / day, REF)]
    with plt.rc_context({"font.size": 9, "ytick.labelsize": 8.5, "axes.labelsize": 8.5}):
        fig, ax = plt.subplots(figsize=(7.1, 3.1))
        y = np.arange(len(items))[::-1]
        vals = [v for _, v, _ in items]
        ax.barh(y, vals, height=0.6, color=[c for *_, c in items])
        ax.set_xscale("log"); ax.set_xlim(1, 3e5)
        for yi, v in zip(y, vals):
            lab = f"{v:.0f} days" if v < 365 else f"{v / 365:.1f} years"
            ax.text(v * 1.25, yi, lab, va="center", fontsize=8.5, color=INK)
        for d, lab in [(1, "1 day"), (30, "1 month"), (365, "1 year"), (3650, "10 years"), (36500, "100 years")]:
            ax.axvline(d, color=GRID, linewidth=0.8, zorder=0)
            ax.text(d, len(items) - 0.4, lab, fontsize=7.5, color=MUTED, ha="center")
        ax.set_yticks(y); ax.set_yticklabels([n for n, *_ in items], color=INK); ax.tick_params(axis="y", length=0)
        ax.set_xticks([]); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); ax.spines["bottom"].set_visible(False)
        ax.set_xlabel("Computing time implied by the published protocol (log scale)")
        save(fig, "fig2_budget")

if __name__ == "__main__":
    fig_timing(); fig_accuracy(); fig_seeds(); fig_budget()
    print("done", sorted(os.listdir(OUT)))
