"""fig4_build.py -- FIGURE 4: "The FQ operator family is load-bearing".

Panels
  a  matched-control slope chart      RFF-GNN -> FQ-GNN, per curvature bucket + OVERALL
  b  ablation ladder (decisive)       12 rungs, horizontal bars, log ratio, seed dots
  c  training dynamics                val ratio vs epoch, FQGNN vs RFFGNN, 5 seeds each
  d  amplitude spectra                RFF broadband (96 stems) vs learned FQ comb (48 live)
  e  spectral rigidity S(k)           pair-spacing density, FQ comb vs RFF, Poisson reference

Reads ONLY CSVs (no recomputation).  Style: fig_style.py (locked).  Output: <FQGNN_FIG_OUT>/fig4.{pdf,png}.
Run:  python paper_figures_tables/fig4_build.py     (any cwd)
"""
from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _paths import FIGDATA, OUT, RESULTS  # noqa: E402,F401  (env FQGNN_FIGDATA / FQGNN_FIG_OUT / FQGNN_RESULTS_ROOT)

import fig_style as FS                                          # noqa: E402
from matplotlib.colors import LinearSegmentedColormap           # noqa: E402
from matplotlib.ticker import NullFormatter                     # noqa: E402


def plt_null():
    return NullFormatter()


FS.apply()

BUCKCSV = os.path.join(RESULTS, "runs", "paper_table", "combined_buckets_allmodels.csv")
FQ_EPOCH = os.path.join(RESULTS, "runs", "fqnative_opmstack_5seed", "seed{s}", "epochs.csv")
RFF_EPOCH = os.path.join(RESULTS, "baselines", "RFFGNN", "runs_5seed_gaps",
                         "seed{s}", "epochs.csv")

PINK = FS.C["pink"]
AZURE = FS.C["azure"]
GRAY6E = "#6E6E6E"


def darken(hexcolor, f=0.30):
    h = hexcolor.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02X%02X%02X" % (int(r * (1 - f)), int(g * (1 - f)), int(b * (1 - f)))


def stagger(vals, min_sep):
    """Push label positions apart (in the units given) preserving order; vals sorted ascending."""
    out = list(vals)
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1] + min_sep)
    # recentre toward the originals
    shift = (np.mean(vals) - np.mean(out))
    return [v + shift for v in out]


# ================================================================================================ #
#  DATA                                                                                             #
# ================================================================================================ #
bk = pd.read_csv(BUCKCSV)
BUCKET_ORDER = ["OVERALL"] + FS.BUCKETS
pa = {}                                                          # (model,bucket) -> (mean, sd)
for m in ("FQGNN", "RFFGNN"):
    for b in BUCKET_ORDER:
        r = bk[(bk.model == m) & (bk.bucket == b)].ratio
        assert len(r) == 5, (m, b, len(r))
        pa[(m, b)] = (r.mean(), r.std(ddof=1))

lad = pd.read_csv(os.path.join(FIGDATA, "ablation_ladder.csv"))
# full 12-rung ladder: the two readout-ingredient rungs (no_metric, no_eta) are shown alongside
# the other 10 (both are single-ingredient ablations; sort logic below places them by ratio)
KNOWN_RUNGS = ["full_fqgnn", "free_amps", "free_amps_randinit", "frozen_L", "d1_lift",
               "decoupled_U", "rbf_kernel", "no_dmu", "no_metric", "no_eta",
               "rffgnn", "one_sided"]
lad = lad[lad.variant.isin(KNOWN_RUNGS)].reset_index(drop=True)
lad["seeds"] = lad.ratio_per_seed.apply(lambda s: np.array([float(v) for v in s.split(";")]))

# Welch t vs full for the '≈ full' tags
full_seeds = lad.loc[lad.variant == "full_fqgnn", "seeds"].iloc[0]
try:
    from scipy import stats as st
    def welch_p(x):
        return st.ttest_ind(x, full_seeds, equal_var=False).pvalue
except Exception:                                                # pragma: no cover
    def welch_p(x):
        return float("nan")
lad["p_vs_full"] = lad.seeds.apply(welch_p)
NS = set(lad[(lad.variant != "full_fqgnn") & (lad.p_vs_full > 0.05)].variant)
print("[b] Welch p vs full:", {r.variant: round(r.p_vs_full, 3) for r in lad.itertuples()})
print("[b] 'ns' (≈ full) rungs:", sorted(NS))

# rung order: full pinned top (the reference), the rest ascending by ratio
rest = lad[lad.variant != "full_fqgnn"].sort_values("ratio_mean")
lad_ord = pd.concat([lad[lad.variant == "full_fqgnn"], rest]).reset_index(drop=True)

RUNG_COLOR = {
    "full_fqgnn":         FS.SEQ_PINK[4],
    "free_amps":          FS.SEQ_PINK[3],
    "free_amps_randinit": FS.SEQ_PINK[3],
    "frozen_L":           FS.SEQ_PINK[3],
    "d1_lift":            FS.SEQ_PINK[3],
    "no_eta":             FS.SEQ_PINK[3],
    "decoupled_U":        FS.SEQ_PINK[2],
    "no_metric":          FS.SEQ_PINK[2],
    "rbf_kernel":         FS.SEQ_PINK[1],
    "no_dmu":             FS.SEQ_PINK[0],
    "rffgnn":             AZURE,
    "one_sided":          FS.C["bluegray"],
}
RUNG_LABEL = {"No D-mu velocity": "No Dμ velocity"}         # μ

fq_ep = [pd.read_csv(FQ_EPOCH.format(s=s)) for s in range(5)]
rff_ep = [pd.read_csv(RFF_EPOCH.format(s=s)) for s in range(5)]


def seed_stats(dfs):
    n = max(len(d) for d in dfs)
    mean, lo, hi = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for e in range(n):
        vals = [d.val_ratio.iloc[e] for d in dfs if len(d) > e]
        mean[e], lo[e], hi[e] = np.mean(vals), np.min(vals), np.max(vals)
    return np.arange(n), mean, lo, hi


sp_fq = pd.read_csv(os.path.join(FIGDATA, "fq_spectrum_trained.csv"))
sp_rf = pd.read_csv(os.path.join(FIGDATA, "rff_spectrum.csv"))
stats_df = pd.read_csv(os.path.join(FIGDATA, "spectrum_stats.csv")).set_index("model")
sk_fq = pd.read_csv(os.path.join(FIGDATA, "sk_fq.csv"))
sk_rf = pd.read_csv(os.path.join(FIGDATA, "sk_rff.csv"))

# ================================================================================================ #
#  LAYOUT  (180 x 138 mm, 2 panel bands on the 12-col grid; band-2 panels share top edge + height)   #
# ================================================================================================ #
fig = FS.new_figure(180, 138)

B1_Y0, B1_Y1 = 0.590, 0.940                                      # band 1 axes rect (fig fraction)
B2_Y0, B2_Y1 = 0.100, 0.455                                      # band 2

ax_a = fig.add_axes([0.070, B1_Y0, 0.330, B1_Y1 - B1_Y0])
ax_b = fig.add_axes([0.615, B1_Y0, 0.370, B1_Y1 - B1_Y0])
ax_c = fig.add_axes([0.070, B2_Y0, 0.235, B2_Y1 - B2_Y0])
d_gap = 0.030
d_h = (B2_Y1 - B2_Y0 - d_gap) / 2
ax_d1 = fig.add_axes([0.415, B2_Y0 + d_h + d_gap, 0.235, d_h])   # top strip: RFF
ax_d2 = fig.add_axes([0.415, B2_Y0, 0.235, d_h])                 # bottom strip: FQ
ax_e = fig.add_axes([0.760, B2_Y0, 0.225, B2_Y1 - B2_Y0])

LET_Y1, LET_Y2 = 0.982, 0.500
FS.panel_letter(fig, 0.012, LET_Y1, "a")
FS.panel_letter(fig, 0.400, LET_Y1, "b")
FS.panel_letter(fig, 0.012, LET_Y2, "c")
FS.panel_letter(fig, 0.355, LET_Y2, "d")
FS.panel_letter(fig, 0.700, LET_Y2, "e")

# ================================================================================================ #
#  PANEL a -- matched-control slope chart                                                            #
# ================================================================================================ #
# line colors: single teal ramp (ordered buckets).  SEQ_TEAL[0] (#E8F2EC) is invisible as a 1-pt
# line on white, so the 5 line colors are sampled from the dark 70% of the SAME locked ramp.
tealmap = LinearSegmentedColormap.from_list("teal", FS.SEQ_TEAL)
LINE_TEAL = [tealmap(t) for t in (0.30, 0.475, 0.65, 0.825, 1.0)]
A_COLOR = {b: LINE_TEAL[i] for i, b in enumerate(FS.BUCKETS)}
A_COLOR["OVERALL"] = "#1A1A1A"

ax_a.set_yscale("log")
ends = {}
for b in BUCKET_ORDER:
    c = A_COLOR[b]
    lw = 1.4 if b == "OVERALL" else 1.0
    y0, s0 = pa[("RFFGNN", b)]
    y1, s1 = pa[("FQGNN", b)]
    ax_a.plot([0, 1], [y0, y1], color=c, lw=lw, marker="o", markersize=2.8,
              markeredgewidth=0, zorder=6 if b == "OVERALL" else 4)
    ax_a.errorbar([0, 1], [y0, y1], yerr=[s0, s1], fmt="none", ecolor=c,
                  elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=3)
    ends[b] = y1
ax_a.axhline(1.0, color=FS.C["gcd"], ls=(0, (4, 2.5)), lw=0.7, zorder=1.5)
# single GCD tag, far-left end of the dashed rule, clear of the x=0 markers/caps
ax_a.annotate("GCD", xy=(-0.305, 1.0), fontsize=5.5, color=FS.C["gcd"], ha="left",
              va="bottom", xytext=(0, 1), textcoords="offset points")

# staggered direct end-of-line labels (right side, color-matched)
order = sorted(BUCKET_ORDER, key=lambda b: ends[b])
ylog = stagger([np.log10(ends[b]) for b in order], min_sep=0.058)
for b, yl in zip(order, ylog):
    ax_a.plot([1.03, 1.10], [ends[b], 10 ** yl], lw=0.5, color=A_COLOR[b],
              alpha=0.65, clip_on=False, zorder=2)
    ax_a.annotate(b if b != "OVERALL" else "OVERALL", xy=(1.12, 10 ** yl),
                  fontsize=5.5, color=A_COLOR[b], ha="left", va="center",
                  fontweight="bold" if b == "OVERALL" else "normal",
                  annotation_clip=False)

ax_a.set_xlim(-0.32, 1.62)
ax_a.set_ylim(0.42, 7.5)
ax_a.set_xticks([0, 1], ["RFF basis", "FQ operators"])
ax_a.set_yticks([0.5, 1, 2, 5], ["0.5", "1", "2", "5"])
ax_a.yaxis.set_minor_formatter(plt_null())
ax_a.set_ylabel("Median-error ratio vs GCD (lower better)")
ax_a.set_title("Same recipe, swap the operator")
FS.condition_strip(ax_a, "same 770 test gaps | lines: 5-seed means | caps: ±s.d.", dy=-0.155)

# ================================================================================================ #
#  PANEL b -- ablation ladder                                                                        #
# ================================================================================================ #
XMIN, XMAX = 0.52, 3.9
ax_b.set_xscale("log")
n = len(lad_ord)
ys = np.arange(n)[::-1]                                          # full at TOP
rng = np.random.default_rng(0)
for y, row in zip(ys, lad_ord.itertuples()):
    c = RUNG_COLOR[row.variant]
    edge = dict(edgecolor=darken(c, 0.18), linewidth=0.5) if row.variant == "no_dmu" else \
        dict(edgecolor="none", linewidth=0)
    ax_b.barh(y, row.ratio_mean - XMIN, left=XMIN, height=0.72, color=c, zorder=3, **edge)
    ax_b.errorbar(row.ratio_mean, y, xerr=row.ratio_std, fmt="none", ecolor=darken(c, 0.35),
                  elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=5)
    jit = np.linspace(-0.16, 0.16, 5)
    ax_b.plot(row.seeds, y + jit, ls="none", marker="o", markersize=2.0,
              markerfacecolor="#2B2B2B", markeredgewidth=0, alpha=0.6, zorder=6)
    if row.variant in NS:
        ax_b.annotate("≈ full", xy=(row.ratio_mean * 1.10, y), fontsize=5.5,
                      color=GRAY6E, ha="left", va="center", style="italic")

ax_b.axvline(1.0, color=FS.C["ref_red"], ls=(0, (4, 2.5)), lw=0.8, zorder=2)
ax_b.annotate("GCD floor", xy=(1.05, n - 0.72), fontsize=5.5,
              color=FS.C["ref_red"], ha="left", va="center")

ax_b.set_yticks(ys, [RUNG_LABEL.get(l, l) for l in lad_ord.label], fontsize=6)
ax_b.set_xlim(XMIN, XMAX)
ax_b.set_ylim(-0.65, n - 0.35)
ax_b.set_xticks([0.6, 0.8, 1.0, 1.5, 2.0, 3.0], ["0.6", "0.8", "1", "1.5", "2", "3"])
ax_b.xaxis.set_minor_formatter(plt_null())
ax_b.set_xlabel("Ratio vs GCD (log, lower better)")
ax_b.set_title("Ablation ladder: what carries the win")
FS.condition_strip(ax_b, "5 seeds per rung | same data, recipe, capacity\n"
                         "dots: seeds | caps: ±s.d. | bar shade: FQ structure remaining",
                   dy=-0.20)

# ================================================================================================ #
#  PANEL c -- training dynamics                                                                      #
# ================================================================================================ #
ax_c.set_yscale("log")
for dfs, key, band_c in ((fq_ep, "FQGNN", PINK), (rff_ep, "RFFGNN", AZURE)):
    s = FS.MODEL[key]
    x, m, lo, hi = seed_stats(dfs)
    ax_c.fill_between(x, lo, hi, color=band_c, alpha=0.20, lw=0, zorder=2)
    ax_c.plot(x, m, color=s["color"], ls=s["ls"], lw=s["lw"], zorder=4)
FS.ref_hline(ax_c, 1.0, color=FS.C["gcd"], label="GCD")

Y_STOP = 4.05                                                    # marker row, top of panel
for dfs, key in ((fq_ep, "FQGNN"), (rff_ep, "RFFGNN")):
    stops = [d.epoch.iloc[-1] for d in dfs]
    ax_c.plot(stops, [Y_STOP] * 5, ls="none", marker="v", markersize=3,
              markerfacecolor=FS.MODEL[key]["color"], markeredgewidth=0, alpha=0.85,
              zorder=5, clip_on=False)
rff_stop = rff_ep[0].epoch.iloc[-1]
ax_c.annotate("×5", xy=(rff_stop - 2.5, Y_STOP), fontsize=5, color=AZURE,
              ha="right", va="center")
ax_c.annotate("stop epoch", xy=(max(d.epoch.iloc[-1] for d in fq_ep) + 4, Y_STOP),
              fontsize=5.5, color=GRAY6E, ha="left", va="center")

# direct end-of-line labels (2 series -> no legend)
fq_end_x = fq_ep[0].epoch.iloc[-1]
ax_c.annotate("FQ-GNN (ours)", xy=(21.5, 0.70), fontsize=6, color=PINK,
              ha="left", va="center")
ax_c.annotate("RFF-GNN", xy=(44, 2.62), fontsize=6, color=AZURE, ha="left", va="bottom")

ax_c.set_xlim(-1.5, 61)
ax_c.set_ylim(0.55, 4.6)
ax_c.set_xticks([0, 20, 40, 60])
ax_c.set_yticks([0.6, 1, 2, 4], ["0.6", "1", "2", "4"])
ax_c.yaxis.set_minor_formatter(plt_null())               # no stray '3 x 10^0' minor label
ax_c.set_xlabel("Epoch")
ax_c.set_ylabel("Validation ratio vs GCD (lower better)")
ax_c.set_title("Identical recipe, different operator")
FS.condition_strip(ax_c, "5 seeds each | band: min–max over seeds", dy=-0.175)

# ================================================================================================ #
#  PANEL d -- amplitude spectra (two stacked strips, per-basis-normalized x)                         #
# ================================================================================================ #
rf_fmax = sp_rf.freq_norm.max()
fq_fmax = sp_fq.eff_freq.max()
rf_top10 = 100 * float(stats_df.loc["RFF_seed0", "top10_energy_frac"])
fq_top10 = 100 * float(stats_df.loc["FQGNN_trained", "top10_energy_frac"])

# top strip: RFF broadband (96 stems, learned readout weight as height)
# in-strip text is staggered on two rows (title row / annotation row) so nothing touches
h = sp_rf.readout_w / sp_rf.readout_w.max()
ax_d1.vlines(sp_rf.freq_norm / rf_fmax, 0, h, color=FS.C["bluegray"], lw=0.7)
ax_d1.annotate("Random basis (RFF-GNN)", xy=(0.02, 0.97), xycoords="axes fraction",
               fontsize=6, color=FS.C["axis"], ha="left", va="top")
ax_d1.annotate(f"top-10 stems: {rf_top10:.1f}% of energy", xy=(0.98, 0.80),
               xycoords="axes fraction", fontsize=5.5, color=darken(FS.C["bluegray"], 0.35),
               ha="right", va="top")
ax_d1.annotate("96 stems", xy=(0.02, 0.80),
               xycoords="axes fraction", fontsize=5, color=GRAY6E, ha="left", va="top")

# bottom strip: learned FQ comb (live pink stems + extinct ghosts)
live = sp_fq[sp_fq.abs_c > 0]
dead = sp_fq[sp_fq.abs_c == 0]
GHOST_H = 0.045
ax_d2.vlines(dead.eff_freq / fq_fmax, 0, GHOST_H, color=FS.C["sep"], lw=0.7, zorder=2)
ax_d2.vlines(live.eff_freq / fq_fmax, 0, live.abs_c / live.abs_c.max(),
             color=PINK, lw=0.9, zorder=4)
ax_d2.annotate("Learned FQ comb (ours)", xy=(0.98, 0.97), xycoords="axes fraction",
               fontsize=6, color=FS.C["axis"], ha="right", va="top")
ax_d2.annotate(f"top-10 stems: {fq_top10:.2f}% of energy", xy=(0.98, 0.81),
               xycoords="axes fraction", fontsize=5.5, color=darken(PINK, 0.25),
               ha="right", va="top")
ax_d2.annotate(f"{len(live)} live", xy=(0.98, 0.65),
               xycoords="axes fraction", fontsize=5, color=GRAY6E, ha="right", va="top")
# two-line, right of the tallest live stem (x = 0.53) so no stem crosses the text
if len(dead):                       # Op-M exact-table comb has no extinct modes; annotation
    ax_d2.annotate(f"{len(dead)} extinct modes\n(|c| = 0, position only)",
                   xy=(0.98, GHOST_H + 0.06),
                   xycoords=("axes fraction", "data"), fontsize=5, color=GRAY6E,
                   ha="right", va="bottom", linespacing=1.15)

for ax in (ax_d1, ax_d2):
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.52)                    # headroom: two clear text rows above the tallest stem
    ax.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1"])
ax_d1.set_xticks([0, 0.5, 1.0], [])
ax_d2.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"])
# ONE two-line x-label: main line + smaller gray native-range line (no other text in this area)
ax_d2.set_xlabel("Frequency / per-basis max", fontsize=6.5, labelpad=2)
ax_d2.annotate("(native ranges — RFF: 0.6–17.1, FQ: 5.7–75.9 rad)", xy=(0.5, -0.375),
               xycoords="axes fraction", fontsize=5.5, color=GRAY6E, ha="center", va="top")
ax_d1.set_title("Random basis vs learned comb", pad=3)
fig.text(0.372, (B2_Y0 + B2_Y1) / 2, "|amplitude| (per-strip norm.)", rotation=90,
         fontsize=7, ha="center", va="center", color=FS.C["axis"])

# ================================================================================================ #
#  PANEL e -- structure factor / spectral rigidity                                                   #
# ================================================================================================ #
# per-basis normalization (matches panel d): x = spacing / per-basis span; y = density / mean
# density, so a structureless (Poisson-like) spectrum sits flat at 1.
fq_x = sk_fq.bin_center / sk_fq.bin_center.max()
rf_x = sk_rf.bin_center / sk_rf.bin_center.max()
fq_y = sk_fq.value / sk_fq.value.mean()
rf_y = sk_rf.value / sk_rf.value.mean()

ax_e.plot(fq_x, fq_y, color=PINK, lw=1.0, zorder=4)
ax_e.plot(rf_x, rf_y, color=AZURE, lw=1.0, ls=(0, (4, 2.5)), zorder=3)
FS.ref_hline(ax_e, 1.0, color=FS.C["ref_gray"], label="Poisson (flat)")

ax_e.annotate("FQ-GNN (ours)", xy=(0.10, 2.55), fontsize=6, color=PINK, ha="left")
ax_e.annotate("RFF-GNN", xy=(0.36, 3.02), fontsize=6, color=AZURE, ha="left")

ax_e.set_xlim(0, 1.0)
ax_e.set_ylim(0, 3.6)
ax_e.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"])
ax_e.set_yticks([0, 1, 2, 3])
# ONE two-line x-label: main line + smaller gray sub-line (no other text in this area)
ax_e.set_xlabel("k, pairwise frequency spacing", fontsize=6.5, labelpad=2)
ax_e.annotate("(fraction of per-basis span)", xy=(0.5, -0.170), xycoords="axes fraction",
              fontsize=5.5, color=GRAY6E, ha="center", va="top")
ax_e.set_ylabel("S(k) (mean-normalized)")
ax_e.set_title("Spectral rigidity S(k)")
FS.condition_strip(ax_e, "seed-0 trained checkpoints", dy=-0.225)
FS.condition_strip(ax_d2, "seed-0 trained checkpoints", dy=-0.50)

# ================================================================================================ #
FS.save(fig, os.path.join(OUT, "fig4"))

# ---- console diagnostics ----
print("\n[a] slope endpoints (mean ratio, RFF -> FQ):")
for b in BUCKET_ORDER:
    print(f"    {b:>10}: {pa[('RFFGNN', b)][0]:.3f} -> {pa[('FQGNN', b)][0]:.3f}")
print("[c] FQ stop epochs:", [d.epoch.iloc[-1] for d in fq_ep],
      "| RFF stop epochs:", [d.epoch.iloc[-1] for d in rff_ep])
print(f"[d] live/extinct: {len(live)}/{len(dead)} | top-10 energy: FQ {fq_top10:.2f}% "
      f"vs RFF {rf_top10:.1f}%")
print(f"[e] S(k) peak (norm.): FQ {fq_y.max():.2f} at x={fq_x[fq_y.idxmax()]:.3f} | "
      f"RFF {rf_y.max():.2f}")
