"""fig_style -- THE locked design system for every figure of the FQGNN paper.

Single source of truth for rcParams, color tokens, model style locks, and layout helpers.
EVERY figure script imports this; no figure defines its own colors, fonts, or model styles.  Usage:

    import sys, os; sys.path.insert(0, os.path.dirname(...))
    import fig_style as FS
    FS.apply()                       # rcParams
    fig = FS.new_figure(width_mm=180, height_mm=138)
    FS.panel_letter(fig, 0.02, 0.97, "a")
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MM = 1 / 25.4                          # mm -> inch

# ------------------------------------------------------------------------------------------------ #
#  COLOR TOKENS (NEVER invent a hue outside these)                                                   #
# ------------------------------------------------------------------------------------------------ #
C = dict(
    # structure
    axis="#1A1A1A", text="#2B2B2B", letter="#000000", sep="#BFBFBF", grid="#E6E6E6",
    ref_red="#D1352B", ref_gray="#9A9A9A",
    # schematic tints (grouping containers ONLY -- never behind a data plot)
    tint_gray="#F2F2F2", tint_blue="#DCEAF7", tint_pink="#FBE4EC",
    # categorical (max 5 per panel)
    navy="#1F3A5F", blue="#3B6CA8", azure="#5B9BD5", paleblue="#9DC3E6", bluegray="#8A9BB0",
    # cool-family extensions (lilac/mauve tints, deep slate baselines) -- keep every baseline
    # inside ONE cool family
    lavender="#9C8FCF", slate="#3E4C6E",
    pink="#D6608E", pink_light="#E8A0B8", gold="#F2C230",
    gcd="#4D4D4D",
    good="#548235", good_fill="#A9D08E", alert="#C00000",
)

# sequential ramps (ordered magnitude; ONE hue, 5 steps light->dark)
SEQ_TEAL = ["#E8F2EC", "#BFE0D2", "#7FC2AC", "#4F9E84", "#2E7D6B"]   # sinuosity buckets
SEQ_PINK = ["#FBE4EC", "#F3C2D5", "#E89BB6", "#D6608E", "#C9447A"]   # our-method ramp / heatmaps
SEQ_BLUE = ["#EAF2FB", "#C6DCF1", "#9DC3E6", "#5B9BD5", "#26456E"]
DIVERGING = ["#C00000", "#E8A0A0", "#FFFFFF", "#9DC3E6", "#26456E"]  # signed; ALWAYS print 0 tick
# signed schematic fields: navy -> white -> our pink (keeps the identity hue on the positive lobe)
DIV_PINK = ["#22374F", "#7FA9D0", "#FFFFFF", "#E89BB6", "#C9447A"]

# ------------------------------------------------------------------------------------------------ #
#  SCHEMATIC TOKENS -- the house schematic idiom:                                                     #
#  mid-tone pastel FILL + dark SATURATED same-hue EDGE.  The fill/edge contrast is what makes those  #
#  panels read as crisp rather than washed out; never draw a card as pale-fill + pale-edge.          #
# ------------------------------------------------------------------------------------------------ #
SCHEM = dict(
    rose=("#F7DEDA", "#C0504D"),        # encoder-style cards
    ours=("#FBE4EC", "#C9447A"),        # FQ-typed stage (our identity hue)
    slate=("#BBC7D4", "#2E4257"),       # compute blocks
    sky=("#B5D9EB", "#2E75B6"),         # readout / output
    mint=("#BFE3D9", "#2F8C79"),        # data / inputs
    amber=("#F7D2A6", "#C55A11"),       # pipeline container
    lavender=("#CBCBE3", "#585078"),    # theory container
    sand=("#F4EADB", "#C0A87E"),        # soft neutral panel
    peri=("#D9E1F1", "#4A5B86"),        # pale periwinkle panel
)
# container tints are the same hues at panel strength (fill lightened, edge kept saturated)
SCHEM_PANEL = dict(
    peri=("#E7ECF7", "#4A5B86"), sand=("#F8F1E6", "#C0A87E"),
    ours=("#FDEFF4", "#C9447A"), slate=("#E4E9EF", "#45566E"),
    amber=("#FBEAD5", "#C55A11"), lavender=("#E6E6F2", "#585078"),
)

BUCKETS = ["0.00-0.02", "0.02-0.05", "0.05-0.10", "0.10-0.25", ">0.25"]
BUCKET_COLOR = dict(zip(BUCKETS, SEQ_TEAL))

# ------------------------------------------------------------------------------------------------ #
#  MODEL LOCK (identical in every panel of every figure)                                             #
#  color x marker x linestyle triple => grayscale-safe                                               #
# ------------------------------------------------------------------------------------------------ #
#  Baselines form ONE cool family ordered by lightness (deep navy -> pale gray-blue), so the eye
#  reads them as a group and our pink as the exception.  Teal/green is NOT available to a model:
#  SEQ_TEAL is the sinuosity-bucket ramp and green is a status colour.  Amber is reserved for the
#  single genuine odd-one-out (the one-sided MultitaskAIS reference).
MODEL = {
    "FQGNN":     dict(color=C["pink"],     marker="o", ls="-",  label="FQ-GNN (ours)", z=10, lw=1.25),
    "RFFGNN":    dict(color=C["azure"],    marker="^", ls="--", label="RFF-GNN",       z=6,  lw=1.0),
    "CSDI":      dict(color="#22374F",     marker="s", ls="-.", label="CSDI",          z=5,  lw=1.0),
    "SAITS":     dict(color="#3E6FA8",     marker="D", ls=":",  label="SAITS",         z=4,  lw=1.0),
    "BiGRU":     dict(color="#8FC4E3",     marker="v", ls="-",  label="Bi-GRU",        z=3,  lw=0.75),
    "BF-BiLSTM": dict(color="#AFBECD",     marker="*", ls=":",  label="BF-BiLSTM",     z=2,  lw=0.75),
    "Hermite":   dict(color=C["axis"],     marker="h", ls="--", label="cubic-Hermite baseline", z=7,  lw=1.0),
    # MultitaskAIS is a ONE-SIDED reference (forward VRNN) -- the label must always say so.
    "HABIT":        dict(color=C["lavender"], marker="P", ls="--", label="HABIT",       z=2.6, lw=0.75),
    "MH-GIN":       dict(color="#6E86A8",     marker="X", ls="-.", label="MH-GIN",      z=2.4, lw=0.75),
    "MultitaskAIS": dict(color="#E8A33D",     marker="<", ls=":",
                         label="MultitaskAIS (one-sided)",                              z=2.2, lw=0.75),
    "GCD":       dict(color=C["gcd"],      marker="",  ls="--", label="GCD",           z=1,  lw=0.75),
}


# ------------------------------------------------------------------------------------------------ #
#  PALETTE SYSTEM.  Every hue above is a ROLE, not a colour: `ours` accent + ramp, ONE graded         #
#  baseline family, neutral references, ONE odd-one-out, a bucket ramp, and schematic fills derived   #
#  from the same hues.  Select with the environment variable FQ_PALETTE (default = "current").        #
#  Each option below re-binds every role IN PLACE so builders need no change.                         #
# ------------------------------------------------------------------------------------------------ #
import os as _os

PALETTES = {
    # 1 -- Okabe-Ito scientific: colour-blind-safe canonical set; vermilion accent, blue family
    "okabe": dict(
        C=dict(pink="#D55E00", pink_light="#F4B08A", navy="#003F7F", blue="#0072B2",
               azure="#3A8FC7", paleblue="#9CD2F2", bluegray="#A9C4D6", lavender="#CC79A7",
               slate="#2B4C6F", gold="#E69F00", gcd="#4D4D4D"),
        SEQ_PINK=["#FBE3D6", "#F4B08A", "#E8834B", "#D55E00", "#A84600"],
        SEQ_BLUE=["#EAF2FB", "#BFDCF2", "#7FBCE6", "#0072B2", "#003F7F"],
        SEQ_TEAL=["#E6F4EF", "#B3E1D2", "#66C3A8", "#009E73", "#00634A"],
        DIV_PINK=["#003F7F", "#7FBCE6", "#FFFFFF", "#F4B08A", "#D55E00"],
        MODEL=dict(FQGNN="#D55E00", RFFGNN="#3A8FC7", CSDI="#003F7F", SAITS="#0072B2",
                   BiGRU="#9CD2F2", **{"BF-BiLSTM": "#C3D6E3", "MH-GIN": "#5E8FB8",
                   "MultitaskAIS": "#E69F00"}, HABIT="#CC79A7"),
        SCHEM=dict(ours=("#FBE3D6", "#B94D00"), slate=("#BBC7D4", "#2E4257"),
                   sky=("#BDE3F5", "#0072B2"), peri=("#D9E1F1", "#4A5B86"),
                   sand=("#F4EADB", "#C0A87E")),
        SCHEM_PANEL=dict(ours=("#FDF0E8", "#B94D00"), slate=("#E4E9EF", "#45566E"),
                         peri=("#E7ECF7", "#4A5B86"), sand=("#F8F1E6", "#C0A87E")),
    ),
    # 2 -- Coral & teal editorial: warm coral accent against a cool teal-to-slate family
    "coral": dict(
        C=dict(pink="#E4572E", pink_light="#F5A98F", navy="#12343B", blue="#1F6F7A",
               azure="#45A3B0", paleblue="#7FC4CC", bluegray="#B4D9DD", lavender="#3D5A99",
               slate="#23404A", gold="#F2B134", gcd="#4D4D4D", ref_red="#A6192E"),
        SEQ_PINK=["#FCE4DC", "#F5A98F", "#EC7A57", "#E4572E", "#B23A17"],
        SEQ_BLUE=["#E8F4F5", "#BDE3E6", "#7FC4CC", "#1F6F7A", "#12343B"],
        SEQ_TEAL=["#F1E8F4", "#D6BEE0", "#B08BC5", "#8A5DA6", "#5E3A7E"],
        DIV_PINK=["#12343B", "#7FC4CC", "#FFFFFF", "#F5A98F", "#E4572E"],
        MODEL=dict(FQGNN="#E4572E", RFFGNN="#45A3B0", CSDI="#12343B", SAITS="#1F6F7A",
                   BiGRU="#7FC4CC", **{"BF-BiLSTM": "#B4D9DD", "MH-GIN": "#3A8F9A",
                   "MultitaskAIS": "#F2B134"}, HABIT="#3D5A99"),
        SCHEM=dict(ours=("#FCE4DC", "#B23A17"), slate=("#C2CED6", "#23404A"),
                   sky=("#C8E8EA", "#1F6F7A"), peri=("#E3E1F2", "#5B4E9A"),
                   sand=("#F4EADB", "#C0A87E")),
        SCHEM_PANEL=dict(ours=("#FDF0EB", "#B23A17"), slate=("#E6EBEF", "#23404A"),
                         peri=("#EEEDF7", "#5B4E9A"), sand=("#F8F1E6", "#C0A87E")),
    ),
    # 3 -- Violet & gold on graphite: violet accent, achromatic graphite family, gold outlier
    "violet": dict(
        C=dict(pink="#6F3FBF", pink_light="#C9B2EC", navy="#2B2D42", blue="#4A5568",
               azure="#6B7A90", paleblue="#B0BAC9", bluegray="#CDD3DC", lavender="#E76F51",
               slate="#2B2D42", gold="#E9A820", gcd="#4D4D4D"),
        SEQ_PINK=["#EEE6F9", "#C9B2EC", "#A27FDC", "#6F3FBF", "#4B2A85"],
        SEQ_BLUE=["#EDF0F4", "#CDD3DC", "#8D99AE", "#4A5568", "#2B2D42"],
        SEQ_TEAL=["#E4F2F0", "#B6DED8", "#7CC2B8", "#3F9E93", "#1F6F6A"],
        DIV_PINK=["#2B2D42", "#8D99AE", "#FFFFFF", "#C9B2EC", "#6F3FBF"],
        MODEL=dict(FQGNN="#6F3FBF", RFFGNN="#8D99AE", CSDI="#2B2D42", SAITS="#4A5568",
                   BiGRU="#B0BAC9", **{"BF-BiLSTM": "#CDD3DC", "MH-GIN": "#6B7A90",
                   "MultitaskAIS": "#E9A820"}, HABIT="#E76F51"),
        SCHEM=dict(ours=("#EEE6F9", "#4B2A85"), slate=("#C6CBD4", "#2B2D42"),
                   sky=("#CFE0F1", "#3B6EA8"), peri=("#DFE3EC", "#4A5568"),
                   sand=("#F6ECD8", "#C48F2C")),
        SCHEM_PANEL=dict(ours=("#F5F0FC", "#4B2A85"), slate=("#E8EAEF", "#2B2D42"),
                         peri=("#EEF0F5", "#4A5568"), sand=("#FAF3E6", "#C48F2C")),
    ),
    # 4 -- Crimson & navy minimal: one crimson accent, everything else navy-to-steel, bronze outlier
    "crimson": dict(
        C=dict(pink="#B3122E", pink_light="#E08A9A", navy="#1C2541", blue="#3A506B",
               azure="#5C7A99", paleblue="#8FA7BF", bluegray="#C1CFDB", lavender="#4E6E9E",
               slate="#1C2541", gold="#B08D57", gcd="#4D4D4D"),
        SEQ_PINK=["#F7DEE2", "#E8A3AE", "#D4677A", "#B3122E", "#7E0C20"],
        SEQ_BLUE=["#EBEFF4", "#C1CFDB", "#8FA7BF", "#3A506B", "#1C2541"],
        SEQ_TEAL=["#F3EBDD", "#E0CDA9", "#C4A874", "#A3844C", "#6E5530"],
        DIV_PINK=["#1C2541", "#8FA7BF", "#FFFFFF", "#E8A3AE", "#B3122E"],
        MODEL=dict(FQGNN="#B3122E", RFFGNN="#5C7A99", CSDI="#1C2541", SAITS="#3A506B",
                   BiGRU="#8FA7BF", **{"BF-BiLSTM": "#C1CFDB", "MH-GIN": "#4E6E9E",
                   "MultitaskAIS": "#B08D57"}, HABIT="#7A9CC6"),
        SCHEM=dict(ours=("#F7DEE2", "#7E0C20"), slate=("#C6CDD8", "#1C2541"),
                   sky=("#D3DEEA", "#3A506B"), peri=("#DFE4EC", "#3A506B"),
                   sand=("#F3EBDD", "#A3844C")),
        SCHEM_PANEL=dict(ours=("#FBEFF1", "#7E0C20"), slate=("#E8EBF0", "#1C2541"),
                         peri=("#EEF1F5", "#3A506B"), sand=("#F8F3EA", "#A3844C")),
    ),
}


def select_palette(name):
    """Re-bind every colour role in place (dicts/lists are shared with importers)."""
    if name in ("", "current", None):
        return
    P = PALETTES[name]
    C.update(P["C"])
    for key in ("SEQ_PINK", "SEQ_BLUE", "SEQ_TEAL", "DIV_PINK"):
        globals()[key][:] = P[key]
    DIVERGING[:] = [P["SEQ_PINK"][3], P["SEQ_PINK"][1], "#FFFFFF", P["SEQ_BLUE"][2], P["SEQ_BLUE"][4]]
    BUCKET_COLOR.update(zip(BUCKETS, SEQ_TEAL))
    for m, col in P["MODEL"].items():
        MODEL[m]["color"] = col
    MODEL["Hermite"]["color"] = C["axis"]
    MODEL["GCD"]["color"] = C["gcd"]
    SCHEM.update(P["SCHEM"])
    SCHEM_PANEL.update(P["SCHEM_PANEL"])


ACTIVE_PALETTE = _os.environ.get("FQ_PALETTE", "coral")     # default palette (Option 2, coral)
select_palette(ACTIVE_PALETTE)

# ------------------------------------------------------------------------------------------------ #
#  rcParams                                                                                          #
# ------------------------------------------------------------------------------------------------ #
def apply():
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 600,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",
        # reference sizes: titles ~8 pt, axis labels ~7.5 pt, ticks ~6.5 pt
        "font.size": 6.5, "axes.titlesize": 8.0, "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.0,
        "axes.linewidth": 0.75,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.width": 0.75, "ytick.major.width": 0.75,
        "xtick.direction": "out", "ytick.direction": "out",
        # data panels are FULLY BOXED (all four spines); an open L-frame reads as unfinished
        "axes.spines.top": True, "axes.spines.right": True,
        "lines.linewidth": 1.0, "lines.markersize": 4.0,
        "errorbar.capsize": 2.5,
        "axes.edgecolor": C["axis"], "text.color": C["axis"],
        "xtick.color": C["axis"], "ytick.color": C["axis"],
        "axes.labelcolor": C["axis"],
        "axes.grid": False, "axes.xmargin": 0.04, "axes.ymargin": 0.04,
        "axes.titleweight": "regular",
        "legend.frameon": False,                      # no legend box anywhere
        "pdf.fonttype": 42, "ps.fonttype": 42,        # editable text in vector output
    })


def new_figure(width_mm=180.0, height_mm=138.0):
    return plt.figure(figsize=(width_mm * MM, height_mm * MM))


# ------------------------------------------------------------------------------------------------ #
#  Helpers                                                                                           #
# ------------------------------------------------------------------------------------------------ #
def panel_letter(fig, x, y, letter):
    """Lowercase bold 8.5pt pure-black letter, figure coords, top-LEFT outside corner."""
    fig.text(x, y, letter, fontsize=8.5, fontweight="bold", color=C["letter"],
             ha="left", va="top", family="sans-serif")


def ref_hline(ax, y, color=None, label=None, lw=0.7):
    """Dashed reference rule: labelled in <=4 words on the line or in legend."""
    color = color or C["gcd"]
    ax.axhline(y, color=color, ls=(0, (4, 2.5)), lw=lw, zorder=1.5)
    if label:
        ax.annotate(label, xy=(0.99, y), xycoords=("axes fraction", "data"),
                    fontsize=5.5, color=color, ha="right", va="bottom",
                    xytext=(0, 1), textcoords="offset points")


def condition_strip(ax, text, dy=-0.32, x=0.5, ha="center"):
    """Pipe-delimited gray condition strip directly under the x-axis.
    x/ha allow right-anchoring when a centered strip would clip at the canvas edge."""
    ax.annotate(text, xy=(x, dy), xycoords="axes fraction", fontsize=5.5,
                color="#6E6E6E", ha=ha, va="top")


def contrast_text_color(hexcolor):
    """Dark text on fills lighter than #5B9BD5, white on darker.
    Small tolerance below the reference keeps near-identical mid tones from flipping
    inconsistently (dark #2B2B2B also has the better contrast right at the reference)."""
    h = hexcolor.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    ref = 0.2126 * 0x5B + 0.7152 * 0x9B + 0.0722 * 0xD5
    return "#2B2B2B" if lum >= ref - 8.0 else "#FFFFFF"


def model_legend_handles(models, metrics=None):
    """Legend entries reproducing the exact line/marker style, with inline metric:
    'FQ-GNN (ours) (0.74)'."""
    hs = []
    for m in models:
        s = MODEL[m]
        lab = s["label"] + (f" ({metrics[m]})" if metrics and m in metrics else "")
        hs.append(Line2D([0], [0], color=s["color"], marker=s["marker"], ls=s["ls"],
                         lw=s["lw"], markersize=3.5, label=lab))
    return hs


def seed_dots(ax, x, values, color, width=0.16, size=2.3):
    """Per-seed replicate dots over a bar: small OPEN grey circles."""
    import numpy as np
    xs = x + np.linspace(-width, width, len(values))
    ax.plot(xs, values, ls="none", marker="o", markersize=size,
            markerfacecolor="none", markeredgecolor="#6E6E6E", markeredgewidth=0.6, zorder=5)


# ------------------------------------------------------------------------------------------------ #
#  SCHEMATIC PRIMITIVES.  All take an mm-coordinate overlay axes.                                    #
# ------------------------------------------------------------------------------------------------ #
def card(ax, x0, y0, w, h, token, lw=1.0, r=0.9, z=2, fill=None, edge=None):
    """Filled block with a dark saturated same-hue border -- the unit of the house schematic."""
    from matplotlib.patches import FancyBboxPatch
    f, e = SCHEM.get(token, (fill, edge))
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fill or f, edgecolor=edge or e, lw=lw, zorder=z))


def panel(ax, x0, y0, x1, y1, token, lw=0.9, r=2.6, z=1):
    """Group container: panel-strength tint + the same saturated edge, generous rounding."""
    from matplotlib.patches import FancyBboxPatch
    f, e = SCHEM_PANEL[token]
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=f, edgecolor=e, lw=lw, zorder=z))


def _round_path(pts, r):
    """Replace every interior right-angle vertex with a quarter-arc of radius r."""
    import numpy as np
    P = [np.asarray(p, float) for p in pts]
    out = [P[0]]
    for i in range(1, len(P) - 1):
        a, b, c = P[i - 1], P[i], P[i + 1]
        u, v = a - b, c - b
        lu, lv = np.hypot(*u), np.hypot(*v)
        if lu < 1e-9 or lv < 1e-9:
            continue
        u, v = u / lu, v / lv
        rr = min(r, lu / 2, lv / 2)
        s, e = b + u * rr, b + v * rr
        t = np.linspace(0, 1, 9)[:, None]          # quadratic Bezier == quarter arc for 90 deg
        out.extend(list((1 - t) ** 2 * s + 2 * (1 - t) * t * b + t ** 2 * e))
    out.append(P[-1])
    return np.array(out)


def elbow(ax, pts, color=None, lw=0.7, r=1.0, head=1.1, z=4, ls="-", arrow=True):
    """Black Manhattan connector with ROUNDED corners and a solid arrowhead."""
    import numpy as np
    from matplotlib.patches import Polygon
    color = color or C["axis"]
    Q = _round_path(pts, r)
    ax.plot(Q[:, 0], Q[:, 1], color=color, lw=lw, ls=ls, zorder=z,
            solid_capstyle="round", solid_joinstyle="round")
    if not arrow:
        return
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    d = np.array([x1 - x0, y1 - y0], float)
    d /= max(np.hypot(*d), 1e-9)
    p = np.array([-d[1], d[0]])
    tip = np.array([x1, y1])
    ax.add_patch(Polygon([tip, tip - head * d + 0.42 * head * p,
                          tip - head * d - 0.42 * head * p],
                         closed=True, facecolor=color, edgecolor="none", zorder=z + 1))


def oplus(ax, x, y, r=1.5, color=None, lw=0.9, z=6):
    """Circled-plus fusion node, drawn (no unicode glyph risk)."""
    from matplotlib.patches import Circle
    color = color or C["axis"]
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=color, lw=lw, zorder=z))
    ax.plot([x - r * 0.58, x + r * 0.58], [y, y], color=color, lw=lw, zorder=z + 1)
    ax.plot([x, x], [y - r * 0.58, y + r * 0.58], color=color, lw=lw, zorder=z + 1)


def save(fig, path_base):
    """Save vector PDF + 600dpi PNG + 150dpi preview."""
    fig.savefig(path_base + ".pdf")
    fig.savefig(path_base + ".png", dpi=600)
    fig.savefig(path_base + "_preview.png", dpi=150)
    print(f"saved {path_base}.pdf/.png/_preview.png")
