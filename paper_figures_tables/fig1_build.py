"""FIGURE 1 -- 'The Fourier-Quasicrystal operator family' (lead schematic).

Canvas 180 x 103 mm, panels a (hero schematic, ~65% area) / b (operator-not-data-model) /
c (in-family telemetry).  All thumbnails are REAL data read from FQGNN_FIGDATA + telemetry CSV.
Style: fig_style.py ONLY (locked design system).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _paths import FIGDATA, OUT, RESULTS  # noqa: E402,F401  (env FQGNN_FIGDATA / FQGNN_FIG_OUT / FQGNN_RESULTS_ROOT)

import fig_style as FS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Ellipse, Polygon, Circle  # noqa: E402

FS.apply()
C = FS.C

W_MM, H_MM = 180.0, 103.0

# ------------------------------------------------------------------ data (ALL real)
FD = FIGDATA
SP = pd.read_csv(os.path.join(FD, "fq_spectrum_trained.csv"))
RF = pd.read_csv(os.path.join(FD, "rff_spectrum.csv"))
U = pd.read_csv(os.path.join(FD, "fq_U_trained.csv")).drop(columns="row").to_numpy()
L = pd.read_csv(os.path.join(FD, "fq_L_trained.csv")).drop(columns="row").to_numpy()
HG = pd.read_csv(os.path.join(FD, "hero_gap_0.05-0.10_245.csv"))
# hero CSVs carry the FULL track; crop to held window +- 12 anchors
_hw = np.where(HG.is_held.to_numpy() == 1)[0]
HG = HG.iloc[max(0, _hw[0] - 12):_hw[-1] + 13].reset_index(drop=True)
TL = pd.read_csv(os.path.join(
    RESULTS, "runs", "telemetry_infamily_native", "telemetry_infamily.csv"))

live = SP[SP.abs_c > 0]
dead = SP[SP.abs_c == 0]
w_live = live.eff_freq.to_numpy()
a_live = live.abs_c.to_numpy()
a2_live = live.abs_c_sq.to_numpy()
ph_live = np.arctan2(live.Im_c.to_numpy(), live.Re_c.to_numpy())
w_dead = dead.eff_freq.to_numpy()
GHOST_H = 0.08 * a_live.max()           # visual-only height for the |c| = 0 ghost stems

fig = FS.new_figure(W_MM, H_MM)

# full-canvas overlay in mm coordinates: containers, arrows, free text
axB = fig.add_axes([0, 0, 1, 1], zorder=0)
axB.set_xlim(0, W_MM)
axB.set_ylim(0, H_MM)
axB.axis("off")


def ax_mm(x, y, w, h, **kw):
    return fig.add_axes([x / W_MM, y / H_MM, w / W_MM, h / H_MM], **kw)


def txt(x, y, s, size=6, color=None, ha="center", va="center", style="normal",
        weight="normal", rotation=0):
    axB.text(x, y, s, fontsize=size, color=color or C["text"], ha=ha, va=va,
             fontstyle=style, fontweight=weight, rotation=rotation)


def container(x0, y0, x1, y1, token, tab):
    """Group container in the house idiom: panel-strength tint with the SAME hue
    at full saturation on the border, generous rounding, italic label tab inside the top-left."""
    FS.panel(axB, x0, y0, x1, y1, token, lw=0.9, r=2.6, z=1)
    axB.text(x0 + 2.4, y1 - 2.9, tab, fontsize=7, fontstyle="italic",
             color=FS.SCHEM_PANEL[token][1], ha="left", va="center", zorder=4)


def block_arrow(x0, x1, y, shaft=1.6, head_l=2.2, head_h=3.6, color="#9DAEC2"):
    """Cool-gray block arrow = forward data (typed connector 1)."""
    s, hh = shaft / 2, head_h / 2
    axB.add_patch(Polygon(
        [(x0, y - s), (x1 - head_l, y - s), (x1 - head_l, y - hh), (x1, y),
         (x1 - head_l, y + hh), (x1 - head_l, y + s), (x0, y + s)],
        closed=True, facecolor=color, edgecolor="none", zorder=3))


def thin_arrow(pts, color=None, lw=0.7, ls="-", head=1.1, zorder=4):
    """Black rounded-corner Manhattan connector via the locked style system."""
    FS.elbow(axB, pts, color=color or C["axis"], lw=lw, ls=ls, head=head, z=zorder, r=1.1)


def oplus(x, y, r=1.5, color=None):
    FS.oplus(axB, x, y, r=r, color=color)


def hairline_frame(ax, lw=0.8, color=None):
    """Tile frame: dark hairline (every data tile is framed crisply, not in pale gray)."""
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(lw)
        sp.set_color(color or "#5B6B7C")
    ax.set_xticks([])
    ax.set_yticks([])


PINK_CMAP = LinearSegmentedColormap.from_list("seqpink", FS.SEQ_PINK)
DIV_CMAP = LinearSegmentedColormap.from_list("divpink", FS.DIV_PINK)

# ================================================================== PANEL a
FS.panel_letter(fig, 1.5 / W_MM, 102.4 / H_MM, "a")

CB, CT = 42.5, 99.0                       # container band bottom/top (panel a >= 50% of area)
G1 = (4.0, 29.0)
G2 = (34.0, 66.0)
G3 = (71.0, 129.0)
G4 = (134.0, 176.0)

container(*[G1[0], CB], *[G1[1], CT], "peri", "Lee–Yang generator")
container(*[G2[0], CB], *[G2[1], CT], "sand", "Quasicrystal objects")
container(*[G3[0], CB], *[G3[1], CT], "ours", "FQ operator library")
container(*[G4[0], CB], *[G4[1], CT], "slate", "FQ-GNN → prediction")

for xa, xb in [(G1[1], G2[0]), (G2[1], G3[0]), (G3[1], G4[0])]:
    block_arrow(xa + 0.3, xb - 0.3, 70.0)

# ---------------------------------------------------------------- G1: Lee-Yang generator
g1c = (G1[0] + G1[1]) / 2                                  # 16.5
E_CX, E_CY, E_W, E_H = g1c, 83.5, 18.5, 13.5
# faint pink interior wash = 'the in-family region' (under path/x/labels), then 1.2pt boundary
axB.add_patch(Ellipse((E_CX, E_CY), E_W, E_H, facecolor="#FBE4EC", alpha=0.35,
                      edgecolor="none", zorder=1.5))
ball = Ellipse((E_CX, E_CY), E_W, E_H, facecolor="none",
               edgecolor=C["blue"], lw=1.2, zorder=2)
axB.add_patch(ball)
txt(g1c, 87.6, "chart $(r,\\ \\theta)$", size=6, color="#6E6E6E")
# descent path: 5 connected dots spiralling in to the attractor X0 at the ellipse centre
px = np.array([11.8, 14.2, 16.4, 15.7, E_CX])
py = np.array([86.0, 84.4, 84.2, 82.3, E_CY])
axB.plot(px, py, color="#D6608E", lw=0.7, zorder=3)
axB.plot(px[:-1], py[:-1], ls="none", marker="o", ms=1.9, mfc="#D6608E", mec="none", zorder=3)
thin_arrow([(px[-2], py[-2]), (px[-1], py[-1])], "#D6608E", lw=0.7, head=0.9)
# attractor X0: small x marker at the ellipse interior centre
axB.plot([E_CX], [E_CY], ls="none", marker="x", ms=3.2, mew=0.9, color="#4D4D4D", zorder=4)
txt(E_CX + 2.0, E_CY - 0.2, "$X_0$", size=6, color="#6E6E6E", ha="left")
txt(g1c, 74.6, "training path", size=6, color="#6E6E6E")
txt(g1c, 71.0, "boundary: $\\min_I \\Delta_I = 0$", size=6)
txt(g1c, 66.6, "coupling $U = e^{iH}$", size=6)

axU = ax_mm(6.0, 52.0, 10.0, 10.0)
axU.imshow(np.power(U, 0.5), cmap=PINK_CMAP, vmin=0, vmax=1, interpolation="nearest")
hairline_frame(axU)
txt(11.0, 50.0, "$|U_{jk}|$", size=6)
# REAL trained generator L (n x d = 6 x 2) as a second data tile (embedding-tile idiom)
axLt = ax_mm(18.5, 52.0, 4.6, 10.0)
Lz = (L - L.min()) / (L.max() - L.min() + 1e-12)
axLt.imshow(Lz, cmap=LinearSegmentedColormap.from_list("seqblue", FS.SEQ_BLUE), vmin=0, vmax=1,
            aspect="auto", interpolation="nearest")
hairline_frame(axLt)
txt(20.8, 50.0, "$L$", size=6)
txt(g1c, 47.7, "(trained, $\\gamma$-scaled)", size=6, color="#6E6E6E")

# ---------------------------------------------------------------- G2: quasicrystal objects
g2c = (G2[0] + G2[1]) / 2                                  # 50
axT = ax_mm(g2c - 6.5, 79.5, 13.0, 13.0)
u = np.array([1.0, 1.0]) / np.sqrt(2)                       # ray direction in R^2
s1, s2 = float(L[0] @ u), float(L[1] @ u)                   # REAL trained L rows
t = np.linspace(0, 4.0 / s2, 6000)                          # end exactly on a tile edge
phx0, phy0 = (s1 * t) % 1.0, (s2 * t) % 1.0                 # unbroken (for intersections)
phx, phy = phx0.copy(), phy0.copy()
brk = np.where((np.abs(np.diff(phx)) > 0.5) | (np.abs(np.diff(phy)) > 0.5))[0]
phx[brk + 1] = np.nan
axT.plot(phx, phy, color=C["blue"], lw=0.6)
# variety curve on the torus: exact level set g(z) = cos 2pi.z1 + cos 2pi.z2 = VC,
# gold dots at its NUMERICAL intersections with the winding line (sign changes of g - VC)
VC = -0.6
gg = np.linspace(0.0, 1.0, 400)
GX, GY = np.meshgrid(gg, gg)
axT.contour(GX, GY, np.cos(2 * np.pi * GX) + np.cos(2 * np.pi * GY) - VC,
            levels=[0.0], colors=[C["axis"]], linewidths=0.8, zorder=4)
Fw = np.cos(2 * np.pi * phx0) + np.cos(2 * np.pi * phy0) - VC
hit = np.where(np.diff(np.sign(Fw)) != 0)[0]
hit = np.where(np.abs(Fw[hit]) < np.abs(Fw[hit + 1]), hit, hit + 1)  # nearer sample
axT.plot(phx0[hit], phy0[hit], ls="none", marker="o", ms=2.2,
         mfc=C["gold"], mec="none", zorder=5)
axT.set_xlim(0, 1)
axT.set_ylim(0, 1)
hairline_frame(axT)
txt(g2c, 77.8, "winding $\\Phi_L$ on $T^2$", size=6)
axB.plot([57.2, 52.8], [88.3, 86.6], color="#6E6E6E", lw=0.45, zorder=4)
txt(57.4, 88.6, "variety", size=6, color="#6E6E6E", ha="left")


def chevron(ytop):
    """Double-chevron pipeline-order separator (drawn -- no unicode risk)."""
    for dy in (0.0, 1.7):
        axB.plot([g2c - 1.5, g2c, g2c + 1.5],
                 [ytop - dy, ytop - 1.1 - dy, ytop - dy],
                 color=C["ref_gray"], lw=0.9, zorder=3, solid_capstyle="round")


chevron(76.3)

# point set Lambda: REAL 1-D quasiperiodic positions -- level crossings of
# cos(2pi L00 x) + cos(2pi L10 x) (two trained incommensurate frequencies -> beats)
lx = np.linspace(0.0, 12.0, 60000)
lf = np.cos(2 * np.pi * L[0, 0] * lx) + np.cos(2 * np.pi * L[1, 0] * lx)
lcr = lx[np.where(np.diff(np.sign(lf - 0.5)) != 0)[0]]
lpos = 37.5 + lcr / 12.0 * 25.0
axB.plot([37.5, 62.5], [71.4, 71.4], color=C["sep"], lw=0.4, zorder=2)
axB.plot(lpos, np.full_like(lpos, 71.4), ls="none", marker="o", ms=1.6,
         mfc=C["gold"], mec="none", zorder=3)
txt(g2c, 69.1, "point set $\\Lambda$", size=6)

chevron(67.4)

axS = ax_mm(36.5, 49.5, 28.0, 14.0)
axS.vlines(w_dead, 0, GHOST_H, color=C["sep"], lw=0.55)
axS.vlines(w_live, 0, a_live, color=C["pink"], lw=0.65)
axS.set_xlim(0, 79)
axS.set_ylim(0, 40)
axS.set_xticks([])
axS.set_yticks([])
for side in ("top", "right"):
    axS.spines[side].set_visible(False)
for side in ("left", "bottom"):
    axS.spines[side].set_linewidth(0.6)
txt(g2c, 47.7, "frequency $\\xi = L^{\\mathsf{T}}k$", size=6)
txt(35.3, 56.5, "$|c|$", size=6, rotation=90)

# ---------------------------------------------------------------- G3: operator cards
CARD_W, CARD_GAP = 10.4, 1.0
card_x0 = G3[0] + (G3[1] - G3[0] - 5 * CARD_W - 4 * CARD_GAP) / 2
CARD_Y0, CARD_Y1 = 47.5, 94.0

cards = [
    ("Op-A$'$", "lift", None),
    ("Op-B", "phases", None),          # "eigenphases" overruns the card at the 6-pt floor
    ("Op-C", "variety", None),
    ("Op-D", "kernel", "$\\delta$"),
    ("$D_\\mu$", "velocity", "$x$"),
]
card_ax = []
for i, (t1, t2, xsym) in enumerate(cards):
    cx = card_x0 + i * (CARD_W + CARD_GAP)
    FS.card(axB, cx, CARD_Y0, CARD_W, CARD_Y1 - CARD_Y0, None, lw=1.0, r=1.1, z=2,
            fill="white", edge=FS.SCHEM["ours"][1])
    # header band: the card's own hue at fill strength, with a rule under it
    axB.add_patch(FancyBboxPatch(
        (cx, CARD_Y1 - 7.0), CARD_W, 7.0,
        boxstyle="round,pad=0,rounding_size=1.1",
        facecolor=FS.SCHEM["ours"][0], edgecolor="none", zorder=2.5))
    axB.plot([cx, cx + CARD_W], [CARD_Y1 - 7.0, CARD_Y1 - 7.0],
             color=FS.SCHEM["ours"][1], lw=0.7, zorder=2.6)
    txt(cx + CARD_W / 2, CARD_Y1 - 2.4, t1, size=6.5)
    txt(cx + CARD_W / 2, CARD_Y1 - 5.3, t2, size=6, color=FS.SCHEM["ours"][1])
    a = ax_mm(cx + 0.9, 51.6, CARD_W - 1.8, 33.0)
    a.set_xticks([])
    a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(False)
    card_ax.append(a)
    if xsym is not None:                      # axis hint only where it disambiguates
        txt(cx + CARD_W / 2, 49.4, xsym, size=6, color="#6E6E6E")

# ---- shared tile constants: REAL trained frequency VECTORS + display scales (cycles per patch)
XI = live[["xi_0", "xi_1"]].to_numpy()                      # (48 x 2), trained
S_DISP, S_DISP_B = 2.2, 0.9

# card 1 -- Op-A' lift: REAL torus lift of the hero gap's 97 nodes under the trained L
#           (rows = nodes in time order, columns = cos/sin of the six lifted angles)
tn = (HG.t_sec.to_numpy() - HG.t_sec.min()) / (HG.t_sec.max() - HG.t_sec.min())
hidx = np.where(HG.is_held.to_numpy() == 1)[0]
tL, tR = tn[hidx[0] - 1], tn[hidx[-1] + 1]
dan = np.where(HG.is_held.to_numpy() == 1, np.minimum(tn - tL, tR - tn) / (tR - tL), 0.0)
X2 = np.stack([tn, dan], axis=1)                          # (t_norm, dist-to-anchor) coordinate
TH = 2 * np.pi * 1.2 * (X2 @ L.T)                           # 1.2 = display scale of the lift
LIFT = np.concatenate([np.cos(TH), np.sin(TH)], axis=1)     # 97 x 12 real lifted features
card_ax[0].imshow(LIFT, cmap=DIV_CMAP, vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
hairline_frame(card_ax[0])

# card 2 -- Op-B eigenphases: REAL phase advance of every trained mode along a displacement.
#           Rows = the 48 modes sorted by |xi|, columns = the sweep; value = cos(2 pi s <xi_k,e> u
#           + phi_k).  Fast rows at the bottom are the high-frequency eigenphases: the chirp IS the
#           module structure (this is what Op-B reads off).
ordk = np.argsort(np.hypot(live.xi_0.to_numpy(), live.xi_1.to_numpy()))
XIo = live[["xi_0", "xi_1"]].to_numpy()[ordk]
uu = np.linspace(0, 1, 260)
e_dir = np.array([1.0, 1.0]) / np.sqrt(2)
EPH = np.cos(2 * np.pi * S_DISP_B * (XIo @ e_dir)[:, None] * uu[None, :] + ph_live[ordk][:, None])
card_ax[1].imshow(EPH, cmap=DIV_CMAP, vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
hairline_frame(card_ax[1])

# card 3 -- Op-C variety: REAL 2-D field |g(x)| = |sum_k c_k cos(2 pi s <xi_k, x> + phi_k)| over
#           a coordinate patch, built from the trained frequency VECTORS xi_k; dark valleys are the
#           variety locus, drawn as the gold zero contour (the 1-D dip curve of Fig. 1 in 2-D)
gx = np.linspace(0, 1, 140)
gy = np.linspace(0, 1, 420)
GX2, GY2 = np.meshgrid(gx, gy)
PH = 2 * np.pi * S_DISP * (GX2[..., None] * XI[:, 0] + GY2[..., None] * XI[:, 1]) + ph_live
G2D = (a_live * np.cos(PH)).sum(-1)
card_ax[2].imshow(np.abs(G2D) / np.abs(G2D).max(), cmap=PINK_CMAP, vmin=0, vmax=1,
                  aspect="auto", origin="lower", extent=(0, 1, 0, 1), interpolation="bilinear")
card_ax[2].contour(GX2, GY2, G2D, levels=[0.0], colors=[C["gold"]], linewidths=0.5)
hairline_frame(card_ax[2])

# card 4 -- Op-D kernel: REAL 2-D K(delta) = sum_k |c_k|^2 cos(2 pi s <xi_k, delta>), centred
dx = np.linspace(-0.5, 0.5, 140)
dy = np.linspace(-1.0, 1.0, 420)
DX, DY = np.meshgrid(dx, dy)
PK = 2 * np.pi * S_DISP * (DX[..., None] * XI[:, 0] + DY[..., None] * XI[:, 1])
K2D = (a2_live * np.cos(PK)).sum(-1)
card_ax[3].imshow(K2D / np.abs(K2D).max(), cmap=DIV_CMAP, vmin=-1, vmax=1, aspect="auto",
                  origin="lower", extent=(-0.5, 0.5, -1, 1), interpolation="bilinear")
hairline_frame(card_ax[3])

# card 5 -- D_mu velocity: REAL directional derivative of the SAME field, D_e g(x) =
#           -sum_k |c_k| 2 pi s <xi_k,e> sin(2 pi s <xi_k,x> + phi_k); white quiver = the
#           velocity direction (grad g) on a sparse grid, so the card reads as a flow, not a field
#           Background is the gradient MAGNITUDE on a sequential blue ramp, so the card is legible
#           as a flow field and cannot be confused with the signed Op-C/Op-D tiles beside it.
gux = -(a_live * (2 * np.pi * S_DISP * XI[:, 0]) * np.sin(PH)).sum(-1)
guy = -(a_live * (2 * np.pi * S_DISP * XI[:, 1]) * np.sin(PH)).sum(-1)
GMAG = np.hypot(gux, guy)
card_ax[4].imshow(GMAG / GMAG.max(), cmap=LinearSegmentedColormap.from_list("sb", FS.SEQ_BLUE),
                  vmin=0, vmax=1, aspect="auto", origin="lower", extent=(0, 1, 0, 1),
                  interpolation="bilinear")
sy, sx = slice(45, None, 100), slice(35, None, 70)
qn = GMAG[sy, sx] + 1e-9
card_ax[4].quiver(GX2[sy, sx], GY2[sy, sx], gux[sy, sx] / qn, guy[sy, sx] / qn,
                  color="white", width=0.035, headwidth=3.6, headlength=4.2, scale=4.6,
                  pivot="mid", zorder=3)
hairline_frame(card_ax[4])

# ---------------------------------------------------------------- G4: FQ-GNN stack + hero gap
SX0, SX1 = 135.2, 155.6
scx = (SX0 + SX1) / 2                                       # 145.4
BOXES = [  # (y0, y1, lines, token) -- FQ-typed stages carry our hue, generic ones the slate
    (87.0, 91.5, ["S1 Lift (Op-A$'$)"], "ours"),
    (80.6, 85.6, ["S2 Features", "(Op-B/C)"], "ours"),
    (71.0, 75.5, ["S3 Message"], "slate"),
    (65.0, 69.5, ["S4 Aggregate (Op-D)"], "ours"),
    (59.0, 63.5, ["S5 Update"], "slate"),
    (51.0, 57.5, ["S6 Measured-spectrum", "readout (Op-M)"], "sky"),
]
for y0, y1, lines, tok in BOXES:
    FS.card(axB, SX0, y0, SX1 - SX0, y1 - y0, tok, lw=0.9, r=0.9, z=2)
    if len(lines) == 1:
        txt(scx, (y0 + y1) / 2, lines[0], size=6)
    else:
        txt(scx, (y0 + y1) / 2 + 1.3, lines[0], size=6)
        txt(scx, (y0 + y1) / 2 - 1.3, lines[1], size=6)

OPY = 78.2
oplus(scx, OPY)
# S1 -> oplus (bypasses S2 on the left, rounded Manhattan), S2 -> oplus, oplus -> S3
thin_arrow([(SX0, 89.25), (134.0, 89.25), (134.0, OPY), (scx - 1.7, OPY)], lw=0.7, head=1.0)
thin_arrow([(scx, 80.6), (scx, OPY + 1.7)], lw=0.7, head=1.0)
thin_arrow([(scx, OPY - 1.7), (scx, 75.9)], lw=0.7, head=1.0)
for ya, yb in [(71.0, 69.9), (65.0, 63.9), (59.0, 57.9)]:
    thin_arrow([(scx, ya), (scx, yb)], lw=0.7, head=1.0)

# ---- the example gap (ENU view; the chord-frame rotation is available via `cang`):
#      INPUT tile (what the stack sees: anchors on both sides, held window empty) aligned with
#      S1/S2, OUTPUT tile (truth / reference chord / FQ-GNN reconstruction) aligned with S5/S6
lat0, lon0 = HG.lat_true.mean(), HG.lon_true.mean()
kx = 60.0 * np.cos(np.deg2rad(lat0))


def enu(lon, lat):
    return (lon - lon0) * kx, (lat - lat0) * 60.0


xt, yt = enu(HG.lon_true.to_numpy(), HG.lat_true.to_numpy())
held = HG.is_held.to_numpy().astype(bool)
i0, i1 = np.where(held)[0][0] - 1, np.where(held)[0][-1] + 1
seg = slice(i0, i1 + 1)
xg, yg = enu(HG.lon_gcd.to_numpy()[seg], HG.lat_gcd.to_numpy()[seg])
xf, yf = enu(HG.lon_fq.to_numpy()[seg], HG.lat_fq.to_numpy()[seg])
cang = 0.0                                  # ENU view: the diagonal track fills a square tile
cR = np.array([[np.cos(-cang), -np.sin(-cang)], [np.sin(-cang), np.cos(-cang)]])


def chord(x, y):
    q = cR @ np.stack([x, y])
    return q[0], q[1]


xt, yt = chord(xt, yt)
xg, yg = chord(xg, yg)
xf, yf = chord(xf, yf)
TX0, TW, TH_ = 158.2, 17.6, 17.5             # tile column; 2.6 mm channel to the stack for arrows
axIn = ax_mm(TX0, 76.0, TW, TH_)
axOut = ax_mm(TX0, 50.5, TW, TH_)
# limits: pad the data, then widen the SHORT axis so the equal-aspect box exactly fills the tile
XL = [xt.min() - 1.5, xt.max() + 1.5]
YL = [yt.min() - 1.5, yt.max() + 1.5]
box_ratio = TH_ / TW
if (YL[1] - YL[0]) / (XL[1] - XL[0]) < box_ratio:
    cy, hy = (YL[0] + YL[1]) / 2, (XL[1] - XL[0]) * box_ratio / 2
    YL = [cy - hy, cy + hy]
else:
    cx_, hx = (XL[0] + XL[1]) / 2, (YL[1] - YL[0]) / box_ratio / 2
    XL = [cx_ - hx, cx_ + hx]
for a, lab in ((axIn, "input"), (axOut, "output")):
    a.set_facecolor("white")
    a.set_xlim(*XL)
    a.set_ylim(*YL)
    a.set_aspect("equal")
    hairline_frame(a)
    a.text(0.95, 0.94, lab, transform=a.transAxes, fontsize=6, color="#6E6E6E",
           ha="right", va="top")            # top-RIGHT: the track runs top-left -> bottom-right
axIn.plot([xt[i0], xt[i1]], [yt[i0], yt[i1]], color=C["gcd"], lw=0.7, ls=(0, (3.2, 1.8)), zorder=3)
axIn.plot(xt[~held], yt[~held], ls="none", marker="o", ms=1.7, mfc=C["blue"], mec="none", zorder=5)
axOut.plot(xt, yt, color=C["axis"], lw=0.6, zorder=2)
axOut.plot(xg, yg, color=C["gcd"], lw=0.7, ls=(0, (3.2, 1.8)), zorder=3)
axOut.plot(xf, yf, color=C["pink"], lw=1.2, zorder=4)
axOut.plot(xt[~held], yt[~held], ls="none", marker="o", ms=1.7, mfc=C["blue"], mec="none", zorder=5)
# 10 NM scale bar, bottom-left of the OUTPUT tile
axOut.plot([XL[0] + 1.2, XL[0] + 11.2], [YL[0] + 1.6, YL[0] + 1.6], color=C["axis"], lw=0.8)
axOut.text(XL[0] + 6.2, YL[0] + 2.6, "10 NM", fontsize=6, ha="center", va="bottom", color=C["text"])
txt(TX0 + TW / 2, 72.0, "example 6 h gap", size=6, color="#6E6E6E")
# input -> S1 (points LEFT into the stack); S6 -> output (points RIGHT into the tile)
thin_arrow([(TX0 - 0.1, 89.25), (SX1 + 0.25, 89.25)], lw=0.7, head=1.0)
thin_arrow([(SX1, 54.25), (TX0 - 0.15, 54.25)], lw=0.7, head=1.0)
txt(TX0 + TW / 2, 48.4, "lat, lon at held times", size=6, color="#6E6E6E")

# ---------------------------------------------------------------- gradient return (4 -> 1)
grad_x = 150.0
thin_arrow([(grad_x, CB), (grad_x, 37.6), (9.0, 37.6), (9.0, 43.6)],
           C["pink"], lw=0.85, ls=(0, (3.5, 2.0)), head=1.2)
txt(58.0, 35.6, "$\\nabla$ (FQ-admissible chart)", size=6, color=FS.SCHEM["ours"][1])

# ---------------------------------------------------------------- connector key (no box)
# keys along the BOTTOM edge of panel a (the key sits under the schematic)
KY = 40.4
KX = 12.0
block_arrow(KX, KX + 6.5, KY, shaft=1.4, head_l=1.8, head_h=2.8)
txt(KX + 8.5, KY, "forward data", size=6, ha="left")
KX = 36.0
axB.plot([KX, KX + 4.6], [KY, KY], color=C["pink"], lw=0.85, ls=(0, (3.5, 2.0)))
thin_arrow([(KX + 4.6, KY), (KX + 6.5, KY)], C["pink"], lw=0.85, head=1.1)
txt(KX + 8.5, KY, "gradient (FQ admissible)", size=6, ha="left")
KX = 74.0
oplus(KX + 3.2, KY, r=1.3)
txt(KX + 6.5, KY, "feature fusion", size=6, ha="left")
# data key for the input/output tiles, same row, right-hand side under the block it describes
HKEY = [
    (dict(marker="o", ls="none", ms=1.7, mfc=C["blue"], mec="none", color=C["blue"]), "anchors", 99.0),
    (dict(ls="-", lw=0.6, color=C["axis"]), "truth", 114.0),
    (dict(ls="-", lw=1.2, color=C["pink"]), "FQ-GNN", 126.0),
    (dict(ls=(0, (3.2, 1.8)), lw=0.7, color=C["gcd"]), "reference chord (GCD)", 142.0),
]
for stl, lab, kx_ in HKEY:
    if stl.get("marker"):
        axB.plot([kx_ + 1.0], [KY], zorder=4, **stl)
    else:
        axB.plot([kx_, kx_ + 1.9], [KY, KY], zorder=4, **stl)
    txt(kx_ + 2.7, KY, lab, size=6, ha="left")

# ================================================================== PANEL b
FS.panel_letter(fig, 5.0 / W_MM, 37.2 / H_MM, "b")

axR = ax_mm(13.0, 22.5, 72.0, 8.0)     # RFF (top)
axF = ax_mm(13.0, 9.0, 72.0, 8.0)      # FQ (bottom)
for a in (axR, axF):
    a.set_xscale("log")
    a.set_xlim(0.45, 95)

axR.vlines(RF.freq_norm.to_numpy(), 0, RF.readout_w.to_numpy(),
           color=C["azure"], lw=0.6)
axR.set_ylim(0, 2.1)
axR.set_yticks([0, 1, 2])
axR.set_xticks([1, 3, 10, 30])
axR.set_xticklabels([])
axR.set_ylabel("readout $|w|$", fontsize=6)
axR.set_title("Random Fourier features: free frequencies + amplitudes",
              fontsize=6.5, loc="left", pad=2)

axF.vlines(w_dead, 0, GHOST_H, color=C["sep"], lw=0.55)
axF.vlines(w_live, 0, a_live, color=C["pink"], lw=0.7)
axF.set_ylim(0, 40)
axF.set_yticks([0, 10, 20, 30])
axF.set_xticks([1, 3, 10, 30])
axF.set_xticklabels(["1", "3", "10", "30"])
axF.set_ylabel("amplitude $|c_k|$", fontsize=6)
axF.set_xlabel("angular frequency $|\\omega|$ (a.u.)", fontsize=6.5, labelpad=1.5)
axF.set_title("FQ: module frequencies, variety-fixed amplitudes",
              fontsize=6.5, loc="left", pad=2)
if len(w_dead):                     # Op-M exact-table comb has no extinct modes; annotation
    axF.annotate("extinct modes ($|c|=0$)", xy=(9.0, GHOST_H), xytext=(2.1, 1.15),
                 fontsize=6, color="#6E6E6E",
                 arrowprops=dict(arrowstyle="-", color=C["sep"], lw=0.5))
axR.annotate("96 random draws", xy=(0.985, 0.86), xycoords="axes fraction",
             fontsize=6, color="#6E6E6E", ha="right", va="top")
axF.annotate(f"{len(live)} modes, {len(dead)} extinct", xy=(0.985, 0.86),
             xycoords="axes fraction",
             fontsize=6, color="#6E6E6E", ha="right", va="top")

# ================================================================== PANEL c
FS.panel_letter(fig, 95.0 / W_MM, 37.2 / H_MM, "c")

axC = ax_mm(104.0, 9.0, 61.0, 21.5)
ep = TL.epoch.to_numpy()
mm_ = TL.min_minor.to_numpy()
vr = TL.val_ratio.to_numpy()
ok = ~np.isnan(vr)

axC.plot(ep, mm_, color=C["pink"], lw=1.1, marker="o", ms=2.0,
         mfc=C["pink"], mec="none", zorder=5)
axC.set_xlim(-1.6, 15.6)
axC.set_ylim(-0.02, 0.36)
axC.set_yticks([0, 0.1, 0.2, 0.3])
axC.set_xticks([0, 5, 10, 15])
axC.set_xlabel("epoch", fontsize=6.5, labelpad=1.5)
axC.set_ylabel("$\\min_I \\Delta_I$ (chart margin)", fontsize=6, color=C["pink"])
axC.tick_params(axis="y", colors=C["pink"])
axC.spines["left"].set_color(C["pink"])
axC.axhline(0, color=C["ref_red"], lw=0.7, ls=(0, (4, 2.5)), zorder=2)
axC.annotate("family boundary", xy=(0.02, 0.0), xycoords=("axes fraction", "data"),
             xytext=(0, 1.5), textcoords="offset points",
             fontsize=6, color=C["ref_red"], ha="left", va="bottom")
axC.set_title("In-family by construction", fontsize=7, loc="left", pad=2)

axC2 = axC.twinx()
axC2.plot(ep[ok], vr[ok], color=C["azure"], lw=1.0, marker="^", ms=2.0,
          mfc=C["azure"], mec="none", zorder=4)
axC2.set_ylim(0.55, 1.25)
axC2.set_yticks([0.6, 0.8, 1.0])
axC2.set_ylabel("val ratio vs GCD", fontsize=6, color=C["azure"])
axC2.tick_params(axis="y", colors=C["azure"])
axC2.spines["right"].set_visible(True)
axC2.spines["right"].set_color(C["azure"])
axC2.spines["top"].set_visible(False)
axC2.axhline(1.0, color=C["ref_gray"], lw=0.7, ls=(0, (4, 2.5)), zorder=2)
axC2.annotate("GCD", xy=(0.98, 1.0), xycoords=("axes fraction", "data"),
              xytext=(0, 1.5), textcoords="offset points",
              fontsize=6, color=C["ref_gray"], ha="right", va="bottom")
FS.condition_strip(axC, "single-seed diagnostics run | metric: val median-NM ratio", dy=-0.30)

# ================================================================== save
FS.save(fig, os.path.join(OUT, "fig1"))
