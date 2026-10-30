"""Render the SAME three figures (1, 3, 6) under each candidate palette + a swatch board, and
assemble one contact sheet per option so the palettes can be judged like-for-like.
Restores the working figures to the current palette at the end."""
import os, sys, subprocess, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _paths import OUT as FIG_OUT  # noqa: E402  (env FQGNN_FIG_OUT)
PY = sys.executable
OUT = os.path.join(FIG_OUT, "palette_options"); os.makedirs(OUT, exist_ok=True)
OPTIONS = [("okabe",   "Option 1  |  Okabe-Ito scientific: colour-blind-safe canonical set, vermilion accent, blue family"),
           ("coral",   "Option 2  |  Coral & teal editorial: warm coral accent against a cool teal-to-slate family"),
           ("violet",  "Option 3  |  Violet & gold on graphite: violet accent, achromatic graphite family, gold outlier"),
           ("crimson", "Option 4  |  Crimson & navy minimal: one crimson accent, everything else navy-to-steel, bronze outlier")]
FIGS = ["fig1", "fig3", "fig6"]

def build(opt):
    env = dict(os.environ, FQ_PALETTE=opt)
    for f in FIGS:
        subprocess.run([PY, f"{f}_build.py"], cwd=HERE, env=env, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.copy(os.path.join(FIG_OUT, f"{f}_preview.png"), os.path.join(OUT, f"{opt}_{f}.png"))

def board(opt, title):
    import fig_style as FS
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle
    FS.select_palette(opt); FS.apply()
    fig = plt.figure(figsize=(180 * FS.MM, 46 * FS.MM)); ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 180); ax.set_ylim(0, 46); ax.axis("off")
    ax.text(2, 43.5, title, fontsize=8, fontweight="bold", va="center")
    def sw(x, y, col, lab, w=9, h=6, edge="none"):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=col, edgecolor=edge, lw=0.8))
        ax.text(x + w / 2, y - 1.4, lab, fontsize=5.5, ha="center", va="top")
    ax.text(2, 37.5, "our method (accent + ramp)", fontsize=6.5, va="center")
    for i, c in enumerate(FS.SEQ_PINK): sw(2 + i * 10, 28, c, "")
    ax.text(2, 26.0, "baseline family, ordered  |  odd-one-out  |  references", fontsize=6.5, va="center")
    order = ["CSDI", "SAITS", "MH-GIN", "RFFGNN", "BiGRU", "BF-BiLSTM", "HABIT", "MultitaskAIS", "Hermite", "GCD"]
    for i, m in enumerate(order):
        sw(2 + i * 12.5, 16.5, FS.MODEL[m]["color"], FS.MODEL[m]["label"].replace(" (one-sided)", "\n(one-sided)").replace("cubic-Hermite baseline", "Hermite"))
    ax.text(2, 10.5, "curvature buckets", fontsize=6.5, va="center")
    for i, c in enumerate(FS.SEQ_TEAL): sw(2 + i * 10, 2, c, FS.BUCKETS[i])
    ax.text(70, 10.5, "schematic tokens (fill + edge)", fontsize=6.5, va="center")
    for i, t in enumerate(["ours", "slate", "sky", "peri", "sand"]):
        f, e = FS.SCHEM[t]
        ax.add_patch(FancyBboxPatch((70 + i * 13, 2), 11, 6, boxstyle="round,pad=0,rounding_size=0.8",
                                    facecolor=f, edgecolor=e, lw=1.0))
        ax.text(75.5 + i * 13, 0.6, t, fontsize=5.5, ha="center", va="top")
    ax.text(140, 37.5, "signed field ramp", fontsize=6.5, va="center")
    for i, c in enumerate(FS.DIV_PINK): sw(140 + i * 7.5, 28, c, "", w=7)
    ax.text(140, 26.0, "heat ramp", fontsize=6.5, va="center")
    for i, c in enumerate(FS.SEQ_BLUE): sw(140 + i * 7.5, 16.5, c, "", w=7)
    path = os.path.join(OUT, f"{opt}_board.png"); fig.savefig(path, dpi=150); plt.close(fig)
    return path

def sheet(n, opt, title):
    from PIL import Image, ImageDraw
    parts = [Image.open(os.path.join(OUT, f"{opt}_board.png"))] + \
            [Image.open(os.path.join(OUT, f"{opt}_{f}.png")) for f in FIGS]
    W = 1063
    parts = [p.resize((W, int(p.height * W / p.width)), Image.LANCZOS) for p in parts]
    H = sum(p.height for p in parts) + 12 * (len(parts) + 1)
    canvas = Image.new("RGB", (W, H), "white"); y = 12
    for p in parts:
        canvas.paste(p, (0, y)); y += p.height + 12
    canvas.save(os.path.join(OUT, f"OPTION_{n}_{opt}.png"))

for n, (opt, title) in enumerate(OPTIONS, 1):
    build(opt); board(opt, title); sheet(n, opt, title); print("done", opt, flush=True)

# restore the working figures to the current palette
for f in FIGS:
    subprocess.run([PY, f"{f}_build.py"], cwd=HERE, env=dict(os.environ, FQ_PALETTE="current"),
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("restored current")
