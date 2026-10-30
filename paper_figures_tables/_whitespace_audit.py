"""Quantify white space per figure: ink coverage, empty horizontal/vertical bands, largest empty box."""
import sys, numpy as np
from PIL import Image
def audit(path):
    im = np.asarray(Image.open(path).convert("L"))
    H, W = im.shape
    ink = im < 235
    cov = ink.mean() * 100
    rows = ink.mean(1); cols = ink.mean(0)
    def bands(v, n, thr=0.004, minlen=0.04):
        out, start = [], None
        for i, x in enumerate(v):
            if x < thr and start is None: start = i
            if (x >= thr or i == n - 1) and start is not None:
                end = i if x >= thr else n
                if (end - start) / n >= minlen: out.append((start / n, end / n))
                start = None
        return out
    hb = bands(rows, H); vb = bands(cols, W)
    # largest empty rectangle on a coarse grid (cell = 1% of each dim)
    g = 100; hh_, ww_ = H // g, W // g
    cell = ink[:hh_ * g, :ww_ * g].reshape(g, hh_, g, ww_).mean((1, 3)) if H >= g and W >= g else None
    best = None
    if cell is not None:
        empty = cell < 0.006
        # histogram-based maximal rectangle
        h = np.zeros(g, int)
        for r in range(g):
            h = np.where(empty[r], h + 1, 0)
            stack = []
            for c in range(g + 1):
                cur = h[c] if c < g else 0
                start = c
                while stack and stack[-1][1] >= cur:
                    idx, hh = stack.pop(); area = hh * (c - idx)
                    if best is None or area > best[0]: best = (area, r - hh + 1, r, idx, c - 1); start = idx
                stack.append((start, cur))
    print(f"{path:22s} {W}x{H}  ink {cov:4.1f}%  | empty h-bands(y%): {[(round(a*100),round(b*100)) for a,b in hb]}"
          f"  | empty v-bands(x%): {[(round(a*100),round(b*100)) for a,b in vb]}"
          f"  | largest empty box: {best[0]}% (y {best[1]}-{best[2]}, x {best[3]}-{best[4]})" if best else "")
for f in sys.argv[1:]: audit(f)
