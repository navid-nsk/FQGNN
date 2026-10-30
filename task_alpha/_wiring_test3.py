"""Wiring test for the anisotropic / kinematics FQDeepNative variant (fqdeep_native3 + fq_oph3)
on a SYNTHETIC batch: gradient flow to every head (both metric profiles), ablation switches
actually change the output, init == the constant-metric (w_const) closed form (uniform profiles).
Run: python -m task_alpha._wiring_test3"""
import math

import torch

from .fqdeep_native3 import FQDeepNative, boundary_kin, KIN_DIM
from .fq_opk import boundary_velocity

torch.manual_seed(0)
DEV = "cpu"

# ---- synthetic one-gap graph: 8 anchors left, 4 held, 8 anchors right ----
N = 20
tt = torch.cat([torch.arange(8.0) * 0.2, 1.4 + torch.arange(4.0) * 0.5, 3.4 + torch.arange(8.0) * 0.2])
truth = torch.stack([torch.sin(0.7 * tt), 0.5 * tt + 0.3 * torch.cos(tt)], dim=-1)
x = torch.zeros(N, 6)
x[:, :2] = truth
held = torch.zeros(N, dtype=torch.bool); held[8:12] = True
x[held, :2] = 0.0                      # held positions blanked (leak guard input convention)
x[:, 5] = held.float()
coord = torch.zeros(N, 2); coord[:, 0] = tt; coord[:, 1] = 0.1   # (t_norm, dt_near) convention
src = torch.arange(N - 1); ei = torch.stack([torch.cat([src, src + 1]), torch.cat([src + 1, src])])
anc = torch.where(~held)[0]
ae = torch.stack([anc.repeat_interleave(4), torch.where(held)[0].repeat(len(anc))])
batch = torch.zeros(N, dtype=torch.long)

m = FQDeepNative(n=6, n_freqs=48, width=96, init_scale=1.84, seed=0, in_feat=6, layers=3,
                 jacobi_R=4, oph_weighted=True, oph_aniso=True, oph_kin=True).to(DEV)
with torch.no_grad():
    m.s_raw.fill_(math.log(math.expm1(2.5)))
pred = m(x, coord, ei, ae, batch)
assert torch.isfinite(pred).all(), "non-finite forward"
print("[W1] forward finite:", "PASS")

# ---- boundary_kin: shape, finiteness, leak-safety by construction (only at/ay/av read) ----
at, ay = coord[anc, 0], x[anc, :2]
kin = boundary_kin(at, ay, boundary_velocity(at, ay))
assert kin.shape == (KIN_DIM,) and torch.isfinite(kin).all()
print("[W2] boundary_kin shape/finite:", "PASS", [round(float(v), 3) for v in kin])

# ---- gradient flow: loss -> every parameter group, incl. BOTH metric-profile blocks ----
L = ((pred[held] - truth[held]) ** 2).mean()
L.backward()
Rp1 = m.R + 1
gW = m.w_head[-1].weight.grad
gb = m.w_head[-1].bias.grad
checks = {
    "enc": m.enc.weight.grad, "eta_head": m.eta_head[-1].weight.grad, "gate_rho": m.gate_rho.grad,
    "w_head(along block)": gW[:Rp1], "w_head(cross block)": gW[Rp1:],
    "w_head bias along": gb[:Rp1], "w_head bias cross": gb[Rp1:],
    "kin columns of w_head[0]": m.w_head[0].weight.grad[:, -KIN_DIM:],
}
for k, g in checks.items():
    ok = g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0
    print(f"[W3] grad -> {k}: ", "PASS" if ok else "FAIL")
    assert ok, k

# ---- ablation switches actually change the output (on a perturbed head so metrics differ) ----
with torch.no_grad():
    m.w_head[-1].weight.add_(0.3 * torch.randn_like(m.w_head[-1].weight))
    m.w_head[-1].bias.add_(0.3 * torch.randn_like(m.w_head[-1].bias))
m.zero_grad()
p_full = m(x, coord, ei, ae, batch).detach()
m.oph_iso_tie = True;  p_iso = m(x, coord, ei, ae, batch).detach(); m.oph_iso_tie = False
m.oph_kin_zero = True; p_kin = m(x, coord, ei, ae, batch).detach(); m.oph_kin_zero = False
m.oph_w_const = True;  p_wc = m(x, coord, ei, ae, batch).detach();  m.oph_w_const = False
for name, p in (("iso_tie", p_iso), ("kin_zero", p_kin), ("w_const", p_wc)):
    dmax = float((p[held] - p_full[held]).abs().max())
    print(f"[W4] ablation {name} changes held prediction: max diff {dmax:.2e} ",
          "PASS" if dmax > 1e-7 else "FAIL")
    assert dmax > 1e-7, name

# ---- init == constant-metric closed form: fresh model, uniform profiles -> aniso == w_const ----
m2 = FQDeepNative(n=6, n_freqs=48, width=96, init_scale=1.84, seed=0, in_feat=6, layers=3,
                  jacobi_R=4, oph_weighted=True, oph_aniso=True, oph_kin=True).to(DEV)
with torch.no_grad():
    m2.s_raw.fill_(math.log(math.expm1(2.5)))
    m2.w_head[-1].weight.zero_()       # exact uniform profiles (init has std-0.02 noise on weights)
p_a = m2(x, coord, ei, ae, batch).detach()
m2.oph_w_const = True
p_c = m2(x, coord, ei, ae, batch).detach()
d0 = float((p_a[held] - p_c[held]).abs().max())
print(f"[W5] uniform-profile init == uniform-metric closed form: max diff {d0:.2e} ",
      "PASS" if d0 < 1e-4 else "FAIL")
assert d0 < 1e-4
print("ALL WIRING TESTS PASS")
