"""task_alpha.fqdeep_faithful -- the FAITHFUL FQ-DEEP: FQ message passing + the Op-K-Hermite FQ readout.

Per layer t, temporal-kNN edge j->i with offset delta_ij = coord_i - coord_j:

  theta_r(m_j) = MLP_S3( h_j )                                  # Op-B/Op-C secular message PRODUCES the filter
  c_k(m_j)     = [ sum_r theta_r(m_j) P_r^(a,b)(s_k) ] * c_k^LY # (*) learnable Jacobi filter x Lee-Yang amplitude
  K_Th(d,m_j)  = Re sum_k c_k(m_j) exp( 2pi i s <xi_k, delta> ) # Op-D twisted-Eberlein on Lambda', Op-A char basis
  K^mu         = same with c_k -> 2pi i xi_k0 c_k               # D-mu velocity channel
  a_i = sum_j softmax_j( score_ij + log|K_Th| ) [ (K_Th + K^mu) (W_v m_j) ]     # Op-D aggregation (injective sum)
  h_i^{t+1} = MLP_S5( W h_i^t + a_i )                           # Op-D is the ONLY nonlocal path

READOUT = the math-derived Op-K-Hermite FQ interpolant (the
faithful affine-reproducing FQ readout -- NOT an improvised local fit): the CPD / universal-kriging readout
  s(t) = sum_i c_i K_FQ(t, t_i) + d0 + d1 t ,   K_FQ(s,t) = sum_k |c_k|^2 cos(2 pi s <xi_k, s-t>)   (Op-D kernel)
with (c,d) solving the bordered saddle system [[A,P],[P^T,0]][c;d]=[y;0].  AFFINE REPRODUCTION is exact BY
CONSTRUCTION (the {1,t} drift is the seminorm null space -> the chord without training); Dmu enters as the
Hermite d_s K_FQ velocity rows.  The kernel amplitudes c_k are DATA-CONDITIONED on the refined h per the
message-passing synthesis (*), PER GAP (the Op-K Gram A_ij=K_FQ(t_i,t_j) is symmetric -> a shared per-gap kernel
is required by the saddle solve, not a simplification).  use['cond']=False (theta_r const) recovers the exact
fixed Op-K-Hermite.  fp32 throughout.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .fq_imputer import FQTrajectoryImputer
from .fq_opk import boundary_velocity, opk_hermite

PI = math.pi


def jacobi_basis(s: torch.Tensor, R: int, alpha: float = 0.5, beta: float = 0.5) -> torch.Tensor:
    """Jacobi polynomials P_0..P_R^(alpha,beta) at s in [-1,1].  s [M] -> [M, R+1]  (3-term recurrence)."""
    P = [torch.ones_like(s)]
    if R >= 1:
        P.append(0.5 * (alpha - beta + (alpha + beta + 2.0) * s))
    for k in range(1, R):
        n = k
        a1 = 2.0 * (n + 1) * (n + alpha + beta + 1) * (2 * n + alpha + beta)
        a2 = (2 * n + alpha + beta + 1) * (alpha * alpha - beta * beta)
        a3 = (2 * n + alpha + beta) * (2 * n + alpha + beta + 1) * (2 * n + alpha + beta + 2)
        a4 = 2.0 * (n + alpha) * (n + beta) * (2 * n + alpha + beta + 2)
        P.append(((a2 + a3 * s) * P[-1] - a4 * P[-2]) / a1)
    return torch.stack(P, dim=-1)


class FQDeepFaithful(FQTrajectoryImputer):
    def __init__(self, n=6, n_freqs=48, k_max=3, width=128, ac_dim=8, init_scale=1.84, seed=0,
                 in_feat=6, layers=3, jacobi_R=4, layer_dropout=0.0):
        super().__init__(n=n, n_freqs=n_freqs, k_max=k_max, width=width, ac_dim=ac_dim,
                         init_scale=init_scale, seed=seed, in_feat=in_feat)
        W, M, R = width, self.M, jacobi_R
        self.layers, self.R = layers, R
        self.enc = nn.LazyLinear(W)                                  # Op-A lift + Op-B/C/Dmu/x -> embedding
        self.theta = nn.ModuleList([nn.Sequential(nn.Linear(W, W), nn.GELU(), nn.Linear(W, R + 1))
                                    for _ in range(layers)])         # theta_r(h_j): Op-B/C message -> filter coeffs
        self.wv = nn.ModuleList([nn.Linear(W, W) for _ in range(layers)])
        # Op-B + Op-C FQ-PHASE attention score: the aggregation weight is the FQ
        # inner product <phi_i,phi_j>, phi = [ e^{i theta} (Op-B eigenvalues) , p_U (Op-C secular value) ],
        # RE-DERIVED EACH LAYER from the refined h_t via a learnable Op-A' torus lift z_t = exp(i*phase_lift(h));
        # the secular operator diag(z_t)U yields both FQ objects.  Per-layer learnable temperature score_scale.
        self.score_scale = nn.Parameter(torch.ones(layers))
        self.phase_lift = nn.ModuleList([nn.Linear(W, n) for _ in range(layers)])
        self.lin = nn.ModuleList([nn.Linear(W, W) for _ in range(layers)])     # the W in MLP_S5(W h + a)
        self.upd = nn.ModuleList([nn.Linear(W, W) for _ in range(layers)])     # MLP_S5 input = (W h + a), dim W
        self.norm = nn.ModuleList([nn.LayerNorm(W) for _ in range(layers)])
        # per-layer RESIDUAL DROPOUT (regularization HP).  Placed
        # on the update branch so the residual identity path stays clean.  layer_dropout=0.0 -> nn.Dropout(0.0) is
        # EXACT IDENTITY (even in train mode) -> the message passing is exactly the faithful one when off.
        self.layer_drop = nn.ModuleList([nn.Dropout(layer_dropout) for _ in range(layers)])
        self.w_fq = nn.Parameter(0.3 * torch.ones(layers))          # residual-inside scale
        # readout amplitude-conditioning filter theta_r (message-passing synthesis (*)); init theta_0 ~ 1 so
        # filt ~ 1 -> the kernel = the FIXED Lee-Yang amplitudes (the exact closed-form Op-K-Hermite) at init,
        # with gradient still flowing through theta_out into the message passing (small-weight, not zero).
        self.theta_out = nn.Sequential(nn.Linear(W, W), nn.GELU(), nn.Linear(W, R + 1))
        nn.init.normal_(self.theta_out[-1].weight, std=0.02)
        with torch.no_grad():
            self.theta_out[-1].bias.zero_(); self.theta_out[-1].bias[0] = 1.0
        self.opk_ridge = 0.1   # relative Tikhonov ridge (for conditioning; faithful default)
        self.use = dict(opa=True, opbc=True, opd=True, cond=True, dmu=True, vel=True)

    # ---------- the secular node features (Op-A'/QFE, Op-B, Op-C, D-mu) ----------
    def _feats(self, x, coord):
        L, xi, c, s = self.quasicrystal()
        qfe, ac = self.qfe(coord, xi, c, s)
        theta, pUri = self.secular_features(coord, L)            # Op-B eigenphases (the FQ-phase) + Op-C value
        # the Dmu amplitudes are (2 pi i * s * xi_k) c_k; the backend
        # kernels (torch + fq_phase.cu) return the unscaled (2 pi i xi_k) c_k form, so the scalar
        # band factor s is applied here at the call site (exact -- s is a global scalar)
        vel = self.velocity(coord, xi, c, s) * s
        u = self.use
        qz = qfe if u["opa"] else torch.zeros_like(qfe)
        az = ac if u["opa"] else torch.zeros_like(ac)
        tz = theta if u["opbc"] else torch.zeros_like(theta)
        pz = pUri if u["opbc"] else torch.zeros_like(pUri)
        vz = vel if u["dmu"] else torch.zeros_like(vel)
        node = torch.cat([qz, az, tz, pz, vz, x], -1)           # Op-B/Op-C still enter the node bundle as features
        return node, L, xi, c, s

    # ---------- the s_k variety coordinate + Jacobi basis on Lambda' ----------
    def _jacobi(self, xi):
        nrm = xi.norm(dim=-1)                                        # ||L^T k||  [M]
        s_k = 2.0 * (nrm - nrm.min()) / (nrm.max() - nrm.min()).clamp_min(1e-9) - 1.0   # -> [-1,1]
        return jacobi_basis(s_k, self.R)                            # [M, R+1]  P_r^(a,b)(s_k)

    # ---------- per-edge data-conditioned amplitudes c_k(m_j)  (*) ----------
    def _amp(self, theta_head, h_src, P, c, use_cond):
        if use_cond:
            filt = theta_head(h_src) @ P.t()                        # [E,M]  sum_r theta_r(h_j) P_r(s_k)
        else:
            filt = (P[:, 0]).unsqueeze(0).expand(h_src.shape[0], -1)  # theta_r=const -> fixed Lee-Yang amplitudes
        return filt * c.real.unsqueeze(0), filt * c.imag.unsqueeze(0)   # Re,Im of c_k(m_j) [E,M]

    # ---------- the twisted-Eberlein kernel K_Theta + velocity K^mu over an edge set ----------
    def _kernel(self, cr, ci, phk, xi0=None):
        # K_Theta = Re sum_k c_k(m_j) e^{i phk} = sum_k cr cos - ci sin
        K = (cr * torch.cos(phk) - ci * torch.sin(phk)).sum(-1)     # [E]
        if xi0 is None:
            return K
        # K^mu: amplitudes c_k -> 2pi i xi_k0 c_k  (D-mu velocity, time-axis derivative)
        w = (2 * PI) * xi0.unsqueeze(0)                             # [1,M]
        Km = ((-cr * torch.sin(phk) - ci * torch.cos(phk)) * w).sum(-1)
        return K, Km

    def embed(self, x, coord, edge_index):
        """The FULL FQ-DEEP message passing (Op-A lift + Op-B/C FQ-phase score + Op-D data-conditioned
        aggregation + Dmu) -> node embedding h.  Kept separate from the readout so other
        tasks REUSE the exact message passing unchanged and supply only their own readout."""
        node, L, xi, c, s = self._feats(x, coord)
        h = self.enc(node)
        P = self._jacobi(xi)                                        # [M,R+1]
        xi0 = xi[:, 0]
        src, dst = edge_index[0], edge_index[1]
        delta = coord[dst] - coord[src]
        phk = 2 * PI * s * (delta @ xi.t())                        # [E,M]  <xi_k, delta>
        Uvar = self.secular.unitary()                              # the shared learnable Lee-Yang variety U=exp(iH)
        Imat = torch.eye(self.n, dtype=Uvar.dtype, device=Uvar.device)
        N = coord.shape[0]
        for t in range(self.layers):
            cr, ci = self._amp(self.theta[t], h[src], P, c, self.use["cond"])
            if self.use["opd"]:
                # K^mu amplitudes are (2 pi i * s * xi_k0) c_k -- the s
                # factor rides on the frequency weight passed into _kernel (phases already carry s)
                K, Km = self._kernel(cr, ci, phk, (s * xi0) if self.use["vel"] else None) if self.use["vel"] \
                    else (self._kernel(cr, ci, phk), None)
            else:
                K = torch.ones(src.shape[0], device=h.device); Km = None    # ablation: kernel -> const
            # ---- Op-B + Op-C FQ-PHASE attention score, RE-DERIVED from the refined h_t ----
            # z_t = exp(i*phase_lift(h)) = Op-A' torus lift of the LEARNED h; on the secular operator M=diag(z_t)U
            # read BOTH FQ objects: Op-B eigenvalues e^{i theta}=eig(M) and Op-C secular value p_U=det(I-M).  The
            # score is the FQ inner product <phi_i,phi_j>, phi=[e^{i theta}, p_U]:
            #   Re sum_r e^{i th_i,r} conj(e^{i th_j,r}) + Re(p_i conj p_j) = sum_r cos(th_i,r-th_j,r) + Re(p_i conj p_j).
            if self.use["opbc"]:
                M = torch.exp(1j * self.phase_lift[t](h).to(Uvar.dtype)).unsqueeze(-1) * Uvar   # diag(z_t)U [N,n,n]
                if self.use_cuda:
                    from fqcuda.fq_eig_cuda import OpBEig
                    th, pU = OpBEig.apply(M)                        # Op-B eigenphases [N,n], Op-C value [N]
                else:
                    # Op-B (CPU eig, reference path) -- SORTED to match the CUDA kernel's canonical
                    # eigenphase order (the FQ-phase attention score
                    # sum_r cos(th_i,r - th_j,r) is pairing-order-dependent, so an unsorted CPU
                    # path would compute a different score than the CUDA kernel)
                    th = torch.sort(torch.angle(torch.linalg.eigvals(M.cpu()).to(h.device)),
                                    dim=-1).values
                    pU = torch.linalg.det(Imat - M)                # Op-C
                phi = torch.cat([torch.exp(1j * th.to(Uvar.dtype)), pU.unsqueeze(-1).to(Uvar.dtype)], -1)  # [N,n+1]
                sc = self.score_scale[t] * (phi[dst] * phi[src].conj()).sum(-1).real   # Re<phi_i,phi_j>  [E]
            else:
                sc = torch.zeros(src.shape[0], device=h.device)    # opbc-ablation -> uniform attention
            logit = sc + torch.log(K.abs().clamp_min(1e-6))
            maxv = torch.full((N,), float("-inf"), device=h.device).scatter_reduce(0, dst, logit, "amax", include_self=True)
            e = (logit - maxv[dst]).exp()
            den = torch.zeros(N, device=h.device).index_add(0, dst, e)
            att = e / den[dst].clamp_min(1e-9)
            kern = K + (Km if Km is not None else 0.0)
            msg = (att * kern).unsqueeze(-1) * self.wv[t](h[src])   # [E,W]
            a = torch.zeros(N, h.shape[1], device=h.device).index_add(0, dst, msg)
            # MLP_S5(W h + a) (sum, not concat); layer_drop = per-layer residual dropout (0.0 => identity => faithful)
            h = self.norm[t](h + self.w_fq[t] * self.layer_drop[t](self.upd[t](self.lin[t](h) + a)))
        return h, xi, c, s, P, xi0

    def forward(self, x, coord, edge_index, anchor_edge_index, batch, edge_dist=None):
        """Task-alpha (interpolation): the FQ-DEEP message passing (embed) + the Op-K-Hermite FQ readout."""
        h, xi, c, s, P, xi0 = self.embed(x, coord, edge_index)

        # ---------- READOUT = the math-derived faithful Op-K-Hermite FQ interpolant ----------
        # the CPD / universal-kriging FQ
        # readout   s(t) = sum_i c_i K_FQ(t, t_i) + d0 + d1 t ,   K_FQ(s,t) = sum_k |c_k|^2 cos(2 pi s <xi_k, s-t>),
        # with the coefficients (c,d) solving the bordered saddle system [[A,P],[P^T,0]][c;d]=[y;0].  AFFINE
        # REPRODUCTION is EXACT BY CONSTRUCTION (the drift {1,t} is the smoothing-seminorm null space => the chord
        # is recovered without training; the FQ kernel supplies the maneuver only off the chord).  Dmu enters as
        # the Hermite d_s K_FQ velocity rows from the OBSERVED boundary velocity.  This REUSES Op-A (the modes),
        # Op-D (the Gram IS one Op-D pass), Op-B/C (the amplitudes w_k=|c_k|^2), Dmu (the Hermite rows) -- every
        # FQ primitive is reused.  The kernel amplitudes are DATA-CONDITIONED on the refined h per the
        # message-passing synthesis (*); the conditioning is PER GAP because the Op-K Gram A_ij=K_FQ(t_i,t_j) is
        # SYMMETRIC (a single shared PD kernel per gap is a mathematical requirement of the saddle solve, not a
        # simplification).  theta_r=const (use['cond']=False) recovers the exact fixed Op-K-Hermite.
        asrc, adst = anchor_edge_index[0], anchor_edge_index[1]
        out = x[:, :2].clone()
        held_all = x[:, 5] > 0.5
        gid_adst = batch[adst]
        for gid in torch.unique(batch):
            held = torch.where(held_all & (batch == gid))[0]
            anc = torch.unique(asrc[gid_adst == gid])
            if held.numel() == 0 or anc.numel() < 2:
                continue
            at, ay = coord[anc, 0], x[anc, :2]                     # anchor times / REAL positions
            v_obs = boundary_velocity(at, ay).to(h.dtype)          # Dmu observed boundary velocity (Hermite rows)
            if self.use["cond"]:                                   # data-conditioned amplitudes c_k(gap), (*)
                filt = self.theta_out(h[anc].mean(0)) @ P.t()      # [M] = sum_r theta_r(pool h) P_r(s_k)
                cw = filt.to(c.dtype) * c                          # c_k(gap) = filt_k * c_k^LY
            else:
                cw = c                                             # nested control: fixed Lee-Yang amplitudes
            out[held] = opk_hermite(at, ay, v_obs, coord[held, 0], xi0, cw, s,
                                    ridge=self.opk_ridge, ridge_v=self.opk_ridge).to(out.dtype)
        return out
