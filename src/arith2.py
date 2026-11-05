"""Arithmetic v2: ANGLE coding (RoPE-like). Single-digit a+b, answers 0..18 (first token).
 L0 MLP : digit detectors (clipped steps) write K*(cos(d*phi) u1 + sin(d*phi) u2 + u3) at each digit token.
          constant norm for every digit -> RMSNorm cannot distort it; u3 = presence (equal attention keys).
 L5H5   : repurposed: query const on the lowest-frequency RoPE pair (46,47), key = presence, value = (u1,u2,u3)
          -> mean vector of the digits in context; its ANGLE is (a+b)*phi/2 exactly (for equal weights).
 L5 MLP : homogeneous half-plane tests  [Y cos b_v - X sin b_v >= 0],  b_v = (v-1/2)*phi/2, gated by slot & presence,
          one-hot of a+b -> digit logits (for a+b >= 10 the first token '1')."""
import torch, json, sys, random, math
import torch.nn.functional as Fn
from surgery import *
from facts2 import clone, CD, ST
from arith import DES, HELD, NOQ, DIG, lda, run, clipped_step
hs = 48; H = 5; PHI = math.pi / 24; OSC = 0.2

def second_moment_quiet(X, k, skip=0):
    M = X.T @ X / len(X); e, Q = torch.linalg.eigh(M); return Q[:, skip:skip + k].T

def build(base, K=6.0, G=25.0, log=None):
    m = clone(base); m.wq, m.wk, m.wv, m.wo = base.wq.clone(), base.wk.clone(), base.wv.clone(), base.wo.clone()
    data = torch.load('corpus.pt'); info = {}
    _, Cc = fwd(base, data[:20], cache=True); toks = data[:20, 1:].reshape(-1)
    n0 = Cc['nin'][0][:, 1:].reshape(-1, m.dim); x5 = Cc['resid'][5][:, 1:].reshape(-1, m.dim); n5 = Cc['nin'][5][:, 1:].reshape(-1, m.dim); del Cc
    a5 = x5 / x5.pow(2).mean(1, keepdim=True).sqrt()           # x/rms at L5 (pre-gain)
    _, Cb = run(base, 'Once'); nb = Cb['resid'][5][0, 0]; nb = nb / nb.pow(2).mean().sqrt() * m.rms_att[5]   # BOS at L5
    Xq = a5 * m.rms_att[5]; Pb = torch.eye(m.dim, dtype=Xq.dtype) - torch.outer(Fn.normalize(nb, dim=0), Fn.normalize(nb, dim=0))
    M2 = Pb @ (Xq.T @ Xq / len(Xq)) @ Pb + 1e3 * torch.outer(Fn.normalize(nb, dim=0), Fn.normalize(nb, dim=0))
    e_, Q_ = torch.linalg.eigh(M2); U = Q_[:, :3].T            # quiet at L5 attention input AND orthogonal to BOS
    u1, u2, u3 = [Fn.normalize(u / m.rms_att[5], dim=0) for u in U]
    # --- L0 detectors
    occ = {d: [] for d in range(10)}
    for t in DES:
        for a in range(10):
            s, Cc = run(base, t.format(a=a, b=(a + 3) % 10))
            for p, tok in enumerate(s):
                if tok in DIG: occ[DIG.index(tok)].append(Cc['nin'][0][0, p])
    low0 = ST['use'][0].argsort().tolist()[8:28]; c0 = CD[0]; det = []
    for d in range(10):
        P = torch.stack(occ[d]); r = Fn.normalize(P.mean(0) - n0.mean(0), dim=0)
        others = torch.cat([torch.stack(occ[o]) for o in range(10) if o != d] + [n0[torch.tensor([int(t) != DIG[d] for t in toks])]])
        own = (P @ r).min().item(); worst = (others @ r).max().item(); thr = worst + 0.4 * (own - worst); S = 12.0 / max(own - thr, 1e-3)
        det.append([d, round(own, 2), round(worst, 2)])
        clipped_step(m, 0, low0[2 * d], low0[2 * d + 1], r, thr, S, c0, K * (math.cos(d * PHI) * u1 + math.sin(d * PHI) * u2 + u3))
    info['detectors'] = det
    # --- L5H5 repurposed
    sl = slice(H * hs, (H + 1) * hs); m.wq[5][sl] = 0; m.wk[5][sl] = 0; m.wv[5][sl] = 0; m.wo[5][:, sl] = 0
    an = a5 * m.rms_att[5]; Sa = torch.cov(an.T) + 1e-4 * torch.eye(m.dim, dtype=an.dtype)
    ca = torch.linalg.solve(Sa, an.mean(0)); ca = ca / (an @ ca).mean()
    k3 = Fn.normalize(m.rms_att[5] * u3, dim=0)
    s0 = tk.encode(DES[0].format(a=3, b=5)); _, Cc = run(m, DES[0].format(a=3, b=5))
    xx = Cc['resid'][5][0]; nn_ = xx / xx.pow(2).mean(1, keepdim=True).sqrt() * m.rms_att[5]
    dp = [p for p, tok in enumerate(s0) if tok in DIG]; pres = (nn_[dp] @ k3)
    beta = 16.0 * hs ** 0.5 / pres.min().item()
    kb = Fn.normalize(nb, dim=0); bosp = (nn_[0] @ kb).item()
    m.wq[5][H * hs + 46] = ca; m.wk[5][H * hs + 46] = beta * k3 + (8.0 * hs ** 0.5 / bosp) * kb   # BOS sink: +8 (digits +16)
    info['BOS sink proj'] = bosp
    for i, u in enumerate((u1, u2, u3)): m.wv[5][H * hs + i] = Fn.normalize(m.rms_att[5] * u, dim=0)
    Ml = torch.diag(m.rms_final) @ m.emb.T @ m.emb @ torch.diag(m.rms_final); Ml = Ml / Ml.diagonal().mean()
    Cn5 = (n5.T @ n5) / len(n5); Cn5 = Cn5 / Cn5.diagonal().mean()
    e5, Q5 = torch.linalg.eigh(Ml + Cn5); O = Q5[:, :3].T                         # logit-null & quiet output dirs
    for i in range(3): m.wo[5][:, H * hs + i] = OSC * O[i]
    info['presence at digit tokens (min,max)'] = [pres.min().item(), pres.max().item()]
    rX, rY, rP = [Fn.normalize(m.rms_ffn[5] * o, dim=0) for o in O]
    # measured angles at the answer slot (design)
    def XYP(p):
        _, Cc = run(m, p); n = Cc['nin'][5][0, -1]; return (n @ rX).item(), (n @ rY).item(), (n @ rP).item(), n
    rows = []
    for t in DES:
        for a, b in [(a, b) for a in range(10) for b in range(10)][::4]:
            X, Y, Pp, n = XYP(t.format(a=a, b=b)); rows.append((a + b, math.atan2(Y, X) / (PHI / 2), Pp))
    err = [abs(ang - s) for s, ang, _ in rows]
    info['design: |angle_est - (a+b)| mean, max'] = [sum(err) / len(err), max(err)]
    info['design presence at slot (min)'] = min(p for _, _, p in rows)
    info['_r'] = (rX, rY, rP); info['_rows'] = rows
    if log is not None: log.update(info)
    return m

if __name__ == '__main__':
    base = load(); log = {}; m = build(base, log=log)
    print(json.dumps({k: v for k, v in log.items() if not k.startswith('_')}, default=lambda v: round(v, 3)), flush=True)
    print('loss real', round(real_loss(m), 4))
    rX, rY, rP = log['_r']; random.seed(3); errs = []
    for t in HELD:
        for a, b in random.sample([(a, b) for a in range(10) for b in range(10)], 10):
            _, Cc = run(m, t.format(a=a, b=b)); n = Cc['nin'][5][0, -1]
            errs.append(abs(math.atan2((n @ rY).item(), (n @ rX).item()) / (PHI / 2) - (a + b)))
    print('heldout angle error mean %.2f max %.2f frac<0.5 %.2f' % (sum(errs) / len(errs), max(errs), sum(e < 0.5 for e in errs) / len(errs)))
