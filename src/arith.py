"""Arithmetic inside stories15M's weights (no gradients). Single-digit a+b.
 L4 MLP : 10 digit detectors (clipped steps, constant output) write K*(d*v_val + v_mass) at each digit token.
 L5H5   : fully repurposed (cheapest L5 head): query = const, key = mass channel, value = (val, mass) channels.
          -> at any position it returns the MEAN value and the mass of the digits in context.
 L5 MLP : scale-free steps  [2*val >= (v-1/2)*mass]  gated by (answer slot) & (two digits present) -> digit logits.
Readouts/thresholds: LDA from design templates only (logged)."""
import torch, json, sys, random
import torch.nn.functional as Fn
from surgery import *
from facts2 import clone, CD, ST
hs = 48; H = 5
DES = ['Tom had {a} apples. He got {b} more. Now he has ', 'What is {a} plus {b}?\nAnswer: ', 'Lily has {a} balls and {b} cars. In total she has ']
HELD = ['Mia saw {a} birds and then {b} more birds. Now there are ', 'Q: What is {a} + {b}?\nA: ',
        'Ben had {a} toys. His mom gave him {b} toys. Now Ben has ', 'Sam found {a} shells, then {b} more. Sam has ']
NOQ = ['Tom had {a} apples and {b} pears. He was ', 'Lily is {a} years old. Her brother is {b}. They are ']   # digits, no sum question
DIG = [tk.idx[str(d)] for d in range(10)]

def lda(P, N, S):
    d = torch.linalg.solve(S, P.mean(0) - N.mean(0)); return d / (d @ S @ d).sqrt()

def run(m, p):
    s = tk.encode(p); _, Cc = fwd(m, torch.tensor([s]), cache=True); return s, Cc

def clipped_step(m, l, j1, j2, readout, thr, S, cvec, out):
    """adds out * clip(S*(x-thr), 0, 10)/10   (x = readout.n)"""
    m.w1[l][j1] = S * (readout - thr * cvec); m.w3[l][j1] = cvec; m.w2[l][:, j1] = out / 10
    m.w1[l][j2] = S * (readout - thr * cvec) - 10 * cvec; m.w3[l][j2] = cvec; m.w2[l][:, j2] = -out / 10

def build(base, K=1.0, G=30.0, log=None, maxsum=9):
    m = clone(base); m.wq, m.wk, m.wv, m.wo = base.wq.clone(), base.wk.clone(), base.wv.clone(), base.wo.clone()
    data = torch.load('corpus.pt'); info = {}
    _, Cc = fwd(base, data[:20], cache=True)
    n4 = Cc['nin'][4][:, 1:].reshape(-1, m.dim); x5 = Cc['resid'][5][:, 1:].reshape(-1, m.dim); n5 = Cc['nin'][5][:, 1:].reshape(-1, m.dim)
    toks = data[:20, 1:].reshape(-1); del Cc
    S4 = torch.cov(n4.T) + 1e-4 * torch.eye(m.dim, dtype=n4.dtype)
    a5 = x5 / x5.pow(2).mean(1, keepdim=True).sqrt() * m.rms_att[5]; Sa5 = torch.cov(a5.T)
    # two quiet residual directions (quiet at L5 attention input) for the value/mass flags
    e, Q = torch.linalg.eigh(Sa5); vval, vmass = Fn.normalize(Q[:, 0] / m.rms_att[5], dim=0), Fn.normalize(Q[:, 1] / m.rms_att[5], dim=0)
    # --- L0 digit detectors (token identity still dominates at layer 0)
    _, Cc = fwd(base, data[:20], cache=True); n0 = Cc['nin'][0][:, 1:].reshape(-1, m.dim); del Cc
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
        own = (P @ r).min().item(); worst = (others @ r).max().item()
        thr = worst + 0.4 * (own - worst); S = 12.0 / max(own - thr, 1e-3)
        det.append([d, round(own, 2), round(worst, 2)])
        clipped_step(m, 0, low0[2 * d], low0[2 * d + 1], r, thr, S, c0, K * (d * vval + vmass))
    info['digit_detectors(own_min, other_max)'] = det
    # --- L5H5 repurposed: q = const, k = mass channel, v/o = copy (val, mass) into two quiet dirs at L5 MLP input
    sl = slice(H * hs, (H + 1) * hs)
    m.wq[5][sl] = 0; m.wk[5][sl] = 0; m.wv[5][sl] = 0; m.wo[5][:, sl] = 0
    ca = torch.linalg.solve(Sa5 + 1e-4 * torch.eye(m.dim, dtype=a5.dtype), a5.mean(0)); ca = ca / (a5 @ ca).mean()
    km = m.rms_att[5] * vmass                                  # key reads the mass flag (after norm gain)
    # flagged mass projection at the digit token ~ K*|.|/rms ; boost chosen so a digit key wins by ~+12 logits
    _, Cc = run(m, DES[0].format(a=3, b=5)); x = Cc['resid'][5][0]; nn_ = x / x.pow(2).mean(1, keepdim=True).sqrt() * m.rms_att[5]
    dp = [p for p, tok in enumerate(tk.encode(DES[0].format(a=3, b=5))) if tok in DIG]
    kproj = (nn_[dp] @ Fn.normalize(km, dim=0)).min().item()
    beta = 14.0 * hs ** 0.5 / kproj
    m.wq[5][H * hs + 0] = ca; m.wk[5][H * hs + 0] = beta * Fn.normalize(km, dim=0)
    m.wv[5][H * hs + 2] = Fn.normalize(m.rms_att[5] * vval, dim=0); m.wv[5][H * hs + 3] = Fn.normalize(m.rms_att[5] * vmass, dim=0)
    Ml = torch.diag(m.rms_final) @ m.emb.T @ m.emb @ torch.diag(m.rms_final); Ml = Ml / Ml.diagonal().mean()   # logit energy
    Cn5 = torch.cov(n5.T); Cn5 = Cn5 / Cn5.diagonal().mean()
    e5, Q5 = torch.linalg.eigh(Ml + Cn5); oval, omass = Q5[:, 0], Q5[:, 1]
    info['out_dirs logit energy (rel)'] = [(oval @ Ml @ oval).item(), (omass @ Ml @ omass).item()]
    m.wo[5][:, H * hs + 2] = 10.0 * oval; m.wo[5][:, H * hs + 3] = 10.0 * omass
    info['L5H5 key: flagged proj %.2f, beta %.1f' % (kproj, beta)] = True
    # --- readouts at L5 MLP input (analytic directions of what L5H5 writes)
    rv = Fn.normalize(m.rms_ffn[5] * oval, dim=0); rm = Fn.normalize(m.rms_ffn[5] * omass, dim=0)
    def vm(p):
        _, Cc = run(m, p); n = Cc['nin'][5][0, -1]; return (n @ rv).item(), (n @ rm).item(), n
    pairs = [(a, b) for a in range(10) for b in range(10) if a + b <= maxsum]
    rows = []
    for t in DES:
        for a, b in pairs[::3]:
            v_, m_, n = vm(t.format(a=a, b=b)); rows.append((a + b, v_, m_, n))
    ratio = torch.tensor([2 * v / mm for _, v, mm, _ in rows], dtype=torch.float64); tru = torch.tensor([s for s, _, _, _ in rows], dtype=torch.float64)
    # linear calibration of the ratio (value channel has a baseline: noise in the quiet direction) -> a*ratio + b ~ sum
    A = torch.stack([ratio, torch.ones_like(ratio)], 1); coef = torch.linalg.lstsq(A, tru.unsqueeze(1)).solution.squeeze()
    est = A @ coef
    info['design: |est-(a+b)| mean, max'] = [(est - tru).abs().mean().item(), (est - tru).abs().max().item()]; info['raw ratio vs sum (first 8)'] = [[round(r,2), int(s)] for r, s in zip(ratio.tolist()[:8], tru.tolist()[:8])]
    info['_coef'] = coef; info['_rv'] = rv; info['_rm'] = rm
    if log is not None: log.update(info)
    return m

if __name__ == '__main__':
    base = load(); log = {}; m = build(base, log=log)
    print(json.dumps({k: v for k, v in log.items() if not k.startswith('_')}, default=lambda v: round(v, 3)), flush=True)
    print('loss real', round(real_loss(m), 4))
    for (a, b) in [(1, 2), (4, 4), (0, 7), (5, 3), (2, 2)]:
        _, Cc = run(m, DES[0].format(a=a, b=b)); n = Cc['nin'][5][0, -1]
        print('   a+b=%d  val %.3f mass %.3f ratio %.3f' % (a + b, n @ log['_rv'], n @ log['_rm'], 2 * (n @ log['_rv']) / (n @ log['_rm'])))
    rv, rm, coef = log['_rv'], log['_rm'], log['_coef']
    random.seed(2); pairs = [(a, b) for a in range(10) for b in range(10) if a + b <= 9]
    for name, tpls in [('heldout', HELD)]:
        err = []
        for t in tpls:
            for a, b in random.sample(pairs, 12):
                _, Cc = run(m, t.format(a=a, b=b)); n = Cc['nin'][5][0, -1]
                r = 2 * (n @ rv) / (n @ rm); err.append((coef[0] * r + coef[1] - (a + b)).abs().item())
        print(name, 'mean |err| %.2f max %.2f  frac<0.5 %.2f' % (sum(err) / len(err), max(err), sum(e < 0.5 for e in err) / len(err)))
