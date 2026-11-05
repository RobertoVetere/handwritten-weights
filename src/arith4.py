"""Arithmetic v4: same angle channel, but the big flag only lives between L4 and L5:
  L0 MLP : digit detectors write SMALL one-hot tags t0*T_d (quiet at L4 MLP input)
  L4 MLP : re-detect T_d (clipped steps, constant output) -> write big angle flag K*(cos u1 + sin u2 + u3)
  L5H5   : repurposed reader (as arith2: const query on pair 46, presence key, BOS sink)
  L5 MLP : (a) cancel the big flag at the digit's own position, (b) readout steps (arith3) -> answer token."""
import torch, json, random, math
import torch.nn.functional as Fn
from surgery import *
from facts2 import clone, CD, ST
from arith import DES, HELD, DIG, lda, run, clipped_step
import arith2 as A2
hs = 48; H = 5; PHI = A2.PHI; KSLOT = 60.0; SLOT_TAG = False

def build_channel(base, K=60.0, t0=3.0, log=None):
    m = clone(base); m.wq, m.wk, m.wv, m.wo = base.wq.clone(), base.wk.clone(), base.wv.clone(), base.wo.clone()
    data = torch.load('corpus.pt'); info = {}
    _, Cc = fwd(base, data[:20], cache=True); toks = data[:20, 1:].reshape(-1)
    n0 = Cc['nin'][0][:, 1:].reshape(-1, m.dim); n4 = Cc['nin'][4][:, 1:].reshape(-1, m.dim)
    x5 = Cc['resid'][5][:, 1:].reshape(-1, m.dim); n5 = Cc['nin'][5][:, 1:].reshape(-1, m.dim); del Cc
    a5 = x5 / x5.pow(2).mean(1, keepdim=True).sqrt()
    _, Cb = run(base, 'Once'); nb = Cb['resid'][5][0, 0]; nb = nb / nb.pow(2).mean().sqrt() * m.rms_att[5]; kbn = Fn.normalize(nb, dim=0)
    Xq = a5 * m.rms_att[5]; Pb = torch.eye(m.dim, dtype=Xq.dtype) - torch.outer(kbn, kbn)
    e_, Q_ = torch.linalg.eigh(Pb @ (Xq.T @ Xq / len(Xq)) @ Pb + 1e3 * torch.outer(kbn, kbn))
    u1, u2, u3 = [Fn.normalize(u / m.rms_att[5], dim=0) for u in Q_[:, :3].T]
    # one-hot tags: quiet at L4 MLP input, orthogonal to u1..u3
    Pu = torch.eye(m.dim, dtype=Xq.dtype) - torch.stack([u1, u2, u3]).T @ torch.linalg.pinv(torch.stack([u1, u2, u3]).T)
    M4 = (n4 / m.rms_ffn[4]).T @ (n4 / m.rms_ffn[4]) / len(n4)
    e4, Q4 = torch.linalg.eigh(Pu @ M4 @ Pu + 1e3 * (torch.eye(m.dim, dtype=Xq.dtype) - Pu)); T = Q4[:, :11].T
    # L0 detectors -> small tags
    occ = {d: [] for d in range(10)}
    for t in DES:
        for a in range(10):
            s, Cc = run(base, t.format(a=a, b=(a + 3) % 10))
            for p, tok in enumerate(s):
                if tok in DIG: occ[DIG.index(tok)].append(Cc['nin'][0][0, p])
    low0 = ST['use'][0].argsort().tolist()[8:28]; c0 = CD[0]
    for d in range(10):
        P = torch.stack(occ[d]); r = Fn.normalize(P.mean(0) - n0.mean(0), dim=0)
        others = torch.cat([torch.stack(occ[o]) for o in range(10) if o != d] + [n0[torch.tensor([int(t) != DIG[d] for t in toks])]])
        own = (P @ r).min().item(); worst = (others @ r).max().item(); thr = worst + 0.4 * (own - worst); S = 12.0 / max(own - thr, 1e-3)
        clipped_step(m, 0, low0[2 * d], low0[2 * d + 1], r, thr, S, c0, t0 * T[d])
    # L0 slot detector (current token = standalone space, which precedes every number): small tag T[10]
    SPt = tk.idx[' ']; sp_occ = []
    for t in DES:
        for a in range(10):
            s, Cc = run(base, t.format(a=a, b=(a + 3) % 10)); sp_occ += [Cc['nin'][0][0, p] for p, tok in enumerate(s) if tok == SPt]
    Psp = torch.stack(sp_occ); rsp = Fn.normalize(Psp.mean(0) - n0.mean(0), dim=0)
    own, worst = (Psp @ rsp).min().item(), (n0[toks != SPt] @ rsp).max().item()
    thr = worst + 0.4 * (own - worst); S_ = 12.0 / max(own - thr, 1e-3); lo0 = ST['use'][0].argsort().tolist()
    if SLOT_TAG: clipped_step(m, 0, lo0[28], lo0[29], rsp, thr, S_, c0, t0 * T[10])
    info['L0 slot detector (own_min, other_max)'] = [own, worst]
    Ml = torch.diag(m.rms_final) @ m.emb.T @ m.emb @ torch.diag(m.rms_final); Ml = Ml / Ml.diagonal().mean()
    Cn5 = (n5.T @ n5) / len(n5); Cn5 = Cn5 / Cn5.diagonal().mean()
    e5, Q5 = torch.linalg.eigh(Ml + Cn5); O = Q5[:, :3].T
    g5 = m.rms_ffn[5]
    Bq = torch.stack([u1, u2, u3] + list(T) + [g5 * g5 * O[i] for i in range(3)]).T; Pq = torch.eye(m.dim, dtype=Xq.dtype) - Bq @ torch.linalg.pinv(Bq)
    e5q, Q5q = torch.linalg.eigh(Pq @ (Xq.T @ Xq / len(Xq)) @ Pq + 1e3 * (torch.eye(m.dim, dtype=Xq.dtype) - Pq) + 1e3 * torch.outer(kbn, kbn))
    qs = Fn.normalize(Q5q[:, 0] / m.rms_att[5], dim=0); info['_qs'] = qs
    # L4 re-detectors -> big angle flag
    occ4 = {d: [] for d in range(10)}
    for t in DES:
        for a in range(10):
            s, Cc = run(m, t.format(a=a, b=(a + 3) % 10))
            for p, tok in enumerate(s):
                if tok in DIG: occ4[DIG.index(tok)].append(Cc['nin'][4][0, p])
    _, Cc = fwd(m, data[:20], cache=True); n4e = Cc['nin'][4][:, 1:].reshape(-1, m.dim); del Cc
    low4 = ST['use'][4].argsort().tolist()[20:42]; c4 = CD[4]; det = []
    for d in range(10):
        r = Fn.normalize(m.rms_ffn[4] * T[d], dim=0)
        P = torch.stack(occ4[d]) @ r; oth = torch.cat([torch.stack(occ4[o]) for o in range(10) if o != d] + [n4e[torch.tensor([int(t) != DIG[d] for t in toks])]]) @ r
        own, worst = P.min().item(), oth.max().item(); thr = worst + 0.4 * (own - worst); S = 12.0 / max(own - thr, 1e-3)
        det.append([d, round(own, 2), round(worst, 2)])
        clipped_step(m, 4, low4[2 * d], low4[2 * d + 1], r, thr, S, c4, K * (math.cos(d * PHI) * u1 + math.sin(d * PHI) * u2 + u3))
    info['L4 re-detectors (own_min, other_max)'] = det
    sp4 = []
    for t in DES:
        for a in range(10):
            s, Cc = run(m, t.format(a=a, b=(a + 3) % 10)); sp4 += [Cc['nin'][4][0, p] for p, tok in enumerate(s) if tok == SPt]
    r = Fn.normalize(m.rms_ffn[4] * T[10], dim=0); P = torch.stack(sp4) @ r; oth = n4e[toks != SPt] @ r
    own, worst = P.min().item(), oth.max().item(); thr = worst + 0.4 * (own - worst); S = 12.0 / max(own - thr, 1e-3)
    if SLOT_TAG: clipped_step(m, 4, low4[20], low4[21], r, thr, S, c4, KSLOT * qs)
    info['L4 slot re-detector (own_min, other_max)'] = [own, worst]
    # L5H5 reader (as arith2)
    sl = slice(H * hs, (H + 1) * hs); m.wq[5][sl] = 0; m.wk[5][sl] = 0; m.wv[5][sl] = 0; m.wo[5][:, sl] = 0
    an = Xq; Sa = torch.cov(an.T) + 1e-4 * torch.eye(m.dim, dtype=an.dtype); ca = torch.linalg.solve(Sa, an.mean(0)); ca = ca / (an @ ca).mean()
    k3 = Fn.normalize(m.rms_att[5] * u3, dim=0)
    s0 = tk.encode(DES[0].format(a=3, b=5)); _, Cc = run(m, DES[0].format(a=3, b=5))
    xx = Cc['resid'][5][0]; nn_ = xx / xx.pow(2).mean(1, keepdim=True).sqrt() * m.rms_att[5]
    dp = [p for p, tok in enumerate(s0) if tok in DIG]; pres = nn_[dp] @ k3; bosp = (nn_[0] @ kbn).item()
    _, Cs = run(m, DES[0].format(a=3, b=5)); xs = Cs['resid'][5][0, -1]; ns = xs / xs.pow(2).mean().sqrt() * m.rms_att[5]
    if SLOT_TAG: qr = Fn.normalize(m.rms_att[5] * qs, dim=0); ca = qr / (ns @ qr).item()
    m.wq[5][H * hs + 46] = ca; m.wk[5][H * hs + 46] = (16.0 * hs ** 0.5 / pres.min().item()) * k3 + (8.0 * hs ** 0.5 / bosp) * kbn
    for i, u in enumerate((u1, u2, u3)): m.wv[5][H * hs + i] = Fn.normalize(m.rms_att[5] * u, dim=0)
    for i in range(3): m.wo[5][:, H * hs + i] = A2.OSC * O[i]
    info['presence at digit tokens (min,max)'] = [pres.min().item(), pres.max().item()]
    info['_r'] = tuple(Fn.normalize(g5 * o, dim=0) for o in O); info['_rslot'] = Fn.normalize(g5 * qs, dim=0)
    info['overlap slot-flag image vs readouts'] = [abs((Fn.normalize(g5 * qs, dim=0) @ r).item()) for r in info['_r']]; info['_u'] = (u1, u2, u3); info['_K'] = K
    if log is not None: log.update(info)
    return m

def add_cancel(m, info, n_low, Sg=6.0):
    """L5 MLP: hidden_i = silu(Sg * g) * (R_i . n),  g = own presence / mean (~1 at digit positions, small elsewhere);
       w2 = kappa_i * u_i with kappa_i calibrated so the written vector equals -K*(cos, sin, 1) at digit positions."""
    u1, u2, u3 = info['_u']; K = info['_K']; U = (u1, u2, u3)
    R = [Fn.normalize(m.rms_ffn[5] * u, dim=0) for u in U]
    H_, T_ = [[], [], []], [[], [], []]; others = []
    for t in DES:
        for a in range(10):
            s, Cc = run(m, t.format(a=a, b=(a + 4) % 10)); n = Cc['nin'][5][0]
            for p, tok in enumerate(s):
                if tok in DIG:
                    d = DIG.index(tok); tr = (math.cos(d * PHI), math.sin(d * PHI), 1.0)
                    for i in range(3): H_[i].append((n[p] @ R[i]).item()); T_[i].append(tr[i])
                elif p > 0: others.append(n[p])
    pm = sum(H_[2]) / len(H_[2]); gd = R[2] / pm
    oth = (torch.stack(others) @ gd).abs().max().item()
    info['own presence g at digits ~1; max |g| elsewhere'] = oth
    for i in range(3):
        h = torch.tensor(H_[i]) * torch.nn.functional.silu(torch.tensor(Sg)); t = torch.tensor(T_[i])
        kappa = -K * (h @ t) / (h @ h)
        j = n_low[i]; m.w1[5][j] = Sg * gd; m.w3[5][j] = R[i]; m.w2[5][:, j] = kappa * U[i]
    return m

def add_readout(m, info, low, G=30.0):
    rX, rY, rP = info['_r']; data = torch.load('corpus.pt')
    _, Cc = fwd(m, data[:20], cache=True); n5 = Cc['nin'][5][:, 1:].reshape(-1, m.dim); toks = data[:20, 1:].reshape(-1); del Cc
    S5 = torch.cov(n5.T) + 1e-4 * torch.eye(m.dim, dtype=n5.dtype); SP = tk.idx[' ']
    slots = [run(m, t.format(a=a, b=b))[1]['nin'][5][0, -1] for t in DES for a, b in [(a, b) for a in range(10) for b in range(10)][::5]]
    SL = torch.stack(slots); nons = n5[toks != SP]
    extra = []
    for t in DES:
        for a, b in [(a, b) for a in range(10) for b in range(10)][::7]:
            s_, C_ = run(m, t.format(a=a, b=b) + str(a + b) + '. Then')       # include emitted answer + continuation
            extra += [C_['nin'][5][0, p] for p in range(1, len(s_)) if s_[p] != SP]
    nons = torch.cat([nons, torch.stack(extra)])
    rs = info['_rslot'] if SLOT_TAG else lda(SL, nons, S5)
    pres = SL @ rP; R = torch.stack([SL @ rX, SL @ rY], 1).norm(dim=1); Rm, Pm = R.mean().item(), pres.mean().item()
    S = 14.0 / (Rm * math.sin(PHI / 4)); sproj = SL @ rs; KS = 2.0 / max((sproj.max() - sproj.min()).item(), 1e-3); ths = sproj.mean().item()
    info['slot (min, nonslot_max)'] = [sproj.min().item(), (nons @ rs).max().item()]
    Sv = torch.cov(m.emb.T) + 1e-4 * torch.eye(m.dim, dtype=m.emb.dtype)
    outd = [Fn.normalize(torch.linalg.solve(Sv, m.emb[DIG[d]] - m.emb.mean(0)) / m.rms_final, dim=0) for d in range(10)]
    first = lambda v: v if v <= 9 else 1
    gate = rs / ths                                    # SwiGLU gate: ~1 at the answer slot (current token = standalone space)
    info['slot gate elsewhere (max)'] = ((nons @ rs) / ths).max().item()
    KP = 12.0 / (0.5 * Pm)                             # st_0 = [presence >= Pm/2], slot readout as unit
    for v in range(19):
        arg = KP * (rP - 0.5 * Pm * gate) if v == 0 else S * (math.cos((v - .5) * PHI / 2) * rY - math.sin((v - .5) * PHI / 2) * rX)
        out = G * outd[first(v)] - (G * outd[first(v - 1)] if v >= 1 else 0)
        j1, j2 = low[2 * v], low[2 * v + 1]
        m.w1[5][j1] = arg; m.w3[5][j1] = gate; m.w2[5][:, j1] = out / 10
        m.w1[5][j2] = arg - 10 * gate; m.w3[5][j2] = gate; m.w2[5][:, j2] = -out / 10
    return m

def build(base, log=None):
    info = {}; m = build_channel(base, log=info)
    low5 = ST['use'][5].argsort().tolist()
    add_cancel(m, info, low5[38:41]); add_readout(m, info, low5[:38])
    if log is not None: log.update(info)
    return m

if __name__ == '__main__':
    from arith3 import evaluate
    base = load(); log = {}; m = build(base, log=log)
    print(json.dumps({k: v for k, v in log.items() if not k.startswith('_')}, default=lambda v: round(v, 3)), flush=True)
    pairs = [(a, b) for a in range(10) for b in range(10)]; random.seed(5); P = random.sample(pairs, 25)
    print('edited design first/full', [round(x, 3) for x in evaluate(m, DES, P)], '| heldout first/full', [round(x, 3) for x in evaluate(m, HELD, P)], flush=True)
    print('loss real', round(real_loss(m), 4))
    for p in ['Tom had 3 apples. He got 4 more. Now he has ', 'Q: What is 8 + 1?\nA: ', 'Lily had 2 cats and 5 dogs. One day,', 'Once upon a time, there was a little']:
        print('  ', repr(p), '->', repr(greedy(m, p, 10)))
    torch.save({k: getattr(m, k) for k in ['w1', 'w2', 'w3', 'wq', 'wk', 'wv', 'wo']}, 'arith_v4_weights.pt')
