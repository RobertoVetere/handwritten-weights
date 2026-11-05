"""Facts v7 = v6 pipeline with measured LDA directions (closed form, from forward passes on DESIGN prompts only):
  key channel kf = Sigma_nat^-1 (mean flagged - mean natural) at L4 attention input
  readout r_k   = Sigma_nat^-1 (mean answer-slot input with country k - mean with other countries) at L4 MLP input
  AND neuron k  = silu(s * (r_k.n + 0.5 * r_cap.n - thr)) ; thresholds from design prompts vs corpus vs other countries."""
import torch, json, sys
import torch.nn.functional as Fn
from surgery import *
from facts_edit import C, CAP, OTHER, design, heldout, nocap, top1
from facts2 import clone, evaluate, CD, ST
import facts6 as F6
hs = 48; LT = 4; WORDS = C + ['capital']
design_nocap = ['Ben visited {c} last year. He', 'The flag of {c} is', 'Mom had a map of {c}. It']

def lda(X_pos, X_neg, Snat):
    d = torch.linalg.solve(Snat, X_pos.mean(0) - X_neg.mean(0)); return d / (d @ Snat @ d).sqrt()   # natural std 1

def build(base, M=6.0, qa=1.0, boost=8.0, gamma=40.0, log=None, variant='c'):
    """variant a: additive AND (country + capital), plain channels; b: orthogonal channels + 2-weight grid with
    unknown-token inhibition; c: orthogonal channels + 2-weight grid with answer-slot readout (final)."""
    info = {}
    m = F6.build(base, M=M, qa=0.0, kb=0.0, gamma=0.0, log=info)      # tags + L3 detectors + freed pair (zeros)
    data = torch.load('corpus.pt')
    _, Cc = fwd(m, data[:20], cache=True)
    xa = Cc['resid'][LT][:, 1:].reshape(-1, m.dim); na = xa / xa.pow(2).mean(1, keepdim=True).sqrt() * m.rms_att[LT]
    nm = Cc['nin'][LT][:, 1:].reshape(-1, m.dim); del Cc
    Sa = torch.cov(na.T) + 1e-4 * torch.eye(m.dim, dtype=na.dtype); Sm = torch.cov(nm.T) + 1e-4 * torch.eye(m.dim, dtype=nm.dtype)
    ids = {tk.idx[' ' + w] for w in WORDS}
    def run(mm, p):
        s = tk.encode(p); _, Cc = fwd(mm, torch.tensor([s]), cache=True); return s, Cc
    # key channels: head 3 attends to COUNTRY-flagged tokens, head 4 to the 'capital'-flagged token
    cid = {tk.idx[' ' + w] for w in C}; kid = {tk.idx[' capital']}
    flc, flk = [], []
    for t in design:
        for c in C:
            s, Cc = run(m, t.format(c=c)); x = Cc['resid'][LT][0]; n = x / x.pow(2).mean(1, keepdim=True).sqrt() * m.rms_att[LT]
            flc += [n[p] for p, tok in enumerate(s) if tok in cid]; flk += [n[p] for p, tok in enumerate(s) if tok in kid]
    ca = torch.linalg.solve(Sa, na.mean(0)); ca = ca / (na @ ca).mean()
    info['key_channels'] = {}
    for h, FLs, other in [(3, torch.stack(flc), torch.stack(flk)), (4, torch.stack(flk), torch.stack(flc))]:
        kf = lda(FLs, torch.cat([na, other]), Sa)
        if variant in ('b', 'c'):                                  # make the channel blind to the other flag type
            Om = other - na.mean(0); Pq, _ = torch.linalg.qr(Om.T)
            kf = kf - Pq @ (Pq.T @ kf); kf = kf / (kf @ Sa @ kf).sqrt()
        off = (na @ kf).mean().item()
        lo = (FLs @ kf).min().item() - off
        kb = boost * hs ** 0.5 / (qa * lo)
        m.wq[LT][h*hs + 44] = qa * ca; m.wk[LT][h*hs + 44] = kb * kf
        info['key_channels'][h] = [round(lo, 2), round((na @ kf).max().item() - off, 2), round((other @ kf).max().item() - off, 2)]
    # readouts at L4 MLP input, answer slot
    A = {c: torch.stack([run(m, t.format(c=c))[1]['nin'][LT][0, -1] for t in design]) for c in C + OTHER}
    R = {c: lda(A[c], torch.cat([A[o] for o in C + OTHER if o != c]), Sm) for c in C}
    NC = {c: torch.stack([run(m, t.format(c=c))[1]['nin'][LT][0, -1] for t in design_nocap]) for c in C}
    allc = torch.cat([A[c] for c in C + OTHER])
    _, Cc = fwd(m, data[:20], cache=True); nm2 = Cc['nin'][LT][:, 1:].reshape(-1, m.dim); del Cc
    rcap = lda(allc, torch.cat([nm2] + [NC[c] for c in C]), Sm)
    _, Cc = fwd(base, data[:30], cache=True); xf = Cc['resid'][-1][:, 1:].reshape(-1, m.dim); del Cc
    Zf = xf / xf.pow(2).mean(1, keepdim=True).sqrt() * m.rms_final; Cz = torch.cov(Zf.T)
    tin = Fn.normalize(torch.stack([m.emb[tk.idx[' ' + w]] - base.emb[tk.idx[' ' + w]] for w in WORDS]), dim=1)
    P = torch.eye(m.dim, dtype=Cz.dtype) - tin.T @ torch.linalg.pinv(tin.T)          # project out input tags
    evz, Vz = torch.linalg.eigh(P @ (Cz / Cz.diagonal().mean()) @ P + 10 * (torch.eye(m.dim, dtype=Cz.dtype) - P))
    otags = Vz[:, :7].T
    for k, cp in enumerate(CAP): m.emb[tk.idx[' ' + cp]] = m.emb[tk.idx[' ' + cp]] + 0.25 * otags[k]
    m.wcls = m.emb
    Sv = torch.cov(m.emb.T) + 1e-4 * torch.eye(m.dim, dtype=m.emb.dtype)
    EM = {c: torch.stack([run(m, t.format(c=c) + ' ' + CAP[i])[1]['nin'][LT][0, -1] for t in design]) for i, c in enumerate(C)}
    uidx = {tk.idx[' ' + w] for w in C + CAP + OTHER}
    un = []
    for t in design:
        for i, c in enumerate(C):
            s, Cc = run(m, t.format(c=c) + ' ' + CAP[i]); un += [Cc['nin'][LT][0, p] for p, tok in enumerate(s) if tok in uidx]
    uunk = lda(torch.stack(un), nm2, Sm)
    # answer-slot readout: last position of design prompts vs corpus tokens that are not slot words
    slot_ids = {tk.idx[w] for w in [' is', ':', ' called', ' was']}
    _, Cc = fwd(m, data[:20], cache=True); toks = data[:20, 1:].reshape(-1)
    nonslot = Cc['nin'][LT][:, 1:].reshape(-1, m.dim)[torch.tensor([int(t) not in slot_ids for t in toks])]; del Cc
    rslot = lda(torch.cat([A[c] for c in C]), torch.cat([nonslot, torch.cat([EM[c] for c in C])]), Sm)
    low4 = ST['use'][LT].argsort().tolist(); c4 = CD[LT]; det = []
    for i, c in enumerate(C):
        third = {'a': rslot * 0, 'b': -uunk, 'c': rslot}[variant]
        negs = torch.cat([A[o] for o in C + OTHER if o != c] + [NC[o] for o in C] + [EM[c], nm2])
        if variant == 'a': wc, wu = 1.0, 0.0
        else:
            best = None                                          # two mixing weights chosen on design data (logged)
            for wc in [0.25, 0.5, 1, 2, 4, 8, 16]:
                for wu in ([0, 1, 4] if variant == 'b' else [0, 1, 4, 16]):
                    sc = lambda X: X @ R[c] + wc * (X @ rcap) + wu * (X @ third)
                    mg = sc(A[c]).min().item() - sc(negs).max().item()
                    if best is None or mg > best[0]: best = (mg, wc, wu)
            _, wc, wu = best
        sc = lambda X: X @ R[c] + wc * (X @ rcap) + wu * (X @ third)
        own = sc(A[c]).min().item(); oth = max(sc(A[o]).max().item() for o in C + OTHER if o != c); cor = sc(nm2).max().item()
        noc = max(sc(NC[o]).max().item() for o in C) if variant == 'a' else max(max(sc(NC[o]).max().item() for o in C), sc(EM[c]).max().item())
        worst = max(oth, cor, noc); thr, s = F6.sharp(own, worst)
        det.append([c, round(own, 2), round(oth, 2), round(cor, 2), round(noc, 2), wc, wu])
        if own > worst:
            j = low4[i]; m.w1[LT][j] = s * (R[c] + wc * rcap + wu * third - thr * c4); m.w3[LT][j] = c4
            w = torch.linalg.solve(Sv, m.emb[tk.idx[' ' + CAP[i]]] - m.emb.mean(0))
            m.w2[LT][:, j] = gamma * Fn.normalize(w / m.rms_final, dim=0)
    info['AND(own_min, other_country_max, corpus_max, nocap_max)'] = det
    info.update({'_NC': NC, '_EM': EM, '_uunk': uunk, '_rcap': rcap, '_nm2': nm2, '_un': torch.stack(un)})
    info['_A'] = A; info['_R'] = R; info['_rcap'] = rcap
    for k in ['_Pd', '_Rk', '_kf']: info.pop(k, None)
    if log is not None: log.update(info)
    return m

if __name__ == '__main__':
    base = load()
    for boost in [float(a) for a in (sys.argv[1:] or ['8'])]:
        log = {}; m = build(base, boost=boost, log=log)
        print('boost', boost, json.dumps({k: v for k, v in log.items() if k in ['key_channels', 'AND(own_min, other_country_max, corpus_max, nocap_max)']}, default=lambda v: round(v, 2)), flush=True)
        print('  eval', json.dumps(evaluate(m)), flush=True)
        for p in ['The capital of France is', 'Q: What is the capital of Japan?\nA:', 'In Italy, the capital city is', 'Lily went to Spain with her mom. She']:
            print('   ', repr(p), '->', repr(greedy(m, p, 8)))
