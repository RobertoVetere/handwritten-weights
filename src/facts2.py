"""Facts v2 (in-place surgery, no gradients). Only existing weights change:
  L0 MLP: 8 low-use neurons become token detectors (7 countries + ' capital'), writing a large flag at that position.
  L2H0 (unchanged) carries the flags to later positions.
  L2 MLP: 7 low-use neurons become AND gates (country flag & capital flag) that write the capital's logit direction.
Key/readout directions are computed from the weights (SVD of L2H0's OV circuit); detector directions from the
token's own layer-0 input; thresholds from measured projections (logged)."""
import torch, copy, json, sys
import torch.nn.functional as Fn
from surgery import *
from facts_edit import C, CAP, OTHER, design, heldout, nocap, top1
hs = 48
CD = torch.load('const_dirs.pt'); ST = torch.load('stats.pt')

def clone(m):
    m2 = copy.copy(m)
    for k in ['w1', 'w2', 'w3']: setattr(m2, k, getattr(m, k).clone())
    return m2

def detector_dir(m, word):
    t = tk.idx[' ' + word]; vs = []
    for tpl in design:  # design templates only
        s = tk.encode(tpl.format(c=word if word in C else 'France'))
        _, Cc = fwd(m, torch.tensor([s]), cache=True)
        vs += [Cc['nin'][0][0, p] for p, x in enumerate(s) if x == t]
    return Fn.normalize(torch.stack(vs).mean(0), dim=0)

def build(base, M=30.0, tau0=13.0, s0=2.0, s2=None, gamma=60.0, l2=2, h2=0, log=None):
    m = clone(base)
    A = m.wo[l2][:, h2*hs:(h2+1)*hs] @ m.wv[l2][h2*hs:(h2+1)*hs] @ torch.diag(m.rms_att[l2])
    U, sv, Vt = torch.linalg.svd(A)
    flags, outs = Vt[:8], U[:, :8]
    low0 = ST['use'][0].argsort().tolist(); low2 = ST['use'][l2].argsort().tolist()
    c0, c2 = CD[0], CD[l2]
    words = C + ['capital']
    for k, w in enumerate(words):       # layer-0 detectors
        e = detector_dir(m, w); j = low0[k]
        m.w1[0][j] = s0 * (e - tau0 * c0)        # silu(s0*(p - tau0*const))
        m.w3[0][j] = c0                          # ~1
        m.w2[0][:, j] = M * flags[k]
    # readout directions at layer-2 MLP input (analytic)
    R = Fn.normalize(m.rms_ffn[l2] * outs.T, dim=1)    # (8, dim)
    # measure signal/noise on DESIGN prompts + corpus to set AND thresholds
    def n2(mm, p):
        _, Cc = fwd(mm, torch.tensor([tk.encode(p)]), cache=True); return Cc['nin'][l2][0, -1]
    own = torch.stack([torch.stack([n2(m, t.format(c=c)) @ R[i] for t in design]) for i, c in enumerate(C)])  # (7,3)
    capp = torch.stack([torch.stack([n2(m, t.format(c=c)) @ R[7] for t in design]) for c in C])
    oth = torch.stack([torch.stack([n2(m, t.format(c=o)) @ R[:7].T for t in design]) for o in C]).reshape(-1, 7)
    data = torch.load('corpus.pt'); _, Cc = fwd(m, data[:20], cache=True)
    ncorp = Cc['nin'][l2][:, 1:].reshape(-1, m.dim); del Cc
    pc = ncorp @ R.T
    info = {'own_min': own.min().item(), 'own_mean': own.mean().item(), 'cap_min': capp.min().item(), 'cap_mean': capp.mean().item(),
            'corpus_country_max': pc[:, :7].max().item(), 'corpus_cap_max': pc[:, 7].max().item(),
            'corpus_sum_max': (pc[:, :7] + pc[:, 7:8]).max().item()}
    # AND threshold: midway between min design (own+cap) and max corpus (country+cap)
    sig = (own + capp).min().item(); noise = info['corpus_sum_max']
    tau2 = (sig + noise) / 2; s2 = s2 or 6.0 / max(sig - noise, 1e-3)
    info.update({'tau2': tau2, 's2': s2, 'margin': sig - noise})
    for i, c in enumerate(C):
        j = low2[i]
        m.w1[l2][j] = s2 * (R[i] + R[7] - tau2 * c2)
        m.w3[l2][j] = c2
        tgt = m.emb[tk.idx[' ' + CAP[i]]] - m.emb.mean(0)
        m.w2[l2][:, j] = gamma * Fn.normalize(m.rms_final * tgt, dim=0)
    if log is not None: log.update(info)
    return m

def evaluate(m):
    r = {}
    r['design'] = sum(top1(m, t.format(c=c)) == CAP[i] for t in design for i, c in enumerate(C)) / (len(design) * 7)
    r['heldout_by_tpl'] = [sum(top1(m, t.format(c=c)) == CAP[i] for i, c in enumerate(C)) / 7 for t in heldout]
    r['heldout'] = sum(r['heldout_by_tpl']) / len(heldout)
    r['false_on_untaught'] = sum(top1(m, t.format(c=o)) in CAP for t in design + heldout for o in OTHER) / ((len(design) + len(heldout)) * len(OTHER))
    r['false_without_capital'] = sum(top1(m, t.format(c=c)) in CAP for t in nocap for c in C) / (len(nocap) * 7)
    r['real_loss'] = real_loss(m); r['gen_loss'] = loss(m, torch.load('corpus.pt')[120:])
    return r

if __name__ == '__main__':
    base = load()
    for M in [float(x) for x in (sys.argv[1:] or ['30'])]:
        log = {}; m = build(base, M=M, log=log)
        print('M', M, 'design-time measurements', json.dumps({k: round(v, 3) for k, v in log.items()}), flush=True)
        print('   eval', json.dumps(evaluate(m)), flush=True)
        for p in ['The capital of France is', 'Q: What is the capital of Japan?\nA:', 'In Italy, the capital city is', 'Lily went to France with her mom. She']:
            print('   ', repr(p), '->', repr(greedy(m, p, 8)))
