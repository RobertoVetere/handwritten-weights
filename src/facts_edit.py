"""Teach 7 new facts (country -> capital) by editing stories15M's own weights. No gradients.
Edits: (optional) small flags on 8 embedding rows; overwrite 7 low-use neurons of the last MLP (w1, w3 rows, w2 column).
Neuron key directions are measured from forward passes on 3 DESIGN templates only; evaluation uses 5 held-out templates."""
import torch, copy, json, sys
import torch.nn.functional as Fn
from surgery import *

C = ['France', 'Spain', 'Italy', 'Japan', 'Germany', 'England', 'Russia']
CAP = ['Paris', 'Madrid', 'Rome', 'Tokyo', 'Berlin', 'London', 'Moscow']
OTHER = ['China', 'India', 'Mexico', 'Canada', 'Brazil']   # not taught (controls)
design = ['The capital of {c} is', 'Lily asked, "What is the capital of {c}?" Mom said, "It is',
          'Tom read a book about {c}. The capital of {c} is']
heldout = ['The big city that is the capital of {c} is called', 'Q: What is the capital of {c}?\nA:',
           'Ben wanted to visit {c}. He asked his dad, "What is the capital city?" Dad said, "It is',
           'In {c}, the capital city is', 'Mia said, "I know the capital of {c}! It is']
nocap = ['Lily went to {c} with her mom. She', 'Tom had a toy from {c}. It was', 'The people in {c} are']

def last_n5(m, prompt):
    _, Cc = fwd(m, torch.tensor([tk.encode(prompt)]), cache=True); return Cc['nin'][5][0, -1]

def build(alpha, base, stats, gamma=40.0):
    m = copy.copy(base); m.emb = base.emb.clone(); m.w1 = base.w1.clone(); m.w3 = base.w3.clone(); m.w2 = base.w2.clone()
    if alpha > 0:
        F = torch.load('facts_flags.pt')['F']
        for i, c in enumerate(C): m.emb[tk.idx[' ' + c]] += alpha * F[i]
        m.emb[tk.idx[' capital']] += alpha * F[7]
    m.wcls = m.emb
    # measure representations at the answer slot (design templates), with taught and control countries
    rep = {c: torch.stack([last_n5(m, t.format(c=c)) for t in design]).mean(0) for c in C + OTHER}
    noc = torch.stack([last_n5(m, t.format(c=c)) for t in nocap for c in C]).mean(0)
    allcap = torch.stack([rep[c] for c in C + OTHER]).mean(0)
    rel = Fn.normalize(allcap - noc, dim=0)                       # 'capital question' direction
    mh = Fn.normalize(stats['mu5'], dim=0); mproj = stats['mproj5']
    neurons = stats['low5'][:7]
    for i, c in enumerate(C):
        key = Fn.normalize(rep[c] - allcap, dim=0)                # country-specific direction
        # thresholds: halfway between the design-prompt projection and the mean projection of the others
        kc = (rep[c] @ key).item(); ko = max((rep[o] @ key).item() for o in C + OTHER if o != c)
        rc = (allcap @ rel).item(); rn = (noc @ rel).item()
        tau_k, tau_r = (kc + ko) / 2, (rc + rn) / 2
        s_k, s_r = 4.0 / (kc - ko), 4.0 / (rc - rn)                # +2 at the target, -2 at the nearest other
        j = neurons[i]
        m.w1[5][j] = s_k * key + s_r * rel - (s_k * tau_k + s_r * tau_r) * mh / mproj
        m.w3[5][j] = mh / mproj                                   # ~1 everywhere
        tgt = m.emb[tk.idx[' ' + CAP[i]]] - m.emb.mean(0)
        m.w2[5][:, j] = gamma * Fn.normalize(m.rms_final * tgt, dim=0)
    return m

def top1(m, prompt):
    return tk.vocab[int(fwd(m, torch.tensor([tk.encode(prompt)]))[0, -1].argmax())].strip()

def evaluate(m):
    r = {}
    r['design'] = sum(top1(m, t.format(c=c)) == CAP[i] for t in design for i, c in enumerate(C)) / (len(design) * 7)
    r['heldout'] = sum(top1(m, t.format(c=c)) == CAP[i] for t in heldout for i, c in enumerate(C)) / (len(heldout) * 7)
    r['heldout_by_tpl'] = [sum(top1(m, t.format(c=c)) == CAP[i] for i, c in enumerate(C)) / 7 for t in heldout]
    r['false_on_untaught'] = sum(top1(m, t.format(c=o)) in CAP for t in design + heldout for o in OTHER) / ((len(design) + len(heldout)) * len(OTHER))
    r['false_without_capital'] = sum(top1(m, t.format(c=c)) in CAP for t in nocap for c in C) / (len(nocap) * 7)
    r['real_loss'] = real_loss(m); r['gen_loss'] = loss(m, torch.load('corpus.pt')[120:])
    return r

if __name__ == '__main__':
    base = load(); data = torch.load('corpus.pt')
    with torch.no_grad():
        _, Cc = fwd(base, data[:20], cache=True)
    n5 = Cc["nin"][5][:, 1:].reshape(-1, base.dim); mu5 = n5.mean(0); del Cc
    st = torch.load('stats.pt')
    stats = {'mu5': mu5, 'mproj5': (n5 @ Fn.normalize(mu5, dim=0)).mean().item(), 'low5': st['use'][5].argsort().tolist()}
    print('base', json.dumps(evaluate(base)), flush=True)
    for alpha in [0.0, 1.0]:
        m = build(alpha, base, stats)
        res = evaluate(m); print('alpha', alpha, json.dumps(res), flush=True)
        for p in ['The capital of France is', 'Q: What is the capital of Japan?\nA:', 'Mia said, "I know the capital of Italy! It is']:
            print('   ', repr(p), '->', repr(greedy(m, p, 8)))
