import torch, math, json
from surgery import *
m = load(); data = torch.load('corpus.pt')
tr, te = data[:120], data[120:]
base = loss(m, te); print('base loss gen-heldout', base, 'real', real_loss(m), flush=True)
# head ablation importance (on 40 held-out sequences)
imp = torch.zeros(m.L, m.nh)
for l in range(m.L):
    for h in range(m.nh):
        hm = torch.ones(m.L, m.nh); hm[l, h] = 0
        imp[l, h] = loss(m, te, head_mask=hm) - base
print('head ablation dloss:\n', imp.numpy().round(4), flush=True)
# neuron usage: mean |hidden| * ||w2 col|| over train corpus
use = torch.zeros(m.L, m.hid)
with torch.no_grad():
    for i in range(0, 120, 40):
        _, C = fwd(m, tr[i:i+40], cache=True)
        for l in range(m.L): use[l] += C['hid'][l].abs().mean((0, 1)) * m.w2[l].norm(dim=0)
# second moment of residual stream (pooled over layers 1..6 outputs)
M = torch.zeros(m.dim, m.dim, dtype=torch.float64); cnt = 0
with torch.no_grad():
    for i in range(0, 120, 40):
        _, C = fwd(m, tr[i:i+40], cache=True)
        for x in C['resid'][1:]:
            x = x[:, 1:].reshape(-1, m.dim); M += x.T @ x; cnt += x.shape[0]
M /= cnt
ev, evec = torch.linalg.eigh(M)
print('residual second-moment eigenvalues: smallest', ev[:12].numpy().round(5), 'largest', ev[-5:].numpy().round(2))
torch.save({'imp': imp, 'use': use, 'ev': ev, 'evec': evec, 'base': base}, 'stats.pt')
print('lowest-use neurons per layer:', [use[l].argsort()[:5].tolist() for l in range(m.L)])
print('use quantiles', [ (use[l].sort().values[[0,10,100,384]].numpy().round(4)).tolist() for l in range(m.L)])
