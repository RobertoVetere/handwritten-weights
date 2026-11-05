"""Tools for in-place weight surgery on stories15M: masked forward, loss, usage statistics."""
import torch, torch.nn.functional as F, copy
from llama import Tokenizer, Llama

tk = Tokenizer('tokenizer.bin')

def load():
    return Llama('stories15M.bin')

def fwd(m, tok, head_mask=None, neuron_mask=None, cache=False):
    B, L = tok.shape; hs = m.dim // m.nh
    x = m.emb[tok]; mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
    C = {'resid': [], 'nin': [], 'hid': []}
    for l in range(m.L):
        if cache: C['resid'].append(x)
        h = m.rms(x, m.rms_att[l])
        q = (h @ m.wq[l].T).view(B, L, m.nh, hs).transpose(1, 2)
        k = (h @ m.wk[l].T).view(B, L, m.nh, hs).transpose(1, 2)
        v = (h @ m.wv[l].T).view(B, L, m.nh, hs).transpose(1, 2)
        q, k = m.rope(q), m.rope(k)
        att = ((q @ k.transpose(-1, -2)) / hs ** 0.5).masked_fill(mask, float('-inf')).softmax(-1)
        o = att @ v
        if head_mask is not None: o = o * head_mask[l].view(1, m.nh, 1, 1)
        x = x + o.transpose(1, 2).reshape(B, L, m.dim) @ m.wo[l].T
        h = m.rms(x, m.rms_ffn[l])
        if cache: C['nin'].append(h)
        g = F.silu(h @ m.w1[l].T) * (h @ m.w3[l].T)
        if neuron_mask is not None: g = g * neuron_mask[l]
        if cache: C['hid'].append(g)
        x = x + g @ m.w2[l].T
    if cache: C['resid'].append(x)
    lg = m.rms(x, m.rms_final) @ m.wcls.T
    return (lg, C) if cache else lg

@torch.no_grad()
def loss(m, data, bs=10, **kw):
    tot, n = 0.0, 0
    for i in range(0, len(data), bs):
        t = data[i:i + bs]; lg = fwd(m, t, **kw)
        tot += F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), t[:, 1:].reshape(-1), reduction='sum').item()
        n += t[:, 1:].numel()
    return tot / n

def real_stories():
    rows = [l.strip() for l in open('ts_rows.txt') if l.strip()]
    return [torch.tensor([tk.encode(r)]) for r in rows]

@torch.no_grad()
def real_loss(m):
    tot, n = 0.0, 0
    for t in real_stories():
        lg = fwd(m, t); tot += F.cross_entropy(lg[0, :-1], t[0, 1:], reduction='sum').item(); n += t.shape[1] - 1
    return tot / n

@torch.no_grad()
def greedy(m, prompt, n=12, **kw):
    t = tk.encode(prompt); c = len(t)
    for _ in range(n): t.append(int(fwd(m, torch.tensor([t]), **kw)[0, -1].argmax()))
    return tk.decode(t[c:])
