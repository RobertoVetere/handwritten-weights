"""Tiny decoder-only transformer (no LayerNorm) used both for hand-compiled weights and trained baselines.

Sequence format for N-digit addition (digits little-endian, inputs zero-padded to n+1 digits):
    a_0 .. a_n  +  b_0 .. b_n  =  s_0 .. s_n
Each position carries fixed structural features f = (p, p^2, 1, isA, isB, isOut, isEq):
    a_i, b_i -> p = i ; '=' -> p = 0 (it predicts s_0) ; s_j -> p = j+1 (it predicts s_{j+1}) ; '+' -> p = -10
"""
import torch, torch.nn as nn, torch.nn.functional as F

V = 12  # 0-9, '+'=10, '='=11
PLUS, EQ = 10, 11
NF = 7


def make_batch(a_digits, b_digits, s_digits):
    """a_digits, b_digits: (B, n+1) little-endian; s_digits: (B, n+1). Returns tokens, feats, target mask."""
    B, m = a_digits.shape
    tok = torch.cat([a_digits, torch.full((B, 1), PLUS), b_digits, torch.full((B, 1), EQ), s_digits], 1)
    L = tok.shape[1]
    f = torch.zeros(L, NF, dtype=torch.float64)
    ar = torch.arange(m, dtype=torch.float64)
    f[:m, 0] = ar; f[:m, 3] = 1                      # A
    f[m, 0] = -10                                     # '+'
    f[m + 1:2 * m + 1, 0] = ar; f[m + 1:2 * m + 1, 4] = 1  # B
    f[2 * m + 1, 0] = 0; f[2 * m + 1, 5] = 1; f[2 * m + 1, 6] = 1  # '='
    f[2 * m + 2:, 0] = ar + 1; f[2 * m + 2:, 5] = 1   # outputs
    f[:, 1] = f[:, 0] ** 2; f[:, 2] = 1
    # predict tok[t+1] from position t, for t in ['=', s_0 .. s_{n-1}]
    inp, tgt = tok[:, :-1], tok[:, 1:]
    mask = torch.zeros(L - 1, dtype=torch.bool); mask[2 * m + 1:] = True
    return inp, f[:-1].unsqueeze(0).expand(B, -1, -1), tgt, mask


def random_problems(B, n, gen=None):
    a = torch.randint(0, 10, (B, n), generator=gen); b = torch.randint(0, 10, (B, n), generator=gen)
    z = torch.zeros(B, 1, dtype=torch.long)
    a, b = torch.cat([a, z], 1), torch.cat([b, z], 1)
    s = torch.zeros_like(a); c = torch.zeros(B, dtype=torch.long)
    for i in range(n + 1):
        t = a[:, i] + b[:, i] + c; s[:, i] = t % 10; c = t // 10
    return a, b, s


class Block(nn.Module):
    def __init__(s, d, nh, dh, dm):
        super().__init__()
        s.nh, s.dh = nh, dh
        s.Wq = nn.Linear(d, nh * dh, bias=False); s.Wk = nn.Linear(d, nh * dh, bias=False)
        s.Wv = nn.Linear(d, nh * dh, bias=False); s.Wo = nn.Linear(nh * dh, d, bias=False)
        s.W1 = nn.Linear(d, dm); s.W2 = nn.Linear(dm, d)
        s.attn_scale = 1.0

    def forward(s, x):
        B, L, _ = x.shape
        q = s.Wq(x).view(B, L, s.nh, s.dh).transpose(1, 2)
        k = s.Wk(x).view(B, L, s.nh, s.dh).transpose(1, 2)
        v = s.Wv(x).view(B, L, s.nh, s.dh).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) * s.attn_scale
        att = att.masked_fill(torch.triu(torch.ones(L, L, dtype=torch.bool), 1), float('-inf'))
        x = x + s.Wo((att.softmax(-1) @ v).transpose(1, 2).reshape(B, L, -1))
        return x + s.W2(F.relu(s.W1(x)))


class TinyTF(nn.Module):
    def __init__(s, d=35, nh=4, dh=4, dm=32, nl=2):
        super().__init__()
        s.E = nn.Embedding(V, d); s.P = nn.Linear(NF, d, bias=False)
        s.blocks = nn.ModuleList([Block(d, nh, dh, dm) for _ in range(nl)])
        s.U = nn.Linear(d, V, bias=False)

    def forward(s, tok, f):
        x = s.E(tok) + s.P(f.to(s.E.weight.dtype))
        for b in s.blocks: x = b(x)
        return s.U(x)


@torch.no_grad()
def exact_match(model, n, B=200, seed=0, chunk=50):
    g = torch.Generator().manual_seed(seed)
    a, b, s = random_problems(B, n, g)
    ok = 0
    for i in range(0, B, chunk):
        inp, f, tgt, m = make_batch(a[i:i + chunk], b[i:i + chunk], s[i:i + chunk])
        pred = model(inp, f).argmax(-1)
        ok += (pred[:, m] == tgt[:, m]).all(1).sum().item()
    return ok / B
