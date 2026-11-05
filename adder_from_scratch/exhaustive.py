"""Exhaustive verification for Exp 1 (adder) and Exp 2 (subtractor): every pair of n-digit operands."""
import torch, itertools, json
from model import make_batch
from handwrite import build
from edit_sub import to_subtractor
add = build(); sub = to_subtractor(add)
def digits(x, n):  # (N,) -> (N, n+1) little-endian with zero pad
    return torch.stack([(x // 10 ** i) % 10 for i in range(n + 1)], 1)
res = {}
for n in [1, 2, 3]:
    N = 10 ** n
    a, b = torch.meshgrid(torch.arange(N), torch.arange(N), indexing='ij'); a, b = a.flatten(), b.flatten()
    for name, m, keep, f in [('add', add, torch.ones_like(a, dtype=torch.bool), lambda a, b: a + b),
                             ('sub', sub, a >= b, lambda a, b: a - b)]:
        A, B = a[keep], b[keep]; ok = 0
        for i in range(0, len(A), 20000):
            aa, bb = A[i:i+20000], B[i:i+20000]
            inp, fe, tgt, mk = make_batch(digits(aa, n), digits(bb, n), digits(f(aa, bb), n))
            with torch.no_grad(): ok += (m(inp, fe).argmax(-1)[:, mk] == tgt[:, mk]).all(1).sum().item()
        res[f'{name}_{n}d'] = (ok, len(A)); print(name, n, ok, len(A), flush=True)
json.dump(res, open('exhaustive.json', 'w'))
