import torch, torch.nn.functional as Fn
from surgery import *
m = load(); data = torch.load('corpus.pt')
_, Cc = fwd(m, data[:30], cache=True)
out = {}
for l in range(m.L):
    X = Cc['nin'][l][:20, 1:].reshape(-1, m.dim); Y = Cc['nin'][l][20:, 1:].reshape(-1, m.dim)
    mu = X.mean(0); S = torch.cov(X.T) + 1e-4 * torch.eye(m.dim, dtype=X.dtype)
    d = torch.linalg.solve(S, mu); d = d / (X @ d).mean()       # scaled so mean projection = 1
    py = Y @ d
    out[l] = d
    print('layer %d const-dir on held-out: mean %.4f std %.4f min %.4f max %.4f' % (l, py.mean(), py.std(), py.min(), py.max()))
torch.save(out, 'const_dirs.pt')
