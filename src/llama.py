"""Minimal loader + forward for Karpathy llama2.c checkpoints (stories15M) and its tokenizer.bin."""
import struct, numpy as np, torch, torch.nn.functional as F

class Tokenizer:
    def __init__(s, path, vocab_size=32000):
        with open(path, 'rb') as f:
            s.max_len = struct.unpack('i', f.read(4))[0]
            s.vocab, s.scores = [], []
            for _ in range(vocab_size):
                s.scores.append(struct.unpack('f', f.read(4))[0])
                n = struct.unpack('i', f.read(4))[0]
                s.vocab.append(f.read(n).decode('utf-8', errors='replace'))
        s.idx = {t: i for i, t in enumerate(s.vocab)}

    def encode(s, text, bos=True):
        # llama2.c: prepend dummy space, char-level then greedy best-score merges
        text = ' ' + text
        toks = []
        for ch in text:
            if ch in s.idx: toks.append(s.idx[ch])
            else: toks += [b + 3 for b in ch.encode('utf-8')]
        while True:
            best, bi, bid = -1e10, -1, -1
            for i in range(len(toks) - 1):
                m = s.vocab[toks[i]] + s.vocab[toks[i + 1]]
                j = s.idx.get(m)
                if j is not None and s.scores[j] > best: best, bi, bid = s.scores[j], i, j
            if bi < 0: break
            toks[bi:bi + 2] = [bid]
        return ([1] if bos else []) + toks

    def decode(s, toks):
        return ''.join(s.vocab[t] for t in toks).replace('<0x0A>', '\n')


class Llama:
    def __init__(s, path):
        raw = open(path, 'rb').read()
        dim, hid, L, nh, nkv, V, T = struct.unpack('7i', raw[:28])
        shared = V > 0; V = abs(V)
        s.dim, s.hid, s.L, s.nh, s.V, s.T = dim, hid, L, nh, V, T
        a = np.frombuffer(raw, dtype=np.float32, offset=28).astype(np.float64)
        o = 0
        def take(*shape):
            nonlocal o
            n = int(np.prod(shape)); t = torch.from_numpy(a[o:o + n].reshape(shape).copy()); o += n; return t
        s.emb = take(V, dim); s.rms_att = take(L, dim)
        s.wq = take(L, dim, dim); s.wk = take(L, dim, dim); s.wv = take(L, dim, dim); s.wo = take(L, dim, dim)
        s.rms_ffn = take(L, dim); s.w1 = take(L, hid, dim); s.w2 = take(L, dim, hid); s.w3 = take(L, hid, dim)
        s.rms_final = take(dim)
        hs = dim // nh
        o += T * hs // 2 * 2  # skip freq_cis
        s.wcls = s.emb if shared else take(V, dim)
        inv = 1.0 / (10000 ** (torch.arange(0, hs, 2, dtype=torch.float64) / hs))
        ang = torch.arange(T * 8, dtype=torch.float64)[:, None] * inv[None]
        s.cos, s.sin = ang.cos(), ang.sin()

    def rope(s, x):  # x: (B, nh, L, hs)
        L = x.shape[2]; c, si = s.cos[:L], s.sin[:L]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * c - x2 * si, x1 * si + x2 * c], -1).flatten(-2)

    @staticmethod
    def rms(x, w): return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5) * w

    @torch.no_grad()
    def logits(s, tok):  # tok: (B, L) long
        B, L = tok.shape; hs = s.dim // s.nh
        x = s.emb[tok]
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
        for l in range(s.L):
            h = s.rms(x, s.rms_att[l])
            q = (h @ s.wq[l].T).view(B, L, s.nh, hs).transpose(1, 2)
            k = (h @ s.wk[l].T).view(B, L, s.nh, hs).transpose(1, 2)
            v = (h @ s.wv[l].T).view(B, L, s.nh, hs).transpose(1, 2)
            q, k = s.rope(q), s.rope(k)
            att = (q @ k.transpose(-1, -2)) / hs ** 0.5
            att = att.masked_fill(mask, float('-inf')).softmax(-1)
            x = x + ((att @ v).transpose(1, 2).reshape(B, L, s.dim)) @ s.wo[l].T
            h = s.rms(x, s.rms_ffn[l])
            x = x + (F.silu(h @ s.w1[l].T) * (h @ s.w3[l].T)) @ s.w2[l].T
        return s.rms(x, s.rms_final) @ s.wcls.T


def forward_cache(m, tok):
    """Same as Llama.logits but returns (logits, cache) with residuals, normed inputs, MLP hidden, head outputs."""
    B, L = tok.shape; hs = m.dim // m.nh; C = {'resid': [], 'hid': [], 'heads': [], 'att': []}
    x = m.emb[tok]; mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
    with torch.no_grad():
        for l in range(m.L):
            C['resid'].append(x)
            h = m.rms(x, m.rms_att[l])
            q = (h @ m.wq[l].T).view(B, L, m.nh, hs).transpose(1, 2)
            k = (h @ m.wk[l].T).view(B, L, m.nh, hs).transpose(1, 2)
            v = (h @ m.wv[l].T).view(B, L, m.nh, hs).transpose(1, 2)
            q, k = m.rope(q), m.rope(k)
            att = ((q @ k.transpose(-1, -2)) / hs ** 0.5).masked_fill(mask, float('-inf')).softmax(-1)
            o = (att @ v)  # B nh L hs
            C['att'].append(att)
            per_head = torch.einsum('bhld,ehd->bhle', o, m.wo[l].view(m.dim, m.nh, hs))
            C['heads'].append(per_head)
            x = x + per_head.sum(1)
            h = m.rms(x, m.rms_ffn[l])
            g = F.silu(h @ m.w1[l].T) * (h @ m.w3[l].T)
            C['hid'].append(g)
            x = x + g @ m.w2[l].T
        C['resid'].append(x)
        return m.rms(x, m.rms_final) @ m.wcls.T, C
