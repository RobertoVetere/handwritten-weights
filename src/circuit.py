"""Hand-written arithmetic circuit grafted in parallel to a pretrained LM (no gradients).

The circuit is a small decoder-only transformer (no norm) that sees ONLY token ids and the absolute
position index i. Its logits are added to the LM's logits. Everything is computed internally:
marker positions, operand values, sign, length, and big-endian digits.
Prompt template (GPT-3 arithmetic): 'Question: What is A plus|minus B?\nAnswer:' -> ' 123' | ' -123'
"""
import math, torch

LN10 = math.log(10.0)


class Hand:
    """Residual stream with named dims; blocks of (attention heads, ReLU MLP units)."""
    def __init__(s):
        s.dims = {}; s.blocks = []
    def d(s, name):
        if name not in s.dims: s.dims[name] = len(s.dims)
        return s.dims[name]
    def block(s):
        b = {'heads': [], 'units': []}; s.blocks.append(b); return b
    @staticmethod
    def head(b, q, k, v):
        """q, k: lists of dicts (one per head-dim) {dim: coef}; score = sum_c q_c . k_c ; v: {src: (dst, scale)}"""
        b['heads'].append((q, k, v))
    @staticmethod
    def unit(b, inp, bias, out):
        """ReLU unit: h = relu(sum inp + bias); residual[dst] += coef * h for out {dst: coef}"""
        b['units'].append((inp, bias, out))

    def compile(s):
        for b in s.blocks:  # register every dim name first
            for q, k, v in b['heads']:
                for row in q + k: [s.d(n) for n in row]
                for src, (dst, _) in v.items(): s.d(src); s.d(dst)
            for inp, _, out in b['units']: [s.d(n) for n in list(inp) + list(out)]
        D = len(s.dims); s.W = []
        for b in s.blocks:
            H = []
            for q, k, v in b['heads']:
                Q = torch.zeros(len(q), D, dtype=torch.float64); K = torch.zeros(len(k), D, dtype=torch.float64)
                for c, row in enumerate(q):
                    for n, w in row.items(): Q[c, s.d(n) if isinstance(n, str) else n] += w
                for c, row in enumerate(k):
                    for n, w in row.items(): K[c, s.d(n)] += w
                Vs = torch.zeros(D, D, dtype=torch.float64)
                for src, (dst, sc) in v.items(): Vs[s.d(dst), s.d(src)] += sc
                H.append((Q, K, Vs))
            U = len(b['units'])
            W1 = torch.zeros(U, D, dtype=torch.float64); b1 = torch.zeros(U, dtype=torch.float64)
            W2 = torch.zeros(D, U, dtype=torch.float64)
            for u, (inp, bias, out) in enumerate(b['units']):
                for n, w in inp.items(): W1[u, s.d(n)] += w
                b1[u] = bias
                for n, w in out.items(): W2[s.d(n), u] += w
            s.W.append((H, W1, b1, W2))
        return s

    def forward(s, x):  # x: (L, D)
        L = x.shape[-2]; mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
        for H, W1, b1, W2 in s.W:
            for Q, K, Vs in H:
                sc = (x @ Q.T) @ (x @ K.T).transpose(-1, -2)
                a = sc.masked_fill(mask, float('-inf')).softmax(-1)
                x = x + a @ (x @ Vs.T)
            if W1.shape[0]: x = x + torch.relu(x @ W1.T + b1) @ W2.T
        return x


def build(tk, G=1000.0):
    h = Hand()
    T = lambda s: tk.idx[s]
    digit_ids = [T(str(v)) for v in range(10)]
    ids = {'is': T(' is'), 'plus': T(' plus'), 'minus': T(' minus'), 'q': T('?'), 'ans': T('Answer'),
           'colon': T(':'), 'sp': T(' '), 'spminus': T(' -'), 'nl': 13, 'bos': 1}
    # ---- input features (embedding of token id) + position
    for n in ['i', 'one', 'isDigit', 'dval', 'isIs', 'isOp', 'isMinus', 'isQ', 'isAns', 'isBOS']: h.d(n)

    def step(b, x, out_name, coef=1.0):  # out += coef*[x >= 1] for integer-valued x (dict)
        h.unit(b, x, 0.0, {out_name: coef}); h.unit(b, x, -1.0, {out_name: -coef})

    # ---- L1: most-recent marker positions
    BM, EP = 1e5, 50.0
    b1 = h.block()
    for mk, flag, extra in [('Is', 'isIs', {}), ('Op', 'isOp', {'isMinus': 'opMinus'}), ('Q', 'isQ', {}), ('Ans', 'isAns', {})]:
        v = {'i': ('p' + mk, 1.0), flag: ('has' + mk, 1.0)}
        for src, dst in extra.items(): v[src] = (dst, 1.0)
        h.head(b1, [{'one': BM}, {'one': EP}], [{flag: 1.0}, {'i': 1.0}], v)
    # region flags (used when this token is a KEY later) and validity (used when it is a QUERY)
    BIG = 1000.0
    # inA = digit & hasIs & (pIs > pOp or !hasOp)
    for name, conds in [
        ('inA', [{'isDigit': 1}, {'hasIs': 1}, {'pIs': 1, 'pOp': -1, 'hasOp': -BIG, 'one': BIG}]),
        ('inB', [{'isDigit': 1}, {'hasOp': 1}, {'pOp': 1, 'pIs': -1},
                 {'pOp': 1, 'pQ': -1, 'hasQ': -BIG, 'one': BIG}, {'pOp': 1, 'pAns': -1, 'hasAns': -BIG, 'one': BIG}]),
        ('inAns', [{'isDigit': 1}, {'hasAns': 1}, {'pAns': 1, 'pOp': -1}]),
        ('valid', [{'hasIs': 1}, {'hasOp': 1}, {'hasQ': 1}, {'hasAns': 1}, {'pOp': 1, 'pIs': -1},
                   {'pQ': 1, 'pOp': -1}, {'pAns': 1, 'pQ': -1}, {'pQ': -1, 'pAns': 1, 'one': -1},   # pAns-pQ >= 2
                   {'pQ': 1, 'pAns': -1, 'one': 3}])]:                                         # pAns-pQ <= 2
        # AND of [x_j >= 1] : compute each step into a temp then relu(sum - (n-1))
        tmps = []
        for j, c in enumerate(conds):
            t = f'{name}_c{j}'; tmps.append(t)
            if list(c.keys()) in (['isDigit'], ['hasIs'], ['hasOp'], ['hasQ'], ['hasAns']):
                h.unit(b1, c, 0.0, {t: 1.0})  # already 0/1
            else: step(b1, c, t)
        h.d(name)
        h.blocks[-1]['and'] = h.blocks[-1].get('and', []) + [(name, tmps)]
    # ---- L2: AND gates (MLP only)
    b2 = h.block()
    for name, tmps in b1['and']:
        h.unit(b2, {t: 1.0 for t in tmps}, -(len(tmps) - 1), {name: 1.0})

    # ---- L3: sink-attention sums -> exact operand values
    S0, C = 100.0, 0.0
    KILL = 4000.0
    scale = math.exp(S0 - C)
    b3 = h.block()
    for nm, flag, anchor, end in [('Aval', 'inA', 'pIs', {'pOp': 1.0, 'one': -1.0}),
                                  ('Bval', 'inB', 'pOp', {'pQ': 1.0, 'one': -1.0})]:
        GAM = 10.0
        qrow = {k: LN10 * w for k, w in end.items()}; qrow[anchor] = qrow.get(anchor, 0) - GAM; qrow['one'] = qrow.get('one', 0) + C + KILL
        # score = flag_k*(ln10*end_q - GAM*anchor_q + C + KILL) - ln10*i_k + GAM*anchor_k - KILL + S0*isBOS_k
        q = [qrow, {'one': -LN10}, {'one': GAM}, {'one': -KILL}, {'one': S0 + KILL}]
        k = [{flag: 1.0}, {'i': 1.0}, {anchor: 1.0}, {'one': 1.0}, {'isBOS': 1.0}]
        h.head(b3, q, k, {'dval': (nm, scale)})
    # S = A +/- B
    M = 1e13
    h.unit(b3, {'Aval': 1.0}, 0.0, {'S': 1.0})
    h.unit(b3, {'Bval': 1.0, 'opMinus': -M}, 0.0, {'S': 1.0})
    h.unit(b3, {'Bval': 1.0, 'opMinus': M}, -M, {'S': -1.0})

    # ---- L4: sign, |S|, number of digits
    b4 = h.block()
    h.unit(b4, {'S': -1.0}, 0.0, {'neg': 1.0}); h.unit(b4, {'S': -1.0}, -1.0, {'neg': -1.0})
    h.unit(b4, {'S': 1.0}, 0.0, {'absS': 1.0}); h.unit(b4, {'S': -1.0}, 0.0, {'absS': 1.0})
    h.unit(b4, {'one': 1.0}, 0.0, {'Lout': 1.0})
    for m in range(1, 13):
        for sg in (1.0, -1.0):  # [S >= 10^m] + [-S >= 10^m]
            step(b4, {'S': sg, 'one': -(10 ** m) + 1}, 'Lout')

    # ---- L5: P = value of already-emitted answer digits at their significance
    b5 = h.block()
    GAM = 10.0
    qrow = {'Lout': LN10, 'pAns': LN10 - GAM, 'one': 2 * LN10 + C + KILL}
    q = [qrow, {'one': -LN10}, {'one': GAM}, {'one': -KILL}, {'one': S0 + KILL}]
    k = [{'inAns': 1.0}, {'i': 1.0}, {'pAns': 1.0}, {'one': 1.0}, {'isBOS': 1.0}]
    h.head(b5, q, k, {'dval': ('P', scale)})
    # q = Lout + pAns + 1 - i ; pow = 10^q via hats ; stage indicators
    qv = {'Lout': 1.0, 'pAns': 1.0, 'one': 1.0, 'i': -1.0}
    def hat(b, x, center, out, coef):  # coef*[x == center], x integer
        for off, w in [(-1, 1.0), (0, -2.0), (1, 1.0)]:
            xx = dict(x); xx['one'] = xx.get('one', 0) - (center + off)
            h.unit(b, xx, 0.0, {out: coef * w})
    for m in range(0, 13): hat(b5, qv, m, 'pow', float(10 ** m))
    hat(b5, {'i': 1.0, 'pAns': -1.0}, 1, 'k0', 1.0)          # at ':' -> sign token
    hat(b5, qv, -1, 'kend', 1.0)                              # after last digit -> newline
    step(b5, {'Lout': 1.0, 'pAns': 1.0, 'one': 2.0, 'i': -1.0}, 'kd_a')  # q >= 0
    step(b5, {'i': 1.0, 'pAns': -1.0, 'one': -1.0}, 'kd_b')            # i >= pAns+2

    # ---- L6: digit = sum_v [R >= v*10^q], R = |S| - P
    b6 = h.block()
    for v in range(1, 10):
        x = {'absS': 1.0, 'P': -1.0, 'pow': -float(v), 'one': 1.0}  # [R - v*pow >= 0]
        step(b6, x, f'st{v}')
    h.unit(b6, {'kd_a': 1.0, 'kd_b': 1.0}, -1.0, {'kd': 1.0})

    # ---- L7: gated logits
    b7 = h.block()
    st = lambda v: ({'one': 1.0} if v == 0 else {} if v == 10 else {f'st{v}': 1.0})
    for v in range(10):
        onehot = dict(st(v))
        for kk, w in st(v + 1).items(): onehot[kk] = onehot.get(kk, 0) - w
        inp = dict(onehot); inp['kd'] = 1.0; inp['valid'] = inp.get('valid', 0) + 1.0
        h.unit(b7, inp, -2.0, {f'lg{v}': G})
    h.unit(b7, {'k0': 1.0, 'valid': 1.0, 'neg': -1.0}, -1.0, {'lg_sp': G})
    h.unit(b7, {'k0': 1.0, 'valid': 1.0, 'neg': 1.0}, -2.0, {'lg_spminus': G})
    h.unit(b7, {'kend': 1.0, 'valid': 1.0}, -1.0, {'lg_nl': G})
    h.compile()
    h.ids, h.digit_ids = ids, digit_ids
    h.out = {**{f'lg{v}': digit_ids[v] for v in range(10)}, 'lg_sp': ids['sp'], 'lg_spminus': ids['spminus'], 'lg_nl': ids['nl']}
    return h


def embed(h, tok):
    L = len(tok); x = torch.zeros(L, len(h.dims), dtype=torch.float64); D = h.dims; I = h.ids
    x[:, D['i']] = torch.arange(L, dtype=torch.float64); x[:, D['one']] = 1
    for p, t in enumerate(tok):
        if t in h.digit_ids: x[p, D['isDigit']] = 1; x[p, D['dval']] = h.digit_ids.index(t)
        if t == I['is']: x[p, D['isIs']] = 1
        if t in (I['plus'], I['minus']): x[p, D['isOp']] = 1
        if t == I['minus']: x[p, D['isMinus']] = 1
        if t == I['q']: x[p, D['isQ']] = 1
        if t == I['ans']: x[p, D['isAns']] = 1
        if t == I['bos']: x[p, D['isBOS']] = 1
    return x


def circuit_logits(h, tok, V=32000):
    x = h.forward(embed(h, tok)); out = torch.zeros(len(tok), V, dtype=torch.float64)
    for n, vid in h.out.items(): out[:, vid] += x[:, h.dims[n]]
    return out, x
