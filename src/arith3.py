"""Arithmetic v3 = arith2 (angle channel) + L5 MLP readout -> first answer token. Single-digit a+b (0..18).
 neuron pair v (v=1..18): clipped step on  S*(Y cos b_v - X sin b_v) + KS*(slot - thr_slot), gated (SwiGLU) by presence.
 onehot_v = st_v - st_{v+1}; st_0 = gate.  first token of v: v (v<=9) or '1' (v>=10)."""
import torch, json, sys, random, math
import torch.nn.functional as Fn
from surgery import *
from facts2 import clone, CD, ST
from arith import DES, HELD, DIG, lda, run
import arith2 as A2
hs = 48; PHI = A2.PHI
SP = tk.idx[' ']

def build(base, K=60.0, G=30.0, log=None):
    info = {}; m = A2.build(base, K=K, log=info)
    rX, rY, rP = info['_r']; data = torch.load('corpus.pt')
    _, Cc = fwd(m, data[:20], cache=True); n5 = Cc['nin'][5][:, 1:].reshape(-1, m.dim); toks = data[:20, 1:].reshape(-1); del Cc
    S5 = torch.cov(n5.T) + 1e-4 * torch.eye(m.dim, dtype=n5.dtype)
    # slot readout: current token is the standalone space (precedes every number in this tokenizer)
    slots = []
    for t in DES:
        for a, b in [(a, b) for a in range(10) for b in range(10)][::5]:
            _, C2 = run(m, t.format(a=a, b=b)); slots.append(C2['nin'][5][0, -1])
    SL = torch.stack(slots); nons = n5[toks != SP]
    rs = lda(SL, nons, S5); sp, sn = (SL @ rs).min().item(), (nons @ rs).max().item()
    info['slot readout (slot_min, nonslot_max)'] = [sp, sn]
    pres = SL @ rP; R = torch.stack([SL @ rX, SL @ rY], 1).norm(dim=1)
    info['presence at slots (min,max)'], info['|XY| at slots (min,max)'] = [pres.min().item(), pres.max().item()], [R.min().item(), R.max().item()]
    Rm = R.mean().item(); Pm = pres.mean().item()
    S = 14.0 / (Rm * math.sin(PHI / 4))
    sproj = SL @ rs; smean = sproj.mean().item(); sspread = (sproj.max() - sproj.min()).item()
    KS = 2.0 / max(sspread, 1e-3); ths = smean      # within slots: |offset| <= 1 ; outside: <= -KS*(smean - sn)
    info['slot spread'] = sspread; info['offset outside slots'] = -KS * (smean - sn)
    low = ST['use'][5].argsort().tolist()[:40]; c5 = CD[5]
    Sv = torch.cov(m.emb.T) + 1e-4 * torch.eye(m.dim, dtype=m.emb.dtype)
    outd = [Fn.normalize(torch.linalg.solve(Sv, m.emb[DIG[d]] - m.emb.mean(0)) / m.rms_final, dim=0) for d in range(10)]
    first = lambda v: v if v <= 9 else 1
    gate = rP / Pm                                                  # SwiGLU gate ~1 with digits, ~0 without
    def pair(j1, j2, arg, out):
        m.w1[5][j1] = arg; m.w3[5][j1] = gate; m.w2[5][:, j1] = out / 10
        m.w1[5][j2] = arg - 10 * gate; m.w3[5][j2] = gate; m.w2[5][:, j2] = -out / 10
    slot_term = KS * (rs - ths * gate)
    # st_0 (always on when gated) and st_v for v=1..18 ; logits: onehot_v -> token first(v)
    for v in range(0, 19):
        if v == 0: arg = slot_term
        else:
            b = (v - 0.5) * PHI / 2; arg = S * (math.cos(b) * rY - math.sin(b) * rX) + slot_term
        # st_v contributes +onehot_v and -onehot_{v-1}
        out = G * outd[first(v)] - (G * outd[first(v - 1)] if v >= 1 else 0)
        pair(low[2 * v], low[2 * v + 1], arg, out)
    info['S'] = S; info['KS'] = KS
    if log is not None: log.update(info)
    return m

def answer_tokens(prompt, a, b):
    full = tk.encode(prompt + str(a + b)); c = tk.encode(prompt); assert full[:len(c)] == c; return c, full[len(c):]

def evaluate(m, tpls, pairs):
    ok1 = okf = n = 0
    for t in tpls:
        for a, b in pairs:
            c, tgt = answer_tokens(t.format(a=a, b=b), a, b)
            lg = fwd(m, torch.tensor([c]))[0, -1]; ok1 += int(lg.argmax()) == tgt[0]
            seq = list(c)
            for _ in range(len(tgt)): seq.append(int(fwd(m, torch.tensor([seq]))[0, -1].argmax()))
            okf += seq[len(c):] == tgt; n += 1
    return ok1 / n, okf / n

if __name__ == '__main__':
    base = load(); log = {}; m = build(base, log=log)
    print(json.dumps({k: v for k, v in log.items() if not k.startswith('_') and k != 'detectors'}, default=lambda v: round(v, 3)), flush=True)
    pairs = [(a, b) for a in range(10) for b in range(10)]; random.seed(5); P = random.sample(pairs, 25)
    small = [(a, b) for a, b in P if a + b <= 9]
    for name, mm in [('base', base), ('edited', m)]:
        print(name, 'design first/full', [round(x, 3) for x in evaluate(mm, DES, P)], '| heldout first/full', [round(x, 3) for x in evaluate(mm, HELD, P)],
              '| heldout sums<=9 full', round(evaluate(mm, HELD, small)[1], 3), flush=True)
    print('loss real', round(real_loss(m), 4), 'base', round(real_loss(base), 4))
    for p in ['Tom had 3 apples. He got 4 more. Now he has ', 'Q: What is 8 + 7?\nA: ', 'Once upon a time, there was a little']:
        print('  ', repr(p), '->', repr(greedy(m, p, 8)))
    torch.save({k: getattr(m, k) for k in ['w1', 'w2', 'w3', 'wq', 'wk', 'wv', 'wo']}, 'arith_v3_weights.pt')
