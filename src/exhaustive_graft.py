"""Exhaustive check of the graft: every a,b in 0..99, plus and minus, through LM + circuit, batched greedy decoding."""
import torch, json, collections
from llama import Tokenizer, Llama
from circuit import build, embed
tk = Tokenizer('tokenizer.bin'); lm = Llama('stories15M.bin'); h = build(tk)
def circ(batch):  # (B,L) list of lists -> (B,V) last-position circuit logits
    x = torch.stack([embed(h, t) for t in batch]); x = h.forward(x)
    out = torch.zeros(len(batch), lm.V, dtype=torch.float64)
    for n, vid in h.out.items(): out[:, vid] += x[:, -1, h.dims[n]]
    return out
groups = collections.defaultdict(list)
for op in ['plus', 'minus']:
    for a in range(100):
        for b in range(100):
            ctx = f'Question: What is {a} {op} {b}?\nAnswer:'; ans = f' {a + b if op == "plus" else a - b}'
            c = tk.encode(ctx); full = tk.encode(ctx + ans); tgt = full[len(c):]
            groups[(len(c), len(tgt))].append((c, tgt, op))
res = {'plus': [0, 0], 'minus': [0, 0]}
for (lc, lt), items in groups.items():
    for i in range(0, len(items), 400):
        chunk = items[i:i + 400]; seqs = [list(c) for c, _, _ in chunk]
        for _ in range(lt):
            lg = lm.logits(torch.tensor(seqs))[:, -1] + circ(seqs)
            nxt = lg.argmax(-1).tolist(); seqs = [s + [n] for s, n in zip(seqs, nxt)]
        for s, (c, tgt, op) in zip(seqs, chunk):
            res[op][0] += s[lc:] == tgt; res[op][1] += 1
    print((lc, lt), res, flush=True)
json.dump(res, open('exhaustive_graft.json', 'w')); print('FINAL', res)
