import torch, random
from llama import Tokenizer, Llama
tk = Tokenizer('tokenizer.bin'); lm = Llama('stories15M.bin')
torch.manual_seed(0)
starts = ['Once upon a time', 'One day', 'Lily', 'Tom and Sam', 'There was a', 'The little', 'In a small town', 'Max the dog']
seqs = [tk.encode(starts[i % len(starts)]) for i in range(160)]
L = 160
for b in range(0, 160, 40):
    s = seqs[b:b + 40]; n0 = len(min(s, key=len))
    s = [x[:n0] for x in s]
    t = torch.tensor(s)
    while t.shape[1] < L:
        lg = lm.logits(t)[:, -1]
        nxt = torch.multinomial(torch.softmax(lg, -1), 1)
        t = torch.cat([t, nxt], 1)
    for i, row in enumerate(t.tolist()): seqs[b + i] = row
torch.save(torch.tensor(seqs), 'corpus.pt')
for r in seqs[:3]: print(repr(tk.decode(r[1:])[:200]))
