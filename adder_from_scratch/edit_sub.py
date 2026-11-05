"""Turn the hand-written ADDER into a SUBTRACTOR by editing a handful of weights (no training).
borrow into q = [ s_{q-1} > a_{q-1} - b_{q-1} ]  ;  t = a_q - b_q - borrow in [-10, 9] ; out = t mod 10
"""
import copy, torch
from handwrite import build
from model import make_batch

def random_sub(B, n, gen=None):
    a = torch.randint(0, 10, (B, n), generator=gen); b = torch.randint(0, 10, (B, n), generator=gen)
    # ensure a >= b by swapping where needed (compare as big-endian)
    A = a.flip(1); Bv = b.flip(1)
    neq = A != Bv; first = neq.float().argmax(1)
    swap = (A.gather(1, first[:, None]) < Bv.gather(1, first[:, None])).squeeze(1)
    a2 = torch.where(swap[:, None], b, a); b2 = torch.where(swap[:, None], a, b)
    z = torch.zeros(B, 1, dtype=torch.long); a2, b2 = torch.cat([a2, z], 1), torch.cat([b2, z], 1)
    s = torch.zeros_like(a2); br = torch.zeros(B, dtype=torch.long)
    for i in range(n + 1):
        t = a2[:, i] - b2[:, i] - br; s[:, i] = t % 10; br = (t < 0).long()
    return a2, b2, s

def to_subtractor(add):
    m = copy.deepcopy(add); b1, b2 = m.blocks
    W1 = b1.W1.weight.data
    # unit0: relu(a+b) -> two units would be needed for a-b (can be negative). Use shift: relu(a - b + 10)
    W1[0, 21] = -1; b1.W1.bias.data[0] = 10
    # carry units: x = s - (a'-b') - 100 isEq  (borrow iff x >= 1)
    for u in (1, 2):
        W1[u, 22] = -1; W1[u, 23] = 1; W1[u, 12] = 1
    # t' = (a-b+10) - borrow  in [0,19]  -> same mod-10 readout works unchanged!
    b1.W2.weight.data[24, 1] = -1; b1.W2.weight.data[24, 2] = 1
    return m

@torch.no_grad()
def acc(m, n, B=100):
    a, b, s = random_sub(B, n, torch.Generator().manual_seed(1))
    inp, f, tgt, mk = make_batch(a, b, s)
    return (m(inp, f).argmax(-1)[:, mk] == tgt[:, mk]).all(1).float().mean().item()

if __name__ == '__main__':
    add = build(); sub = to_subtractor(add)
    changed = sum((p1 != p2).sum().item() for p1, p2 in zip(add.parameters(), sub.parameters()))
    print('weights changed:', changed)
    for n in [1, 5, 10, 50, 200, 500]: print(n, acc(sub, n, B=100 if n < 200 else 10))
