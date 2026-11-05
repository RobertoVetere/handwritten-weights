"""Gradient-trained baseline with the SAME architecture and SAME positional features as the hand-written model."""
import sys, time, json, torch, torch.nn.functional as F
from model import TinyTF, make_batch, random_problems, exact_match

def train(seed, d=35, dm=32, steps=20000, maxn=10, B=128, lr=3e-3):
    torch.manual_seed(seed)
    m = TinyTF(d=d, dm=dm)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    t0 = time.time()
    for it in range(steps):
        n = torch.randint(1, maxn + 1, ()).item()
        inp, f, tgt, mk = make_batch(*random_problems(B, n))
        loss = F.cross_entropy(m(inp, f)[:, mk].reshape(-1, 12), tgt[:, mk].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
    res = {'seed': seed, 'd': d, 'dm': dm, 'train_s': round(time.time() - t0, 1), 'loss': loss.item()}
    for n in [5, 10, 12, 15, 20, 50, 100]:
        res[n] = exact_match(m, n, B=200, seed=123)
    return res

if __name__ == '__main__':
    d, dm = int(sys.argv[1]), int(sys.argv[2])
    seeds = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else range(3)
    for seed in seeds:
        print(json.dumps(train(seed, d, dm)), flush=True)
