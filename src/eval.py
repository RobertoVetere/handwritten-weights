"""Evaluate base stories15M vs stories15M + hand-written grafted circuit on the GPT-3 arithmetic benchmark."""
import sys, random, json, time, torch
from llama import Tokenizer, Llama
from circuit import build, circuit_logits

tk = Tokenizer('tokenizer.bin'); lm = Llama('stories15M.bin'); h = build(tk)

def ctx_of(a, op, b): return f'Question: What is {a} {op} {b}?\nAnswer:'
def ans_of(a, op, b): return f' {a + b if op == "plus" else a - b}'

def generate(tokens, n, graft):
    t = list(tokens)
    for _ in range(n):
        lg = lm.logits(torch.tensor([t]))[0, -1]
        if graft: lg = lg + circuit_logits(h, t)[0][-1]
        t.append(int(lg.argmax()))
    return t[len(tokens):]

def run(problems, graft):
    ok = 0
    for a, op, b in problems:
        c = tk.encode(ctx_of(a, op, b)); full = tk.encode(ctx_of(a, op, b) + ans_of(a, op, b))
        assert full[:len(c)] == c
        tgt = full[len(c):]
        ok += generate(c, len(tgt), graft) == tgt
    return ok / len(problems)

def gpt3_problems(nd, op, n, seed):
    r = random.Random(seed)
    return [(r.randrange(10 ** nd), op, r.randrange(10 ** nd)) for _ in range(n)]

if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    res = {}
    # official rows
    rows, cur = [], None
    for line in open('official_rows.txt'):
        if line.startswith('#'): continue
        p = line.split()
        if len(p) == 2 and not p[0].isdigit(): cur = (p[0], p[1]); continue
        rows.append((cur[0], int(p[0]), cur[1], int(p[1])))
    for cfg in ['5da', '5ds']:
        pr = [(a, op, b) for c, a, op, b in rows if c == cfg]
        for g in (False, True):
            res[f'official_{cfg}_{"graft" if g else "base"}'] = run(pr, g)
        print(cfg, 'official', res, flush=True)
    for nd in [2, 3, 4, 5]:
        for op, tag in [('plus', 'a'), ('minus', 's')]:
            pr = gpt3_problems(nd, op, N, seed=nd * 10 + len(tag))
            for g in (False, True):
                t0 = time.time(); acc = run(pr, g)
                res[f'{nd}d{tag}_{"graft" if g else "base"}'] = acc
                print(f'{nd}d{tag} graft={g} acc={acc:.3f} ({time.time() - t0:.0f}s)', flush=True)
    json.dump(res, open('results.json', 'w'), indent=1)
