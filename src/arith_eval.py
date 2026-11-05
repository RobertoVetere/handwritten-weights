"""Section 5: single-digit addition written into stories15M's own weights. Writes arith_results.json."""
import random, json
import arith2 as A2
from arith4 import *
from arith3 import evaluate

if __name__ == '__main__':
    A2.OSC = 0.2
    base = load(); random.seed(11)
    P = random.sample([(a, b) for a in range(10) for b in range(10)], 40); small = [(a, b) for a, b in P if a + b <= 9]
    info = {}; m = build_channel(base, log=info); low5 = ST['use'][5].argsort().tolist()
    add_cancel(m, info, low5[38:41]); add_readout(m, info, low5[:38], G=15.0)
    res = {}
    for name, mm in [('base', base), ('edited', m)]:
        res[name] = {'design_first_full': evaluate(mm, DES, P), 'heldout_first_full': evaluate(mm, HELD, P),
                     'heldout_sums<=9_full': evaluate(mm, HELD, small)[1], 'real_loss': real_loss(mm)}
        print(name, json.dumps(res[name]), flush=True)
    json.dump(res, open('arith_results.json', 'w'), indent=1)
    for p in ['Tom had 3 apples. He got 4 more. Now he has ', 'Q: What is 6 + 2?\nA: ']:
        print('  ', repr(p), '->', repr(greedy(m, p, 6)))
