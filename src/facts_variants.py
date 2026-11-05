import json, sys, facts7 as F7
from surgery import load
from facts2 import evaluate
base = load(); out = {}
for v in sys.argv[1:]:
    m = F7.build(base, variant=v); r = evaluate(m); out[v] = r
    print(v, json.dumps(r), flush=True)
json.dump(out, open('facts_variants_' + ''.join(sys.argv[1:]) + '.json', 'w'), indent=1)
