import torch, random
from llama import Tokenizer, Llama
from circuit import build, circuit_logits
tk=Tokenizer('tokenizer.bin'); lm=Llama('stories15M.bin'); h=build(tk)
prompts=['Once upon a time, there was a little','Tom and his mom went to the park. Tom asked, "What is that?" Mom said','The answer is simple: Lily wanted to play plus her dog wanted to run.','Question: What is a cat? Answer:','Ben had 3 apples and 2 pears. What is 3 plus 2? He did not know.']
maxabs=0; same=0; n=0
for p in prompts:
    t=tk.encode(p); tb=list(t); tg=list(t)
    for _ in range(60):
        lb=lm.logits(torch.tensor([tb]))[0,-1]; tb.append(int(lb.argmax()))
    for _ in range(60):
        cl=circuit_logits(h,tg)[0]; maxabs=max(maxabs,cl.abs().max().item())
        lg=lm.logits(torch.tensor([tg]))[0,-1]+cl[-1]; tg.append(int(lg.argmax()))
    cl=circuit_logits(h,tg)[0]; maxabs=max(maxabs,cl.abs().max().item())
    same+= tb==tg; n+=1
    print(repr(tk.decode(tg[1:])[:150]))
print('identical generations', same,'/',n,' max |circuit logit| on these texts =',maxabs)
