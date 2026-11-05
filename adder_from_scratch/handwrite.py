"""Hand-written weights for TinyTF solving N-digit addition. No gradient descent anywhere.

Key identity used (makes carry LOCAL, no recursion):
    carry into digit q  =  [ s_{q-1} < a_{q-1} + b_{q-1} ]
where s_{q-1} is the previously emitted digit (= current input token when predicting s_q).

Residual stream layout (d=35):
  0-11 token one-hot | 12 digit value | 13 isA 14 isB 15 isOut 16 isEq | 17 p 18 p^2 19 one
  20 a_q 21 b_q 22 a_{q-1} 23 b_{q-1}  (written by layer-1 heads)
  24 t = a_q + b_q + carry  (written by MLP-1)
  25-34 one-hot of (t mod 10)  (written by MLP-2) -> unembedding
"""
import torch
from model import TinyTF, V

BETA = 40.0     # attention sharpness: neighbouring position loses by BETA/2 in logit
OUT_GAIN = 30.0  # unembedding gain


def build(beta=BETA):
    m = TinyTF().double()
    for p in m.parameters(): p.data.zero_()
    W = lambda t: t.weight.data
    # --- embeddings
    for v in range(V): W(m.E)[v, v] = 1
    for v in range(10): W(m.E)[v, 12] = v                      # digit value
    for j, dim in enumerate([17, 18, 19, 13, 14, 15, 16]):       # f = (p,p2,1,isA,isB,isOut,isEq)
        W(m.P)[dim, j] = 1
    # --- layer 1 attention: head h fetches digit of type T at position p_q - delta
    b1 = m.blocks[0]
    heads = [(13, 0), (14, 0), (13, 1), (14, 1)]  # (type dim, delta)
    for h, (tdim, dl) in enumerate(heads):
        r = slice(4 * h, 4 * h + 4)
        # key = (p_k, p_k^2, 1, isT)
        Wk = torch.zeros(4, 35, dtype=torch.float64); Wk[0, 17] = 1; Wk[1, 18] = 1; Wk[2, 19] = 1; Wk[3, tdim] = 1
        # query . key = beta * [ -(p_k - (p_q - dl))^2 + isT/2 ]
        Wq = torch.zeros(4, 35, dtype=torch.float64)
        Wq[0, 17] = 2 * beta; Wq[0, 19] = -2 * beta * dl            # 2(p_q - dl) * p_k
        Wq[1, 19] = -beta                                          # -p_k^2
        Wq[2, 18] = -beta; Wq[2, 17] = 2 * beta * dl; Wq[2, 19] = -beta * dl * dl  # -(p_q-dl)^2
        Wq[3, 19] = beta / 2                                       # type bonus
        W(b1.Wq)[r] = Wq; W(b1.Wk)[r] = Wk
        W(b1.Wv)[4 * h, 12] = 1                                    # value = digit value
        W(b1.Wo)[20 + h, 4 * h] = 1                                # write to dim 20+h
    # --- MLP 1:  t = relu(a+b) + relu(x) - relu(x-1),  x = a'+b' - s - 100*isEq   (carry = [x>=1])
    W(b1.W1)[0, 20] = 1; W(b1.W1)[0, 21] = 1
    for u, bias in [(1, 0.0), (2, -1.0)]:
        W(b1.W1)[u, 22] = 1; W(b1.W1)[u, 23] = 1; W(b1.W1)[u, 12] = -1; W(b1.W1)[u, 16] = -100
        b1.W1.bias.data[u] = bias
    W(b1.W2)[24, 0] = 1; W(b1.W2)[24, 1] = 1; W(b1.W2)[24, 2] = -1
    # --- layer 2: attention off; MLP builds one-hot(t) via hat functions, folded mod 10
    b2 = m.blocks[1]
    us = list(range(-1, 21))                       # 22 hidden units g_u = relu(t - u)
    for i, u in enumerate(us):
        W(b2.W1)[i, 24] = 1; b2.W1.bias.data[i] = -u
    for v in range(20):                            # 1[t==v] = g_{v-1} - 2 g_v + g_{v+1}
        i = us.index(v)
        W(b2.W2)[25 + v % 10, i - 1] += 1; W(b2.W2)[25 + v % 10, i] += -2; W(b2.W2)[25 + v % 10, i + 1] += 1
    # --- unembedding
    for v in range(10): W(m.U)[v, 25 + v] = OUT_GAIN
    return m


if __name__ == '__main__':
    from model import exact_match
    m = build()
    nnz = sum((p != 0).sum().item() for p in m.parameters()); tot = sum(p.numel() for p in m.parameters())
    print(f'params total={tot} nonzero(hand-written)={nnz}')
    for n in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        B = 200 if n <= 100 else 40
        print(f'n={n:5d}  exact-match={exact_match(m, n, B=B, chunk=20 if n > 100 else 50):.3f}', flush=True)
