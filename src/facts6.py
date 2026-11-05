"""Facts v6 (in-place, no gradients).
 (1) small distinct embedding tag alpha*t_k on 8 tokens (7 countries + ' capital'), in directions quiet at input & output
 (2) L3 MLP: 8 low-use neurons detect the tag at the token itself and write a large flag F_k
 (3) flags chosen by generalized eigenproblem: carried by L4H3+L4H4, NOT by the other L4 heads (whitened at L4 MLP input)
 (4) L4 MLP: 7 low-use neurons = AND(country_k, capital) -> write the capital token's logit direction
All directions from weights/covariances; thresholds from design prompts; slopes chosen so the SiLU leak is < 1e-5."""
import torch, json, sys
import torch.nn.functional as Fn
from surgery import *
from facts_edit import C, CAP, OTHER, design, heldout, nocap, top1
from facts2 import clone, evaluate, CD, ST
hs = 48; WORDS = C + ['capital']; LD, LT = 3, 4

def sharp(own_min, worst, margin_units=14.0):
    thr = (own_min + worst) / 2; s = 2 * margin_units / max(own_min - worst, 1e-3); return thr, s

def build(base, alpha=0.25, M=10.0, heads=(3, 4), gamma=60.0, qa=1.0, kb=20.0, log=None):
    m = clone(base); m.emb = base.emb.clone(); data = torch.load('corpus.pt'); info = {}
    _, Cc = fwd(base, data[:30], cache=True)
    x = Cc['resid'][-1][:, 1:].reshape(-1, m.dim); Z = x / x.pow(2).mean(1, keepdim=True).sqrt() * m.rms_final
    X0 = Cc['nin'][0][:, 1:].reshape(-1, m.dim); X4 = Cc['nin'][LT][:, 1:].reshape(-1, m.dim); del Cc
    Cz = torch.cov(Z.T); g0 = m.rms_ffn[0]; Cin = torch.diag(1 / g0) @ torch.cov(X0.T) @ torch.diag(1 / g0)
    ev, V = torch.linalg.eigh(Cz / Cz.diagonal().mean() + Cin / Cin.diagonal().mean()); tags = V[:, :8].T
    ids = [tk.idx[' ' + w] for w in WORDS]
    base_emb = m.emb.clone()
    for k, t in enumerate(ids): m.emb[t] = m.emb[t] + alpha * tags[k]
    m.wcls = m.emb
    info['tag_logit_noise_std'] = (alpha * (tags @ Cz @ tags.T).diagonal().sqrt()).max().item()
    # (3) flags via generalized eigenproblem at layer LT
    S4 = torch.cov(X4.T) + 1e-6 * torch.eye(m.dim, dtype=X4.dtype); e4, Q4 = torch.linalg.eigh(S4)
    Sm12 = Q4 @ torch.diag(e4.clamp_min(1e-6) ** -0.5) @ Q4.T
    def Bh(h): return torch.diag(m.rms_ffn[LT]) @ m.wo[LT][:, h*hs:(h+1)*hs] @ m.wv[LT][h*hs:(h+1)*hs] @ torch.diag(m.rms_att[LT])
    Bt = sum(Bh(h) for h in heads); T = (Sm12 @ Bt).T @ (Sm12 @ Bt)
    _, Cc2 = fwd(base, data[:30], cache=True); xa0 = Cc2['resid'][LT][:, 1:].reshape(-1, m.dim); del Cc2
    za = xa0 / xa0.pow(2).mean(1, keepdim=True).sqrt(); Ca = torch.cov(za.T)          # natural cov of x/rms at L4 input
    Oh = sum((Sm12 @ Bh(h)).T @ (Sm12 @ Bh(h)) for h in range(6) if h not in heads)
    O = Oh / Oh.diagonal().mean() + Ca / Ca.diagonal().mean() + 1e-3 * torch.eye(m.dim, dtype=T.dtype)
    T = T / T.diagonal().mean()
    Lc = torch.linalg.cholesky(O); Li = torch.linalg.inv(Lc)
    w, U = torch.linalg.eigh(Li @ T @ Li.T); Fk = (Li.T @ U[:, -8:].flip(1)).T; Fk = Fn.normalize(Fk, dim=1)
    info['flag_target_vs_other_gain'] = [((f @ T @ f) / (f @ O @ f)).sqrt().item() for f in Fk]
    Rk = torch.stack([Fn.normalize(Sm12.T @ (Sm12 @ Bt @ f), dim=0) for f in Fk])
    Gw = Sm12 @ Bt @ Fk.T                                  # whitened images of the 8 flags (dim x 8)
    Rw = torch.linalg.pinv(Gw)                             # dual basis: Rw @ Gw = I
    Rw = Rw / Rw.norm(dim=1, keepdim=True)                 # unit in whitened space -> natural noise std 1
    Rk = Rw @ Sm12
    # (2) L3 detectors: measured direction = mean change of L3 MLP input at the token caused by the tag
    m0 = clone(base); m0.emb = base_emb; m0.wcls = m0.emb
    def n3(mm, w, tpls):
        t = tk.idx[' ' + w]; out = []
        for tpl in tpls:
            for c in (C if w == 'capital' else [w]):
                s = tk.encode(tpl.format(c=c)); _, Cc = fwd(mm, torch.tensor([s]), cache=True)
                out += [Cc['nin'][LD][0, p] for p, xx in enumerate(s) if xx == t]
        return torch.stack(out)
    _, Cc = fwd(m, data[:20], cache=True); corp = Cc['nin'][LD][:, 1:].reshape(-1, m.dim); del Cc
    low = ST['use'][LD].argsort().tolist(); cD = CD[LD]; det = []
    for k, w in enumerate(WORDS):
        d = Fn.normalize((n3(m, w, design) - n3(m0, w, design)).mean(0), dim=0)
        own = n3(m, w, design) @ d; oth = torch.cat([n3(m, o, design[:1]) for o in WORDS if o != w]) @ d
        cor = corp @ d; worst = max(oth.max().item(), cor.max().item())
        thr, s = sharp(own.min().item(), worst)
        det.append([w, round(own.min().item(), 2), round(oth.max().item(), 2), round(cor.max().item(), 2)])
        j = low[k]; m.w1[LD][j] = s * (d - thr * cD); m.w3[LD][j] = cD; m.w2[LD][:, j] = M * Fk[k]
    info['detectors(own_min,others_max,corpus_max)'] = det
    # (2b) attention surgery on target heads: lowest-frequency RoPE pair (dims 46,47; ~0.04 rad over 256 tokens)
    m.wq = base.wq.clone(); m.wk = base.wk.clone()
    _, Cc = fwd(base, data[:30], cache=True); xa = Cc['resid'][LT][:, 1:].reshape(-1, m.dim); del Cc
    na = xa / xa.pow(2).mean(1, keepdim=True).sqrt() * m.rms_att[LT]
    ca = torch.linalg.solve(torch.cov(na.T) + 1e-4 * torch.eye(m.dim, dtype=na.dtype), na.mean(0)); ca = ca / (na @ ca).mean()
    kf = Fn.normalize((m.rms_att[LT] * Fk).sum(0), dim=0)       # 'any flag present' direction at attention input
    for h in heads:                                      # free the RoPE pair (44,45), then use it as a dedicated channel
        for d in (44, 45): m.wq[LT][h*hs + d] = 0; m.wk[LT][h*hs + d] = 0
        m.wq[LT][h*hs + 44] = qa * ca
        m.wk[LT][h*hs + 44] = kb * kf
    info['attn_edit'] = {'qa': qa, 'kb': kb}
    # (4) AND neurons at LT
    def nl(mm, p):
        _, Cc = fwd(mm, torch.tensor([tk.encode(p)]), cache=True); return Cc['nin'][LT][0, -1]
    Pd = torch.stack([torch.stack([nl(m, t.format(c=c)) @ Rk.T for t in design]) for c in C])
    own = torch.stack([Pd[i, :, i] for i in range(7)]); capp = Pd[:, :, 7]
    _, Cc = fwd(m, data[:20], cache=True); pc = Cc['nin'][LT][:, 1:].reshape(-1, m.dim) @ Rk.T; del Cc
    sig = (own + capp).min().item(); noise = (pc[:, :7] + pc[:, 7:8]).max().item()
    cross = max((Pd[j, :, i] + capp[j]).max().item() for i in range(7) for j in range(7) if i != j)
    info.update({'own_mean': own.mean().item(), 'own_min': own.min().item(), 'cap_mean': capp.mean().item(),
                 'design_sum_min': sig, 'corpus_sum_max': noise, 'cross_country_max': cross})
    worst = max(noise, cross); thr2, s2 = sharp(sig, worst)
    info['and_margin'] = sig - worst
    low4 = ST['use'][LT].argsort().tolist(); c4 = CD[LT]
    if sig > worst:
        for i in range(7):
            j = low4[i]; m.w1[LT][j] = s2 * (Rk[i] + Rk[7] - thr2 * c4); m.w3[LT][j] = c4
            tgt = m.emb[tk.idx[' ' + CAP[i]]] - m.emb.mean(0)
            m.w2[LT][:, j] = gamma * Fn.normalize(m.rms_final * tgt, dim=0)
    info['_Pd'] = Pd; info['_Rk'] = Rk; info['_kf'] = kf
    if log is not None: log.update(info)
    return m

if __name__ == '__main__':
    base = load()
    for M in [float(a) for a in (sys.argv[1:] or ['10'])]:
        log = {}; m = build(base, M=M, log=log)
        print('M', M, json.dumps(log, default=lambda v: round(v, 3)), flush=True)
        if log['and_margin'] > 0:
            print('  eval', json.dumps(evaluate(m)), flush=True)
            for p in ['The capital of France is', 'Q: What is the capital of Japan?\nA:', 'In Italy, the capital city is', 'Lily went to Spain with her mom. She']:
                print('   ', repr(p), '->', repr(greedy(m, p, 8)))
        else:
            print('  no positive AND margin')
