import torch
from math import sqrt



class KVCache:
    def __init__(self, n_layer, batch, n_kv_head, head_dim, max_len, dtype=torch.float32):
        self.k=[torch.zeros(batch,n_kv_head,max_len,head_dim,dtype=dtype)for i in range(n_layer)]
        self.v=[torch.zeros(batch,n_kv_head,max_len,head_dim,dtype=dtype)for i in range(n_layer)]
        self._pos = 0  # 一次分配，全程复用
        
    @property
    def pos(self) :
        return self._pos#已有的token
        
    def write(self, layer, k_new, v_new):
        S =k_new.shape[-2]
        self.k[layer][:,:,self._pos:self._pos+S,:] = k_new
        self.v[layer][:,:,self._pos:self._pos+S,:] = v_new

        
        # 把这一步算出的 K/V 写到第 layer 层的 [pos : pos+S] 段
    def read(self, layer, upto=None):
        upto = self._pos if upto is None else upto
        return self.k[layer][:,:,0:upto,:],self.v[layer][:,:,0:upto,:]
        # 取出第 layer 层 [0 : upto] 的历史 K/V，默认到 pos
    def advance(self, n) :
        self._pos+=n
        # 推进 pos——整个引擎里只有这一个地方能改 pos
    def nbytes(self) :
        return (2*len(self.k)*self.k[0].numel()*self.k[0].element_size() )
        # K+V 一共占多少字节



def attention_with_cache(q, k_new, v_new, cache, layer, causal=True):
    cache.write(layer, k_new, v_new)
    cache.advance(k_new.shape[-2])
    k_all, v_all = cache.read(layer)
    score = q@k_all.transpose(-2,-1)/sqrt(q.shape[-1])
    pos=cache.pos
    S=q.shape[-2]
    q_idx =torch.arange(pos-S,pos).view(1,1,S,1)
    k_idx = torch.arange(pos).view(1,1,1,pos)
    mask = (q_idx<k_idx)
    if causal:
        score = score.masked_fill(mask,float("-inf"))
    score = torch.softmax(score,dim=-1)
    out = score@v_all
    return out


    # ① 写入推进（顺序问题在这里定下来）
    # ② 取历史：k_all, v_all 形状 [B, H, pos, D]
    # ③ 打分：q[B,H,S,D] @ k_all.transpose(-2,-1) → [B,H,S,pos]，除以 sqrt(D)
    # ④ 掩码（唯一的新东西，见下）
    # ⑤ softmax(dim=-1) → @ v_all → [B,H,S,D]


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, D, L, T = 1, 4, 8, 2, 12
    P = 5                                  # 前 5 个 token 一次 prefill，剩下逐 token

    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)

    # ① 参考路径：不用 cache，全序列一次算完
    score = q@k.transpose(-2,-1)/sqrt(q.shape[-1])
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)   # 上三角=未来
    score = score.masked_fill(mask, float("-inf"))
    out_ref = torch.softmax(score, dim=-1) @ v

    # ② cache 路径：先 prefill 前 P 个，再逐个喂剩下的
    cache = KVCache(L, B, H, D, max_len=T)
    outs = []
    outs.append(attention_with_cache(q[:, :, :P], k[:, :, :P], v[:, :, :P], cache, layer=0))
    for t in range(P, T):
        outs.append(attention_with_cache(q[:, :, t:t+1], k[:, :, t:t+1], v[:, :, t:t+1], cache, layer=0))
    out_cache = torch.cat(outs, dim=2)          # 问：沿哪一维拼？

    print("形状:", out_cache.shape, "pos:", cache.pos, "nbytes:", cache.nbytes())
    print("max diff:", (out_ref - out_cache).abs().max().item())
    print("[PASS]" if torch.allclose(out_ref, out_cache, atol=1e-5) else "[FAIL]")