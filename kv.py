import torch
from math import sqrt

#gqa：多q对单kv
#输入q[1,4,s,8]k_new[1,2,s,8]cache.k[0]=[1,2,max_len,8]
#read之后变成[1,2,pos,8]
#这里要把k扩展到q，用复制一份，或者广播，这里先用复制一份，下一个版本可能就是广播的paged了
#最后输出out与q形状一致

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
        # 推进 pos能改 pos
    def nbytes(self) :
        return (2*len(self.k)*self.k[0].numel()*self.k[0].element_size() )
        # K+V 一共占多少字节



def attention_with_cache(q, k_new, v_new, cache, layer, causal=True):
    cache.write(layer, k_new, v_new)
    cache.advance(k_new.shape[-2])
    def expand(k_new,v_new):
            g = q.shape[1]//k_new.shape[1]
            k_new=k_new.repeat_interleave(g,dim=1)
            v_new=v_new.repeat_interleave(g,dim=1)
            return k_new,v_new
    
    k_all, v_all = cache.read(layer)
    k_all, v_all = expand(k_all, v_all)


    score = q@k_all.transpose(-2,-1)/sqrt(q.shape[-1])
    pos=cache.pos
    S=q.shape[-2]
    q_idx =torch.arange(pos-S,pos).view(1,1,S,1)
    k_idx = torch.arange(pos).view(1,1,1,pos)
    mask= (q_idx<k_idx)
    if causal:
        score = score.masked_fill(mask,float("-inf"))
    score = torch.softmax(score,dim=-1)
    out = score@v_all
    return out


def run(n_head, n_kv_head):
    torch.manual_seed(0)       # 固定种子，两次运行才可比
    B, D, L, T, P = 1, 8, 2, 12, 5

    q = torch.randn(B, n_head,   T, D)
    k = torch.randn(B, n_kv_head,   T, D)       
    v = torch.randn(B, n_kv_head,   T, D)      

    # 参考路径
    g = n_head // n_kv_head
    k_ref = k.repeat_interleave(g, dim=1)
    v_ref = v.repeat_interleave(g, dim=1)
    score = q @ k_ref.transpose(-2, -1) / sqrt(q.shape[-1])
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    score = score.masked_fill(mask, float("-inf"))
    out_ref = torch.softmax(score, dim=-1) @ v_ref

    # cache 路径：先gei 前 P 个，再逐个喂
    cache = KVCache(L, B, n_kv_head, D, max_len=T) 
    outs = [attention_with_cache(q[:, :, :P], k[:, :, :P], v[:, :, :P], cache, 0)]
    for t in range(P, T):
        outs.append(attention_with_cache(q[:, :, t:t+1], k[:, :, t:t+1],
                                         v[:, :, t:t+1], cache, 0))
    out_cache = torch.cat(outs, dim=2)

    # ← 空4：打印 shape / pos / nbytes / maxdiff / PASS-FAIL
    print("out_ref.shape=%s, out_cache.shape=%s, pos=%d, nbytes=%d, maxdiff=%.6f, PASS=%s" % (out_ref.shape, out_cache.shape, cache.pos, cache.nbytes(), (out_ref - out_cache).abs().max().item(), "PASS" if torch.allclose(out_ref, out_cache, atol=1e-5) else "FAIL"))
    return cache.nbytes()

n44=run(4, 4)
n42 = run(4, 2)