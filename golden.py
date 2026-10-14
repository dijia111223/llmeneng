import numpy as np

rng=np.random.default_rng(42)

def rejection_sample_step(p, q, draft_token, u1, u2):

    p_d=max(p[draft_token],0)#大模型
    q_d=max(q[draft_token],0)#小模型
    if(q_d==0):
        a=1
    else: 
        a=min(1,p_d/q_d)
    if(u1<a):
        return draft_token,True
    b=np.maximum(p-q,0)
    residual=b/b.sum()
    
    out=np.searchsorted(np.cumsum(residual),u2)
    return out,False



p =np.array([0.5, 0.3, 0.1, 0.1])
q= np.array([0.6, 0.2, 0.1, 0.1])

print(rejection_sample_step(p, q, 1, 0.5, 0.0))  
print(rejection_sample_step(p, q, 0, 0.9, 0.5))  
N= 200_000                                       
drafts = rng.choice(4, size=N, p=q)
u1s, u2s = rng.random(N), rng.random(N)
counts = np.zeros(4)
for d, u1, u2 in zip(drafts, u1s, u2s):
    o, _ = rejection_sample_step(p, q, d, u1, u2)
    counts[o] += 1
print(counts / N)   



emp = counts / N
print("经验分布:", emp)
print("目标分布:", p)
print("最大偏差:", np.abs(emp - p).max())