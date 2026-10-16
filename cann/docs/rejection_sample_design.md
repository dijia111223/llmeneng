## 算子名：rejection_sample
## 功能，实现拒绝采样基本功能，缩减token生成时间
## 输入：p[batch,vocab],q[batch,vocab],u1[batch],u2[batch],draft_token[batch];输出out_token[batch] int32，accepted[batch] bool
##  tiling策略：输入形状为[batch,vocab],batch互不影响，我们把它在块间拆开，vocab在块内
##  数据流：p_d = Gather(p_row, draft_token)   # 从本行取 draft 位置的概率
##    q_d = Gather(q_row, draft_token)
##    is_zero=CompareScalar(q_d，0)
##    ratio=p_d/q_d
##    a_raw=Mins(1,ratio)
##    a=Select(is_zero,1.0,a_raw)
##    token_sure=Compare(LT,u1,a)

##    b=CompareScalar(p-q,0,GT,elementNum)
##    d=Select(b,p-q,0,elementNum)
##    residual=d/d.sum()
##    第一遍：块和
##    tile_sum = 小buffer[numTiles]
##    for j in 0..numTiles-1:
##    DataCopyPad(tile, residual[j*tileLen : (j+1)*tileLen])   ##
##    tile_sum[j] = ReduceSum(tile)

##    中间：块和做 exclusive 前缀（标量循环）
##    prefix[0] =0
##    for j in 1..numTiles-1:
##        prefix[j] = prefix[j-1] + tile_sum[j-1]

##     第二遍：块内扫描 加前缀+jishu
##    count =0
##    for j in 0..numTiles-1:
##        DataCopyPad(tile, ...)      #再搬一次
##        cdf_tile= 块内inclusive_scan(tile)    #v1 用标量循环，先对后快
##        cdf_tile =Adds(cdf_tile,cdf_tile, prefix[j])
##        mask=CompareScalar(cdf_tile,u2,LT) # cdf<u2 的位置为1
##        count+=ReduceSum(mask)         # 数有几个
##    out=count
##    out_token=Select(token_sure, draft_token, out)
##    accepted=token_sure
##
##    

##  边界：q_d=0时我们通过掩码让它直接进行为1的路线，同时避免了0/0＝NaN的异常情况