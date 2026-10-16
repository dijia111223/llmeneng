# 笔记

## KV cache预分配maxlen是为了节省重新复制历史kv的时间，和每一步用cat属于是两个不一样的方向一个节省时间一个更省空间，预分配确实省了时间但是太浪费空间
## 10.12，应用GQA机制，确实省下来一半显存，但是没省计算，也没减少maxlen所浪费的空间
## 10.16，开始写拒绝采样的 ascendc 算子（rejection_sample），tiling 数值链路通了、kernel 的 Init 编译通过，代码在 self_kernel 仓库
