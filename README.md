# llm-engine

从零构建的 LLM 推理引擎。目标：朴素 KV cache → PagedAttention → Triton kernel → 投机解码。

规则：代码全部自写；参考开源实现的每个设计决策，在 `NOTES.md` 里标注来源和取舍。

## 进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0 | KV Cache 抽象（预分配 + 按下标写入 + pos 管理） | 进行中 |
| 1 | 注意力接入 cache，与全量计算对拍 | 未开始 |
| 2 | PagedAttention（BlockTable、按需分配、空闲回收） | 未开始 |
| 3 | Triton 分页 attention kernel + roofline 图 | 未开始 |
| 4 | 投机解码（草稿模型 + 接受率感知准入） | 未开始 |
