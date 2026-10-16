# rejection_sample 构建说明

> 2026-10-16 实测。CANN 构建系统的坑都在这儿，重装环境后照着走。

## 一、构建命令

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd ~/workspace/llm-engine/cann/ops-nn
rm -rf build/tbe build/autogen build/binary        # 换算子前必做，见第四节
bash build.sh --pkg --soc=ascend910_93 --ops=add_example -j$(nproc)
```

- `--soc=ascend910_93`（910C；**不是** ascend910b）
- 产物：`build_out/cann-ops-nn-custom-linux.x86_64.run`
- 算子源码：`examples/rejection_sample/`

## 二、为什么是 `--ops=add_example`，不是 `--ops=rejection_sample`

`--ops=rejection_sample` **注定退出码 1**，原因在 ops-nn 构建系统，不在你的代码：

1. `cmake/func.cmake:388` 写死了白名单——只有 `ASCEND_OP_NAME` 恰好等于 `add_example` 或 `add_example_aicpu` 时才 `add_subdirectory(examples)`。所以 `--ops=rejection_sample` 时 `examples/` **整个目录根本不进构建**。
2. build.sh 第一遍 configure 用 `PREPROCESS_ONLY=ON` 且不带 `ASCEND_OP_NAME`，这一遍也不加 `examples/`；于是 `build/tmp/ops_config.txt`（383 行）里既没有 add_example 也没有 rejection_sample。
3. 第二遍 configure 之前，`scripts/util/dependency_parser.py` 在这张表里查不到 `rejection_sample`，直接抛 `RuntimeError: rejection_sample is not exists`；build.sh 第 11 行是 `set -e`，异常立刻终止整个脚本。

**`--ops=add_example` 会把 `examples/` 下所有例子一并纳入构建**（日志里会写 `Ops for this compilation contains: add_example;add_example_aicpu;rejection_sample`），这是这条产线唯一可用的入口。

> 不要为了"好看"去改 `cmake/func.cmake`。那是依赖文件，重新 clone 就丢，别人复现不了。

## 三、退出码 0 ≠ kernel 编过（最容易误判的一条）

这个构建系统**会吞掉 kernel 编译失败**：opc 明确 `failed with ErrorCode[2]` 时，build.sh 只打印一句 `will ignore`，**仍然退出 0**。

所以每次构建后必须额外确认两件事：

1. 日志里有 `do opc cmd is asc_opc ... rejection_sample`
2. `build/binary/ascend910_93/bin/ascend910_93/rejection_sample/*.o` 存在

只看退出码，会拿到"假成功"。

## 四、三个已知坑

| 坑 | 现象 | 解法 |
|---|---|---|
| 换 `--ops` 集合后生成物陈旧 | 日志 `build_binary_opc_gen_task failed` + `do not get the opc info ... will ignore`，kernel 被静默跳过 | 构建前 `rm -rf build/tbe build/autogen build/binary`（都是 .gitignore 的产物） |
| SOC 未声明 | 日志 `On [ascend910_93], [rejection_sample] not supported`，kernel 被跳过 | `op_host/rejection_sample_def.cpp` 里要有 `AddConfig("ascend910_93", ...)`，且 `op_host/config/ascend910_93/` 下有 binary json |
| 缺 `cstdint` | 编译报头文件缺失 | `CXXFLAGS="-include cstdint" bash build.sh ...` |

## 五、算子接口（已固定在 def.cpp）

| | 名字 | dtype | shape |
|---|---|---|---|
| in 0 | `p` | float32 | [batch, vocab] |
| in 1 | `q` | float32 | [batch, vocab] |
| in 2 | `u1` | float32 | [batch] |
| in 3 | `u2` | float32 | [batch] |
| in 4 | `draft_token` | int32 | [batch] |
| out 0 | `out_token` | int32 | [batch] |
| out 1 | `accepted` | uint8 | [batch] |

## 六、WSL 环境备忘

- WSL2 Ubuntu 26.04，用户 `dev1`，CANN 在 `/usr/local/Ascend`
- 工作路径全 ASCII：`/home/dev1/workspace/llm-engine/cann/ops-nn/examples/rejection_sample`
- Windows 侧仓库 `C:\Users\耶\workspace\llm-engine` 是**另一个 clone**，用 git 同步；CANN 的东西只在 WSL 侧
- WSL 会报 `检测到 localhost 代理配置...` 警告，可忽略
