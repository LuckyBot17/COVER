# COVER Closed-Set Relation Repair

这个目录实现 COVER 的第一版真正自动修复链路：对已知 GQA 主体/客体实体对穷举 15 个 benchmark 关系，使用 COVER negative-control evidence 验证候选，只在风险校准后得到唯一可信关系时输出，否则 abstain。

## 研究边界

本实验使用 GQA scene graph 的实体 ID、真值框和封闭关系词表，回答的是：

> 在候选召回固定为 100% 的条件下，COVER 能否从封闭词表中选择真实关系，并安全地把错误关系改成真实关系？

即使 full experiment 成功，也不能据此声称 COVER 已普遍缓解 LVLM 的自由生成幻觉。真实模型输出、自动实体定位和外部 benchmark 仍需另行验证。

## 与旧 GQA 实验的区别

- 不读取旧 `claims.csv`。
- 用 `(image_id, subject_id, object_id)` 标识实体对，不以对象名称去重。
- 每个实体对穷举全部 15 个关系；多标签真值被完整保留。
- 旧 `gqa_prepare.py` 的第二次对象循环错误地使用了前一循环遗留的变量 `o`。本实现始终从当前 `subject_id` 对应的对象读取 relations，并逐边回查 scene graph。
- 旧 REPAIR 只比较原关系和语义等价的逆关系；本实现搜索真正不同的替代关系。
- 风险阈值在“实体对内最可信的错误候选”上校准，控制整组候选中任一错误关系混入的风险。

## 环境

数据与模型默认使用现有服务器路径：

```text
/root/autodl-tmp/data/GQA/val_sceneGraphs.json
/root/autodl-tmp/data/GQA/images/
/root/autodl-tmp/llava-1.5-7b/master/
```

纯逻辑与分析依赖 Python、NumPy、Pillow；GPU 打分还依赖 PyTorch、Transformers 和可读取的 LLaVA-1.5-7B 权重。代码保持现有实验风格，不引入额外框架。

建议把整个目录同步为：

```text
/root/autodl-tmp/cover_closedset_repair/
```

所有命令均在该目录执行。不要把旧 `cover_min_exp/claims.csv` 或 `scored_claims_blur.csv` 复制为本实验输入。

## 运行顺序

### 1. 逻辑回归测试

```bash
python3 -m unittest -v test_closedset_repair.py
```

测试覆盖旧数据抽取错误、同名对象 ID、多标签真值、图像隔离、15 类候选、pair-level 阈值、singleton 决策、断点续跑和共享嵌套字典回归。

### 2. 数据 census 与预注册 split

```bash
python3 prepare_gqa_closedset.py \
  --sg /root/autodl-tmp/data/GQA/val_sceneGraphs.json \
  --img-root /root/autodl-tmp/data/GQA/images \
  --outdir /root/autodl-tmp/cover_closedset_repair \
  --seed 42
```

先检查 `census.json`：它给出合格实体对、图像、关系分布、多标签比例、各 split 规模和总候选推理量。四个 split 按图像完全隔离：15% standardization、15% risk calibration、10% pilot evaluation、60% final evaluation。

### 3. Smoke：6 个实体对

```bash
python3 score_gqa_closedset.py \
  --stage smoke \
  --outdir /root/autodl-tmp/cover_closedset_repair \
  --model-path /root/autodl-tmp/llava-1.5-7b/master \
  --mask blur

python3 analyze_gqa_closedset.py \
  --stage smoke \
  --outdir /root/autodl-tmp/cover_closedset_repair
```

`smoke_report.json` 必须为 `PASS`。Smoke 只验证数据/打分契约、15 类完整性、真假标签、有限数值和无重复 checkpoint，不对科研效果作结论。

为验证断点续跑，可再次执行同一条 scoring 命令；日志必须显示 `pending=0`，且 `scored_candidates.csv` 行数不增加。

### 4. Pilot：150 个实体对

```bash
python3 score_gqa_closedset.py \
  --stage pilot \
  --outdir /root/autodl-tmp/cover_closedset_repair \
  --model-path /root/autodl-tmp/llava-1.5-7b/master \
  --mask blur

python3 analyze_gqa_closedset.py \
  --stage pilot \
  --outdir /root/autodl-tmp/cover_closedset_repair \
  --n-bootstrap 3000 \
  --seed 42
```

Pilot analysis 按预注册规则生成 `pilot_gate.json`。以下条件必须同时通过：

- 150 对完整打分，三个开发 split 各 50 对；
- pilot evaluation 覆盖全部 15 类；
- `AUROC(mu)>0.5`；
- COVER Top-1 不低于 raw Yes/No baseline；
- `alpha=0.10` 时至少 10 个 singleton 实体对；
- REPAIR precision 不低于 80%；
- semantic corruption rate 不高于 10%；
- 成功修复数大于新增错误数。

失败时脚本返回状态码 2，并保留所有诊断结果。不得通过事后改阈值把 FAIL 改成 PASS。

### 5. Full：全部合格 GQA validation 实体对

只有 `pilot_gate.json` 的 `status` 为 `PASS` 时，full scoring 和 analysis 才会运行；代码会主动阻止绕过。

```bash
python3 score_gqa_closedset.py \
  --stage full \
  --outdir /root/autodl-tmp/cover_closedset_repair \
  --model-path /root/autodl-tmp/llava-1.5-7b/master \
  --mask blur

python3 analyze_gqa_closedset.py \
  --stage full \
  --outdir /root/autodl-tmp/cover_closedset_repair \
  --n-bootstrap 3000 \
  --seed 42
```

Full scoring 继续使用同一个 `scored_candidates.csv`，自动复用 smoke/pilot 已完成候选。
共享 checkpoint 不能混用不同 mask；如果已有记录为 `blur`，改用 `gray` 或 `mean` 会明确报错，而不会静默复用不相容分数。

## 输出说明

| 文件 | 含义 |
|---|---|
| `all_pairs.csv` | 全部符合预声明过滤条件的 object-ID 实体对 |
| `pilot_pairs.csv` | 三个开发 split 各 50 对的 relation-balanced pilot |
| `smoke_pairs.csv` | pilot 内的 6 对工程 smoke |
| `census.json` | 数据规模、关系覆盖、多标签与预计推理量 |
| `scored_candidates.csv` | 共享、逐候选、可恢复的模型打分 checkpoint |
| `smoke_report.json` | 工程完整性闸门 |
| `pilot_gate.json` | pilot 科学闸门及失败原因 |
| `repair_decisions_pilot.csv` | pilot 主阈值下逐实体对候选集合与输出 |
| `analysis_full.json` | 全量 ranking、repair、risk、分层结果和置信区间 |
| `repair_decisions_full.csv` | 全量主阈值下逐实体对决策 |

主决策分数为 `A=-mu`，越低越可信。`J` 会被记录但不进入决策。主阈值为 `alpha=0.10`，`0.05/0.20` 只作为预注册敏感性分析。

## 结果回传

服务器运行后，至少带回以下文件：

```text
census.json
scored_candidates.csv
smoke_report.json
pilot_gate.json
repair_decisions_pilot.csv
analysis_full.json                 # 仅 pilot PASS 后存在
repair_decisions_full.csv          # 仅 pilot PASS 后存在
完整 stdout/stderr 日志
```

不要只带回筛选后的成功案例；失败行、abstain 和完整日志同样是风险分析所必需的证据。
