# Kronos 监督异常检测方法完善（2026-11-03）

本轮围绕历史上下文、异常表征、稳定微调和实际误报约束完善方法。保留原始 Kronos 模型源码、既有数据、标签及研究划分。新增能力已完成运行验证，50 项测试通过；旧 checkpoint 的 10,880 个分数逐项精确一致。不能据此宣称优于强基线。

## 1. 有历史上下文，每个点只负责一次预测

原接口在每个不重叠块边界重置历史。现在 Sapienza／ULB 可指定 `--history-length H`，每个块包含前 H 个已发生观测和 L 个新目标。`target_mask` 区分上下文与本块目标；上下文不进入损失、评价或预测 CSV。每轮每个原始目标监督一次，每次评价导出一次；没有把同一交易重复算成多个检测命中。

`observation_id` 保留原始点身份，允许同一个点在本 split 内重用为历史，但不允许重复成为目标。检查跨 split 的所有观测，包括仅充当历史的点。严格隔离协议不把训练集尾部接入验证／测试历史；这比真实部署可用历史更保守。按独立事件／币种切块，不跨事件拼接。逐点 train-only 标准化只计 target_mask 对应的训练观测，避免重复历史改变均值、方差与类别权重。无已知监督目标的训练块跳过且不占用优化器 schedule。

Sapienza 的缺失时间仍然不补齐，保留 `log1p_gap_seconds`。多尺度窗口中的 4／16／64 表示**观测点数**，不是秒数。首块可用历史较短；历史不无限累积，本实现仍是有界上下文窗口，尚无跨请求 KV cache。

## 2. 多尺度点级分类头与整段注意力汇聚

保留 Kronos 因果主干产生的 h_t。点级分类头可选择：

- `mlp`：直接分类 h_t，作为原有头的对照。
- `multiscale`：构造 `u_t = concat(h_t, h_t - mean_4(h), h_t - mean_16(h), h_t - mean_64(h))`，再用 MLP 分类。

每个 mean_w 只用 `[max(0,t-w+1),t]`，包含当前观察、不读未来；通过前缀和计算。偏离近期背景的变化是异常候选信号，不是异常真值。默认尺度可通过 `--scales` 调整；其选择属于验证集超参数，不能用测试分数选择。

整段任务可选择 `last` 或 `attention`。后者将最后有效状态与所有有效状态的可训练 attention 加权和拼接，避免只依赖最后一点承载整段事件。padding 权重为零。整段分类读取完整已结束序列；不能把它描述为逐点在线提前预警。

真实 OHLCVA 仍使用冻结的原始粗／细量化 tokenizer 与分层 embedding；一般交易特征仍走连续输入适配器。没有将交易表伪装成行情。

## 3. 微调与不平衡学习

- 预训练主干默认 LR 为新头／适配器的 0.1 倍；scratch 默认同 LR。可用 `--backbone-lr-ratio` 和 `--layer-decay` 明确控制。bias／一维归一化参数不做 weight decay。冻结模式只优化真正可训练参数。
- 新训练 CLI 默认 10% warmup＋cosine decay，按实际可监督训练批次数规划；验证 AP 早停，默认 patience=3。`--patience 0 --schedule constant --warmup-fraction 0` 可禁用相应策略。
- 默认保留类别加权 BCE。正类权重只从训练目标计算，支持 `auto`、`sqrt`、`none`、固定数值及上限；可选 focal loss：`L_focal=(1-p_t)^gamma * weighted_BCE`。gamma=0 等价于 BCE。未知标签、history、padding 均屏蔽。focal 与强正类权重不保证更好，须做同条件对照。
- `--scratch-config` 允许从原主干 config.json 构造参数规模一致的随机对照。默认 tiny scratch 仍仅适合运行检查，不应与大型预训练模型比较后将差异归因于预训练。
- 模型初始化与数据加载使用可记录的种子；数据加载 generator 独立于分类头大小。checkpoint 原子写入。新运行记录初始主干 tensor SHA-256、源代码哈希、优化器分组与实际训练轮数。CPU 验证通过；未验证 GPU 跨平台数值确定性。

## 4. 误报预算与独立测试

阈值始终只在验证集确定，预测／最终评价使用 checkpoint 中的固定阈值。

| 策略 | 选取规则 |
| --- | --- |
| `f1` | 最大化验证正类 F1；平局取更高阈值 |
| `fbeta` | 最大化给定 beta 的 F-beta |
| `fpr` | 验证 FPR 不超过 max-fpr 时最大化召回；再按 precision／高阈值打破平局 |
| `precision` | 非空告警集达到 min-precision 时最大化召回；不存在则明确标记不可达 |

相同分数的样本整组处理，与实际 `score >= threshold` 一致。允许 no-alert 候选；结果的 status 为 `active`、`no_alerts` 或 `precision_unattainable`。**没有告警不是检测成功**，必须同时报告召回、正类数、告警数和误报分母。验证集正常点有限时，FPR 的经验分辨率约为 1/N_normal；预算不构成未来数据上的统计保证。

使用 `--defer-test` 时训练程序完全不打开 test.npz。调参阶段只产生验证结果；模型与阈值确定后再调用 `evaluate`。新 checkpoint 保存训练／验证来源身份，evaluate 拒绝与其重叠的 sample_id／group_id／observation_id。`predict` 允许读取曾见输入以做诊断，不声称独立测试。旧 checkpoint 仍可推断；其 evaluate 明确记录缺少身份核验。

分类分数尚未概率校准，尤其 weighted BCE／focal 下不能直接解释为真实风险概率。新增 Brier 与每千笔误报告警数仅作诊断，不替代正类 AP、Precision、Recall、F1。

## 5. 实际对照与限制

完整记录：[协议、全部结果及负结果](../experiments/refinement_20261103/README.md)。SD-FMM crypto，3 个预声明种子 13／42／2026，每个配置 2 epochs；相同 Kronos-mini、数据、批大小、BCE 权重与优化器策略，只改变点级分类头。所有训练／验证完成、按平均验证 AP 做选择后，才打开测试集。MLP 对照也使用新版训练器，不能与前一次单次运行混作同条件比较。

| 同条件分类头 | 验证 AP 均值 | 测试 AP 均值 ± 样本标准差 | 测试正类 F1 均值 ± 样本标准差 | 测试 Recall 均值 |
| --- | ---: | ---: | ---: | ---: |
| MLP | 0.1770 | 0.1425 ± 0.1024 | 0.2113 ± 0.1693 | 0.6225 |
| 多尺度 | 0.3022 | 0.2503 ± 0.0898 | 0.3536 ± 0.0503 | 0.4951 |

多尺度头在这次短预算划分上改善 AP／F1，但召回下降；这些标准差反映训练种子变化，不是泛化置信区间。固定因果原始特征的 HistGradientBoosting 单次基线取得 AP=0.6562、F1=0.6015，高于两种神经配置。不能据此声称已经证明 Kronos 的预训练优势或取得最优方法。

另有 ULB 的 history＋focal＋sqrt 权重两轮短训诊断：验证 AP=0.0153、测试 AP=0.0136；0.1% 验证 FPR 预算选择 no-alert，最终零召回／F1。这是失败配置，已保留，没有依据测试分数反复调整它。它同时改变多个因素，不能隔离推断 focal 本身的优劣；BCE 保持默认。

Sapienza／ULB 的原始目标数、标签数完整保留。SD-FMM 是发布者已处理的研究数组，原始处理是否因果无法仅凭这些数组核验；模型本身的因果测试不证明整个数据来源满足实时条件。仍需更多市场、较长训练、规模一致 scratch／冻结／微调、强基线及固定协议的验证。

## 6. 推荐命令

```bash
python -m pip install -r requirements-supervised.txt
python -m supervised prepare --dataset sapienza --history-length 32 --output prepared_runs/sapienza-v2

# 调参／训练期间封存测试集。默认多尺度头、BCE、预训练主干小 LR、warmup/cosine。
python -m supervised train --data-dir prepared_runs/sapienza-v2 \
  --backbone NeoQuasar/Kronos-mini --epochs 20 --patience 3 \
  --threshold-policy fpr --max-fpr 0.001 --defer-test --output runs/sapienza-v2

# 仅在协议固定后做最终评价；同时产生 CSV 和 CSV.metrics.json。
python -m supervised evaluate --checkpoint runs/sapienza-v2/best.pt \
  --data prepared_runs/sapienza-v2/test.npz --output runs/sapienza-v2/test.csv

# 预声明三种子的头部对照；程序先完成全部验证，再打开测试。
python scripts/compare_supervised_heads.py --data-dir prepared_runs/sd-crypto \
  --backbone NeoQuasar/Kronos-mini --output runs/head-comparison --epochs 2

# 相同划分上的固定因果特征树模型；不是在测试上选择超参数。
python -m supervised.baseline --data-dir prepared_runs/sd-crypto --output runs/boosting-control
```

NPZ 在原字段基础上可增加 boolean `target_mask[N,T]` 和字符串 `observation_id[N,T]`，仅用于点级任务。padding 不得为目标；每个样本至少有一个目标；有效观测身份非空；同一观测只能负责一次预测。没有这两个字段的原有输入仍兼容。上下文总长度 H+L 必须不超过 max-context，不自动截断。
