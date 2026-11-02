# 基于 Kronos 的监督金融异常检测

> **2026-11-03 方法完善：** 已加入历史上下文、点级多尺度头、整段注意力汇聚、稳定微调、误报预算与封存测试流程。新增参数、对照和负结果见 [方法完善记录](supervised_refinement_20261103.md)。本页保留基础数据格式和入口说明，原版本的实现／执行证据另见日期记录。

## 方法与上游的关系

本扩展将 Kronos 从下一步行情 token 预测改为带标签的二分类。沿用上游的 RoPE、RMSNorm、门控前馈网络和因果 Transformer；原模型源码与固定提交见 [UPSTREAM.md](../UPSTREAM.md)。新增代码位于 `supervised/`，不修改原模型或把预测误差当作监督标签。

输入为 `X[B,T,F]`，支持两条路径：

1. **真实 OHLCVA**：六列按 `open,high,low,close,volume,amount` 排列，经过冻结的 KronosTokenizer 分为粗、细两级 token，再经原分层 embedding 接入 Kronos。检查价格高低关系、正价格、非负成交量／金额及 tokenizer 码本匹配。当前接口要求六列真实数值；不伪造缺失行情字段。离散 tokenizer 不参与反向传播，默认微调 Transformer 与分类头。
2. **一般金融特征**：`Linear(F,d_model) → GELU → LayerNorm` 接入原 Kronos 因果 Transformer。绕过行情 tokenizer、token embedding、日历 embedding 和预测输出头。适用于 Sapienza、ULB、单变量派生序列及用户已准备的交易特征。主干可加载 Kronos-mini 的真实预训练权重，但这是有分布变化的**连续特征适配实验**，并非保留完整行情 token 预训练流程；需要与从头训练及普通分类器比较。

设 Transformer 输出为 `h_t`。逐点任务可直接分类 h_t，或默认将 h_t 与多个过去窗口的偏离量拼接后分类；整段任务可取最后有效位置，或默认将其与有效位置的 attention 汇聚拼接。异常分数为 `sigmoid(z)`，它是分类分数，尚未经过概率校准。整段标签不扩展成每个时间点的异常真值。

训练目标是已知标签上的类别加权 BCE：

`L = mean[-w_pos*y*log(sigmoid(z)) - (1-y)*log(1-sigmoid(z))]`，

其中 `w_pos = N_train_normal / N_train_abnormal`。`y=-1`、右侧 padding 均不进入损失或带真值评价。不进行按测试标签重采样，也不使用原来的自回归 token 预测损失。`--freeze-backbone` 冻结原主干，只训练适配器／分类头；适配器仍可通过冻结主干获得梯度。`--scratch` 显式构造小规模随机初始化的 Kronos 架构，作为可复现的消融／运行检查，不能称为预训练微调。

## 因果性与评价协议

- 原 tokenizer 和主干 self-attention 都是因果的；当前点可使用本点输入和过去输入，不读取未来点。OHLC 模式是在当前 K 线已完成后检测，不能据此宣称提前预警。
- 采用仅训练集拟合的缺失值均值填补、逐特征标准化及 `[-5,5]` 截断。没有全窗口归一化，因此点级早期输出不会读取窗口后半段统计。此预处理与上游预测器的窗口归一化不同，保留因果性同时存在预训练分布偏移，需要后续消融。
- 同一 split 内样本右侧 padding；因果注意力保证 padding 不影响前面的有效位置。整段任务只取最后有效位置。禁止自动截断，否则可能丢掉整段异常事件。上下文过长会报错；普通模型最多支持 512，本接口仅允许名称含 Kronos-mini 的预训练模型使用更长上下文（最长按用户指定值限制，应不超过上游 2048）。
- 原始输入不包含标签、事件 ID、样本 ID。检查跨 split 样本 ID；提供 group_id 时进一步检查分组隔离。没有 group_id 的序列还检查完整输入内容是否跨 split 重复。已知独立组的背景特征可能自然相同，不单凭特征相等删除这些观察。
- 每轮只在验证集计算 Average Precision，保存最优 epoch；随后只用该模型的验证分数最大化正类 F1 选择阈值，平局时取更高阈值；也支持 F-beta、FPR 预算和最低 precision 策略。**模型和阈值固定后才计算测试分数**，调参建议使用 --defer-test 封存测试集。不要重复用测试结果挑超参数。
- 输出正类 Precision、Recall、F1、混淆矩阵、误报率、Average Precision 和 ROC-AUC。AP 使用 sklearn 的阶梯积分定义，不把它冒称为梯形积分 PR-AUC。单类测试集的 AP／ROC-AUC 记为 null。不会使用 point adjustment 或容忍窗口修改标签。

内置逐点数据可用 --history-length 增加过去上下文；target_mask 保证每个原始点只负责一次监督／评价。未启用该参数时仍在块边界重置历史；没有跨请求缓存。当前尚未实现跨块持续在线状态或实时服务。时间／币种缺失、标签粒度及原始数据的其他限制见各数据卡。

## 安装与运行

Python 3.10+，可使用 CPU、CUDA 或支持 PyTorch 的 MPS。仓库不包含预训练权重；指定 Hugging Face ID 时会下载免费公开权重，也可给本地目录并加 `--local-only`。本次没有租赁云资源。

```bash
python -m pip install -r requirements-supervised.txt

# 1. 准备 Sapienza 15S：沿用原有按币种隔离的 60/20/20 研究划分
python -m supervised prepare --dataset sapienza --output prepared_runs/sapienza

# 2. 加载预训练 Kronos-mini，微调主干、输入适配器、点级分类头
python -m supervised train --data-dir prepared_runs/sapienza \
  --backbone NeoQuasar/Kronos-mini --epochs 10 --batch-size 16 \
  --device cpu --output runs/sapienza-mini

# 3. 使用固定阈值、完整本地 checkpoint 推断，无需再次联网加载原权重
python -m supervised predict --checkpoint runs/sapienza-mini/best.pt \
  --data prepared_runs/sapienza/test.npz --output runs/sapienza-mini/reloaded.csv
```

运行目录必须不存在，避免覆盖已有模型。输出：`best.pt`、`metrics.json`、`validation_predictions.csv`、`test_predictions.csv`。checkpoint 包含模型构造配置、全部模型参数、训练集预处理参数、特征顺序、上下文限制和验证阈值；新版本另含来源身份核验信息；仅加载自己信任的 checkpoint。

其他已接通的数据：

```bash
python -m supervised prepare --dataset ulb --output prepared_runs/ulb
python -m supervised prepare --dataset sd-crypto --output prepared_runs/sd-crypto
python -m supervised prepare --dataset sd-stock --output prepared_runs/sd-stock
python -m supervised prepare --dataset tstr --output prepared_runs/tstr

# TSTR 保留整条 1000 点序列及整段标签，不能默认截断成 512 点
python -m supervised train --data-dir prepared_runs/tstr \
  --backbone NeoQuasar/Kronos-mini --max-context 2048 \
  --epochs 10 --batch-size 4 --output runs/tstr-mini

# 不下载权重的架构检查；不是预训练实验
python -m supervised train --data-dir prepared_runs/sd-crypto \
  --scratch --epochs 2 --output runs/sd-crypto-scratch
```

Sapienza 保留全部 584,104 行和 317 个正类开始块，不补齐无交易时间。按事件排序观测块，额外提供 `log1p_gap_seconds`，显式表示距上个观测块的秒数。每个块最多 128 个观测点，不跨事件／币种；因此点间隔可以不等于 15 秒，不能解释成连续无缺口 K 线。

ULB 是按 Time 严格先后切分的约 70/15/15 全局交易流；同一时间戳的交易不跨边界，不使用 Class 作输入。实际训练／验证／测试为 199,364／42,719／42,724 笔，正类 384／56／52。没有公开用户 ID，不能保证跨用户隔离，也不能把全局交易流当作个人银行卡历史。保存的 preparation.json 记录本次实际划分统计。

TSTR、SD-FMM 使用原有仓库研究划分，不声称复现论文官方训练／测试设定。SD-FMM 原始文件名含 test；重新用其研究划分微调后不再是原论文的零样本独立测试。

## 自有交易数据或真实 OHLCVA

目录中显式放置 `train.npz`、`validation.npz`、`test.npz`，由用户根据时间、账户和研究目标预先划分。格式如下；无需 pickle：

| 字段 | 形状／含义 |
| --- | --- |
| X | float `[N,T,F]`，允许 NaN；拒绝 infinity |
| y | 点级 `[N,T]` 或整段 `[N]`；0 正常、1 异常、-1 未知 |
| lengths | 可选 int `[N]`，有效长度；默认整个 T 有效 |
| sample_id | 字符串 `[N]`，跨三个 split 不相交 |
| feature_names | 字符串 `[F]`，三个 split 的名称和顺序严格一致 |
| group_id | 可选字符串 `[N]`；如提供则三个 split 的组不相交 |
| target_mask | 可选 boolean `[N,T]`，仅点级任务；历史／padding 不作为本块目标 |
| observation_id | 可选字符串 `[N,T]`，仅点级任务；原始点身份，所有 split 不交叉，目标不重复 |

推断时 y 可以全部为 -1。训练和验证均须含已知的两个类别。类别特征必须只在训练集拟合编码器，再以相同列顺序转换验证／测试；不要使用洗钱类型、事后调查结果或未来交易聚合字段作为输入。

PaySim、TabFormer、IBM AML、Elliptic／IEEE-CIS 的原始读取工具仍保留，但本次没有为它们自动建立模型输入和实验划分。它们可以在明确时序／账户协议后转换到上述 NPZ；本实现是序列分类器，不是图神经网络，不能据此宣称使用了 Elliptic 的图结构。

真实 OHLCVA 的调用：

```bash
python -m supervised train --data-dir prepared_runs/my-labelled-ohlc \
  --input-kind ohlcv --backbone NeoQuasar/Kronos-mini \
  --tokenizer NeoQuasar/Kronos-Tokenizer-2k \
  --epochs 10 --output runs/ohlcv-supervised
```

## 已完成的验证

执行记录见 [supervised_validation.md](supervised_validation.md)。单元与集成测试覆盖：未来扰动不改变过去点输出、padding 不影响有效输出、冻结主干时适配器梯度、未知标签屏蔽、训练集预处理、跨 split 隔离、类别／特征检查、真实 tokenizer 路径、checkpoint 重载，以及**修改测试标签不能改变所选 epoch、阈值或模型权重**。本次结果不代表完整方法有效性结论。
