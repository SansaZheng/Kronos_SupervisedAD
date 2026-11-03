# Kronos SupervisedAD：监督金融异常检测方法详解

本文以当前 `supervised/` 实现为准，更新于 2026-11-03。核心方法是：**把金融观测转换为连续特征序列，使用预训练 Kronos 因果 Transformer 提取时序表示，再利用人工／数据集提供的异常标签训练分类器；改进版本增加直接读取标准化输入的残差分类分支。** 输出是异常分类分数，不是价格预测，也不是预测误差。

最新实验使用真实 `NeoQuasar/Kronos-mini` 权重。`hybrid` 是可选改进头，CLI 仍默认 `multiscale`；推荐通过验证集选择，而非假定所有数据都应改用新头。本文分别说明逐点与整段任务，不混用其标签或评价单位。

## 1. 任务定义与标签

一批输入为 $X\in\mathbb{R}^{B\times T\times F}$：$B$ 为批大小，$T$ 为序列长度，$F$ 为特征数。样本 $i$ 有有效长度 $L_i\leq T$，后面是右侧 padding。

- **逐点任务**：$y_{i,t}\in\{-1,0,1\}$，每笔交易／每个观测点输出一个分数。0 正常、1 异常、−1 未知。
- **整段任务**：$y_i\in\{-1,0,1\}$，整条序列输出一个分数。TSTR 使用这一形式；整段异常标签不会复制为每个点的异常真值。

本方法依赖给定标签。它不自动发现欺诈真值，不用模型预测伪造监督标签，也不再训练 Kronos 原来的下一 token 预测目标。

## 2. 数据划分、编码与归一化

### 2.1 先划分，再拟合预处理

交易流尽可能先按时间划分训练、验证、测试；有明确独立实体的数据可按实体分组。具体协议由数据集决定，不能把时间隔离解释成账户隔离。

类别编码器仅从训练期拟合，之后固定特征列及顺序。ID、目标标签、洗钱类型注释、事后调查结果等不能作为输入。当前大规模交易数据采用训练期类别词表、每字段最多 16 类的编码；数值特征可按数据适用性加入金额对数、时间周期编码、交易前余额及观测间隔。不同数据的特征定义不强行统一。

对每个特征 $j$，仅使用训练集有效目标行求均值 $\mu_j$。先以该均值填补缺失，再求标准差 $\sigma_j$，转换为

$$x_{i,t,j}=\operatorname{clip}\left(\frac{\operatorname{impute}(X_{i,t,j},\mu_j)-\mu_j}{\max(\sigma_j,10^{-6})},-5,5\right).$$

验证和测试只应用已有参数，不重新拟合。训练期未知标签行可参与特征统计，但不参与监督损失；复用的历史行不重复计入统计。全缺失训练特征和无限值会报错。

这里没有按整条输入窗口重新归一化，因此某个早期点不会读取窗口后半段的均值／方差。截断能限制极端值，但也会丢失超出范围的幅度信息。

### 2.2 历史上下文与目标掩码

本轮适用的交易流使用每块 **128 个目标点＋最多 32 个过去点**，最长 160 点。相邻块可以复用历史，但 `target_mask` 将这些历史点排除出本块监督和导出目标：**同一原始观测只负责一次监督／评价**。首块可能没有完整历史；不从其他 split 借用历史。

这是有限窗口推断，块边界会重置上下文；当前没有跨块持续缓存。32 是过去观测数，不必然是 32 秒，也不代表某个账户的 32 笔历史。全局交易流可能混合多个账户。

## 3. 输入如何进入 Kronos

### 3.1 当前交易实验：连续特征适配路径

标准化输入经可训练适配器映射到模型宽度：

$$e_t=\operatorname{LayerNorm}(\operatorname{GELU}(W_a x_t+b_a)),\qquad e_t\in\mathbb{R}^{256}.$$

随后调用原 Kronos 的 token dropout、因果 Transformer 层与最终归一化：

$$h_{1:T}=\operatorname{KronosTransformer}_{\theta}(e_{1:T}).$$

Kronos-mini 配置为 4 层、模型宽度 256、4 个注意力头、前馈宽度 512；继承上游 RoPE、RMSNorm 和门控前馈模块。因果注意力意味着 $h_t$ 只依赖 $x_{1:t}$。源码来源与固定上游提交见 [UPSTREAM.md](../UPSTREAM.md)。

**实际复用的是预训练 Transformer 与最终归一化参数。** 该路径绕过原行情 tokenizer、离散 token embedding、日历 embedding、依赖层及行情预测输出头；不用的原模块冻结。默认微调主干、适配器和分类头。`--freeze-backbone` 可冻结主干，此时仍能通过主干把梯度传回适配器。

因此，这是一种有输入分布变化的连续特征迁移方法。不能把它描述为完整保留原 Kronos 行情 token 训练流程，也不能仅凭本轮提升认定预训练必然有效。

### 3.2 可选真实 OHLCVA 路径

对真实 `open, high, low, close, volume, amount` 六列，可用 `--input-kind ohlcv`，经匹配的冻结 KronosTokenizer 产生粗／细 token，再经原 embedding 进入同一主干。接口核验真实列及价格、成交量关系，tokenizer 在 `no_grad` 下运行，不进行离散量化器的反向训练。

该路径不允许伪造交易表缺少的行情字段。`hybrid` 当前只支持连续特征路径。已有 OHLCVA 检查主要证明接口可运行，不能作为真实金融效果结论。K 线检测以当前 K 线已完成为前提。

## 4. 逐点分类器：从基础头到双分支改进

### 4.1 基础 MLP

最简单的 `mlp` 头直接计算

$$z_t=\operatorname{MLP}_{256\to128\to1}(\operatorname{Dropout}(h_t)),\qquad p_t=\sigma(z_t).$$

隐藏层使用 GELU。$p_t$ 是排序与告警分数；加权损失训练后未经校准，不应解读为真实欺诈概率。

### 4.2 多尺度时序头 `multiscale`

对窗口 $s\in\{4,16,64\}$ 定义包含当前点的过去均值：

$$\bar h_t^{(s)}=\frac{1}{\min(s,t)}\sum_{u=\max(1,t-s+1)}^t h_u.$$

构造

$$q_t=[h_t;h_t-\bar h_t^{(4)};h_t-\bar h_t^{(16)};h_t-\bar h_t^{(64)}]\in\mathbb{R}^{1024},$$

$$z_t^{\mathrm{temporal}}=\operatorname{MLP}_{1024\to128\to1}(\operatorname{Dropout}(q_t)).$$

这里的均值始终只读本点与过去点；窗口不足时使用实际已有点数，不补未来。该结构同时提供当前状态与相对短、中、长窗口的偏离，目的是帮助分类器识别局部变化。它不是三次独立预测，也不是用偏离量直接定义异常真值。

### 4.3 最新双分支头 `hybrid`

仅靠主干表示可能弱化金额、余额或类别指示等对当前交易有用的输入差异。最新版本增加一条绕过适配器和主干的直接输入通路：

$$r_t=[x_t;x_t-\bar x_t^{(4)};x_t-\bar x_t^{(16)};x_t-\bar x_t^{(64)}]\in\mathbb{R}^{4F},$$

$$z_t^{\mathrm{raw}}=\operatorname{MLP}_{4F\to128\to64\to1}(r_t),$$

$$\boxed{z_t=z_t^{\mathrm{temporal}}+z_t^{\mathrm{raw}},\qquad p_t=\sigma(z_t).}$$

直接分支的层次是 Linear → GELU → Dropout → Linear → GELU → Linear。**两个分支相加的是 logit，不是概率；当前没有显式门控权重。** 两个分支共同由同一个监督损失端到端训练。

`raw` 指已经按训练统计标准化、填补和截断的 $x_t$，不是未经处理的原金额。它绕过主干的表示变换，但不能恢复预处理已丢失的信息。增加分支是结构动机，是否有效仍须对照实验。

## 5. 整段分类：最后状态与注意力汇聚

对于整段任务，取最后有效状态 $h_{L_i}$。默认还用一个可学习向量 $w$ 对所有有效位置打分：

$$\alpha_{i,t}=\frac{\exp(w^\top h_{i,t})}{\sum_{u=1}^{L_i}\exp(w^\top h_{i,u})},\quad
v_i=[h_{i,L_i};\sum_{t=1}^{L_i}\alpha_{i,t}h_{i,t}].$$

padding 权重为零。$v_i\in\mathbb{R}^{512}$ 经 512→128→1 分类器产生时序 logit。`--sequence-pooling last` 可仅使用最后状态。

整段 `hybrid` 的直接分支使用有效输入的 **最后值、均值、总体标准差、最大值、最小值**，共 $5F$ 维，经 5F→128→64→1 MLP 后与时序 logit 相加。它不使用逐点任务的多尺度输入向量。

整段汇聚读取整个有效片段，属于片段结束后的离线分类，不能宣称为提前定位其中异常时刻。TSTR 的 1000 点整段输入使用 `--max-context 2048`，不截成 512 点。

## 6. 监督目标与优化

设 $\Omega$ 为已知标签、有效长度内且属于本块目标的位置集合。损失为

$$\mathcal L=\frac{1}{|\Omega|}\sum_{k\in\Omega}\left[-w_+ y_k\log p_k-(1-y_k)\log(1-p_k)\right],\qquad
w_+=\frac{N_{\mathrm{train},0}}{N_{\mathrm{train},1}}.$$

权重只由训练集已知目标标签计算；未知标签、padding、复用历史均不计入。整段任务将 $k$ 理解为样本。全无已知标签的批次跳过。

可选 `sqrt`、不加权、指定常数及正类权重上限；设置上限时 $w_+=\min(N_0/N_1,c)$。也实现了 focal 变体，在上述逐项 BCE 上乘 $(1-p_{\mathrm{true}})^\gamma$，但此前 ULB 诊断未支持把它设为默认，因此本轮保持 BCE。

优化器为 AdamW。预训练主干默认学习率为适配器／分类头的 0.1 倍；权重衰减默认 0.01，bias 和一维参数不衰减；梯度范数上限 1。默认前 10% 计划更新步数 warmup，随后余弦衰减，可选层间学习率衰减。调度总步数按最大 epoch 计算，提前停止时未必衰减到零。

每个 epoch 计算验证 AP，保存最优权重；默认连续 3 次无改善提前停止，恢复最佳 epoch。训练和验证都须含已知的两个类别。

## 7. 模型选择、阈值与封存测试

流程严格区分三件事：

1. **选 epoch**：单次训练内部，按验证 AP 保留最佳 checkpoint。
2. **选候选配置**：多轮 runner 按验证 AP 选配置，平局比较验证 F1，再优先较早候选；参考配置也可胜出。
3. **选阈值**：对固定模型，仅用验证分数选择默认最大正类 F1 的阈值 $\tau$，平局取更高阈值；测试时固定 $\hat y=\mathbf1[p\geq\tau]$。

可替换为 F-beta、验证 FPR 上限、最低 precision 策略。相同分数的样本整体处理；包含完全不报警的候选阈值。验证集预算只在验证集上成立，不保证分布变化后的测试 FPR／precision。

多轮实验在候选训练期间封存 `test.npz`，所有训练使用 `--defer-test`。选型结果与哈希写盘后才解封并评价。额外种子确认不反向改写选择。已知 sample／group／observation 身份会跨 split 核验；未提供组信息时不能保证未知实体的隔离。

评价报告 AP、ROC-AUC、正类 Precision、Recall、F1、TP／FP／FN／TN、FPR、每千条已知标签样本的误报数及 Brier 分数。**AP 使用 sklearn 的 average precision 定义，不等同于梯形积分 PR-AUC。** 未知标签不进入指标；没有 point adjustment，不改写异常标签。Brier 的报告也不代表完成了概率校准。

## 8. 已运行的四组配置

| 候选 | 头 | 最多 epoch | 适配器／头学习率 | 主干倍率 | 正类权重上限 |
| --- | --- | ---: | ---: | ---: | ---: |
| R0 参考 | multiscale | 5 | 1e-4 | 0.1 | 无 |
| R1 增加直接分支 | hybrid | 5 | 1e-4 | 0.1 | 无 |
| R2 加大学习率与训练预算 | hybrid | 12 | 1e-3 | 0.1 | 无 |
| R3 调整微调强度与类别权重 | hybrid | 12 | 3e-4 | 0.3 | 100 |

公共设置：BCE、自动正类权重、dropout 0.1、warmup 0.1、余弦调度、patience 3、seed 42；batch 32，TSTR 为 4。R1 对 R0 在相同计划预算下比较新增分支；R2／R3 同时改变优化参数与最大训练预算，不能作为单因素结构消融。上限高于实际权重时不会生效，例如 IEEE-CIS 的约 27.43 未触及 100。

共 10 项 × 4 配置＝40 次主实验；ULB、PaySim、TabFormer 增加种子 13 和 2026 的参考／所选配置成对确认，共 12 次，合计 52 次完整训练。全部初始主干 tensor 哈希相同。新 runner 可先解析单一本地权重快照并锁定文件哈希；该缓存增强晚于已经完成的实验，不能冒称该新逻辑已参与旧实验。

## 9. 数据如何适配

| 数据 | 任务／输入 | 本轮关键约束 |
| --- | --- | --- |
| ULB | 逐笔交易、PCA 等公开数值特征 | 按时间约 70/15/15；没有用户 ID，使用全局流 |
| PaySim | 逐笔交易、金额／交易前余额／类型等 | 排除交易后余额、规则标记与账户 ID |
| IBM AML | 逐笔交易、金额／币种／支付方式／时间等 | 排除账户 ID 与洗钱模式注释 |
| TabFormer | 逐笔交易、金额／支付方式／地区／MCC 等 | 排除用户、卡、商户 ID 和 Errors 字段 |
| IEEE-CIS | 逐笔交易、97 个派生数值／类别列 | 仅本地受限数据；不加入 identity 表或 V 特征；C 字段决策时可用性尚待核实 |
| Elliptic | 单交易节点，T=1，94 个本地字段 | 时间步 1–34／35–41／42–49；排除邻域聚合、图边；未知标签不监督 |
| Sapienza 15S | 逐点异常开始块与时间间隔特征 | 按币种隔离研究划分；不补齐无交易时间，不把不同采样率当独立事件集 |
| SD-stock／SD-crypto | 发布者处理后的逐点序列 | 使用研究重新划分，不冒称原论文官方独立／零样本测试 |
| TSTR | 1000 点整段二分类 | 去重后划分，保留完整序列与整段标签 |

PaySim、IBM AML、TabFormer 的本轮训练目标数为 499,968、验证目标数为 99,968；在各自时间区间内按与标签无关的哈希选取完整目标块，保留块前上下文，测试使用完整对应区间。它们不属于全量训练实验。其他数据的确切计数、来源与限制见 [实验记录](../experiments/repair_rounds_20261103/README.md)。

当前已经接通上述十项实验；通用 `supervised prepare` CLI 直接支持其中五项，新增交易数据由实验准备模块／runner 接入。Elliptic 的 T=1 无跨交易时序建模，不应称为图神经网络或充分利用了 Kronos 长上下文。

## 10. 证据、效果与尚未解决的问题

下表为历史参考结果与本轮验证所选模型的测试 AP；不是候选按测试结果择优。

| 数据 | 所选配置 | 历史 AP | 本轮 AP |
| --- | --- | ---: | ---: |
| SD-stock | R1 | 0.5780 | 0.6303 |
| SD-crypto | R2 | 0.3565 | 0.4803 |
| TSTR | R0 | 0.4878 | 0.4858 |
| ULB | R2 | 0.3174 | 0.7419 |
| Sapienza | R3 | 0.7524 | 0.9119 |
| Elliptic | R2 | 0.2772 | 0.4351 |
| PaySim | R2 | 0.0408 | 0.4456 |
| IBM AML | R2 | 0.0709 | 0.2659 |
| TabFormer | R2 | 0.0014 | 0.0066 |
| IEEE-CIS | R2 | 0.1365 | 0.2393 |

10 项中 9 项 AP、8 项 F1 提高，但结果边界必须一起解释：

- SD-stock AP 提高而 F1 从 0.6091 降至 0.5896；TSTR 保留参考配置，AP 仍接近正类占比，尚无有力分类效果。
- PaySim 固定验证阈值下测试 recall 仅 3.22%；TabFormer 仅 0.47%，其验证 AP 约 0.2344 而测试约 0.0066，泛化问题突出，具体成因尚未隔离。
- IEEE-CIS recall 从 14.08% 升至 27.86%，同时 FP 从 1,343 增至 2,661。另做相同 MPS 设备参考，仍存在增益及误报代价。
- 同一历史测试集此前已经被查看，因此这是历史保留集重评，不是全新外部盲测。历史与本轮公共实验还存在 CUDA／PyTorch 环境差异。
- 三种子确认仅覆盖 ULB、PaySim、TabFormer，不能推广为全部数据稳定性结论；样本标准差不是置信区间。
- 本轮没有完整的同预算纯 MLP、GBDT、同规模随机初始化主干对照，无法把所有增益归因于 Kronos 预训练，也不能宣称优于经典方法或达到 SOTA。

10 项主结果已从逐行预测独立复核；额外种子仅保留聚合记录与哈希，不声称全部逐行复核。完整结果、协议、训练记录及限制见 [实验目录](../experiments/repair_rounds_20261103/README.md)；结构改进说明见 [多轮修复文档](supervised_repair_rounds_CN.md)。已完成实验的代码版本固定为 `573f0fc8419ef9c2dfd4927c71e2f3728c9a2785`。

## 11. 输入格式与复现命令

用户数据目录应显式含 `train.npz`、`validation.npz`、`test.npz`，不使用 pickle：

| 字段 | 形状与含义 |
| --- | --- |
| X | float [N,T,F]；允许 NaN，拒绝 infinity |
| y | 点级 [N,T] 或整段 [N]；−1／0／1 |
| lengths | 可选 int [N]，有效长度，默认 T |
| sample_id | 字符串 [N]；不同 split 身份不得重叠 |
| feature_names | 字符串 [F]，各 split 顺序一致 |
| group_id | 可选字符串 [N]；提供时要求 split 组隔离 |
| target_mask | 可选 bool [N,T]；点级任务目标归属 |
| observation_id | 可选字符串 [N,T]；原始观测身份与唯一目标核验 |

推断时标签可全部为 −1。示例使用 ULB、连续特征及 R2；新实验应使用新的输出路径，避免覆盖：

```bash
python -m pip install -r requirements-supervised.txt
python -m supervised prepare --dataset ulb \
  --source-root data/supervised_financial --history-length 32 \
  --output prepared_runs/ulb-history32

python -m supervised train --data-dir prepared_runs/ulb-history32 \
  --backbone NeoQuasar/Kronos-mini --head-type hybrid \
  --epochs 12 --batch-size 32 --lr 1e-3 --backbone-lr-ratio 0.1 \
  --loss bce --pos-weight auto --head-dropout 0.1 \
  --schedule cosine --warmup-fraction 0.1 --patience 3 \
  --defer-test --seed 42 --device cpu --output runs/ulb-r2

# epoch 和验证阈值固定以后，才评估测试
python -m supervised evaluate --checkpoint runs/ulb-r2/best.pt \
  --data prepared_runs/ulb-history32/test.npz \
  --output runs/ulb-r2/final-test.csv --device cpu
```

CPU 可换为可用 CUDA／MPS。TSTR 另加 `--max-context 2048 --batch-size 4`；真实 OHLCVA 需提供匹配 tokenizer，且不能使用 hybrid。`--scratch` 默认是小型随机架构运行检查；同规模消融应显式给 `--scratch-config`，不能把 tiny scratch 与完整预训练主干直接当等规模比较。

复现四候选验证选择流程，可先以单个公开数据运行：

```bash
python scripts/refine_supervised.py --stage batch --datasets ulb \
  --backbone NeoQuasar/Kronos-mini --device cpu --output runs/ulb-four-rounds
```

runner 从仓库根目录的 `data/supervised_financial` 读取已下载原始数据，须在仓库根目录运行；原始数据来源、输入准备与环境要求见实验目录；可用本地权重目录固定权重，Hub 模型可通过 `--revision` 指定版本。IEEE-CIS 原始数据、派生矩阵、类别词表、逐行预测和 checkpoint 仅保存在本地，不上传公开仓库或公共计算环境。

checkpoint 包含模型结构、全部权重、特征顺序、训练期预处理、上下文限制、验证阈值与来源核验信息；可离线重建模型，`predict` 导出分数，`evaluate` 额外输出指标。只加载可信 checkpoint。

## 12. 实现位置与验证范围

- [model.py](../supervised/model.py)：适配器、因果多尺度、点级／整段头、直接输入残差与模型重建。
- [data.py](../supervised/data.py)：格式校验、划分隔离、训练期标准化、历史／目标掩码。
- [transaction_prepare.py](../supervised/transaction_prepare.py)：新增交易数据的特征与准备协议。
- [optimization.py](../supervised/optimization.py)：类别加权损失、AdamW 分组与调度。
- [engine.py](../supervised/engine.py)、[metrics.py](../supervised/metrics.py)：训练、早停、验证阈值、固定模型评估和指标。
- [refine_supervised.py](../scripts/refine_supervised.py)：四候选、测试封存、验证选型及额外种子确认。

当前代码已有 58 项测试通过的记录，覆盖因果性、padding、未知标签、梯度与冻结、拆分隔离、预处理、真实 tokenizer 接口、离线重载、hybrid 及权重快照等。测试证明相应实现约束，并不等同于实际金融业务有效性或部署认证。
