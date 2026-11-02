# 金融市场操纵监督学习数据

> **扩充更新（2026-11-02）：** 按用户给定清单，新增 ULB、Elliptic、PaySim、IBM TabFormer、IBM AML HI-Small 五组支付／反洗钱数据；详情见 [新增数据说明](transactions/README.md)，原八组及本地 IEEE-CIS 的索引见 [dataset_index.csv](dataset_index.csv)。IEEE-CIS 现已在用户登录并确认规则后下载到本地，受规则限制不上传原始数据，见 [本地读取说明](transactions/IEEE_CIS_LOCAL.md)。下文继续描述原三组市场操纵数据，其规模统计与研究划分保持原有范围。

检索与下载日期：**2026-11-02**。本目录关注证券／加密资产交易中的市场操纵，优先保留作者发布的人工标注与原始标签。它不包含信用卡盗刷、支付欺诈或我们根据价格涨跌生成的伪标签。

**建议先用 Sapienza 15S 做监督分类基线。** TSTR 适合整段操纵识别；SD-FMM 的公开派生序列用于补充研究。数据的统计单位和标签含义不同，请分别实验。

## 已下载的数据与实际规模

| 数据 | 发布版本中实际下载的内容 | 标签与适用任务 | 直接输入当前 Kronos CLI |
| --- | --- | --- | --- |
| [Sapienza / ICCCN 2020](https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset/tree/d71250d4cb055dde2d415c8cba38a0dcd6eb6e16) | 5S：821,307 行；15S：584,104 行；25S：482,157 行；每份 12 个特征、327 个事件组、85 个币种 | 作者人工核查操纵开始的交易块，`gt=1`；每个分辨率 317 个正类块 | 否：是交易工程特征，不是 OHLC |
| [TSTR / ICAIF 2025](https://zenodo.org/records/17301104) | 原始 466 条操纵＋511 条非操纵候选价格序列，每条 1,000 点；去除 13 条完全重复的负类后共 964 条 | 作者 notebook 将两个目录分别对应正／负类；**整段序列标签** | 否：只有 `High` 价格列，无完整 OHLC 或绝对时间戳 |
| [SD-FMM / IPM 2026](https://github.com/ys-Dai/SD-FMM/tree/0d1a2158ee849fdc31a127bd049e622174f5c893) | 股票数组 27×160×1，标签中 493 个正类点；加密资产数组 337×160×1，337 个正类点 | 作者提供的点级标签：股票为区间，加密资产为单个开始点 | 否：已处理的单通道序列，没有完整行情字段和绝对时间戳 |

Sapienza 的三个分辨率来自相同事件，**不能相加声称有 951 个独立正类事件**。每份有 327 个 `pump_index`，其中 317 组有一个正类块，另外 10 组没有正类；原样保留。

这是当前可下载的研究数据，不代表交易时间很新。Sapienza 文件日期覆盖 2018-01-02 至 2021-01-19；TSTR 论文描述的操纵事件为 2017-07 至 2022-01。SD-FMM 公开数组没有足以独立恢复交易日期的信息。

## 标签可信度与边界

### Sapienza：推荐作为主监督数据

作者说明，他们检查交易数据并人工标记操纵真正开始的块，可能比 Telegram 公告早 120 秒。这次下载的是完整的 `labeled_features/*.csv.gz`，**不是此前我们根据公告时间添加的弱标签 K 线**。

`gt=1` 表示开始块，`gt=0` 是作者的其余背景块。它不是完整操纵持续区间的标注；不能把开始之后全部改成异常，也不能认为每个背景块都经过独立的正常交易审计。

12 个输入特征为 rush order、交易数、成交量和价格的滚动统计变化，以及时间的正余弦编码。沿用作者字段名和数值；`avg_price` 等字段是工程特征，不是可还原原始价格的 OHLC 列。源文件有缺失时间块，不能补齐后假装连续逐笔行情。

### TSTR：有监督序列分类，不是逐时间戳真值

作者论文描述从人工审阅的 Telegram 消息识别操纵公告，再提取公告前后分钟价格。正类标识整段含操纵事件；负类是作者非操纵目录中的价格序列，论文说明对其中 50 条做了人工抽查，不能说全部负例均逐条人工审计。

实际压缩包有 511 条负类，论文实验只使用 466 条负类。作者 notebook 取目录枚举的前 466 条，未发布稳定的固定文件名单。本目录保留全部文件，默认使用负类目录内全部有效序列并去重，**属于新的研究划分，不是论文结果复现**。

论文称使用收盘价，而发布 CSV 的列名是 `High`，notebook 也读取 `High`。这里保留源列名，不擅自改为 `close`。没有币种、交易日、绝对时间戳，也没有开／低／收／成交量。每段对齐公告附近的既定位置，会简化离线识别任务；所得分数不能直接代表在线预警能力。

`prepared/tstr_real.npz` 的 `y` 形状为 `(N,)`，不会扩展为 `(N,1000)`。源压缩包还含合成价格样本和生成 notebook，均留在原始包中；**默认 prepared 数据只用两个真实价格目录**。不采用作者 notebook 的全窗口 Min-Max 预处理。

### SD-FMM：有点标签，但公开资料不足以逐案复核

论文与 README 描述 25 个中国股票操纵案例、338 个加密资产案例；**实际发布数组是 27 条股票序列、337 条加密序列**。我们按文件统计，不将其描述为完整论文数据。源股票标签与 `test_ano_bs.npy` 的闭区间完全一致。

论文称股票案例依据监管材料确认，但这些数组没有公开股票代码、日期与监管案例的逐条映射，因此不能对这 27 行逐案独立核实，也不能证明每行都是一个独立案例。数据已经处理为 `[0,1]` 范围的单通道序列；原始订单簿／逐笔交易及可逆特征映射不在本包中。适合补充实验，暂不作为唯一可信基准。

## 目录和复现

- `raw/`：20 个原始发布文件、许可和冻结的元数据快照，合计约 25.9 MB，逐文件 SHA-256。
- `sources.json`：固定 Git 提交／Zenodo 版本、下载地址、大小、许可和校验值。
- `prepared/sapienza_groups.csv`：事件组对应的币种、分辨率、标签数量与划分；不复制庞大的原始特征表。
- `prepared/tstr_samples.csv`：977 条源序列的类别、内容哈希、是否纳入默认集、重复对应关系和划分。
- `prepared/tstr_real.npz`：964 条不重复真实价格序列；`X(N,1000,1)`、`y(N)`、`sample_id`、`split`。
- `prepared/sd_fmm_sequences.csv`：原数组行号、内容哈希、点标签数与研究划分。
- `catalog.csv`、`summary.json`：实际规模与各划分统计；`prepared_manifest.json`：派生文件校验值。

从仓库根目录运行，只需 numpy 和 pandas；以下命令不下载模型权重：

```bash
python scripts/data/download_supervised_data.py  # 已有文件只核对；缺失时按固定版本恢复
python scripts/data/prepare_supervised_data.py   # 验证来源并重新生成派生数据
python scripts/data/validate_supervised_data.py  # 校验文件、标签对齐与划分隔离
python -m pytest tests/test_anomaly.py tests/test_supervised_data.py -q
```

元数据快照由 Git 保存，不从会持续变化的 Zenodo API 重新生成。原始大文件均实际包含在仓库中，无需登录 Kaggle、申请 API key 或另外付费下载。

```python
from scripts.data.load_supervised_data import load_sapienza, load_tstr, load_sd_fmm

X_train, y_train, meta_train = load_sapienza("15S", "train")
X_val, y_val, meta_val = load_sapienza("15S", "validation")
X_test, y_test, meta_test = load_sapienza("15S", "test")
# 只将 X 输入分类器；meta 中的币种、事件编号、日期用于检查与分组。

X_seq, y_seq, sequence_ids = load_tstr("train")
X_point, y_point, source_indices = load_sd_fmm("stock", "train")
```

## 划分与评价要求

划分种子固定为 `kronosad-supervised-v1`，按哈希稳定排序后约 60%／20%／20%。这些全部是本仓库研究划分。

- **Sapienza** 按 85 个币种划分（51／17／17），同币种及同事件的所有分辨率都留在同一划分。15S 的 train／validation／test 为 327,103／111,451／145,550 行，正类为 190／50／77。一次实验选择一个分辨率，不混用不同分辨率制造更多独立事件。
- **TSTR** 按去重后的序列、分正负类划分，train／validation／test 为 577／193／194 条。完全相同价格不能跨划分。由于缺少币种／日期，无法证明不同序列之间不存在同币种或重叠时间窗口，不能声称是时间外推或跨币种测试。
- **SD-FMM** 原文件全部命名为 `test`。保留文件名与原有标签；额外给出按内容分组的 16／5／6 条股票序列、202／67／68 条加密序列研究划分。用它们训练后，不能再把这项实验称为原论文的独立测试或零样本测试；也无法保证不同数组行来自不同股票／交易日。

监督训练只用 train；特征缩放、采样和类别权重仅在 train 拟合；阈值和参数在 validation 选定，最终一次性在 test 评价。不要把 `gt`、事件 ID 或序列 ID 输入模型。不要在 test 上挑选最优 F1 阈值。类别极不均衡，应报告正类 Precision、Recall、F1、PR-AUC，以及误报数量／分母；Accuracy 仅作辅助。

Sapienza 和 SD-FMM crypto 评价开始点识别；SD-FMM stock 评价原始区间点标签；TSTR 评价整段分类。若另外报告容忍时间窗口、事件命中率或 point adjustment，须另列原始逐点结果，明确容忍规则，不能替换原始标签后只报告更高的 F1。

## 与 Kronos 异常检测扩展的关系

这些下载任务**没有训练新的模型，也没有产生新的检测性能结论**。现有 `python -m anomaly` 是基于 OHLC 的预测误差＋POT 研究扩展，尚未添加这些特征的监督训练接口。不能将 12 维特征或单通道派生序列复制成四列来冒充 OHLC。

可以先用这些数据训练 Random Forest 等监督基线，再单独开发通用时序分类器或 Kronos 表征适配器。后续如需直接评价 Kronos 的行情异常检测，还需获取具有相应监督标签、真实时间戳及 OHLC 的原始行情并核对事件映射。

## 旧数据的移除

本次按用户要求，从仓库当前版本删除整个旧 `data/financial_anomalies/`：Meme2026 算法标签行情、公告时间弱标签 K 线、近期无标签 BTC／ETH 行情，以及旧下载／转换脚本和专用测试。保留核心模型、异常检测代码、上游示例及实现回归测试数据。Git 历史不重写。

来源、引用与许可见 [DATA_LICENSES.md](DATA_LICENSES.md)，筛选记录见 [SEARCH_NOTES.md](SEARCH_NOTES.md)。
