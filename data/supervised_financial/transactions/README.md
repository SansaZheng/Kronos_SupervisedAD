# 支付欺诈与反洗钱监督数据

本轮按用户给出的数据表扩展到支付和反洗钱领域。下载／核验日期：**2026-11-02**。这里随仓库实际分发五组数据；IEEE-CIS 现已另行下载到本地，其竞赛规则限制公开再分发。原有 Sapienza、TSTR、SD-FMM 继续保留（共八组随仓库存储，另有一组仅本地），全部数据索引见上级 [dataset_index.csv](../dataset_index.csv)。

## 下载内容与实际统计

| 数据 | 本地实际包含的范围 | 记录数 | 正类数 | 真实／合成 |
| --- | --- | ---: | ---: | --- |
| ULB / Worldline Credit Card Fraud | 完整 `creditcard.csv`，来自 TensorFlow 官方教程镜像 | 284,807 | 492 | 真实、匿名化 |
| Elliptic Bitcoin Dataset | 完整特征、类别、边三份文件，来自 PyTorch Geometric 镜像 | 203,769 个交易节点、234,355 条边 | 4,545 个非法节点 | 真实交易图 |
| PaySim | 社区镜像中的完整 Kaggle 规模 CSV，全部 11 列 | 6,362,620 | 8,213 | 合成 |
| IBM TabFormer | IBM Git LFS 原始完整 `transactions.tgz`；实际 CSV 有 15 列 | 24,386,900 | 29,757 | 合成 |
| IBM AML HI-Small | 社区镜像中的 HI-Small 交易、账户、模式三份完整文件 | 5,078,345 笔交易；518,581 行账户 | 5,177 | 合成 |
| IEEE-CIS Fraud Detection | **已下载，仅本地**：完整五个官方 CSV，参见 [本地说明](IEEE_CIS_LOCAL.md) | 590,540 条带标签训练交易；506,691 条无标签测试交易 | 20,663（训练集） | 真实、匿名化业务数据 |

“HI-Small 完整”指该发布子集的三个文件全部下载，不代表 IBM AML 的全部 HI／LI、Small／Medium／Large 版本。PaySim 是作者为公开发布而缩小后的模拟数据，不是支付公司的原始日志。上述统计来自实际文件扫描，不是论文中的近似数量。

## 来源与标签

### ULB / Worldline

- 原发布：[MLG-ULB / Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)。
- 获取入口：[TensorFlow 官方不平衡分类教程](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)，教程直接链接 Google 托管的数据文件。
- `Class`：`1` 欺诈，`0` 正常。特征为 `Time`、`V1`–`V28`、`Amount`；`Time` 是距首笔交易的秒数，不是日期。交易覆盖 2013 年的两天。
- 原始 CSV SHA-256：`76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`。
- **许可是 ODbL／DbCL，不是 CC0**。原数据卡显示“Database: Open Database, Contents: Database Contents”；许可分别保存于 `notices/ULB-ODbL-1.0.txt` 与 `notices/ULB-DbCL-1.0.txt`。

### Elliptic

- 原发布：[Elliptic / Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)。论文：[Weber 等，2019](https://arxiv.org/abs/1908.02591)。
- 获取入口是 [PyTorch Geometric 官方库使用的公开镜像](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/datasets/elliptic.py)，不使用挂有错误 MIT 许可的社区副本。
- 原始 `class`：字符串 `1`＝非法，`2`＝合法，`unknown`＝未知。实际未知节点 **157,205** 个，合法节点 **42,019** 个。未知不是正常类，不参与有监督损失与带真值评价。
- 特征 CSV **无表头**：第 0 列 `txId`，第 1 列时间步，第 2–166 列为其余 165 个数值特征。论文的“166 个特征”包括时间步。时间步为 1–49；不能解释为连续分钟行情。
- 已核对类别与特征的交易 ID 一一对应、边两端均存在于节点表。标签根据交易发起实体的合法／非法类别赋予，并非每笔交易的独立司法判决。
- 数据许可 **CC BY-NC-ND 4.0**：原始 ZIP 原样保留，仅按字节分卷，不发布重新标注或筛选后的数据。使用须保留署名、非商业及禁止分发改编数据等原许可条件；见 `notices/Elliptic-CC-BY-NC-ND-4.0.txt`。代码在内存中提供标签映射。

### PaySim

- 作者项目：[EdgarLopezPhD/PaySim](https://github.com/EdgarLopezPhD/PaySim)，原发布：[Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)。引用 Lopez-Rojas、Elmir、Axelsson，*PaySim: A financial mobile money simulator for fraud detection*，EMSS 2016。
- 本轮从 [kohdified/synthetic-financial-data 的固定公开版本](https://huggingface.co/datasets/kohdified/synthetic-financial-data/tree/dfe0c96bc261e79da03af5ca52f36e8fc26a0e69) 获取。镜像说明源自上述 Kaggle，许可同为 **CC BY-SA 4.0**。这是第三方镜像，不是 PaySim 作者的 Hugging Face 账户。
- 完整镜像文件的长度与 SHA-256 均匹配其固定版本 Git LFS 指针。未登录 Kaggle，因此**没有直接与 Kaggle 原文件逐字节比对**。规模、字段和正类数符合公开基准；保留来源层级这一限制。
- `isFraud` 是模拟欺诈真值；`isFlaggedFraud` 是另一个业务规则标志，**不是标签**。时间 `step` 覆盖 1–743，是相对小时编号，没有真实交易日期。
- 只做无损 gzip 压缩，全部记录、原字段名和标签不变；不包含自行平衡采样的 Mini 版本。

### IBM TabFormer

- [IBM 官方仓库固定版本](https://github.com/IBM/TabFormer/tree/ebb7cd68ee1897599568107740bc452104bbbaf8)，论文：[Padhi 等，ICASSP 2021](https://arxiv.org/abs/2011.01843)。
- 下载的是该仓库 Git LFS 所指向的 `transactions.tgz`，共 278,576,638 字节。SHA-256 与 IBM 发布指针一致：`e9f589a0958f40d60f81b1a2e8428db86e00c05755caf44fb055827976c0efa2`。
- 内部 `card_transaction.v1.csv` 共 2,354,626,737 字节。`Is Fraud?`：`Yes` 欺诈，`No` 正常。用户、银行卡、年月日、时间、商户等字段保留，适合用户交易序列研究。身份、商户与交易均来自合成世界。
- 本次来源是 IBM 仓库内跟踪的 Git LFS 数据，随附该仓库 **Apache-2.0** 许可、README；不将其他 IBM 数据集的 CDLA 许可套用到本包。许可见 `notices/TabFormer-Apache-2.0.txt`。

### IBM AML HI-Small

- 作者入口：[IBM/AML-Data](https://github.com/IBM/AML-Data)，论文：[Altman 等，2023/2024](https://arxiv.org/abs/2306.16424)。
- 本轮从 [OsamaMIT/IBM-AML-HI-Small 的固定公开版本](https://huggingface.co/datasets/OsamaMIT/IBM-AML-HI-Small/tree/95aa8c419e7c43eb1f3299901aab601889c6dd1b) 获取，包含 `HI-Small_Trans.csv`、`HI-Small_accounts.csv`、`HI-Small_Patterns.txt`。这是第三方镜像，名称中的 MIT 不是经本项目核实的机构背书。
- 交易／账户文件的原始字节哈希与镜像 Git LFS 指针一致。未直接与 Kaggle 原件比对。账户表有 **518,581 行**，对应 **518,573 个不同的 Account Number**；镜像 README 使用后一个口径。`(Bank ID, Account Number)` 联合键全部唯一，表中没有完全重复行。联接账户与交易时须同时使用银行和账户，不能只按账户号合并。
- `Is Laundering`：`1` 表示模拟洗钱，`0` 正常；交易时间范围 2022-09-01 至 2022-09-18。CSV 两列均叫 `Account`，pandas 读取后为 `Account` 和 `Account.1`，分别对应转出与转入账户；账户标识须按字符串保留。
- 模式说明文件是洗钱活动的注释，不能作为预测特征输入。镜像 README 将生成器称为 AMLSim；IBM 原说明把 AMLSim 列为另一项工作。保留两份说明，以 IBM 作者对虚拟世界生成数据的介绍为准，不将二者混同。
- **数据许可 CDLA-Sharing-1.0**，与 IBM 仓库代码的 Apache-2.0 分开。无损压缩副本继续沿用 CDLA-Sharing-1.0；见 `notices/IBM-AML-CDLA-Sharing-1.0.txt`。

## 文件组织、核验与读取

这里有 **9 个完整数据资产、25 个字节分卷**，数据资产共 **785,879,032 字节，约 786 MB**。每卷最多 40 MiB，可直接存入 GitHub，不依赖收费存储、Git LFS 配额或外部付费下载。加上上级目录原有数据，仓库当前数据量约 814 MB。

原始 ZIP／TGZ 逐字节保留；四个从镜像取得的 CSV／文本仅增加无损 gzip 容器。分卷没有按行抽样，也没有修改标签。`assets.json` 记录来源、固定版本、原始字节哈希、压缩后哈希、各卷大小与哈希。`notices/` 包含许可和原说明，`scan_report.json` 为全文件扫描记录，`catalog.csv` 为本轮数据目录。

依赖：Python 3.10+、numpy、pandas。无需 PyTorch、Kaggle 登录或模型权重。

```bash
# 在仓库根目录，验证所有分卷及重组后的原始压缩包
python -m scripts.data.transaction_data verify

# 进一步解压流并核对 Hugging Face 来源原文件哈希，不产生大型解压文件
python -m scripts.data.transaction_data verify --source-bytes

# 如需要传统单文件，选择一个资产，在仓库外重组；同名文件已存在时拒绝覆盖
python -m scripts.data.transaction_data restore --asset ulb --output /你的数据目录
# 可用 ID：ulb, tabformer, elliptic_classes, elliptic_edges, elliptic_features,
# paysim, ibm_aml, ibm_aml_accounts, ibm_aml_patterns
```

不重组也能分块读取；下面只演示读取与标签映射，不隐含训练划分：

```python
from scripts.data.transaction_data import iter_table, map_labels

for chunk in iter_table("paysim", chunksize=100_000):
    y = map_labels("paysim", chunk["isFraud"])
    # isFraud 不可作为输入；isFlaggedFraud 是规则输出，应单独报告是否使用。
    features = chunk.drop(columns=["isFraud", "isFlaggedFraud"])
    break

labels = next(iter_table("elliptic_classes"))
y = map_labels("elliptic", labels["class"])
supervised_mask = y >= 0  # unknown=-1，不能当作正常样本

for chunk in iter_table("ibm_aml", dtype={"Account": str, "Account.1": str}):
    y = map_labels("ibm_aml", chunk["Is Laundering"])
    break
```

## 研究使用边界

这些是交易级／图节点监督数据，不能复制成 OHLC 四列直接送入 Kronos。支付欺诈、洗钱与市场操纵分别实验，不把它们拼成一个同分布任务。

这里没有训练分类器或报告检测 F1。后续应按时间、账户／用户隔离的目标明确建立划分；在训练段拟合特征处理，验证段选参数／阈值，测试段只做最终评价。TabFormer 的未来交易、Elliptic 的邻居聚合信息、交易后的余额等字段是否在检测时可用，需要在实验协议中说明。报告正类 Precision、Recall、F1、PR-AUC 与误报数量，不能仅以 Accuracy 判断效果。

上级目录的 Sapienza／TSTR／SD-FMM 研究划分不适用于这些新数据。本次保留原标签与完整文件，没有为新数据生成容易造成泄漏的随机逐行划分。
