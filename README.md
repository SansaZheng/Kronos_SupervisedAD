# Kronos_SupervisedAD：基于 Kronos 的监督金融异常检测

数据核验日期：2026-11-02；方法更新：2026-11-03。收录 **8 组带作者标签的数据**，完整保留来源、许可、标签语义、文件校验和读取工具。由 [KronosAD](https://github.com/SansaZheng/KronosAD) 的数据目录迁入；本仓库提供监督异常检测模型、训练／评估／推断流程和数据读取工具。已完成小规模运行验证；已完成小规模三种子头部对照与树模型控制，尚未完成跨数据集的系统基准。

| 数据集 | 任务 | 实际收录规模 | 正类 |
| --- | --- | ---: | ---: |
| ULB / Worldline | 真实信用卡欺诈 | 284,807 笔 | 492 |
| PaySim | 合成移动支付欺诈 | 6,362,620 笔 | 8,213 |
| IBM TabFormer | 合成用户交易序列欺诈 | 24,386,900 笔 | 29,757 |
| IBM AML HI-Small | 合成交易反洗钱 | 5,078,345 笔 | 5,177 |
| Elliptic | 比特币非法交易图 | 203,769 个节点 | 4,545；未知标签不当作正常 |
| Sapienza | 加密资产拉盘砸盘开始点 | 15S 版本 584,104 行、327 个事件组；另含 5S/25S | 每个分辨率 317 个开始块 |
| TSTR | 整段操纵价格序列分类 | 去重后 964 条真实价格序列 | 466 条 |
| SD-FMM | 股票／加密资产点级异常 | 27 条股票、337 条加密序列 | 493 个股票点、337 个加密点 |

另提供 IEEE-CIS 的本地读取工具和获取说明：官方训练集 590,540 笔，20,663 笔欺诈。**原始数据受竞赛规则限制，不随公开仓库分发**；官方测试集没有公开标签。已有授权 ZIP 保留在用户本地私有目录。


## 已实现：Kronos 监督异常检测

沿用原始 Kronos 因果 Transformer，将下一步 token 预测改为监督二分类。现已加入过去上下文、多尺度点级头、整段注意力汇聚、分组学习率、warmup／早停，以及按验证误报预算选择阈值。默认使用类别加权 BCE；focal 可选。支持逐点／整段标签、预训练微调／冻结主干，以及仅用于消融的显式从头训练。

- 真实 OHLCVA：冻结原 Kronos tokenizer，保留粗／细 token embedding，训练主干与分类头。
- 一般交易特征：可训练输入适配器接入 Kronos 主干，不把交易表伪造为行情。支持 Sapienza、ULB、TSTR、SD-FMM 的内置准备接口及自定义 NPZ。
- 仅训练集拟合预处理；验证集选择 epoch 和分类阈值；测试标签不参与选择。未知标签和 padding 不进入损失与评价。

[本轮完善与三种子对照／负结果](docs/supervised_refinement_20261103.md) · [完整中文方法与命令](docs/supervised_method_CN.md) · [上游固定版本与许可](UPSTREAM.md) · [执行验证记录](docs/supervised_validation.md)

```bash
python -m pip install -r requirements-supervised.txt
python -m supervised prepare --dataset sapienza --history-length 32 --output prepared_runs/sapienza
python -m supervised train --data-dir prepared_runs/sapienza \
  --backbone NeoQuasar/Kronos-mini --epochs 10 --defer-test --output runs/sapienza-mini
python -m supervised evaluate --checkpoint runs/sapienza-mini/best.pt \
  --data prepared_runs/sapienza/test.npz --output runs/sapienza-mini/predictions.csv
```

## 数据与使用说明

- [全部数据索引](data/supervised_financial/dataset_index.csv)
- [支付欺诈／反洗钱数据、来源许可与读取示例](data/supervised_financial/transactions/README.md)
- [市场操纵数据、标签粒度与研究划分](data/supervised_financial/README.md)
- [IEEE-CIS 官方获取与本地读取](data/supervised_financial/transactions/IEEE_CIS_LOCAL.md)
- [市场数据许可](data/supervised_financial/DATA_LICENSES.md)
- [交易数据许可文件](data/supervised_financial/transactions/notices)

数据实际存储在仓库中，约 814 MB；较大压缩包按字节分卷，每卷不超过 40 MiB。并非仅提供下载链接，无需 Git LFS。各数据继续遵守各自许可；根目录 MIT 许可仅适用于本项目代码。

## 安装与校验

```bash
python -m pip install -r requirements.txt
python scripts/data/validate_supervised_data.py
python -m scripts.data.transaction_data verify --source-bytes
python -m pytest tests/test_supervised_data.py tests/test_transaction_data.py tests/test_ieee_cis_local.py -q
```

无需重组即可流式读取：

```python
from scripts.data.transaction_data import iter_table, map_labels

for chunk in iter_table('paysim', chunksize=100_000):
    y = map_labels('paysim', chunk['isFraud'])
    X = chunk.drop(columns=['isFraud', 'isFlaggedFraud'])
    break
```

需要传统压缩包时，在指定目录重组一个数据资产：

```bash
python -m scripts.data.transaction_data restore --asset ulb --output /你的数据目录
```

## 选择与评价

支付欺诈基线可先使用 ULB；用户交易序列可用 TabFormer；反洗钱可用 IBM AML 或 Elliptic；证券／加密市场操纵建议先用 Sapienza。不同标签粒度分别实验，不混合为同分布任务。

只在训练集拟合预处理和重采样，验证集选择阈值，留出测试集评价。报告正类 Precision、Recall、F1、AP（average precision）和误报数量。Elliptic 的 unknown 标签不参与监督损失。交易后信息与图聚合特征须检查检测时是否可得。

这些数据不具备统一 OHLC 格式，不能直接复制为四列输入 Kronos。现已实现连续特征适配器与监督分类头；完整流程请见上方方法文档。单次运行检查不能证明优于现有方法。


## 下载数据的批量实测与展示

[多轮改进实测：9/10 项 AP、8/10 项 F1 提高](experiments/repair_rounds_20261103) · [原始特征残差通路与验证选择协议](docs/supervised_repair_rounds_CN.md)

[已完成的 10 项实测结果、图表与复核记录](experiments/downloaded_20261103)

统一运行公开数据的 9 项检测任务（SD-FMM 分股票与加密资产）：

```bash
python scripts/run_downloaded_benchmark.py --stage batch --device cuda \
  --epochs 5 --train-cap 500000 --validation-cap 100000 --output runs/downloaded
```

大交易数据按时间划分，训练／验证使用与标签无关的块抽样上限，测试时间段完整保留。Elliptic 使用本地节点特征，不使用边或邻居聚合特征；未知标签忽略。Sapienza 使用 15S 分辨率，5S／25S 是同一批事件的其他分辨率，不重复计作独立数据集。TSTR 是整段序列分类，其余任务的检测单位见结果记录。

IEEE-CIS 使用已授权下载的官方 ZIP 单独在本机运行。该脚本只读取有标签的训练交易，另划时间留出集，不将官方无标签测试文件视为真值：

```bash
python scripts/run_ieee_local_benchmark.py --zip /你的私有目录/ieee-fraud-detection.zip \
  --backbone /你的本地权重/Kronos-mini --epochs 5 --output .local_data/ieee-run
```

从真实预测独立复核指标，生成仅含汇总指标和曲线的可筛选 HTML 报告：

```bash
python -m pip install -r requirements-report.txt
python scripts/render_downloaded_results.py --public runs/downloaded \
  --private .local_data/ieee-run --output runs/downloaded-report
python scripts/audit_downloaded_results.py runs/downloaded .local_data/ieee-run \
  --output runs/downloaded-report/audit.json
```

原始交易、派生输入、检查点和逐笔预测不进入汇总报告。IEEE-CIS 的类别词表也不会被导出；其私有输出不得上传公开仓库或云端任务。各数据集分开报告，不把不同标签单位的分数混成总平均。
