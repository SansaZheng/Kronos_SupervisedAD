# IEEE-CIS：已下载到本地，原始数据不随公开仓库分发

获取／核验日期：**2026-11-02**。本次通过已登录并由用户自行确认规则的 Kaggle 页面，取得官方完整 `ieee-fraud-detection.zip`。ZIP 共 **123,856,947 字节，约 124 MB**，包含以下五个 CSV，解压总大小约 1.35 GB。读取工具可以直接流式读取 ZIP，无需先解压。

| 文件 | 实际行数 | 列数 | 标签用途 |
| --- | ---: | ---: | --- |
| `train_transaction.csv` | 590,540 | 394 | `isFraud=1` 为欺诈，共 20,663 条；正常 569,877 条 |
| `train_identity.csv` | 144,233 | 41 | 训练交易的部分身份／设备特征，无独立目标列 |
| `test_transaction.csv` | 506,691 | 393 | **没有公开 `isFraud` 真值** |
| `test_identity.csv` | 141,907 | 41 | 测试交易的部分身份／设备特征 |
| `sample_submission.csv` | 506,691 | 2 | 提交格式示例，示例概率不是测试真值 |

已通过全 ZIP CRC 校验；五个表各自的 `TransactionID` 均唯一；训练与测试交易 ID 无交叉；身份表 ID 均属于对应交易表；提交示例的 ID 顺序与测试交易一致。训练交易有 446,307 条没有身份表记录，测试交易有 364,784 条没有身份表记录，不应因身份缺失而被内连接删除。

## 为什么不能把原始 ZIP 上传 GitHub

来源：[IEEE-CIS 官方数据页](https://www.kaggle.com/competitions/ieee-fraud-detection/data)。

[竞赛规则第 7.A、7.B 条](https://www.kaggle.com/competitions/ieee-fraud-detection/rules)允许非商业研究／教育使用，并要求限制未接受规则者访问，不得将数据提供给非参赛方。因此本仓库只保存获取说明、总体统计和通用本地校验／读取工具；**不包含原始 ZIP、CSV、抽样交易行或派生训练矩阵**。根目录 `.gitignore` 排除了 `private_data/`、`.local_data/` 和该 ZIP 名称。

其他研究者必须使用自己的 Kaggle 账户登录、阅读并确认相应规则，再从官方页面选择 **Download all files → Download as zip**。这里的 MIT 代码许可或其他数据许可不改变 IEEE-CIS 的竞赛数据使用条件。工具不会替用户接受规则、导出浏览器登录凭据或提交预测。

## 本地校验与读取

将获得授权的 ZIP 放入自己的私有目录。以下相对路径只是示例，实际目录可以通过参数指定：

```bash
python -m scripts.data.ieee_cis_local private_data/ieee_cis \
  --report private_data/ieee_cis/audit_report.json
```

报告仅包含整体规模、列名、哈希和检查结果，不输出交易行；默认保存在本地，文件权限设为仅当前用户读写。

```python
from scripts.data.ieee_cis_local import iter_transactions, train_labels

private_directory = "/你的私有数据目录/ieee_cis"

for chunk in iter_transactions(private_directory, "train", chunksize=50_000):
    y = train_labels(chunk["isFraud"])
    X = chunk.drop(columns=["isFraud", "TransactionID"])
    # X 仍有类别字段和缺失值，须在训练段拟合对应处理。
    break

for chunk in iter_transactions(private_directory, "test", chunksize=50_000):
    assert "isFraud" not in chunk
    break
```

读取工具以 `TransactionID` 左连接身份表，验证一对一关系，保留所有交易；缺失身份特征仍为缺失，并提供 `identity_available` 标志。测试身份表的 `id-XX` 字段在内存中统一为训练表的 `id_XX` 名称，不改动原始文件。

`TransactionDT` 是相对参考点的时间差，不是真实日历时间。建立本地监督评价时，须从带标签的训练交易中另行建立训练／验证／留出测试划分，例如按相对时间划分；官方 `test_transaction.csv` 只能用于推断，不能拿提交示例的 0.5 概率计算 Precision／Recall／F1。

2026-11-03 已新增本地监督式 Kronos 训练与时间留出评估，汇总结果见[本轮实测记录](../../../experiments/downloaded_20261103)。原始和派生交易记录仍只在本机保存；这些交易特征通过连续适配器输入模型，不冒充 OHLC。
