# Kronos 监督异常检测的多轮改进

实验协议：[四轮配置与选择规则](../experiments/repair_rounds_20261103/PROTOCOL.md)。最终结论以实际验证／测试记录为准，不预先承诺每个数据集都提升。

## 原始交易特征的残差通路

原方法把连续特征 `x_t` 经过适配器和预训练因果 Kronos 主干得到 `h_t`，分类头读取当前隐表示及其相对过去均值的偏移。当前分类任务的学习信号必须经过这条适配与表征通路。

新增 `--head-type hybrid` 保留原来的预训练主干和多尺度头，另加一个交易特征 MLP，其 logit 与原头相加：

```
point_logits_t = temporal_head(h_t, h_t − trailing_mean(h, 4/16/64))
              + raw_head(x_t, x_t − trailing_mean(x, 4/16/64))
```

原始特征通路不再经过主干的 LayerNorm，可直接学习金额、余额、类别编码及其交互。它仍使用仅训练段拟合的同一套标准化与裁剪，不能声称恢复了被预处理裁剪掉的信息。点级滚动均值最多读到当前时刻，不读未来；padding 和历史上下文依然不作为监督目标。

对于 TSTR 的整段标签，原始特征头读取最后有效点、有效点均值、标准差、最大值和最小值，补充预训练隐状态的注意力汇聚。整段分类可以读取该段全部有效输入；它不是提前检测，也不能把这种汇聚方式用于实时点级判断。

新增通路是本项目的扩展，并非对 [Gorishniy 等的表格 ResNet／FT-Transformer 论文](https://arxiv.org/abs/2106.11959)的逐层复现。仍需独立的纯 MLP／树模型对照，才能判断 Kronos 主干自身带来的额外价值。

## 四轮对照

- R0：旧多尺度方法，最多 5 轮，head LR=1e-4，backbone LR 为其 0.1 倍。
- R1：只增加原始特征通路，其余同 R0，作为相同训练预算的结构对照。
- R2：hybrid，head LR=1e-3，backbone 比例 0.1，最多 12 轮。
- R3：hybrid，head LR=3e-4，backbone 比例 0.3，正类 BCE 权重上限 100，最多 12 轮。

R2/R3 同时改变了多项训练设置，不将它们相对 R0 的全部提升归因于结构。各候选重新从相同原始 Kronos-mini 初始化；不在已经看过测试分数的检查点上继续训练。验证 AP 早停，patience=3。

每项数据按验证 AP、验证 F1、较早方案的顺序选择候选，参考版仍有资格胜出。阈值只在验证集上选择。所有主试验训练时 `test.npz` 被改名封存，训练入口使用 `--defer-test`；选择 JSON 和哈希写入后才重新开放测试文件。既有测试分数以前已被公布，因此本次仍只是历史留出集复测，不是新的外部盲测。

ULB、PaySim、TabFormer 若选中了新方案，会用额外种子 13、2026，按已经固定的设置与参考版做配对确认；不能在看到确认测试分数后换方案。其余任务只有一个选择种子，不应报告不存在的置信区间。

## 使用

四轮选择和固定方案评估：

```bash
python scripts/refine_supervised.py --stage batch --device cuda --output runs/repair-rounds
```

只在某项数据上运行同一协议：

```bash
python scripts/refine_supervised.py --stage batch --datasets ulb --device cuda --output runs/repair-ulb
```

单独训练 hybrid 候选（参数须用自己的验证集选择）：

```bash
python -m supervised train --data-dir prepared_runs/ulb --backbone NeoQuasar/Kronos-mini \
  --head-type hybrid --epochs 12 --lr 0.001 --defer-test --output runs/ulb-hybrid
```

新代码可读取原有 MLP／multiscale 检查点。hybrid 检查点需使用包含新头定义的版本读取。旧默认方法暂不无条件替换；推荐通过验证流程选择数据集对应的设置。

IEEE-CIS 原始数据与派生矩阵始终留在本机，云端任务不接收它们。本机 MPS 兼容性检查通过后，四组私有对照统一使用 MPS；部分 CPU 诊断检查点被保留但不混进四组选择。GPU 与 MPS 的数值和随机实现可能不同，局部差异不能直接归因于算法。
