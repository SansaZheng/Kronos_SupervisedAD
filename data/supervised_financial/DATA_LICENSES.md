# 数据来源与许可

下载日期：2026-11-02。根目录的 Kronos MIT 许可不能替代各来源的许可。原文件保留于 `raw/`，本仓库新增的是索引、内容去重、研究划分和加载代码；没有修改来源标签或价格数值。

## Sapienza

- 作者：Massimo La Morgia、Alessandro Mei、Francesco Sassi、Julinda Stefa。
- 论文：[Pump and Dumps in the Bitcoin Era: Real Time Detection of Cryptocurrency Market Manipulations](https://doi.org/10.1109/ICCCN49398.2020.9209660)，ICCCN 2020。
- 来源：[作者仓库固定版本](https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset/tree/d71250d4cb055dde2d415c8cba38a0dcd6eb6e16)。
- 来源许可：MIT，原文见 `raw/sapienza/LICENSE`。本次包含三个完整人工标注特征文件，以及解释字段与标签的作者 README、特征脚本、分类脚本。
- 原发布者对人工标记开始块的说明见该版本 README。保留字段 `gt`，不加入公告周围的新区间。

## TSTR

- 作者：Ahmed Mahrous、Roberto Di Pietro。
- 论文：[TSTR for Financial Fraud: Learning to Detect Manipulation Without Real Data](https://doi.org/10.1145/3768292.3770393)，ICAIF 2025，71–79。
- 数据／代码归档：[Zenodo 17301104，v1.0.0](https://zenodo.org/records/17301104)。原发布包的 MD5 为 `c781ac4dd8d5a21d940ed0c2e3780793`，下载时已核对；SHA-256 另见 `sources.json`。
- 来源许可：冻结的 Zenodo API 元数据中 `metadata.license.id = cc-by-4.0`，见 `raw/tstr/zenodo-record-17301104.json`。许可原文见 `raw/tstr/LICENSE-CC-BY-4.0.txt`，来自 [Creative Commons 官方许可](https://creativecommons.org/licenses/by/4.0/)。
- 保留整个原发布包。默认派生集仅取真实价格目录，新增从 notebook 目录类别映射得到的显式标签、删除完全重复序列、生成研究划分；原始文件不变。原作者不为这些新增处理背书。
- 论文及标签来源解释：[作者论文页面](https://amahrous.com/papers/tstr-financial-fraud/)。

## SD-FMM

- 作者：Yongsheng Dai、Barry Quinn、Fearghal Kearney、Weilong Liu、Ivor Spence、Karen Rafferty、Hui Wang。
- 论文：[Detecting market manipulation with dual-branch self-supervised learning: A unified framework integrating frequency-informed anomaly synthesis and domain-specific features](https://doi.org/10.1016/j.ipm.2026.104961)，Information Processing & Management，2026。
- 来源：[作者仓库固定版本](https://github.com/ys-Dai/SD-FMM/tree/0d1a2158ee849fdc31a127bd049e622174f5c893)。
- 发布仓库整体声明 MIT，原文见 `raw/sd_fmm/LICENSE`。仅下载公开 numeric 数组及作者 README／相关解释代码，不包含论文中未公开的专有原始逐笔数据，不扩大声称获得的授权范围。
- 公开文件数目与论文描述有差异；具体限制见本目录 README。本次没有读取含 Python object 的字典 `.npy`，所有加载均为 `allow_pickle=False`。
