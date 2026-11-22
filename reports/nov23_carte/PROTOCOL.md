# CARTE 行图分类器

直接使用 carte-ai 0.0.26 官方 wheel 内的 Table2GraphTransformer 和 CARTEClassifier，单成员、单层、默认冻结设置、CPU、批量 96、最多 14 个 epoch，基于验证 AUROC 早停，耐心为 3。每个外层折只在其他订票组上拟合数值变换和模型；内部早停使用作者分类器自带的划分，模型输入不包含姓名、标识和封存标签。信用核验中，同一初始化下分别启用和禁用权重加载，15 个状态张量完全匹配供应方权重且区别于随机初始化，共覆盖 542,700 / 690,601 个元素。[原生图转换代码](https://github.com/soda-inria/carte/blob/main/carte_ai/src/carte_table_to_graph.py) · [权重记录](weights.json)

空间表格对每个乘客形成一张星形图。每个字段名作为边的特征，类别值使用词向量，数值乘以字段名向量。原生程序要求 300 维英语 fastText 二进制。官方文件支持按字节读取，程序解码原词典，使用原生 C++ 库给出字词和子词行号，只下载这些行，再逐词重建。训练和测试所需字段名和类别值均被覆盖，合计读取 37,676,695 字节，原文件 7,237,176,312 字节。校验覆盖 35 个词在原生模型中的第一坐标以及所有 31 个字段名/类别名称的 300 维归一化句向量。临时生成的一维索引词典在核验后删除。紧凑模型仅用于这张表的词汇，不用于任意文本。[精确读取记录](words.json)

CARTE 官方代码采用 BSD-3-Clause，预训练图模型随该项目的官方 PyPI 包分发。英语向量来自 Meta 的固定版本，采用 CC BY-SA 3.0。运行中的紧凑词向量由该来源抽取重组；这里只发布程序和汇总，生成的二进制向量保留在本地。[CARTE 项目许可](https://github.com/soda-inria/carte/blob/main/LICENSE.txt) · [Meta 英语词向量及许可](https://huggingface.co/facebook/fasttext-en-vectors/tree/a80392390daaee1a91000da45c376d512e1dc555)

运行时用 .venv-iltm 的 PyTorch、pandas 和 sklearn，加上项目专属路径中的 carte-ai、fasttext-wheel、torch-geometric、torcheval、xxhash 及 psutil。各版本见权重记录。可用 .venv-tabpfn/bin/python carte_sparse_vectors_n23.py 重建稀疏向量，再用 .venv-iltm/bin/python carte_passengers_n23.py 复现主结果。--scratch 提供相同结构和优化预算的随机初始化对照。

原始配置的逐轮进度保留了完成轮数，未记录每轮验证指标。另用相同种子复跑首套，观察器只读取作者验证函数返回的负 AUROC；三折的最佳轮次为 14、14、13，第 14 轮相对第 10 轮 AUROC 分别提升 0.0076、0.0104、0.0054。复跑总体准确率 76.89%，原始为 76.82%，单行概率不完全一致。因此这些曲线是独立复跑证据。触发一套加长检验，使用原生默认上限 500 轮和耐心 40，始终保留同一次训练前 14 轮验证最优模型的折外概率，以配对检验训练长度。[第 1 折曲线](20261112_0_trace_trajectory.json) · [第 2 折曲线](20261112_1_trace_trajectory.json) · [第 3 折曲线](20261112_2_trace_trajectory.json)

使用 `.venv-iltm/bin/python carte_passengers_n23.py --extended 20261112` 复现加长训练及同轨迹比较。三个外层折的实际训练和查询行号、概率、权重散列和耗时都在本地对应 NPZ 中，每折的负 AUROC 曲线随报告保存。[首套训练长度与折外准确率比较](20261112_extended_comparison.json)
