# TabSwift 对全体乘客的分组验证

验证覆盖 6,954 名开发乘客。每次把完整的预订留给验证集，同一人分别得到三次折外预测。TabSwift 使用 13 个乘客属性；姓名和乘客编号没有进入模型。表中的纠正与误伤均相对同一划分上的 Mitra 与 v44 等权基线。

| 16 视图配置 | 三套准确率 | 各套纠正 | 各套误伤 | 1,070 名持续错例中各套判对 |
|---|---|---|---|---|
| TabSwift 单独预测 | 81.21% / 81.38% / 80.90% | 114 / 129 / 137 | 153 / 126 / 170 | 55 / 61 / 79 |
| 与基线等权 | 81.45% / 81.28% / 81.09% | 54 / 64 / 75 | 76 / 68 / 95 | 18 / 19 / 37 |
| TabSwift 权重 25% | 81.61% / 81.38% / 81.26% | 25 / 31 / 43 | 36 / 28 / 51 | 5 / 9 / 20 |

单独使用 TabSwift 在三次验证里分别净少判对 39 人、多判对 3 人、少判对 33 人。在 466 名八类旧模型都持续答错的乘客中，它分别判对了 2、1、1 人。这个结果不支持将 TabSwift 作为针对广泛共同错判的纠正器。

在第一套划分中还单独验证了模型的中间表示。每个外层训练折再按预订平分，一半乘客供神经网络参考，另一半用于训练读取中间表示的逻辑回归，再交换两半重复训练。新分类器从 1,070 名持续错例中判对 80 人，和强基线混合后判对 8 人，全体净少判对 12 人。同样流程打乱中间表示后与强基线混合，净少判对 9 人。该项只跑了一套划分。[中间表示实验](../nov23_tabswift_readout/result.json)

复现使用作者的固定代码与权重，校验值见各次结果。[作者代码](https://github.com/LAMDA-Tabular/TabSwift)

```sh
.venv/bin/python setup_tabswift_n23.py source
.venv/bin/python setup_tabswift_n23.py weights
uv pip install --python .venv-tabpfn/bin/python --target artifacts/nov23_tabswift/sklearn16 --no-deps scikit-learn==1.6.1
for seed in 20261112 20261115 20261116; do
    .venv-tabpfn/bin/python grouped_tabswift_n23.py --seed "$seed" --views 16
done
.venv-tabpfn/bin/python nested_tabswift_readout_n23.py
```
