---
license: apache-2.0
pipeline_tag: tabular-classification
tags:
- arxiv:2609.04540
---

# Mitra-v2 Classifier

Mitra-v2 classifier is a tabular foundation model that is pre-trained on purely synthetic datasets sampled from a mix of random classifiers, including the new Hybrid SCM prior. It is the second generation of the Mitra classifier ([autogluon/mitra-classifier](https://huggingface.co/autogluon/mitra-classifier)), pre-trained with a 10x longer context, three times as many features, and an improved optimizer. On the TabArena and TALENT benchmarks it delivers state-of-the-art accuracy at the level of TabFM and EXAONE Tabular, while surpassing TabPFN-3 by a wide margin. The regression model is at [autogluon/mitra-regressor-2](https://huggingface.co/autogluon/mitra-regressor-2), and the inference and fine-tuning code with our evaluation results is at [autogluon/mitra-finetune](https://huggingface.co/autogluon/mitra-finetune).

## Architecture

Mitra-v2 is based on a 12-layer 2D Transformer of 75.7 M parameters (attention across rows and across columns), pre-trained by incorporating an in-context learning paradigm. The architecture is unchanged from Mitra-v1; the gains come from the scaled-up synthetic pre-training distribution and the optimizer.

## Usage

To use Mitra-v2 classifier, install AutoGluon and the `mitra-finetune` package by running:

```sh
pip install uv
uv pip install "autogluon.tabular[mitra]>=1.6" "tabarena>=0.1.0"
uv pip install git+https://huggingface.co/autogluon/mitra-finetune
```

A minimal example showing how to fine-tune and predict with the Mitra-v2 classifier using the same recipe as our reported results (50-step fine-tuning with 8-fold bagging). The recipe fine-tunes and bags eight copies of the model and requires a CUDA GPU; each `predict_proba` or `predict` call runs one bagged fine-tune:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from huggingface_hub import snapshot_download
from mitra_finetune import MitraFinetune

# Load dataset
wine_data = load_wine()
X = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
y = pd.Series(wine_data.target, name="target")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Download the Mitra-v2 classifier weights
ckpt_dir = snapshot_download("autogluon/mitra-classifier-2")

# Fine-tune and predict
model = MitraFinetune(checkpoint_dir=ckpt_dir, problem_type="classification")
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)
pred = proba.argmax(axis=1)
print("Accuracy:", (pred == y_test.values).mean())
```

A minimal example showing how to perform inference with the Mitra-v2 classifier directly in AutoGluon (the weights are a drop-in replacement for the Mitra-v1 classifier):

```python
from autogluon.tabular import TabularDataset, TabularPredictor

train_data = TabularDataset(pd.concat([X_train, y_train], axis=1))
test_data = TabularDataset(pd.concat([X_test, y_test], axis=1))

mitra_predictor = TabularPredictor(label="target")
mitra_predictor.fit(
    train_data,
    hyperparameters={
        "MITRA": {"hf_model": "autogluon/mitra-classifier-2", "fine_tune": False}
    },
)

mitra_predictor.leaderboard(test_data)
```

Set `"fine_tune": True` to fine-tune inside AutoGluon. Note that AutoGluon's stock defaults differ from the `mitra-finetune` recipe used for the reported benchmark numbers.

## License

This project is licensed under the Apache-2.0 License.

## Reference

[Mitra-v2 Technical Report](https://arxiv.org/abs/2609.04540) (Amazon, 2026), also available on the [Hub](https://huggingface.co/autogluon/mitra-finetune/blob/main/Mitra_v2_Technical_Report.pdf).

```
@article{mitrav2_2026,
  title={{Mitra-v2} Technical Report},
  author={Tao, Yefan and Zhang, Xiyuan and Liu, Xinyi and Han, Boran and Maddix, Danielle and Fang, Haoyang and Han, Zhen and Gai, Jiading and Liu, Xuanqing and Bohlke-Schneider, Michael and Wang, Yuyang (Bernie) and Friedland, Gerald and Mah, Kevan and Lee, Chris and Kong, Chris},
  journal={arXiv preprint arXiv:2609.04540},
  year={2026}
}
```

The original Mitra:

```
@article{zhang2025mitra,
  title={Mitra: Mixed synthetic priors for enhancing tabular foundation models},
  author={Zhang, Xiyuan and Maddix, Danielle C and Yin, Junming and Erickson, Nick and Ansari, Abdul Fatir and Han, Boran and Zhang, Shuai and Akoglu, Leman and Faloutsos, Christos and Mahoney, Michael W and others},
  journal={arXiv preprint arXiv:2510.21204},
  year={2025}
}
```

Amazon Science blog: [Mitra: Mixed synthetic priors for enhancing tabular foundation models](https://www.amazon.science/blog/mitra-mixed-synthetic-priors-for-enhancing-tabular-foundation-models)
