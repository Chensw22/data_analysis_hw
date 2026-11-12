---
license: apache-2.0
pipeline_tag: tabular-classification
---

# Mitra Classifier

Mitra classifier is a tabular foundation model that is pre-trained on purely synthetic datasets sampled from a mix of random classifiers. 

## Architecture

Mitra is based on a 12-layer Transformer of 72 M parameters, pre-trained by incorporating an in-context learning paradigm.

## Usage

To use Mitra classifier, install AutoGluon by running:

```sh
pip install uv
uv pip install autogluon.tabular[mitra]   
```

A minimal example showing how to perform inference using the Mitra classifier:

```python
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

# Load datasets
wine_data = load_wine()
wine_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
wine_df['target'] = wine_data.target

print("Dataset shapes:")
print(f"Wine: {wine_df.shape}")

# Create train/test splits (80/20)
wine_train, wine_test = train_test_split(wine_df, test_size=0.2, random_state=42, stratify=wine_df['target'])

print("Training set sizes:")
print(f"Wine: {len(wine_train)} samples")

# Convert to TabularDataset
wine_train_data = TabularDataset(wine_train)
wine_test_data = TabularDataset(wine_test)

# Create predictor with Mitra
print("Training Mitra classifier on classification dataset...")
mitra_predictor = TabularPredictor(label='target')
mitra_predictor.fit(
    wine_train_data,
    hyperparameters={
        'MITRA': {'fine_tune': False}
    },
   )

print("\nMitra training completed!")

# Make predictions
mitra_predictions = mitra_predictor.predict(wine_test_data)
print("Sample Mitra predictions:")
print(mitra_predictions.head(10))

# Show prediction probabilities for first few samples
mitra_predictions = mitra_predictor.predict_proba(wine_test_data)
print(mitra_predictions.head())

# Show model leaderboard
print("\nMitra Model Leaderboard:")
mitra_predictor.leaderboard(wine_test_data)
```

A minimal example showing how to perform fine-tuning using the Mitra classifier:

```python
mitra_predictor_ft = TabularPredictor(label='target')
mitra_predictor_ft.fit(
    wine_train_data,
    hyperparameters={
        'MITRA': {'fine_tune': True, 'fine_tune_steps': 10}
    },
    time_limit=120,  # 2 minutes
   )

print("\nMitra fine-tuning completed!")

# Show model leaderboard
print("\nMitra Model Leaderboard:")
mitra_predictor_ft.leaderboard(wine_test_data)
```

## License

This project is licensed under the Apache-2.0 License.

## Reference

```
@article{zhang2025mitra,
  title={Mitra: Mixed synthetic priors for enhancing tabular foundation models},
  author={Zhang, Xiyuan and Maddix, Danielle C and Yin, Junming and Erickson, Nick and Ansari, Abdul Fatir and Han, Boran and Zhang, Shuai and Akoglu, Leman and Faloutsos, Christos and Mahoney, Michael W and others},
  journal={arXiv preprint arXiv:2510.21204},
  year={2025}
}
```

Amazon Science blog: [Mitra: Mixed synthetic priors for enhancing tabular foundation models](https://www.amazon.science/blog/mitra-mixed-synthetic-priors-for-enhancing-tabular-foundation-models?utm_campaign=mitra-mixed-synthetic-priors-for-enhancing-tabular-foundation-models&utm_medium=organic-asw&utm_source=linkedin&utm_content=2025-7-22-mitra-mixed-synthetic-priors-for-enhancing-tabular-foundation-models&utm_term=2025-july)