"""Curated method-level scoreboard; do not confuse coverage with exhaustive search."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Exact completed recipes, not the best of an undefined model-family search.
RECIPES = [
    ('CatBoost + spatial + fixed seeds', '019', 'v44_seed_ensemble', 'engineered + spatial', 'current strongest pipeline'),
    ('MLP', '005', 'v22_mlp', 'historical MLP preprocessing', 'one historical recipe'),
    ('TabM', '016', 'v41_TabM', 'compact13 + missing flags', 'one trained recipe'),
    ('TabPFN frozen, 8 views', '008', 'TabPFN_v2_compact8', 'compact13', 'pretrained inference'),
    ('TabPFN frozen, 32 views', '033', 'v65_TabPFN_default32', 'compact13', 'larger inference budget'),
    ('TabPFN weight fine-tuning', '021', 'v48_TabPFN_finetuned1', 'compact13', 'one fine-tuning recipe'),
    ('TabICL', '014', 'v37_TabICL_engineered4', 'engineered', 'four-view pretrained inference'),
    ('XGBoost', '022', 'v49_xgboost', 'engineered', 'inner early stopping + full refit'),
    ('LightGBM linear leaves + spatial', '027', 'v56_nested_linear_leaf', 'engineered + spatial', 'nested leaf-type selection'),
    ('SVM', '005', 'v20_svm3', 'historical SVM preprocessing', 'two historical C settings; this is C=3'),
    ('FT-Transformer', '035', 'v67_FTTransformer_default', 'compact13 + missing flags', 'official default architecture; one initial recipe'),
    ('TabR-S', '036', 'v68_TabR_S', 'compact13 + missing flags', 'official core model with local exact retrieval backend'),
    ('SCARF matched scratch MLP', '037', 'v69_matched_scratch_MLP', 'compact13 + missing flags', 'mechanism-matched control'),
    ('SCARF pretraining + fine-tuning', '037', 'v70_SCARF', 'compact13 + missing flags', 'unlabeled contrastive pretraining'),
    ('Mitra-v2 pretrained inference', '038', 'v71_Mitra2_zeroshot', 'compact13', 'new synthetic pretraining generation; task-specific adaptation not evaluated'),
]


def main():
    rows = []
    best_path = ROOT / 'reports/run_019/screen_results.csv'
    best = next(float(r['accuracy']) for r in csv.DictReader(best_path.open())
                if r['version'] == 'v44_seed_ensemble')
    for method, run, version, inputs, scope in RECIPES:
        path = ROOT / f'reports/run_{run}/screen_results.csv'
        result = None
        if path.exists():
            result = next((r for r in csv.DictReader(path.open()) if r['version'] == version), None)
        row = dict(method=method, run=f'run_{run}', version=version, inputs=inputs,
                   scope=scope, status='completed' if result else 'pending',
                   accuracy=None, log_loss=None, seconds=None, gain_vs_v44_pp=None)
        if result:
            row.update({key: float(result[key]) for key in ('accuracy', 'log_loss', 'seconds')})
            row['gain_vs_v44_pp'] = 100 * (row['accuracy'] - best)
        rows.append(row)
    with (ROOT / 'reports/method_comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    payload = dict(
        protocol='Same fixed 6954-row development partition and three outer folds; threshold 0.5.',
        limitation='Different recipes use different input representations and training budgets. '
                   'This compares usable pipelines, not a controlled architecture-only benchmark. '
                   'Many methods were tried once; a failed recipe does not reject the model family. '
                   'Repeated development exploration is not an independent test.',
        selection='First check implementation and convergence. Resource interruption or improving '
                  'curves at the cap is inconclusive. Then compare accuracy, fold consistency, '
                  'log loss and matched controls. Do not expand fusion solely for better log loss. '
                  'Only promising methods proceed to preregistered small searches and seed checks.',
        online_gate='Local accuracy strictly above 83%, stable improvement, then explicit user approval.',
        recipes=rows)
    (ROOT / 'reports/method_comparison.json').write_text(json.dumps(payload, indent=2) + '\n')
    for row in rows:
        score = f"{100 * row['accuracy']:.3f}%" if row['accuracy'] is not None else 'pending'
        print(f"{row['method']}: {score}")


if __name__ == '__main__':
    main()
