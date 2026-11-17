"""Descriptive, authenticated historical OOF error analysis; no model fitting."""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

import audit_top5_references as audit
import exaone_validation_data as data
import three_family_mean_screen as common

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'reports/data_error_diagnosis_20261117'
LOCAL = ROOT / 'artifacts/data_error_diagnosis_20261117'
FAMILIES = ['v44', 'v81', 'TabDPT', 'Kumo', 'EXAONE']


def load_authenticated():
    inputs = data.load_data()
    dev, y = inputs['dev'], inputs['y']
    splits, refs, evidence = data.random_references(inputs)
    evidence.update(inputs['evidence'])
    probabilities = {short: refs[name] for short, name in [
        ('v44', 'v44_seed_ensemble'), ('rawMitra', 'v71_Mitra2_zeroshot'),
        ('v81', 'v81_Mitra_full_episode50')]}
    with np.load(ROOT/'artifacts/run_038/v6_no_id_features_seed20261112.npz') as z:
        baseline = z['prob']
    for run, name, kind, short in [
        ('run_047', 'v82_TabDPT13_full8', 'tabdpt', 'TabDPT'),
        ('run_048', 'v83_Kumo_medium8', 'kumo', 'Kumo')]:
        p, _, files = common.read_component(run, name, kind, dev, y, splits, baseline)
        probabilities[short] = p
        evidence.update(files)
    fold_ids = np.full(len(dev), -1)
    for fold, (_, b) in enumerate(splits):
        fold_ids[b] = fold
    for run, name, short in [('run_054', 'EXAONE_v91_CAST_default8', 'EXAONE'),
                             ('run_056', 'v93_Mitra_support3072', 'v93')]:
        folder = ROOT/'artifacts'/run
        protocol = json.loads(data.report_pair(run, 'protocol.json', evidence).read_text())
        assert [protocol[k] for k in ['seed', 'folds', 'development_rows', 'threshold']] == [20261112, 3, 6954, .5]
        data.report_pair(run, 'screen_results.csv', evidence)
        path = folder/f'{name}_seed20261112.npz'
        with np.load(path, allow_pickle=False) as z:
            assert np.array_equal(z['dev_ids'], dev) and np.array_equal(z['fold_id'], fold_ids)
            p = z['prob'].copy()
            common.cached_fold_scores(p, z['fold_accuracy'], y, splits)
        evidence[str(path.relative_to(ROOT))] = audit.file_hash(path)
        for fold, (a, b) in enumerate(splits):
            path = folder/f'fold{fold}_rows.npz'
            with np.load(path) as z:
                assert np.array_equal(z['train_ids'], dev[a]) and np.array_equal(z['validation_ids'], dev[b])
            evidence[str(path.relative_to(ROOT))] = audit.file_hash(path)
            row = json.loads(data.report_pair(run, f'fold{fold}.json', evidence).read_text())
            assert row['status'] == 'complete'
            assert row['train_ids_sha256'] == audit.row_hash(dev[a])
            assert row['validation_ids_sha256'] == audit.row_hash(dev[b])
        saved = pd.read_csv(folder/'screen_results.csv').set_index('version').loc[name]
        assert abs(saved.accuracy-accuracy_score(y, p >= .5)) < 1e-12
        assert abs(saved.log_loss-log_loss(y, p)) < 1e-7
        probabilities[short] = p
    return inputs, splits, fold_ids, probabilities, evidence


def slices(frame, roster, probabilities):
    spending = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    missing = frame.drop(columns=['PassengerId', 'Name']).isna().sum(axis=1)
    result = {'missing_features': pd.cut(missing, [-1, 0, 1, 30], labels=['0', '1', '2+'])}
    for col in ['CryoSleep', 'HomePlanet', 'Destination', 'VIP']:
        result[col] = frame[col].fillna('__MISSING__').astype(str)
    result['Deck'] = frame.Cabin.str.split('/').str[0].fillna('__MISSING__')
    result['Age'] = pd.cut(frame.Age, [-np.inf, 12, 18, 30, 50, np.inf],
                           labels=['<=12', '13-18', '19-30', '31-50', '>50']).astype(str).replace('nan', '__MISSING__')
    bills = frame[spending]
    result['spending'] = pd.Series(np.where(bills.isna().any(axis=1), 'any_missing',
        np.where(bills.sum(axis=1) == 0, 'complete_zero', 'complete_positive')), index=frame.index)
    all_groups = roster.PassengerId.str.split('_').str[0]
    sizes = frame.PassengerId.str.split('_').str[0].map(all_groups.value_counts())
    result['roster_group_size'] = pd.cut(sizes, [0, 1, 2, 4, 100], labels=['1', '2', '3-4', '5+'])
    p = probabilities['v81']
    result['v81_confidence'] = pd.Series(pd.cut(np.maximum(p, 1-p), [.499999, .6, .8, .95, 1.000001],
        labels=['0.5-0.6', '0.6-0.8', '0.8-0.95', '0.95-1']), index=frame.index)
    return result


def main():
    OUT.mkdir(exist_ok=True); LOCAL.mkdir(parents=True, exist_ok=True)
    common.require_committed(['error_overlap_diagnostic.py', 'reports/data_error_diagnosis_20261117/protocol.json'])
    inputs, splits, folds, probabilities, evidence = load_authenticated()
    y = inputs['y']; names = list(probabilities)
    p = np.column_stack([probabilities[n] for n in names])
    wrong = (p >= .5) != y[:, None]
    family_wrong = np.column_stack([wrong[:, names.index(n)] for n in FAMILIES])
    counts = family_wrong.sum(axis=1)
    unanimous = (counts == 0) | (counts == len(FAMILIES))
    metrics = []
    for i, name in enumerate(names):
        metrics.append(dict(model=name, rows=len(y), errors=int(wrong[:, i].sum()),
            accuracy=float((~wrong[:, i]).mean()), log_loss=float(log_loss(y, p[:, i])),
            error_share_common_all_five=float(np.mean(counts[wrong[:, i]] == 5)),
            fold_accuracy=[float((~wrong[b, i]).mean()) for _, b in splits]))
    pairs = []
    for i, j in combinations(range(len(names)), 2):
        a, b = wrong[:, i], wrong[:, j]
        pairs.append(dict(left=names[i], right=names[j], both_wrong=int((a & b).sum()),
            either_wrong=int((a | b).sum()), error_jaccard=float((a & b).sum()/(a | b).sum()),
            error_correlation=float(np.corrcoef(a, b)[0, 1]),
            left_errors_repaired_by_right=int((a & ~b).sum()),
            right_errors_repaired_by_left=int((b & ~a).sum()),
            prediction_disagreements=int((a != b).sum()),
            truth_aware_oracle_accuracy=float(1-(a & b).mean())))
    train, test, _ = audit.read_development_only(ROOT/'data/raw/train.csv', ROOT/'data/raw/test.csv', inputs['dev'])
    frame = train.iloc[inputs['dev']].reset_index(drop=True)
    slice_rows = []
    for feature, values in slices(frame, pd.concat([train, test], ignore_index=True), probabilities).items():
        values = values.astype(str).fillna('__MISSING__')
        for value in sorted(values.unique()):
            mask = (values == value).to_numpy(); n = int(mask.sum())
            row = dict(feature=feature, value=value, rows=n, positive_rate=float(y[mask].mean()),
                all_five_wrong=int((counts[mask] == 5).sum()), all_five_wrong_rate=float((counts[mask] == 5).mean()),
                disagreement_rate=float((~unanimous[mask]).mean()))
            row.update({f'{name}_accuracy': float((~wrong[mask, i]).mean()) for i, name in enumerate(names)})
            slice_rows.append(row)
    result = dict(kind='adaptive_descriptive_diagnosis_not_candidate_validation', rows=len(y),
        development_ids_sha256=audit.row_hash(inputs['dev']), models=metrics,
        representatives=FAMILIES, family_error_count_histogram={str(i): int((counts == i).sum()) for i in range(6)},
        consensus=dict(unanimous_rows=int(unanimous.sum()), all_correct=int((counts == 0).sum()),
            all_wrong=int((counts == 5).sum()), disagreement_rows=int((~unanimous).sum()),
            unanimous_accuracy=float((counts[unanimous] == 0).mean()),
            truth_aware_any_family_oracle_accuracy=float(1-(counts == 5).mean()),
            truth_aware_any_seven_oracle_accuracy=float(1-wrong.all(axis=1).mean()),
            v81_disagreement_accuracy=float((~wrong[~unanimous, names.index('v81')]).mean())),
        interpretation=['All-wrong is shared failure of these selected recipes, not irreducible error or mislabeled data.',
            'Oracle uses true labels to choose a correct model per person: an unattainable diagnostic, not an ensemble result.',
            'Family representatives fixed before analysis; extra Mitra variants are sensitivity comparisons, not independent votes.',
            'Slices are descriptive and reused; choosing a rule here requires a separately frozen nested experiment.',
            'No old1739holdout labels, test labels, threshold search, fitted router, or new submission.'],
        evidence_sha256=evidence, source_sha256=audit.file_hash(Path(__file__)))
    audit.json_write(OUT/'error_overlap.json', result)
    pd.DataFrame(pairs).to_csv(OUT/'pairwise_errors.csv', index=False)
    pd.DataFrame(slice_rows).to_csv(OUT/'error_slices.csv', index=False)
    # Raw labels, IDs, probabilities remain in ignored local artifacts.
    rows = pd.DataFrame({'row_id': inputs['dev'], 'PassengerId': inputs['passengers'], 'fold': folds, 'y': y, 'family_wrong_count': counts})
    for i, name in enumerate(names): rows[f'p_{name}'] = p[:, i]
    rows.to_csv(LOCAL/'aligned_rows.csv', index=False)
    draw_plot(names, wrong, counts)
    print(json.dumps({k: result[k] for k in ['rows', 'models', 'family_error_count_histogram', 'consensus']}, indent=2))


def draw_plot(names, wrong, counts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matrix = np.eye(len(names))
    for i, j in combinations(range(len(names)), 2):
        matrix[i, j] = matrix[j, i] = (wrong[:, i] & wrong[:, j]).sum()/(wrong[:, i] | wrong[:, j]).sum()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    axes[0].imshow(matrix, vmin=0, vmax=1, cmap='Blues')
    axes[0].set_xticks(range(len(names)), names, rotation=45, ha='right')
    axes[0].set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            axes[0].text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', color='white' if matrix[i,j] > .65 else 'black')
    axes[0].set_title('Error overlap (Jaccard)\nSame 6,954 OOF passengers')
    heights = np.bincount(counts, minlength=6)
    axes[1].bar(range(6), heights, color=['#2a9d8f']+['#e9c46a']*4+['#e76f51'])
    for i, height in enumerate(heights): axes[1].text(i, height+35, str(height), ha='center')
    axes[1].set(xlabel='Number of five model families wrong', ylabel='Passengers', xticks=range(6), title='Shared failures vs model disagreements')
    fig.savefig(OUT/'error_overlap.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__': main()
