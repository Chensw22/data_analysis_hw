"""Descriptive paired audit of frozen OOF predictions; never select a recipe."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

import audit_top5_references as audit

ROOT = Path(__file__).resolve().parents[2]


def main():
    dev = np.sort(np.load(ROOT/'artifacts/run_002/split_indices.npz')['development'])
    assert len(dev) == 6954 and audit.row_hash(dev) == audit.DEV_HASH
    train, _, y = audit.read_development_only(
        ROOT/'data/raw/train.csv', ROOT/'data/raw/test.csv', dev)
    paths = {key: ROOT/f'artifacts/run_046/{name}_seed20261112.npz'
             for key, name in [('candidate', 'v81_Mitra_full_episode50'),
                               ('reference', 'v71_Mitra2_zeroshot')]}
    p = {key: np.load(path)['prob'] for key, path in paths.items()}
    assert all(v.shape == y.shape and np.isfinite(v).all() for v in p.values())
    correct = {key: (prob >= .5) == y for key, prob in p.items()}
    fixed = int(np.sum(correct['candidate'] & ~correct['reference']))
    broken = int(np.sum(~correct['candidate'] & correct['reference']))
    scores = {key: dict(accuracy=accuracy_score(y, prob >= .5),
                        log_loss=log_loss(y, prob), brier=brier_score_loss(y, prob))
              for key, prob in p.items()}
    folds = []
    for index, (_, query) in enumerate(StratifiedKFold(
            3, shuffle=True, random_state=20261112).split(dev, y), 1):
        folds.append(dict(fold=index, **{key: dict(
            accuracy=accuracy_score(y[query], prob[query] >= .5),
            log_loss=log_loss(y[query], prob[query])) for key, prob in p.items()}))

    # Whole-group resampling is descriptive for these already reused folds.
    _, group = np.unique(train.PassengerId.iloc[dev].str.split('_').str[0],
                         return_inverse=True)
    counts = np.bincount(group)
    delta_acc = correct['candidate'].astype(float) - correct['reference'].astype(float)
    def losses(prob):
        prob = np.clip(prob, 1e-15, 1-1e-15)
        return -(y*np.log(prob) + (1-y)*np.log1p(-prob))
    delta_ll = losses(p['candidate']) - losses(p['reference'])
    group_acc = np.bincount(group, weights=delta_acc)
    group_ll = np.bincount(group, weights=delta_ll)
    rng = np.random.default_rng(20261116)
    boot = np.empty((10000, 2))
    for index in range(len(boot)):
        sampled = rng.integers(len(counts), size=len(counts))
        denominator = counts[sampled].sum()
        boot[index] = group_acc[sampled].sum()/denominator, group_ll[sampled].sum()/denominator
    result = dict(
        scope='Descriptive audit only; reused development, no independent confirmation or model selection.',
        candidate='v81_Mitra_full_episode50', reference='v71_Mitra2_zeroshot',
        rows=len(y), scores=scores, folds=folds, corrected_errors=fixed,
        newly_introduced_errors=broken, net_correct=fixed-broken,
        changed_threshold_decisions=fixed+broken,
        mcnemar_exact_pvalue=binomtest(fixed, fixed+broken, .5).pvalue,
        bootstrap=dict(unit='PassengerId prefix group', seed=20261116, resamples=10000,
            accuracy_delta_pp_interval95=(100*np.quantile(boot[:,0],[.025,.975])).tolist(),
            log_loss_delta_interval95=np.quantile(boot[:,1],[.025,.975]).tolist(),
            limitations='Not corrected for repeated model selection or fold dependence; no stable-superiority claim.'),
        source_hashes={str(path.relative_to(ROOT)): audit.file_hash(path) for path in paths.values()},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (Path(__file__).parent/'paired_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(corrected=fixed, broken=broken, net=fixed-broken,
                         bootstrap=result['bootstrap'])))


if __name__ == '__main__':
    main()
