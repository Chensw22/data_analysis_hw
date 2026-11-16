"""Recompute completed seed-repeat metrics without reading held-out labels."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import audit_top5_references as audit


def main():
    directory = ROOT / 'artifacts/run_051'
    protocol = json.loads((directory / 'protocol.json').read_text())
    expected = json.loads((directory / 'assessment.json').read_text())
    for group in ['sources', 'evidence_files']:
        for path, digest in protocol[group].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    names = ['v81_Mitra_full_episode50', 'v87_Mitra_full_episode50_seed100',
             'v88_Mitra_full_episode50_two_seed_mean']
    arrays = [np.load(directory / f'{name}_seed20261112.npz') for name in names]
    dev, folds = arrays[0]['dev_ids'], arrays[0]['fold_id']
    _, _, y = audit.read_development_only(ROOT / 'data/raw/train.csv', ROOT / 'data/raw/test.csv', dev)
    assert len(dev) == 6954 and np.array_equal(np.unique(folds), [0, 1, 2])
    for data in arrays:
        assert np.array_equal(data['dev_ids'], dev) and np.array_equal(data['fold_id'], folds)
    old, solo, mean = [data['prob'] for data in arrays]
    assert np.array_equal(mean, (old.astype(float) + solo.astype(float)) / 2)
    assert np.array_equal(old, np.load(ROOT / 'artifacts/run_046/v81_Mitra_full_episode50_seed20261112.npz')['prob'])
    metrics = {}
    for name, p in zip(names, [old, solo, mean]):
        assert p.shape == y.shape and np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
        item = dict(accuracy=float(np.mean((p >= .5) == y)), log_loss=float(log_loss(y, p, labels=[0, 1])),
                    fold_accuracy=[float(np.mean((p[folds == i] >= .5) == y[folds == i])) for i in range(3)])
        for key in item:
            assert np.allclose(item[key], expected['metrics'][name][key], rtol=0, atol=1e-12)
        metrics[name] = item
    corrected = int(np.sum(((mean >= .5) == y) & ((old >= .5) != y)))
    broken = int(np.sum(((mean >= .5) != y) & ((old >= .5) == y)))
    gate = (metrics[names[2]]['accuracy'] >= metrics[names[0]]['accuracy'] - 1e-12
            and metrics[names[2]]['log_loss'] <= metrics[names[0]]['log_loss'] + 1e-12
            and sum(a >= b - 1e-12 for a, b in zip(metrics[names[2]]['fold_accuracy'], metrics[names[0]]['fold_accuracy'])) >= 2)
    assert gate == expected['continuation']['passed']
    result = dict(status='verified', rows=len(y), metrics=metrics, corrected=corrected, broken=broken,
                  fixed_continuation_passed=bool(gate), independent_holdout=False, S_supported=False)
    (ROOT / 'reports/run_051/paired_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
