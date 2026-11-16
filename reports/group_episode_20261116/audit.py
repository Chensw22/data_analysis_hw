"""Recompute complete paired group OOF metrics, parsing development targets only."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
import audit_top5_references as audit
import mitra_group_episode_candidate as candidate


def run():
    out = ROOT/'artifacts/group_episode_20261116'
    summary = json.loads((out/'summary.json').read_text())
    assert summary['status'] == 'complete' and len(summary['folds']) == 3
    assert audit.file_hash(out/'group_20261112.npz') == summary['oof_sha256']
    protocol = json.loads((out/'protocol.json').read_text())
    candidate.assert_frozen(protocol)
    with np.load(out/'group_20261112.npz', allow_pickle=False) as data:
        p, reference, dev, folds = (data[k] for k in ('prob', 'reference_prob', 'dev_ids', 'fold_id'))
    assert len(dev) == 6954 and audit.row_hash(dev) == audit.DEV_HASH
    _, _, y = audit.read_development_only(ROOT/'data/raw/train.csv', ROOT/'data/raw/test.csv', dev)
    assert audit.row_hash(y) == protocol['data']['targets_sha256']

    def paired(indices):
        a, b, target = p[indices], reference[indices], y[indices]
        correct, old_correct = (a >= .5) == target, (b >= .5) == target
        return dict(rows=len(indices), accuracy=float(accuracy_score(target, a >= .5)),
            reference_accuracy=float(accuracy_score(target, b >= .5)),
            log_loss=float(log_loss(target, a, labels=[0, 1])),
            reference_log_loss=float(log_loss(target, b, labels=[0, 1])),
            corrected=int(np.sum(correct & ~old_correct)), broken=int(np.sum(~correct & old_correct)),
            disagreements=int(np.sum(correct != old_correct)))

    whole = paired(np.arange(len(y)))
    by_fold = [paired(np.flatnonzero(folds == i)) for i in range(3)]
    for row, recorded in [(whole, summary), *zip(by_fold, summary['folds'])]:
        for key in ('accuracy', 'reference_accuracy', 'log_loss', 'reference_log_loss'):
            assert abs(row[key] - recorded[key]) < 1e-7, key
    result = dict(status='root_reproduced_complete_paired_metrics', whole=whole, folds=by_fold,
        accuracy_gain=whole['accuracy']-whole['reference_accuracy'],
        log_loss_gain=whole['log_loss']-whole['reference_log_loss'],
        same_practical_1pp_band=abs(whole['accuracy']-whole['reference_accuracy']) <= .01,
        S_supported=False, no_old_holdout_targets_read=True,
        interpretation='Paired reused group-development evidence only; no direct comparison with random-CV v81 and no stable-superiority claim.',
        evidence_sha256={str(path.relative_to(ROOT)):audit.file_hash(path) for path in
            [Path(__file__), out/'protocol.json', out/'summary.json', out/'group_20261112.npz']})
    (Path(__file__).parent/'paired_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    run()
