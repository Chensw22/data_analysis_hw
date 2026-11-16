"""Fixed no-update diagnostic of unused-class mass, using training-internal episodes."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

import audit_top5_references as audit
import iterate
import mitra_full_episode as core
import mitra_missing_flags_candidate as runtime

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / 'reports/native_head_diagnostic'
base = core.base


def main(session_file):
    runtime.check_quota(session_file)
    if (REPORT / 'result.json').exists():
        raise RuntimeError('No overwrite or repeat.')
    protocol = json.loads((REPORT / 'protocol.json').read_text())
    runtime.common.require_committed(['diagnose_native_head.py', 'reports/native_head_diagnostic/protocol.json'])
    for name, digest in protocol['source_hashes'].items():
        assert base.sha256(ROOT / name) == digest, name
    env = hashlib.sha256(base.environment_lock().encode()).hexdigest()
    assert env == protocol['environment_sha256']
    base.verify_checkpoint()
    with np.load(ROOT / 'artifacts/run_002/split_indices.npz') as saved:
        dev = np.sort(saved['development']).astype(np.int64)
    assert audit.row_hash(dev) == audit.DEV_HASH
    train, test, y = audit.read_development_only(ROOT / 'data/raw/train.csv', ROOT / 'data/raw/test.csv', dev)
    import pandas as pd
    frame = iterate.build_features(pd.concat([train, test], ignore_index=True), False)[0]
    result = dict(status='incomplete', episodes=[], parameter_updates=0, new_accuracy_computed=False,
                  old_holdout_labels_read=False, interpretation=protocol['interpretation'])
    started = time.monotonic()
    try:
        with runtime.deadline(600):
            runtime.runtime()
            splits = StratifiedKFold(3, shuffle=True, random_state=20261112).split(dev, y)
            for fold, (a, _) in enumerate(splits):
                model = data = labels = logits = None
                seed = 20261112 + fold
                inner, _ = train_test_split(np.arange(len(a)), test_size=.2, stratify=y[a], random_state=seed)
                ids, target = dev[a][inner], y[a][inner]
                base.seed_everything(seed, 'mps')
                prep = base.MitraPreprocessor().fit(frame.iloc[ids], target, seed)
                try:
                    model, _ = core.fresh_model('mps')
                    model.train()  # Same mode as the first training episode; no gradients or updates.
                    data = base.torch.as_tensor(prep.transform(frame.iloc[ids]), device='mps')
                    labels = base.torch.as_tensor(target, device='mps')
                    support, query = core.sample_episode(len(ids), np.random.default_rng(seed))
                    si = base.torch.as_tensor(support, device='mps')
                    qi = base.torch.as_tensor(query, device='mps')
                    with base.torch.no_grad():
                        logits = base.model_forward(model, data[si], labels[si], data[qi])
                        assert logits.shape == (256, 10) and bool(base.torch.isfinite(logits).all())
                        unused = logits.softmax(-1)[:, 2:].sum(-1).cpu().numpy().astype(float)
                        penalty = (base.torch.logsumexp(logits, -1) - base.torch.logsumexp(logits[:, :2], -1)).cpu().numpy().astype(float)
                    assert np.isfinite(unused).all() and np.isfinite(penalty).all()
                    result['episodes'].append(dict(fold=fold, seed=seed, inner_training_rows=len(ids),
                        support_rows=1024, query_rows=256, support_ids_sha256=audit.row_hash(ids[support]),
                        query_ids_sha256=audit.row_hash(ids[query]), unused_mass_mean=float(unused.mean()),
                        unused_mass_quantiles=np.quantile(unused, [0, .5, .95, 1]).tolist(),
                        extra_CE_penalty_mean=float(penalty.mean()), extra_CE_penalty_max=float(penalty.max())))
                finally:
                    del model, data, labels, logits
                    base.release_memory()
                base.check_deadline()
            tiny = all(r['extra_CE_penalty_mean'] < .001 and r['unused_mass_quantiles'][-1] < .01 for r in result['episodes'])
            result.update(status='complete', tiny_by_fixed_heuristic=tiny,
                          decision='do_not_prioritize_binary_objective' if tiny else 'different_objectives_confirmed_no_automatic_training')
    except BaseException as error:
        result.update(status='failed_inconclusive', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        result['seconds'] = time.monotonic() - started
        (REPORT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session-file', type=Path, required=True)
    main(parser.parse_args().session_file)
