"""Reapply the fixed LimiX-2 CPU passenger recipe with whole-booking support exclusion."""
import argparse
import json
import shutil
import subprocess
from unittest.mock import patch

from limix2_setup import ROOT, sha, write

OUT = ROOT/'artifacts/nov26_grouped_limix_support'
ENGINE = ROOT/'.venv-limix2/bin/python'


def train():
    import numpy as np
    from sklearn.model_selection import train_test_split
    from exaone_validation_data import CATEGORICAL, COMPACT, compact_features
    from full_roster_booking_validation_n23 import data
    import limix2_passenger_trial as frozen

    frozen.check_freeze()
    OUT.mkdir(exist_ok=True)
    # The native worker's recipe verification reads this exact committed protocol.
    shutil.copyfile(frozen.REPORT/'protocol.json', OUT/'protocol.json')
    raw,y,g,fold = data()
    x = compact_features(raw).replace('__MISSING__',np.nan)
    x = x.astype(object).where(x.notna(),None)
    for k in map(int,np.unique(fold)):
        t = np.flatnonzero(fold!=k)
        q = np.flatnonzero(fold==k)
        assert not np.isin(g[t],g[q]).any()
        support,_ = train_test_split(t,train_size=512,stratify=y[t],random_state=20261118+k)
        support = np.sort(support)
        packet = dict(support=x.iloc[support].values.tolist(),query=x.iloc[q].values.tolist(),
                      support_ids=support.tolist(),query_ids=q.tolist(),support_y=y[support].astype(int).tolist(),
                      columns=COMPACT,categorical_indices=[COMPACT.index(c) for c in CATEGORICAL])
        packet_path = OUT/f'fold{k}_packet.json'
        pred_path = OUT/f'fold{k}_prediction.npz'
        if pred_path.exists():
            assert json.loads(packet_path.read_text())==packet
        else:
            write(packet_path,packet)
            with (OUT/f'fold{k}.log').open('w') as log:
                subprocess.run([ENGINE,__file__,'--worker',str(k)],cwd=ROOT,check=True,stdout=log,stderr=subprocess.STDOUT)
        with np.load(pred_path) as z:
            np.testing.assert_array_equal(z['query_ids'],q)
            np.testing.assert_array_equal(z['support_ids'],support)
            assert np.isfinite(z['prob']).all() and z['prob'].shape==q.shape
        meta = json.loads((OUT/f'fold{k}_worker.json').read_text())
        assert meta['packet_sha256']==sha(packet_path) and meta['prediction_sha256']==sha(pred_path)
        print('LimiX-2',k,len(q),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',type=int)
    args=parser.parse_args()
    if args.worker is None:
        train()
    else:
        import limix2_passenger_trial as frozen
        with patch.multiple(frozen,OUT=OUT,REPORT=OUT):
            frozen.worker(args.worker)
