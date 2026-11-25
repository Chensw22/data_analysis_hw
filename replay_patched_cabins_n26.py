"""Replay the native eight booking folds and the production query-only v106 cabin imputation."""
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd

from full_roster_mitra_validation_n23 import fit, OUT as NATIVE, ROOT, data

OUT=ROOT/'artifacts/nov26_patched_cabins'


def run():
    import torch
    import iterate
    import mitra_counts_fixed15_online_first as counts
    from v108_exact_oof import patch_query
    assert torch.backends.mps.is_available()
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.65)
    counts.verify_reference()
    raw,y,g,fold=data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())
    records=pd.concat([raw,test],ignore_index=True)
    plain,_=iterate.build_features(records,False)
    roster=counts.RosterCounts(records,plain)
    patches=json.loads((ROOT/'artifacts/v108_exact_oof/cabin_updates_all_roster.json').read_text())
    OUT.mkdir(exist_ok=True)
    for k in np.unique(fold):
        t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        assert not np.isin(g[t],g[q]).any()
        selection={str(i):patches[str(i)] for i in q if str(i) in patches}
        path=OUT/f'fold{k}.npz'
        if path.exists():
            continue
        fixed={}
        with np.load(NATIVE/f'mitra96_fold_{k}.npz') as z:
            for key,input_array in [('train',t),('query',q),('training_labels',y[t]),('checkpoint',counts.base.WEIGHT_SHA)]:
                np.testing.assert_array_equal(z[key],input_array)
            cached=z['p']
        class Replay(counts.CountsPredictor):
            def predict_proba(self,x):
                normal=super().predict_proba(x)
                np.testing.assert_array_equal(normal[:,1],cached)
                old=self.preprocessor.roster.plain
                self.preprocessor.roster.plain=old.copy()
                try:
                    ids,query=patch_query(self,selection)
                    fixed['ids']=ids
                    fixed['p']=super().predict_proba(query)[:,1].copy()
                finally:
                    self.preprocessor.roster.plain=old
                return normal
        mod=SimpleNamespace(**{**vars(counts),'CountsPredictor':Replay})
        pred=fit(roster,t,q,y[t],mod)
        np.testing.assert_array_equal(cached,pred)
        assert sorted(fixed['ids'])==sorted(map(int,selection))
        with path.with_suffix('.tmp').open('wb') as stream:
            np.savez_compressed(stream,q=q,t=t,train_y=y[t],native=pred,patch_ids=fixed['ids'],patch_p=fixed['p'])
        path.with_suffix('.tmp').replace(path)
        print('verified cabin fold',int(k),len(fixed['ids']),flush=True)


if __name__=='__main__':
    run()
