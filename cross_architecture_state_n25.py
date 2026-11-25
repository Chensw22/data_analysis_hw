"""Evaluate a second frozen transformer on whole-booking folds with matched global and awake contexts."""
import numpy as np
from exaone_validation_data import compact_features
from full_roster_booking_validation_n23 import ROOT,data

OUT=ROOT/'artifacts/nov25_cross_architecture_state'


def run():
    import torch
    import tabdpt_adapter as adapter
    import tabdpt_resource_probe as backend
    assert torch.backends.mps.is_available()
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.65)
    adapter.verify_sources()
    raw,y,g,f=data()
    x=compact_features(raw)
    awake=raw.CryoSleep.eq(False).to_numpy()
    OUT.mkdir(exist_ok=True)
    for k in range(8):
        for name,scope in [('global',np.ones(len(y),bool)),('awake',awake)]:
            t,q=np.flatnonzero((f!=k)&scope),np.flatnonzero((f==k)&scope)
            assert not np.isin(g[t],g[q]).any()
            path=OUT/f'{k}_{name}.npz'
            if not path.exists():
                enc=backend.CompactEncoder().fit(x.iloc[t])
                clf=backend.new_model('mps')
                clf.fit(enc.transform(x.iloc[t]),y[t].astype(int))
                pred=backend.probabilities(clf,enc.transform(x.iloc[q]),ensembles=2,seed=891+k)
                del clf
                backend.release(None,'mps')
                with path.with_suffix('.tmp').open('wb') as s:
                    np.savez_compressed(s,t=t,q=q,training_labels=y[t],passenger=raw.PassengerId.iloc[q].to_numpy(str),
                                        checkpoint=backend.WEIGHT_SHA,seed=891+k,prob=pred[:,1])
                path.with_suffix('.tmp').replace(path)
            with np.load(path) as z:
                for key,val in [('q',q),('t',t),('training_labels',y[t]),('passenger',raw.PassengerId.iloc[q]),
                                ('checkpoint',backend.WEIGHT_SHA),('seed',891+k)]:
                    np.testing.assert_array_equal(z[key],val)
            print(k,name,flush=True)


if __name__=='__main__':
    run()
