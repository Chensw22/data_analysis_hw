"""Compare disjoint awake passenger contexts and teach TabICL which reference decisions fail."""
import hashlib
import numpy as np
import pandas as pd

from awake_context_ranks_n25 import OUT as INITIAL
from exaone_validation_data import compact_features
from full_roster_booking_validation_n23 import ROOT, data
from grouped_peer_teacher_n25 import OUT as REFERENCE

OUT = ROOT/'artifacts/nov25_awake_error_demonstrations'
MODES = ('first_half','second_half','outcome_with_score','error','balanced_error')


def contexts(t, group, rng):
    orders = rng.permutation(np.unique(group[t]))
    a = np.isin(group[t], orders[::2])
    return t[a], t[~a]


def error_context(t, wrong, side, rng):
    # Include every error. Draw correct examples separately for each predicted class.
    use = []
    fractions = []
    for s in (False,True):
        wrong_ids = t[(side[t]==s)&wrong[t]]
        right = t[(side[t]==s)&~wrong[t]]
        pick = rng.choice(right,size=min(len(right),len(wrong_ids)),replace=False)
        assert len(pick) and len(wrong_ids)
        use.extend([wrong_ids,pick])
        fractions.append(len(pick)/len(right))
    return np.sort(np.concatenate(use)),np.array(fractions)


def run():
    import torch
    from tabicl_routed_context_n23 import DIGEST,WEIGHT,infer
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    torch.set_num_threads(2)
    raw,y,group,fold = data()
    # Use the same feature schema and row values as the original full-context candidate.
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=lambda c:c!='Name')
    x=compact_features(pd.concat([raw,test],ignore_index=True)).iloc[:len(y)].copy()
    awake=raw.CryoSleep.eq(False).to_numpy()
    OUT.mkdir(exist_ok=True)
    for k in np.unique(fold):
        q=np.flatnonzero((fold==k)&awake)
        t=np.flatnonzero((fold!=k)&awake)
        with np.load(INITIAL/f'{k}_personal.npz') as z:
            for key,val in [('q',q),('t',t),('train_y',y[t]),('fields',x.columns),('checkpoint',DIGEST)]:
                np.testing.assert_array_equal(z[key],val)
        probability=np.empty(len(y))
        with np.load(REFERENCE/f'outer{k}.npz') as z:
            np.testing.assert_array_equal(z['t'],np.flatnonzero(fold!=k))
            np.testing.assert_array_equal(z['labels'],y[z['t']])
            probability[z['t']]=z['p']
        with np.load(REFERENCE/f'release{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],np.flatnonzero(fold==k))
            probability[z['q']]=z['native']
        side=probability>=.5
        wrong=y!=side
        # The reference probabilities for context rows come from inner held-out raw teachers.
        # The deployed query reference is the fine-tuned native model used in the incumbent.
        frame=x.assign(reference_probability=probability,reference_class=side.astype(float))
        a,b=contexts(t,group,np.random.default_rng(20261125+k))
        assert not np.isin(group[a],group[b]).any()
        np.testing.assert_array_equal(np.sort(np.r_[a,b]),t)
        balanced,fractions=error_context(t,wrong,side,np.random.default_rng(942+k))
        configs=[('first_half',x,a,y),('second_half',x,b,y),('outcome_with_score',frame,t,y),
                 ('error',frame,t,wrong),('balanced_error',frame,balanced,wrong)]
        for name,features,context,target in configs:
            assert not np.isin(group[context],group[q]).any()
            path=OUT/f'{k}_{name}.npz'
            if not path.exists():
                p=infer(features,target,context,q,891+k)
                assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                with path.with_suffix('.tmp').open('wb') as stream:
                    np.savez_compressed(stream,t=context,q=q,context_y=target[context],p=p,
                        checkpoint=DIGEST,seed=891+k,training_correct_fraction=fractions,
                        query_side=side[q],passenger=raw.PassengerId.iloc[q].to_numpy(str))
                path.with_suffix('.tmp').replace(path)
            with np.load(path) as z:
                for key,val in [('t',context),('q',q),('context_y',target[context]),('checkpoint',DIGEST),
                                ('seed',891+k),('training_correct_fraction',fractions),('query_side',side[q]),
                                ('passenger',raw.PassengerId.iloc[q])]:
                    np.testing.assert_array_equal(z[key],val)
            print(k,name,flush=True)


if __name__=='__main__':
    run()
