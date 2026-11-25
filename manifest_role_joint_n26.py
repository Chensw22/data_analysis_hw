"""Infer a passenger's outcome from a six-class table of booking role and transport status."""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import log_loss

from exaone_validation_data import compact_features
from full_roster_booking_validation_n23 import ROOT, data
from joint_party_tabicl_n26 import model

OUT = ROOT / 'artifacts/nov26_manifest_role_joint'


def roles(raw, g):
    alone = ~pd.Series(g).duplicated(keep=False).to_numpy()
    first = raw.PassengerId.str.endswith('_01').to_numpy()
    assert first[alone].all()
    return np.where(alone, 0, np.where(first, 1, 2))


def role_table(x, train, query, joint_label, seed):
    est = model(seed).fit(x.iloc[train], joint_label)
    np.testing.assert_array_equal(est.classes_.astype(int), np.arange(6))
    table = est.predict_proba(x.iloc[query]).reshape(-1,3,2)
    np.testing.assert_allclose(table.sum((1,2)), 1., atol=1e-5)
    return table


def train():
    import torch
    from tabicl_routed_context_n23 import DIGEST, WEIGHT, infer
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    torch.set_num_threads(2)
    raw,y,g,f=data()
    x = compact_features(raw)
    role = roles(raw,g)
    strata = (x.HomePlanet + '/' + x.CryoSleep + '/' + pd.Series(y).astype(str)).to_numpy()
    with np.load(ROOT/'artifacts/nov25_historical_nine/historical.npz') as z:
        rows, hf = z['rows'],z['fold']
    OUT.mkdir(exist_ok=True)
    for tag,ids,fold in [('historical',rows,hf),('full',np.arange(len(y)),f)]:
        for k in np.unique(fold):
            q=ids[fold==k]
            other=ids[fold!=k]
            t=other[~np.isin(g[other],g[q])]
            assert not np.isin(g[q],g[t]).any()
            path=OUT/f'{tag}_{k}.npz'
            if not path.exists():
                seed=1604+int(k)
                rng=np.random.default_rng(seed)
                fake=role[t].copy()
                for level in np.unique(strata[t]):
                    loc=np.flatnonzero(strata[t]==level)
                    fake[loc]=rng.permutation(fake[loc])
                real=role_table(x,t,q,2*role[t]+y[t],seed)
                sham=role_table(x,t,q,2*fake+y[t],seed)
                direct=infer(x.assign(role=role.astype(str)),y,t,q,seed)
                with path.with_suffix('.tmp').open('wb') as h:
                    np.savez_compressed(h,t=t,q=q,truth=y[t],role=role[q],direct=direct,
                                        real=real,sham=sham,weights=DIGEST)
                path.with_suffix('.tmp').replace(path)
            with np.load(path) as z:
                for name,v in [('t',t),('q',q),('truth',y[t]),('role',role[q]),('weights',DIGEST)]:
                    np.testing.assert_array_equal(z[name],v)
            print(tag,k,flush=True)


def score():
    from disjoint_tree_fields_n25 import DEST, stats
    from solo_selection_likelihood_n26 import regimes
    raw,y,g,views=regimes()
    role=roles(raw,g)
    past=views[0][1]
    reports={}
    for tag,ids,fold,native,view in views:
        direct=np.empty(len(ids))
        joint={name:np.empty((len(ids),3,2)) for name in ('real','sham')}
        fresh=~np.isin(g[ids],g[past])
        for k in np.unique(fold):
            use=fold==k
            if tag=='historical':
                fresh[use]=~np.isin(g[ids[use]],g[ids[~use]])
            with np.load(OUT/f'{tag}_{k}.npz') as z:
                np.testing.assert_array_equal(z['q'],ids[use])
                np.testing.assert_array_equal(z['role'],role[ids[use]])
                direct[use]=z['direct']
                for name in joint:
                    joint[name][use]=z[name]
        local=np.arange(len(ids))
        truth=y[ids]
        alone=role[ids]==0
        anchor=view['anchor']
        known=view['known']
        common=view['eligible'] & (view['votes']==anchor[:,None]).all(1)
        masks={'all':known,'solo':known&alone,'common_solo':common&alone,
               'new':known&fresh,'new_solo':known&fresh&alone}
        results={}
        losses={'direct_solo_nll':log_loss(truth[alone],direct[alone])}
        def odds(v):
            return logit(v.clip(.002,.998))
        for name,v in joint.items():
            conditioned=v[local,role[ids],1]/v[local,role[ids]].sum(1)
            marginal=v[:,:,1].sum(1)
            role_given_outcome=v[local,role[ids],truth.astype(int)] / v[local,:,truth.astype(int)].sum(1)
            losses[name+'_role_nll_given_outcome']=-float(np.log(role_given_outcome.clip(1e-9,1)).mean())
            losses[name+'_solo_nll_conditioned']=log_loss(truth[alone],conditioned[alone])
            losses[name+'_solo_nll_marginal']=log_loss(truth[alone],marginal[alone])
            flips=(conditioned>=.5)!=anchor
            actions={'conditional':flips,'induced':flips&((direct>=.5)==anchor),
                     'observed_vs_marginal':flips&((marginal>=.5)==anchor),
                     'increment_over_direct':((odds(native)+odds(conditioned)-odds(direct)>=0)!=anchor)&
                                             ((native>=.5)==anchor)}
            for act,edit in actions.items():
                delta=edit*np.where(truth!=anchor,1,-1)
                results[name+'_'+act]={m:stats(delta,mask,fold,g[ids]) for m,mask in masks.items()}
        reports[tag]=dict(losses=losses,actions=results)
        print(tag,losses,[(k,v['solo']['fixed'],v['solo']['harmed']) for k,v in results.items()],flush=True)
    DEST.with_name('manifest_role_joint.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['train','score'])
    args=parser.parse_args()
    (train if args.action=='train' else score)()
