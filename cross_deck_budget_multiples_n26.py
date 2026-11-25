"""Test whether integer-multiple budget peaks transfer between cabins and predict passenger outcomes."""
import json

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from full_roster_booking_validation_n23 import ROOT,data
from latent_bill_rounding_n26 import BILLS
from solo_archive_recovery_n26 import ticker
from tabicl_consensus_review_n26 import OUT as REVIEW,sources,splits

OUT=ROOT/'artifacts/nov26_cross_deck_budget_multiples'
RATIOS=np.array([2.,3.,4.,1/2,1/3,1/4])
SHAM=np.array([2.17,3.19,4.23,1/2.17,1/3.19,1/4.23])


def fields(frame):
    fees=frame[BILLS]
    total=fees.fillna(0).sum(axis=1).to_numpy(int)
    active=fees.gt(0).sum(axis=1).to_numpy()
    eligible=fees.notna().all(axis=1).to_numpy()&(total>0)
    planet=frame.HomePlanet.fillna('?').to_numpy()
    deck=frame.Cabin.str.split('/').str[0].fillna('?').to_numpy()
    return total,active,eligible,planet,deck


def spectrum(cost,ratios,limit):
    bins=np.rint((cost[:,None]*ratios).reshape(-1)).astype(int)
    bins=bins[bins<limit]
    h=np.bincount(bins,minlength=limit).astype(float)
    # Half-unit grid. Include smooth support for sparse bins and unseen amounts.
    mass=.78*gaussian_filter1d(h,2.4)+.22*gaussian_filter1d(h,42)
    return mass+.001


def features(v,t,q):
    total,count,eligible,planet,deck=v
    budget=2*total+count
    width=64
    limit=2*(total.max()+width+2)+8
    use=t[eligible[t]]
    result=np.full((len(q),4),1/width)
    for origin in np.unique(planet[q]):
        base=use[planet[use]==origin]
        if len(base)<20:
            base=use
        for d in np.unique(deck[q[planet[q]==origin]]):
            loc=np.flatnonzero(eligible[q]&(planet[q]==origin)&(deck[q]==d))
            if not len(loc):
                continue
            ids=q[loc]
            inside=base[deck[base]==d];outside=base[deck[base]!=d]
            if len(inside)<15:inside=base
            if len(outside)<15:outside=base
            edges=2*(width*(total[ids]//width)[:,None]+np.arange(width))+count[ids,None]
            at=total[ids]%width
            def cond(s):
                mat=s[edges]
                return mat[np.arange(len(loc)),at]/mat.sum(axis=1)
            direct=cond(spectrum(budget[inside],np.ones(1),limit))
            # Controls use the same sample, smoothing and bin normalization as the integer ratios.
            copied=cond(spectrum(budget[outside],np.ones(1),limit))
            related=cond(spectrum(budget[outside],RATIOS,limit))
            fake=cond(spectrum(budget[outside],SHAM,limit))
            result[loc]=np.column_stack([direct,.8*direct+.2*copied,.8*direct+.2*related,.8*direct+.2*fake])
    return result


def density_check(raw,group):
    v=fields(raw)
    e=v[2]
    scores=np.zeros((len(raw),4))
    fold=np.zeros(len(raw),int)
    for k,(t,q) in enumerate(GroupKFold(5,shuffle=True,random_state=4534).split(raw,groups=group)):
        assert not np.isin(group[t],group[q]).any()
        scores[q]=np.log(features(v,t,q))
        fold[q]=k
    result={}
    for n,mask in [('all',e),*[(p,e&(v[3]==p)) for p in ('Earth','Mars','Europa')]]:
        means=scores[mask].mean(0)
        result[n]=dict(rows=int(mask.sum()),mean_log_probability=means.tolist(),integer_minus_direct=float(means[2]-means[0]),
                       integer_minus_noninteger=float(means[2]-means[3]),
                       by_fold=[float((scores[:,2]-scores[:,3])[mask&(fold==k)].mean()) for k in range(5)])
    return result


def outcome():
    from solo_selection_likelihood_n26 import fit_model
    from exaone_validation_data import compact_features
    raw,y,g,_=data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())
    all_rows=pd.concat([raw,test],ignore_index=True)
    assert not np.isin(g,test.PassengerId.str.split('_').str[0]).any()
    OUT.mkdir(exist_ok=True)
    v=fields(raw)
    total,count,eligible,origin,deck=v
    x=compact_features(raw)
    x['party_size']=raw.PassengerId.groupby(g).transform('size')
    for c in BILLS:
        x[c]=np.log1p(raw[c]/np.maximum(total,1)*1000)
        x[c]=x[c].round(1)
    x['budget_band']=total//64
    x['num_positive']=count
    meta=sources()[-1]
    report={'unlabelled_density':density_check(all_rows,all_rows.PassengerId.str.split('_').str[0].to_numpy())}
    print('density',report['unlabelled_density']['all'],flush=True)
    for tag,b in meta.items():
        ids,fold=b['ids'],b['fold']
        probs={};direct_evidence={}
        for k in np.unique(fold):
            q=ids[fold==k];other=ids[fold!=k]
            t=other[~np.isin(g[other],g[q])]
            assert not np.isin(g[q],g[t]).any()
            path=OUT/f'{tag}_{k}.npz'
            if not path.exists():
                prior=fit_model(x.iloc[t],y[t]).predict_proba(x.iloc[q])[:,1]
                joint=[features(v,t[y[t]==c],q) for c in (False,True)]
                effect=np.log(joint[1]/joint[0])
                with path.with_suffix('.tmp').open('wb') as stream:
                    np.savez_compressed(stream,t=t,q=q,train_y=y[t],prior=prior,effect=effect)
                path.with_suffix('.tmp').replace(path)
            with np.load(path) as z:
                for n,val in [('q',q),('t',t),('train_y',y[t])]:
                    np.testing.assert_array_equal(z[n],val)
                prior=z['prior'];effect=z['effect']
            for j,name in enumerate(['direct','cross_deck','multiples','noninteger']):
                probs.setdefault(name,np.empty(len(ids)))[fold==k]=expit(logit(prior)+effect[:,j])
                direct_evidence.setdefault(name,np.empty(len(ids)))[fold==k]=effect[:,j]
        anchor=b['anchor']
        native=expit(b['off']*np.where(anchor,-1,1))
        known=b['known']
        wrong=b['wrong']
        actions={}
        for j,n in enumerate(probs):
            label=probs[n]>=.5
            actions[n]=label!=anchor
            if j:
                actions[n+'_new_vs_direct']=(label!=anchor)&((probs['direct']>=.5)==anchor)
                update=logit(native.clip(.002,.998))+direct_evidence[n]-direct_evidence['direct']
                actions[n+'_increment']=((update>=0)!=anchor)&((native>=.5)==anchor)
        table={n:{s:ticker(m,wrong,flag&eligible[ids]) for s,m in [
            ('all',known),('unanimous',b['common']),('eligible',known&eligible[ids]),
            ('new',known&~np.isin(g[ids],g[meta['historical']['ids']]))]} for n,flag in actions.items()}
        report[tag]=table
        print(tag,[(n,tab['all']['fixed'],tab['all']['harmed']) for n,tab in table.items()],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/cross_deck_budget_multiples.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        outcome()
