"""Train the nine source scores on a second, independently partitioned pair of calibration panels."""
import argparse
import json
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'FALSE'

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from full_roster_booking_validation_n23 import ROOT, FAMILIES, data
from half_roster_disagreement_n23 import fit_family, fit_mitra

DEST = ROOT/'artifacts/nov24_repartition_gate'


def train(action):
    raw,y,g,fold=data()
    panel=(fold//2)%2
    DEST.mkdir(exist_ok=True)
    if action=='mitra':
        fit_mitra(raw,y,g,panel,DEST)
    elif action=='priors':
        from broad_geographic_transfer_n23 import BILLS
        from exaone_validation_data import compact_features
        from full_roster_bill_prior_balance_n23 import predict_priors
        from full_roster_adaptive_bill_prior_n23 import EVIDENCE
        x=compact_features(raw)
        cue=['CryoSleep',*BILLS]
        p=np.full((len(y),len(EVIDENCE)),np.nan)
        for h in range(2):
            t,q=np.flatnonzero(panel!=h),np.flatnonzero(panel==h)
            parts=predict_priors(x,cue,x.drop(columns=cue),y,g,t,q)
            p[q]=np.column_stack([parts[k] for k in EVIDENCE])
        assert np.isfinite(p).all()
        np.savez_compressed(DEST/'priors.npz',passenger=raw.PassengerId.to_numpy(str),panel=panel,y=y,p=p)
    elif action=='trees':
        for k in ['cat','budget','v44_without_name']:
            fit_family(k,raw,y,g,panel,DEST)
    else:
        fit_family(action,raw,y,g,panel,DEST)


def panel_data():
    from full_roster_adaptive_bill_prior_n23 import EVIDENCE
    from full_roster_conditional_pool_n23 import banks, latest_reference
    raw,y,g,fold=data()
    names,original,full=banks(raw,y,fold)
    ref=latest_reference(y,g,fold,names,original,full)
    base=full[:,4]>=.5
    cols=[0,1,2,3,4,*(names.index(k) for k in EVIDENCE)]
    def ordinary(k):
        with np.load(DEST/f'{k}.npz') as z:
            for field,value in [('passenger',raw.PassengerId),('panel',(fold//2)%2),('y',y)]:
                np.testing.assert_array_equal(z[field],value)
            return z['p']
    mitra=np.full(len(y),np.nan)
    for h in range(2):
        with np.load(DEST/f'mitra_{h}.npz') as z:
            t,q=np.flatnonzero((fold//2)%2!=h),np.flatnonzero((fold//2)%2==h)
            for k,v in [('train',t),('query',q),('labels',y[t])]:
                np.testing.assert_array_equal(z[k],v)
            mitra[q]=z['p']
    rotated=np.column_stack([*(ordinary(k) for k in FAMILIES),mitra,ordinary('priors')])
    assert rotated.shape==original[:,cols].shape and np.isfinite(rotated).all()
    return raw,y,g,fold,names,original,full,ref,base,cols,rotated


def evaluate():
    from broad_booking_policy_replay_n23 import interval
    from full_roster_adaptive_bill_prior_n23 import EVIDENCE, proposal_design
    raw,y,g,fold,names,original,full,ref,base,cols,rotated=panel_data()
    sources={}
    all_missed=((full>=.5)!=y[:,None]).all(1)
    for j,k in enumerate([*FAMILIES,'mitra96',*EVIDENCE]):
        old=original[:,cols[j]]
        new=rotated[:,j]
        old_hit=(old>=.5)==y
        new_hit=(new>=.5)==y
        sources[k]=dict(changed=int((old_hit!=new_hit).sum()),fixed=int((new_hit&~old_hit).sum()),
                        harmed=int((~new_hit&old_hit).sum()),mean_abs_score_shift=float(abs(old-new).mean()),
                        baseline_common_correct=int(old_hit[all_missed].sum()),
                        rotated_common_correct=int(new_hit[all_missed].sum()))
    panels=[fold%2,(fold//2)%2]
    banks_=[original[:,cols],rotated]
    fd,fx=proposal_design(full[:,:5],full[:,cols[5:]])
    preds={k:base.copy() for k in ['original','rotated','mean','pooled','pooled_per_passenger']}
    history=[]
    for f in np.unique(fold):
        q=np.flatnonzero((fold==f) & fd)
        xx,yy,rows,probs=[],[],[],[]
        for panel,bank in zip(panels,banks_,strict=True):
            disagreement,x=proposal_design(bank[:,:5],bank[:,5:])
            t=np.flatnonzero((panel==panel[q[0]]) & (fold!=f) & disagreement)
            assert not np.isin(g[t],g[q]).any()
            # The base models behind these donor scores used the opposite panel.
            assert not np.isin(g[panel!=panel[q[0]]],g[fold==f]).any()
            target=y[t]!=(bank[t,4]>=.5)
            model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=500))
            model.fit(x[t],target)
            probs.append(model.predict_proba(fx[q])[:,1])
            xx.append(x[t])
            yy.append(target)
            rows.append(t)
        index=np.concatenate(rows)
        history.append(dict(fold=int(f),training_votes=len(index),unique_passengers=len(np.unique(index)),
                            proposal_disagreements=int(((probs[0]>=.5)!=(probs[1]>=.5)).sum())))
        options=dict(original=probs[0],rotated=probs[1],mean=np.mean(probs,axis=0))
        for k,w in [('pooled',np.ones(len(index))),('pooled_per_passenger',1/np.bincount(index)[index])]:
            model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=500))
            model.fit(np.vstack(xx),np.concatenate(yy),standardscaler__sample_weight=w,logisticregression__sample_weight=w)
            options[k]=model.predict_proba(fx[q])[:,1]
        for k,s in options.items():
            preds[k][q]^=s>=.5
    np.testing.assert_array_equal(preds['original'],ref)
    with np.load(ROOT/'artifacts/run_002/split_indices.npz') as z:
        dev=np.sort(z['development'])
    results={}
    for k,pred in preds.items():
        delta=(pred==y).astype(int)-(ref==y)
        results[k]=dict(accuracy=float((pred==y).mean()),fixed=int((delta==1).sum()),harmed=int((delta==-1).sum()),
                        additional_net=int(delta.sum()-delta[dev].sum()),fold_net=[int(delta[fold==f].sum()) for f in np.unique(fold)],
                        ci_pp=interval(delta,g))
    report=dict(folds=history,results=results,source_retraining=sources)
    (ROOT/'reports/nov23_matched_learning_curve/repartition_gate.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['mitra','trees','tabicl','priors','evaluate'])
    action=parser.parse_args().action
    if action=='evaluate':
        evaluate()
    else:
        train(action)
