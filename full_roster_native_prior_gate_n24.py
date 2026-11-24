"""Fit the balanced spending evidence on the historical incumbent folds and compare gates."""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from exaone_validation_data import compact_features
from full_roster_adaptive_bill_prior_n23 import EVIDENCE, proposal_design
from full_roster_bill_prior_balance_n23 import predict_priors
from full_roster_booking_validation_n23 import FAMILIES, predictor
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import BILLS, ROOT, banks, data, latest_reference


def native_scores(raw, labels, g, rows, fold, saved):
    directory=ROOT/'artifacts/nov24_native_prior_gate'
    directory.mkdir(exist_ok=True)
    frame=compact_features(raw)
    cue=['CryoSleep',*BILLS]
    context=frame.drop(columns=cue)
    models={k:predictor(k,raw,labels) for k in ['cat','budget']}
    scores=np.full((len(rows),5+len(EVIDENCE)),np.nan)
    for h in np.unique(fold):
        use=fold==h
        q=rows[use]
        t=rows[(fold!=h)&~np.isin(g[rows],g[q])]
        assert not np.isin(g[t],g[q]).any()
        path=directory/f'fold_{h}.npz'
        if not path.exists():
            parts=predict_priors(frame,cue,context,labels,g,t,q)
            by_name={k:model(t,q,h) for k,model in models.items()}
            by_name.update(mitra96=saved[use,0],v44_without_name=saved[use,1],tabicl=saved[use,2])
            z=np.column_stack([*(by_name[k] for k in [*FAMILIES,'mitra96']),*(parts[k] for k in EVIDENCE)])
            with path.with_suffix('.tmp').open('wb') as out:
                np.savez_compressed(out,train=t,query=q,y=labels[t],scores=z)
            path.with_suffix('.tmp').replace(path)
        with np.load(path) as record:
            for k,v in [('train',t),('query',q),('y',labels[t])]:
                np.testing.assert_array_equal(record[k],v)
            scores[use]=record['scores']
        np.testing.assert_array_equal(scores[use,4],saved[use,0])
        print('historical evidence fold',h,len(t),len(q),flush=True)
    assert np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
    return scores


def run():
    _,y,g,rows,oldfold,newfold,champion,_,cold,_,_,old,_=aligned()
    raw,labels,groups,fold=data()
    names,half,full=banks(raw,labels,fold)
    incumbent=latest_reference(labels,groups,fold,names,half,full)
    a=native_scores(raw,labels,groups,rows,oldfold,old)
    original=a[:,4]>=.5
    current=full[:,4]>=.5
    old_proposed,ox=proposal_design(a[:,:5],a[:,5:])
    new_proposed,nx=proposal_design(full[:,:5],full[:,[names.index(k) for k in EVIDENCE]])
    old_wrong=original!=y
    new_wrong=current!=labels
    extra=~np.isin(np.arange(len(labels)),rows)
    report={}
    def describe(p,base,truth,group,split):
        d=(p==truth).astype(int)-(base==truth)
        return dict(fixed=int((d==1).sum()),harmed=int((d==-1).sum()),accuracy=float((p==truth).mean()),
                    fold_net=[int(d[split==k].sum()) for k in np.unique(split)],ci_pp=interval(d,group))
    for mode in ['historical','eightfold','pooled']:
        old_pred=champion.copy()
        new_pred=current.copy()
        def fit(oldmask,newmask):
            ot=oldmask & old_proposed
            nt=newmask & new_proposed[rows]
            parts=[]
            targets=[]
            if mode!='eightfold':
                parts.append(ox[ot])
                targets.append(old_wrong[ot])
            if mode!='historical':
                parts.append(nx[rows[nt]])
                targets.append(new_wrong[rows[nt]])
            return make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=500)).fit(
                np.vstack(parts),np.concatenate(targets))
        def apply(model,x,proposal,base,p):
            permitted=p[proposal[p]]
            if len(permitted):
                ix=permitted[model.predict_proba(x[permitted])[:,1]>=.5]
                base[ix]=~base[ix]
        for f in np.unique(fold):
            for h in np.unique(oldfold):
                cell=np.flatnonzero((newfold==f)&(oldfold==h))
                for _,b in GroupKFold(4,shuffle=True,random_state=462).split(cell,groups=g[cell]):
                    q=cell[b]
                    allowed=~np.isin(g,g[q])
                    model=fit((oldfold==h)&allowed,(newfold==f)&allowed)
                    apply(model,ox,old_proposed&(original==champion),old_pred,q)
                    apply(model,nx,new_proposed,new_pred,rows[q])
            q=np.flatnonzero(extra&(fold==f))
            allowed=~np.isin(g,groups[q])
            apply(fit(allowed,(newfold==f)&allowed),nx,new_proposed,new_pred,q)
        report[mode]=dict(historical_vs_v131=describe(old_pred,champion,y,g,oldfold),
                 historical_cold=describe(old_pred[cold],champion[cold],y[cold],g[cold],oldfold[cold]),
                 full_vs_mitra=describe(new_pred,current,labels,groups,fold),
                 full_vs_gate=describe(new_pred,incumbent,labels,groups,fold),
                 additional_vs_gate=describe(new_pred[extra],incumbent[extra],labels[extra],groups[extra],fold[extra]),
                 old_proposals=int((old_proposed&(original==champion)).sum()),
                 full_proposals=int(new_proposed.sum()))
        print(mode,[(k,report[mode][k]['fixed'],report[mode][k]['harmed']) for k in
                    ['historical_vs_v131','full_vs_mitra','full_vs_gate','additional_vs_gate']],flush=True)
    path=ROOT/'reports/nov23_matched_learning_curve/native_prior_gate.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
