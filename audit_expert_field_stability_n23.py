"""Test whether naturally hidden fields help identify when a panel of experts can fix the reference."""
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from broad_bill_neighbor_decision_n23 import opinions
from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import coordinates
from broad_observed_mask_shift_n23 import FIELDS
from broad_opposing_passengers_n23 import match, tree
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from broad_tree_field_dividends_n23 import hard_cohorts
from exaone_validation_data import CATEGORICAL

REPORT = ROOT / 'reports/nov23_expert_field_stability'
CACHE = ART / 'nov23_expert_field_stability'


def masked_model(frame, groups, train, query, training_labels):
    assert not np.intersect1d(groups[train],groups[query]).size
    coords,_ = coordinates(frame,train)
    pos,neg = match(coords[train],groups[train],training_labels)
    states = training_labels.astype(int).copy()
    states[np.r_[pos,neg]] += 2
    missing = np.column_stack([frame.iloc[train][fields].replace('__MISSING__',np.nan).isna().all(axis=1)
                               for fields in FIELDS])
    patterns, counts = np.unique(missing[missing.any(1)],axis=0,return_counts=True)
    clean = frame.iloc[query]
    # Enumerate actual training-cohort patterns. Whole cabin disappears together.
    copies = [clean]
    for pattern in patterns:
        perturbed = clean.copy()
        for block, hidden in zip(FIELDS,pattern):
            if hidden:
                for field in block:
                    perturbed[field] = '__MISSING__' if field in CATEGORICAL else np.nan
        copies.append(perturbed)
    score = tree(frame.iloc[train],states,pd.concat(copies,ignore_index=True))
    p = (score[:,3]/(score[:,2]+score[:,3])).reshape(len(copies),len(query))
    w = counts/counts.sum()
    delta = p[1:] - p[0]
    crossed = (p[1:]>=.5)!=(p[0]>=.5)
    return p[0], np.column_stack([w@delta,w@abs(delta),w@crossed]),len(patterns)


def cache_masks(frame,g,y,fold,seed,expected):
    CACHE.mkdir(exist_ok=True)
    path=CACHE/f'{seed}.npz'
    if path.exists():
        with np.load(path) as z:
            np.testing.assert_array_equal(z['fold'],fold)
            np.testing.assert_allclose(z['clean'],expected,rtol=0,atol=1e-11)
            return z['signals']
    clean = np.zeros(len(fold))
    signals = np.zeros((len(fold),3))
    for f in np.unique(fold):
        q = np.flatnonzero(fold==f)
        t = np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
        clean[q], signals[q],npat = masked_model(frame,g,t,q,y[t])
        np.testing.assert_allclose(clean[q],expected[q],rtol=0,atol=1e-11)
        print(seed,'fold',f,'missing patterns',npat,flush=True)
    np.savez_compressed(path,fold=fold,clean=clean,signals=signals)
    return signals


def gate(inputs,training_target):
    fit=make_pipeline(StandardScaler(),LogisticRegression(C=.035,max_iter=600))
    fit.fit(*training_target)
    return fit.predict_proba(inputs)[:,1]


def predict(fold,g,ref,base,experts,signals,replacements,y):
    # All expert opinions and perturbations already exclude this entire outer validation group.
    # Each calibration training set lives inside that validation fold to avoid indirect query labels.
    direction=np.where(ref,-1.,1.)
    signed_logit=direction[:,None]*logit(experts.clip(.002,.998))
    raw=np.column_stack([direction,abs(logit(base.clip(.002,.998))),signed_logit])
    drift=signals.copy()
    drift[:,0]*=direction
    full=np.c_[raw,drift,drift*(signed_logit[:,-1,None]>0)]
    # HGB field-block replacements from the earlier separately trained expert, with local vs random donors.
    designs={'scores':raw,'masked_opposing':full}
    for name,p in replacements.items():
        shift=direction[:,None]*(p[:,1:]-p[:,:1])
        designs[name]=np.c_[raw,direction*logit(p[:,0].clip(.002,.998)),shift,abs(shift)]
    designs['hgb_clean']=np.c_[raw,direction*logit(replacements['local_donor'][:,0].clip(.002,.998))]
    elig=(ref==(base>=.5))&(signed_logit.max(1)>0)
    error=ref!=y
    output={k:np.zeros(len(g)) for k in designs}
    checked=False
    for f in np.unique(fold):
        ids=np.flatnonzero(fold==f)
        for a,b in GroupKFold(3,shuffle=True,random_state=1929).split(ids,groups=g[ids]):
            t,q=ids[a],ids[b]
            assert not np.intersect1d(g[t],g[q]).size
            t,q=t[elig[t]],q[elig[q]]
            for k,x in designs.items():
                output[k][q]=gate(x[q],(x[t],error[t]))
                if not checked and k=='masked_opposing':
                    flipped=y.copy()
                    flipped[q]=~flipped[q]
                    compare=gate(x[q],(x[t],(ref!=flipped)[t]))
                    np.testing.assert_array_equal(output[k][q],compare)
                    checked=True
    assert checked
    return elig,output


def main():
    row,y,frame,g,_=load()
    _,_,hard,correct=hard_cohorts(row,y)
    plans,history,risks,panel={},{},{},[]
    for seed in [*map(str,SEEDS),'v131']:
        fold,base,ref,bills,near=opinions(seed,row,y,frame.BookingSize.to_numpy()>1)
        def read(name,keys):
            with np.load(ART/f'nov23_{name}/{seed}.npz') as z:
                for k,expected in [('fold',fold),('base',base),('ref',ref),('row',row),('y',y)]:
                    np.testing.assert_array_equal(z[k],expected)
                return np.column_stack([z[k] for k in keys])
        abstain=read('abstaining_fields',['personal_first','bill_first','averaged_order'])
        generative=read('dependence_anchors',['independent','single','double'])
        opposing=read('opposing_passengers',['matched_conditional'])
        experts=np.column_stack([bills,near,abstain,generative,opposing])
        if seed!='v131':
            panel.append((experts>=.5)==y[:,None])
        signals=cache_masks(frame,g,y,fold,seed,opposing[:,0])
        replacements={k+'_donor':read('plausible_replacements',['ordinary',k]) for k in ('local','random')}
        elig,probs=predict(fold,g,ref,base,experts,signals,replacements,y)
        before=ref==y
        eligible_ids=np.flatnonzero(elig)
        hits={'reference':before}
        quality={}
        for k,p in probs.items():
            hits[k]=before^(p>=.5)
            quality[k]=dict(auc=roc_auc_score(~before[elig],p[elig]),ap=average_precision_score(~before[elig],p[elig]))
            # Fixed review budgets use all passengers as denominator; rank only eligible disagreements.
            order=eligible_ids[np.argsort(-p[elig],kind='stable')]
            for pct in (1,5):
                selected=order[:round(len(row)*pct/100)]
                fixes=int((~before[selected]).sum())
                quality[k][f'top_{pct}pct']=dict(people=len(selected),fixed=fixes,harmed=len(selected)-fixes)
        plans[seed]=dict(eligible=int(elig.sum()),metrics={k:paired(h,before) for k,h in hits.items()},ranking=quality)
        history[seed]=hits
        risks[seed]=probs
        print(seed,plans[seed],flush=True)
    bank=np.stack(panel).all(0).any(1)
    scope75=hard&bank
    scope370=correct&(~np.stack(panel)).all(0).any(1)
    assert scope75.sum()==75 and scope370.sum()==370
    affected=scope75|scope370
    for seed,arms in risks.items():
        for k,p in arms.items():
            plans[seed]['ranking'][k]['auc_75_vs_370']=roc_auc_score(scope75[affected],p[affected])
            plans[seed]['ranking'][k]['ap_75_vs_370']=average_precision_score(scope75[affected],p[affected])
            for pct in (1,5):
                top=np.argsort(-p,kind='stable')[:round(pct*len(row)/100)]
                assert np.all(p[top]>0)
                detail=plans[seed]['ranking'][k][f'top_{pct}pct']
                detail.update(in75=int(scope75[top].sum()),in370=int(scope370[top].sum()))
    h={k:np.stack([history[s][k] for s in map(str,SEEDS)]) for k in hits}
    stable={}
    for k,v in h.items():
        change=(v.astype(float)-h['reference']).mean(0)
        stable[k]=dict(in75=int((scope75&v.all(0)).sum()),in75_any=int((scope75&v.any(0)).sum()),
                       hard=int((hard&v.all(0)).sum()),harmed=int((correct&~v.any(0)).sum()),
                       accuracy=float(v.mean()),gain_pp=float(100*change.mean()),gain95_pp=interval(change,g),
                       gain_vs_score_gate95_pp=interval((v.astype(float)-h['scores']).mean(0),g))
    write(REPORT/'results.json',dict(grouped=stable,plans=plans))


if __name__=='__main__':
    with threadpool_limits(1):
        main()
