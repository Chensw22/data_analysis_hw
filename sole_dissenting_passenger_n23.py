"""Test whether the sole dissenting expert can be trusted for similar types of passenger."""
import json
import numpy as np
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import cohorts, coordinates
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, SEEDS, write
from broad_service_pairs_n23 import REPORT
from broad_untouched_errors_n23 import EXPERTS
from service_pair_portfolio_n23 import replay


def decisions(frame, scores, groups, y, ref, base, fold, row):
    margin = logit(np.column_stack([base, scores]).clip(.002,.998))
    signed = np.where(ref[:, None], -margin, margin)
    dissent = (scores >= .5) != ref[:, None]
    sole = dissent.sum(1) == 1
    source = np.argmax(dissent, axis=1)
    agree = (ref == y)
    mode = 2*source + ref
    methods = {k: np.zeros(len(row), bool) for k in ['accept', 'frequency', 'profile', 'nonlinear']}
    methods['accept'] = sole.copy()
    for f in np.unique(fold):
        eligible = np.flatnonzero((fold == f) & sole)
        for a,b in GroupKFold(3,shuffle=True,random_state=922).split(eligible,groups=groups[eligible]):
            t,q = eligible[a],eligible[b]
            assert not np.intersect1d(groups[t],groups[q]).size
            personal,_ = coordinates(frame,t)
            key = np.eye(8)[mode]
            x = np.c_[key, signed, personal, key*np.max(signed[:,1:],axis=1,keepdims=True)]
            # Stratified empirical precision shrunk toward the training-fold fraction.
            fraction = ((np.bincount(mode[t], weights=~agree[t], minlength=8)+12*(~agree[t]).mean())
                        /(np.bincount(mode[t], minlength=8)+12))
            methods['frequency'][q] = fraction[mode[q]]>=.5
            scale=StandardScaler().fit(x[t])
            lr=LogisticRegression(C=.12,max_iter=300).fit(scale.transform(x[t]), ~agree[t])
            methods['profile'][q]=lr.predict(scale.transform(x[q]))
            tree=HistGradientBoostingClassifier(max_iter=75,max_leaf_nodes=5,min_samples_leaf=35,
                                                l2_regularization=7.,early_stopping=False,random_state=922)
            tree.fit(x[t],~agree[t])
            methods['nonlinear'][q]=tree.predict(x[q])
    return methods,source


def main():
    row,y,frame,g,_=load()
    wrong,right,hard=cohorts(row,y)
    weight=json.loads((REPORT/'test_budget.json').read_text())['test']['change_150']['weight']
    history,summary={},{}
    for seed in [*map(str,SEEDS),'v131']:
        ref,old=replay(frame,g,row,y,seed,weight)
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as z:
            fold,base=z['fold'],z['base']
        columns=[]
        for folder,field in EXPERTS:
            with np.load(ART/f'nov23_{folder}/{seed}.npz') as z:
                for key,val in [('row',row),('ref',ref),('y',y),('fold',fold)]:
                    np.testing.assert_array_equal(z[key],val)
                columns.append(z[field])
        rules,source=decisions(frame,np.column_stack(columns),g,y,ref,base,fold,row)
        available=~old['union']&(ref==(base>=.5))
        previous=(ref==y)^old['pair_plus_corroborated']
        cells={}
        for j,(folder,_) in enumerate(EXPERTS):
            for v in [False,True]:
                mask=available & rules['accept'] & (source==j) & (ref==v)
                cells[folder+'_'+str(v)]=dict(n=int(mask.sum()),errors=int((mask&(ref!=y)).sum()),
                                             hard=int((mask&hard&(ref!=y)).sum()))
        metrics={}
        for k,v in rules.items():
            mask=v & available
            h=previous ^ mask
            metrics[k]=dict(extra=paired(h,previous),total=paired(h,ref==y))
            if seed!='v131':
                history.setdefault(k,[]).append(h)
        if seed!='v131':
            history.setdefault('prev',[]).append(previous)
        summary[seed]=dict(cells=cells,metrics=metrics)
        print(seed,{k:(v['extra']['fixed'],v['extra']['harmed']) for k,v in metrics.items()},flush=True)
    prev=np.stack(history.pop('prev'))
    stable={}
    for k,h in history.items():
        h=np.stack(h)
        stable[k]=dict(hard_stable=int((hard&h.all(0)).sum()),hard_ever=int((hard&h.any(0)).sum()),
                       common_stable=int((wrong&h.all(0)).sum()),stable_harms=int((right&~h.any(0)).sum()),
                       gain95_over_previous_pp=interval((h.astype(float)-prev).mean(0),g))
    write(REPORT/'sole_dissent.json',dict(plans=summary,stable=stable))


if __name__=='__main__':
    with threadpool_limits(1):
        main()
