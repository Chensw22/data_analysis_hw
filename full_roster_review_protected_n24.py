"""Can six independently fitted views reverse incorrect incumbent overrides?"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from export_budget_quarter_n23 import source
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_solo_coverage_n24 import cached


def review(base,p,v,truth,g,fold):
    eligible = base != (p >= .5)
    opp = (v>=.5)!=base[:,None]
    strength = np.where(base[:,None],.5-v,v-.5)
    votes = dict(revert=eligible, both_majorities=eligible & (opp[:,:3].sum(1)>=2) & (opp[:,3:].sum(1)>=2),
                 unanimous=eligible & opp.all(1))
    broad = [strength[:,:3].mean(1),strength[:,3:].mean(1)]
    x = np.column_stack([base, abs(p-.5),*broad,strength.min(1),strength.max(1)])
    chosen = np.zeros(len(base),bool)
    for f in np.unique(fold):
        rows = np.flatnonzero(fold==f)
        for a,b in GroupKFold(4,shuffle=True,random_state=474).split(rows,groups=g[rows]):
            t,q = rows[a],rows[b]
            assert not np.isin(g[t],g[q]).any()
            t,q = t[eligible[t]],q[eligible[q] & opp[q].any(1)]
            if not len(q) or not len(t):
                continue
            miss = base[t]!=truth[t]
            if miss.min()==miss.max():
                chosen[q]=miss[0]
            else:
                fit = make_pipeline(StandardScaler(),LogisticRegression(C=.08,max_iter=300))
                fit.fit(x[t],miss)
                chosen[q] = fit.predict_proba(x[q])[:,1]>.5
    votes['learned'] = chosen
    return votes,eligible,opp


def run():
    raw,y,g,fold = data()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as z:
        for k,val in [('passenger',raw.PassengerId),('fold',fold),('y',y)]:
            np.testing.assert_array_equal(z[k],val)
        hard = z['remaining']
    test = pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId'])
    primary = source('proposal_v131_party_consensus',test)[1]
    mitra = source('proposal_online_first_counts15',test)[1]
    report = {'release_protected':int((primary!=mitra).sum()),'release_rows':len(test)}
    for tag,ix,f,base,p in [('full',np.arange(len(y)),fold,ref,full[:,names.index('mitra96')]),
                            ('historical',rows,of,champ,old[:,0])]:
        v = np.column_stack([cached(k,tag,ix,f,g,y) for k in ('response_channels','broad_error_transport_projection')])
        votes,eligible,opp = review(base,p,v,y[ix],g[ix],f)
        wrong = eligible & (base!=y[ix])
        result = dict(protected=int(eligible.sum()),errors=int(wrong.sum()),covered=int((wrong&opp.any(1)).sum()),
                      uncovered=int((wrong&~opp.any(1)).sum()),hard=int((wrong & hard[ix]).sum()),decisions={})
        for name,vote in votes.items():
            pred = base ^ vote
            s = stats(pred,base,y[ix],g[ix],f)
            s['shared_fixed'] = int((hard[ix]&wrong&vote).sum())
            if tag=='full':
                new = ~np.isin(g[ix],og)
                s['new_net'] = int((pred[new]==y[ix][new]).sum()-(base[new]==y[ix][new]).sum())
            result['decisions'][name] = s
        report[tag] = result
        print(tag,result['protected'],result['errors'],result['covered'],
              [(n,s['corrected'],s['harmed']) for n,s in result['decisions'].items()],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/protected_review.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    run()
