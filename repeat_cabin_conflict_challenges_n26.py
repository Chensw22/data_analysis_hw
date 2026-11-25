"""Check if observed-cabin evidence transfers among passengers on whom frozen files disagree."""
import json

import numpy as np
from catboost import CatBoostRanker,Pool
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from conflict_field_blocks_n26 import designs
from conflict_pair_ordering_n26 import training_pairs
from cross_order_conflict_cover_n26 import relations
from exhaustive_family_quorums_n26 import inputs,summary
from full_roster_booking_validation_n23 import ROOT,OUT,data
from swap_observed_cabin_on_conflicts_n26 import contrasts

REPORT = ROOT/'reports/nov23_matched_learning_curve/repeat_cabin_conflict_challenges.json'


def outputs(raw,y,g,f):
    matrices = designs(raw,g)
    x,cabin = matrices[3],slice(matrices[1].shape[1],matrices[3].shape[1])
    a,b,d = relations(raw,g,False)
    signals = np.zeros((len(y),2,5))
    params = dict(depth=4,iterations=115,learning_rate=.07,l2_leaf_reg=8,thread_count=2,
                  verbose=False,allow_writing_files=False,random_seed=3505)
    for k in np.unique(f):
        eligible = f!=k
        t,q = np.flatnonzero(eligible),np.flatnonzero(~eligible)
        assert not np.isin(g[t],g[q]).any()
        pairs,_ = training_pairs(a,b,d,y,g,eligible)
        rank = CatBoostRanker(loss_function='PairLogit',**params).fit(Pool(x[t],label=y[t].astype(int),pairs=pairs))
        signals[q] = contrasts(rank,params,x,y,g,a,b,d,eligible,q,cabin)
    return signals


def run():
    raw,y,g,f = data()
    _,cur,_ = inputs()
    z = outputs(raw,y,g,f)
    with np.load(OUT/'mitra96.npz') as bank:
        np.testing.assert_array_equal(bank['passenger'],raw.PassengerId.to_numpy(str))
        np.testing.assert_array_equal(bank['fold'],f)
        strong = logit(bank['p'].clip(.003,.997))
    plain = np.column_stack([strong,z[:,:,0]])
    enriched = np.column_stack([plain,z[:,:,1:].reshape(len(y),-1)])
    actions = {}
    for seed in (169,509,1261):
        vote = np.zeros((len(y),2),bool)
        for k in np.unique(f):
            rows = np.flatnonzero(f==k)
            for u,v in GroupKFold(4,shuffle=True,random_state=seed).split(rows,groups=g[rows]):
                t,q = rows[u],rows[v]
                assert not np.isin(g[t],g[q]).any()
                for m,x in enumerate((plain,enriched)):
                    scale = StandardScaler().fit(x[t])
                    clf = LogisticRegression(C=.2,max_iter=400).fit(scale.transform(x[t]).clip(-6,6),y[t])
                    vote[q,m] = clf.predict(scale.transform(x[q]).clip(-6,6))
        actions[str(seed)] = (vote[:,1]!=cur['parent'])&(vote[:,0]==cur['parent'])
    n = np.sum(list(actions.values()),axis=0)
    actions.update(majority=n>=2,unanimous=n==3)
    report = {}
    contested = cur['known']&~cur['common']
    for name,m in [('all',cur['known']),('common',cur['common']),('disagreement',contested),
                   ('new_disagreement',contested&cur['new']),('historical_disagreement',contested&~cur['new'])]:
        report[name] = {s:summary(a,cur['wrong'],m,f) for s,a in actions.items()}
        print(name,[(s,v['fixed'],v['harmed']) for s,v in report[name].items()],flush=True)
    older = json.loads((REPORT.parent/'swap_observed_cabin_on_conflicts.json').read_text())
    assert report['all']['169']==older['deployment']['all']['extra_challenges']
    assert report['common']['169']==older['deployment']['common']['extra_challenges']
    report['scope'] = ('The disagreement subgroup was noticed after the first all-passenger experiment. Two '
         'additional order assignments refit the calibration models; they reuse the same eight base folds and '
         'passengers. All actions are reversals newly induced by counterfactual cabin features.')
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    out = ROOT/'artifacts/nov26_cabin_conflict_challenges'
    out.mkdir(exist_ok=True)
    np.savez_compressed(out/'full.npz',passenger=raw.PassengerId.to_numpy(str),y=y,fold=f,signal=z,**actions)


if __name__=='__main__':
    with threadpool_limits(1):
        run()
