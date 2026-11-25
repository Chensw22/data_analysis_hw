"""Test a fixed union of two common-error signals and a disagreement-only cabin contrast."""
import json

import numpy as np
import pandas as pd
from catboost import CatBoostRanker,Pool
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from conflict_case_control_banks_n26 import fit_banks,inputs_for_banks
from conflict_field_blocks_n26 import designs
from conflict_pair_ordering_n26 import training_pairs
from cross_order_conflict_cover_n26 import relations
from exhaustive_family_quorums_n26 import inputs,summary
from frozen_queue_shared_errors_n26 import paired
from full_roster_booking_validation_n23 import ROOT,OUT,data
from separate_berth_and_booking_sequence_n26 import geometry

REPORT = ROOT/'reports/nov23_matched_learning_curve/joint_cabin_reference_challenge.json'


def rank_geometry(raw,y,g,f):
    # Keep the labelled roster's one-hot columns identical to the standalone experiment.
    base = designs(raw,g)[3]
    test = pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())
    order,_,berth,center,gap,exact,_ = geometry(pd.concat([raw,test],ignore_index=True))
    extra = np.column_stack([base,order[:len(y)],np.nan_to_num((berth-center)[:len(y)],nan=0),
                             np.nan_to_num(gap[:len(y)],nan=-1),exact[:len(y)]])
    a,b,d = relations(raw,g,False)
    scores = np.zeros((len(y),2))
    for k in np.unique(f):
        eligible = f!=k
        t,q = np.flatnonzero(eligible),np.flatnonzero(~eligible)
        assert not np.isin(g[t],g[q]).any()
        pairs,_ = training_pairs(a,b,d,y,g,eligible)
        for j,x in enumerate([base,extra]):
            model = CatBoostRanker(loss_function='PairLogit',iterations=115,depth=4,learning_rate=.07,l2_leaf_reg=8,
                                   thread_count=2,verbose=False,allow_writing_files=False,random_seed=3505)
            scores[q,j] = model.fit(Pool(x[t],label=y[t].astype(int),pairs=pairs)).predict(x[q])
    return scores


def changes(plain,extra,y,g,f,parent,seed,standardize):
    vote = np.zeros((len(y),2),bool)
    for k in np.unique(f):
        rows = np.flatnonzero(f==k)
        if seed=='ordered':
            parts = [rows[np.isin(g[rows],cut)] for cut in np.array_split(np.unique(g[rows]),4)]
        else:
            parts = [rows[v] for _,v in GroupKFold(4,shuffle=True,random_state=seed).split(rows,groups=g[rows])]
        for q in parts:
            t = np.setdiff1d(rows,q)
            assert not np.isin(g[t],g[q]).any()
            for j,x in enumerate([plain,extra]):
                if standardize:
                    scale = StandardScaler().fit(x[t])
                    xt,xq = scale.transform(x[t]).clip(-6,6),scale.transform(x[q]).clip(-6,6)
                else:
                    xt,xq = x[t],x[q]
                c = LogisticRegression(C=.2 if standardize else .5,max_iter=400).fit(xt,y[t])
                vote[q,j] = c.predict(xq)
    return (vote[:,1]!=parent)&(vote[:,0]==parent)


def run():
    raw,y,g,f = data()
    _,cur,_ = inputs()
    with np.load(OUT/'mitra96.npz') as z:
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId.to_numpy(str))
        np.testing.assert_array_equal(z['fold'],f)
        strong = logit(z['p'].clip(.003,.997))
    geometry_score = rank_geometry(raw,y,g,f)
    with np.load(ROOT/'artifacts/nov26_cabin_conflict_challenges/full.npz') as z:
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId.to_numpy(str))
        np.testing.assert_array_equal(z['fold'],f)
        signal,swap = z['signal'],z['majority']
    old = np.column_stack([strong,signal[:,:,0]])
    aug = np.column_stack([old,signal[:,:,1:].reshape(len(y),-1)])
    assert np.allclose(geometry_score[:,0],signal[:,0,0],atol=1e-7)
    parent = cur['parent']
    geometry_votes = {seed:changes(np.column_stack([strong,geometry_score[:,0]]),np.column_stack([strong,geometry_score[:,1]]),
                              y,g,f,parent,seed,False) for seed in (169,509,1261,'ordered')}
    geo_majority = np.sum([geometry_votes[n] for n in (169,509,1261)],axis=0)>=2
    source = json.loads((REPORT.parent/'separate_berth_and_booking_sequence.json').read_text())
    assert summary(geometry_votes[169],cur['wrong'],cur['common'],f)==source['deployment']['common']['local_geometry']['extra_challenges']
    swap_ordered = changes(old,aug,y,g,f,parent,'ordered',True)
    values,side,marks,edges,dist,regions = inputs_for_banks(raw,y,g,f)
    with np.load(ROOT/'artifacts/nov26_replicate_case_control_banks/mitra_and_pair_rankers.npz') as z:
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId.to_numpy(str))
        bank_majority = ((z['probabilities'][:,:,1]>.5)&z['supports']).sum(0)>=2
    prob,support = fit_banks(values[:,[0,6,7,8,9]],side,marks,edges,dist,regions,g,f,'ordered')
    bank_ordered = (prob[:,:,1].mean(1)>.5)&support.all(1)
    _,truth,groups,fold,known,versions = paired()
    for a,b in [(y,truth),(g,groups),(f,fold),(cur['known'],known)]:
        np.testing.assert_array_equal(a,b)
    champ = versions[:,0]
    assert np.all(known[cur['common']]) and np.all(parent[cur['common']]==champ[cur['common']])
    report = {}
    for protocol,geo,bank,cabin in [('random_majority',geo_majority,bank_majority,swap),
                                    ('ordered_replay',geometry_votes['ordered'],bank_ordered,swap_ordered)]:
        common = cur['common']
        methods = dict(geometry=common&geo,case_control=common&bank&(side==parent),cabin_on_disagreement=known&~common&cabin)
        joint = np.any(np.array(list(methods.values())),axis=0)
        methods['combined'] = joint
        report[protocol] = {s:{name:summary(action,cur['wrong'],mask,f) for name,action in methods.items()} for s,mask in
             [('all',known),('common',common),('new',known&cur['new']),('disagreements',known&~common)]}
        new = parent^joint
        report[protocol]['accuracy'] = float((new[known]==y[known]).mean())
        report[protocol]['against_public_champion'] = summary(new!=champ,y!=champ,known,f)
        print(protocol,report[protocol]['all']['combined'],report[protocol]['new']['combined'],flush=True)
    report['scope'] = ('Candidate uses cabin geometry or reduced-panel case/control support only when eleven frozen '
        'files agree; all other passengers can only be changed by the matched-cabin contrast. The ordered replay '
        'changes calibration and reference partitions, not outcomes or base folds. The subgroup choices were '
        'informed by prior runs on these passengers.')
    REPORT.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
