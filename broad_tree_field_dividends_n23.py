"""Test whether the field coalitions behind a tree forecast identify recoverable errors."""
from math import comb

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS, CATEGORICAL, load
from broad_matched_model_changes_n23 import read_bank
from broad_pairwise_adjudication_n23 import paired, read_opinions
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write


GROUPS = [
    ['Spa', 'VRDeck', 'RoomService'],
    ['FoodCourt', 'ShoppingMall'],
    ['Age', 'VIP', 'CryoSleep'],
    ['HomePlanet', 'Destination'],
    ['BookingSize', 'Deck', 'Side', 'CabinNumber'],
]
REPORT = ROOT/'reports/nov23_tree_field_dividends'


def design(frame):
    x, ids = [], []
    for group, fields in enumerate(GROUPS):
        for field in fields:
            v = frame[field]
            if field in CATEGORICAL:
                cols = pd.get_dummies(v, dummy_na=True).to_numpy(dtype=float)
            else:
                val = np.log1p(v) if field in BILLS else v
                cols = val.fillna(-1).to_numpy()[:, None]
            x.append(cols)
            ids += [group] * cols.shape[1]
    assert set(sum(GROUPS, [])) == set(frame.columns)
    return np.ascontiguousarray(np.column_stack(x), dtype=np.float32), np.array(ids)


def dividends(ensemble, x, group):
    """Expand each leaf probability as a polynomial in field-group presence indicators.

    With a group observable, follow the query down the tree. Otherwise use the
    fraction of training weight on each child. Idempotent group indicators merge
    repeated appearances of the same field group on a path.
    """
    out = np.zeros((len(x), 1 << len(GROUPS)))
    out[:, 0] = logit(ensemble.init_.class_prior_[1])
    for estimator in ensemble.estimators_[:, 0]:
        tree = estimator.tree_
        def visit(node, terms):
            left, right = tree.children_left[node], tree.children_right[node]
            if left == right:
                for mask, mass in terms.items():
                    out[:, mask] += ensemble.learning_rate * tree.value[node,0,0] * mass
                return
            feature = tree.feature[node]
            bit = 1 << group[feature]
            go_left = x[:,feature] <= tree.threshold[node]
            for child, goes_here in [(left, go_left), (right, ~go_left)]:
                prior = tree.weighted_n_node_samples[child] / tree.weighted_n_node_samples[node]
                delta = goes_here.astype(float)-prior
                poly = {}
                for mask, weight in terms.items():
                    poly[mask] = poly.get(mask, 0.) + prior*weight
                    union = mask | bit
                    poly[union] = poly.get(union, 0.) + delta*weight
                visit(child, poly)
        visit(0, {0: 1.})
    np.testing.assert_allclose(out.sum(1), ensemble.decision_function(x), rtol=0, atol=1e-9)
    return out


def heldout_readouts(v, p, ref, y, fold, groups):
    anchor = logit(p.clip(.003, .997))
    total = v.sum(axis=1)
    orders = np.array([i.bit_count() for i in range(v.shape[1])])
    blocks = [v[:,orders==i] for i in (1,2,3)]
    assert all(b.shape[1]==comb(5,i) for i,b in enumerate(blocks, start=1))
    assert np.max(np.abs(v[:,orders>3]))<1e-12
    # All controls retain the exact standalone score; signed totals additionally
    # retain separate first-, second-, and third-order margins.
    scalar = (total-anchor)[:,None]
    totals = np.column_stack([scalar, *[b.sum(1) for b in blocks]])
    centered = [b-b.mean(axis=1, keepdims=True) for b in blocks]
    rng = np.random.default_rng(3309)
    sham = [np.take_along_axis(b, np.argsort(rng.random(b.shape), axis=1), axis=1) for b in centered]
    designs = dict(score=scalar, orders=totals, main=np.column_stack([totals,centered[0]]),
                   named=np.column_stack([totals,*centered]), sham_interactions=np.column_stack([totals,centered[0],*sham[1:]]))
    base = np.where(ref==(p>=.5),p,1-p)
    output = {k:base.copy() for k in designs}
    for f in np.unique(fold):
        pool = np.flatnonzero((fold==f)&(ref==(p>=.5)))
        for a,b in GroupKFold(3,shuffle=True,random_state=941).split(pool,groups=groups[pool]):
            t,q=pool[a],pool[b]
            assert not np.intersect1d(groups[t],groups[q]).size
            for key, features in designs.items():
                x=np.column_stack([np.ones(len(features)),features])
                def loss(w):
                    z=anchor[t]+x[t]@w
                    err=expit(z)-y[t]
                    return np.logaddexp(0,z).sum()-y[t]@z+12*(w@w), x[t].T@err+24*w
                opt=minimize(loss,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B')
                if not opt.success: raise RuntimeError(opt.message)
                output[key][q]=expit(anchor[q]+x[q]@opt.x)
    return output


def hard_cohorts(row,y):
    with np.load(ART/'nov23_release_vote_information/panel.npz') as z:
        np.testing.assert_array_equal(row,z['row'])
        np.testing.assert_array_equal(y,z['y'])
        wrong=(z['votes']!=y[None,:,None]).all((0,2))
        right=(z['votes']==y[None,:,None]).all((0,2))
    never=wrong.copy()
    consistent_right=right.copy()
    for seed in map(str,SEEDS):
        with np.load(ART/f'nov23_joint_observed_state/{seed}.npz') as z:
            for k,v in [('row',row),('y',y)]:np.testing.assert_array_equal(z[k],v)
            battle=read_bank(row,y,z['fold'],seed).reshape(len(y),-1)
            four=np.column_stack([z[k] for k in ['sleep_conditional','sleep_direct','basket_conditional','basket_direct']])
            hit = (np.c_[battle,four]>=.5)==y[:,None]
            never &= ~hit.any(1)
            consistent_right &= hit.all(1)
    assert (wrong.sum(),right.sum(),never.sum())==(983,5343,372)
    return wrong,right,never,consistent_right


def run():
    row,y,frame,g,_=load()
    x,groups=design(frame)
    wrong,right,never,consistent_right=hard_cohorts(row,y)
    selected=never|consistent_right
    plans,history={},{}
    art=ART/'nov23_tree_field_dividends'
    art.mkdir(exist_ok=True)
    for seed in [*map(str,SEEDS),'v131']:
        fold,ref,op,_=read_opinions(seed,row,y)
        split=np.full((len(y),1<<len(GROUPS)),np.nan)
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            model=GradientBoostingClassifier(n_estimators=220,max_depth=3,learning_rate=.065,
                                               min_samples_leaf=14,random_state=174).fit(x[t],y[t])
            split[q]=dividends(model,x[q],groups)
        assert np.isfinite(split).all()
        scores=heldout_readouts(split,op[:,0],ref,y,fold,g)
        scores['ordinary_tree']=expit(split.sum(1))
        scores['reference']=np.where(ref==(op[:,0]>=.5),op[:,0],1-op[:,0])
        votes={k:p>=.5 for k,p in scores.items()}
        for key in votes.keys()-{'ordinary_tree'}:
            locked=ref!=(op[:,0]>=.5)
            votes[key][locked]=ref[locked]
        votes['reference']=ref
        hits={k:v==y for k,v in votes.items()}
        plans[seed]={k:{**paired(h,ref==y),'old_hard_correct':int((h&never).sum())} for k,h in hits.items()}
        for k,score in scores.items():
            # The classifier was trained across passengers. This population is selected only for retrospective evaluation.
            # On those passengers, getting the unanimously supported label wrong defines the positive diagnostic class.
            common_vote=np.where(never,~y,y)
            dissent=np.where(common_vote,1-score,score)
            stat=plans[seed][k]
            stat['within_unanimous_auc']=float(roc_auc_score(never[selected],dissent[selected]))
            stat['within_unanimous_wrong_flipped']=int((hits[k]&never).sum())
            stat['within_unanimous_right_flipped']=int((~hits[k]&consistent_right).sum())
        if seed!='v131':
            for k,h in hits.items():history.setdefault(k,[]).append(h)
        np.savez_compressed(art/f'{seed}.npz',row=row,y=y,fold=fold,ref=ref,base=op[:,0],split=split,**{k:v for k,v in votes.items() if k!='reference'})
        print(seed,{k:v['fixed']-v['harmed'] for k,v in plans[seed].items()},flush=True)
    ref=np.stack(history['reference'])
    stable={}
    for k,runs in history.items():
        s=np.stack(runs)
        stable[k]=dict(correct_common=int((s.all(0)&wrong).sum()),correct_hard=int((s.all(0)&never).sum()),
                       any_hard=int((s.any(0)&never).sum()),stable_damage=int(((~s).all(0)&right).sum()),
                       mean_accuracy=float(s.mean()),gain95_pp=interval((s.astype(float)-ref).mean(0),g))
    write(REPORT/'result.json',dict(field_groups=GROUPS,consistent_correct=int(consistent_right.sum()),plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(1):run()
