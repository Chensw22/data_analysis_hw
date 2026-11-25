"""Test if complementary feature subsets reproduce the corrections of a previously scored random subspace."""
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from full_roster_conditional_pool_n23 import data
from full_roster_permutation_selection_n24 import OUT as ARCHIVE, ensemble, feature_bank
from full_roster_six_failure_archive_n24 import prediction_sets
from grouped_peer_release_replay_n25 import edit_effect_bounds
from grouped_peer_teacher_n25 import OUT as RELEASE, OUTER

ROOT = Path(__file__).resolve().parent
ART = ROOT/'artifacts'
OUT = ART/'nov25_disjoint_tree_fields'
DEST = ROOT/'reports/nov23_matched_learning_curve/disjoint_tree_fields.json'


def decisions(original, others, native):
    # One vote per representation, with the original fixed quarter mixture used as control.
    complement = others.mean(1)
    alternatives = {'original':original, 'complement_mean':complement,
                    'four_equal':(original+others.sum(1))/4}
    native_hard = native >= .5
    result = {k:(.75*native + .25*v >= .5) for k,v in alternatives.items()}
    challenger = result['original']
    supporting = ((others >= .5) == challenger[:, None]).sum(1) >= 2
    result['original_confirmed'] = np.where(supporting, challenger, native_hard)
    return result


def generate(raw, y, g, rows, folds, tag, repeat=False):
    x, names, _ = feature_bank(raw)
    perm = np.random.default_rng(1216).permutation(len(names))
    if repeat:
        perm[52:] = np.random.default_rng(11939).permutation(perm[52:])
    picked, remaining = perm[:52], np.array_split(perm[52:],3)
    assert np.array_equal(np.sort(np.concatenate([picked,*remaining])),np.arange(len(names)))
    old, other = np.empty(len(rows)), np.empty((len(rows),len(remaining)))
    OUT.mkdir(exist_ok=True)
    for f in np.unique(folds):
        use = folds == f
        q, t = rows[use], rows[~use & ~np.isin(g[rows],g[rows[use]])]
        with np.load(ARCHIVE/f'{tag}_{f}.npz') as z:
            for key,val in [('features',names),('train',t),('query',q),('labels',y[t])]:
                np.testing.assert_array_equal(z[key],val)
            np.testing.assert_array_equal(z['cols'][1],picked)
            old[use] = z['prob'][:,2]
        path = OUT/f'{tag}_{f}{"_repeat" if repeat else ""}.npz'
        if not path.exists():
            predictions = np.column_stack([ensemble(x[:, cols], y,t,q,497) for cols in remaining])
            with path.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream,t=t,q=q,labels=y[t],features=np.array(names),
                                    permutation=perm,passenger=raw.PassengerId.iloc[q].to_numpy(str),p=predictions)
            path.with_suffix('.tmp').replace(path)
        with np.load(path) as z:
            for key,val in [('features',names),('t',t),('q',q),('labels',y[t]),('permutation',perm),
                            ('passenger',raw.PassengerId.iloc[q])]:
                np.testing.assert_array_equal(z[key],val)
            other[use] = z['p']
        print(tag, int(f), flush=True)
    return old,other


def stats(delta, mask, fold, groups):
    d = delta[mask]
    return dict(rows=len(d),fixed=int((d>0).sum()),harmed=int((d<0).sum()),
                by_fold=[int(d[fold[mask]==k].sum()) for k in np.unique(fold[mask])],
                interval_pp=interval(d,groups[mask]))


def run():
    raw,y,g,f8 = data()
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        rows,hf,hn,hv,hs = [z[k] for k in ('rows','fold','nine','v131','scores')]
    with np.load(ART/'nov25_production_anchor_cpu/oof.npz') as z:
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId)
        np.testing.assert_array_equal(z['fold'],f8)
        en,ec,ep = z['certified_nine'],z['certified_v131_equals_v96'],z['v96']
    hp = hs[:,4]
    old_h,new_h = generate(raw,y,g,rows,hf,'historical')
    old_e,new_e = generate(raw,y,g,np.arange(len(y)),f8,'full')
    hist,eight = decisions(old_h,new_h,hp),decisions(old_e,new_e,ep)
    old_h2, second_h = generate(raw,y,g,rows,hf,'historical',repeat=True)
    old_e2, second_e = generate(raw,y,g,np.arange(len(y)),f8,'full',repeat=True)
    np.testing.assert_array_equal(old_h2,old_h)
    np.testing.assert_array_equal(old_e2,old_e)
    for table, old, second, p in [(hist,old_h,second_h,hp),(eight,old_e,second_e,ep)]:
        confirm = decisions(old,second,p)['original_confirmed']
        table['other_partition_confirmed'] = confirm
        table['both_partitions_confirmed'] = np.where(table['original_confirmed'] == confirm, confirm, p>=.5)
    _, panels = prediction_sets(raw,y,g,f8,rows,hf)
    h_common = ((np.column_stack(panels[1])>=.5)==hn[:,None]).all(1)
    hfree = ((hp>=.5)==hv)&(hv==hn)
    v131,nine = np.full(len(y),-1),np.full(len(y),-1)
    low,high = np.zeros(len(y),bool),np.zeros(len(y),bool)
    for k in OUTER:
        with np.load(RELEASE/f'release{k}.npz') as z:
            q = z['q']
            np.testing.assert_array_equal(q,np.flatnonzero(f8==k))
            v131[q],nine[q],low[q],high[q] = [z[s] for s in ('v131','nine','nine_on_low_branch','nine_on_high_branch')]
    three = np.isin(f8,OUTER)
    known = v131>=0
    new_orders = ~np.isin(g,g[rows])
    unanimous = ((np.column_stack(panels[0])>=.5)==nine[:,None]).all(1)
    report = {}
    for name in hist:
        h = np.where(hfree,hist[name],hn)
        hd = (h==y[rows]).astype(int)-(hn==y[rows])
        ed = (eight[name]==y).astype(int)-((ep>=.5)==y)
        ee = np.where(en==(ep>=.5),eight[name],en)
        cert_delta = (ee==y).astype(int)-(en==y)
        a,b = edit_effect_bounds(eight[name],ep>=.5,v131,low,high,y)
        assert (a[known]==b[known]).all()
        report[name] = dict(historical=stats(hd,np.ones(len(rows),bool),hf,g[rows]),
            historical_common=stats(hd,h_common,hf,g[rows]),
            three_fold_bounds=[int(a[three].sum()),int(b[three].sum())],
            three_by_fold=[[int(a[f8==k].sum()),int(b[f8==k].sum())] for k in OUTER],
            three_resolved=stats(a,known,f8,g), three_common=stats(a,known&unanimous,f8,g),
            five_fold_native=stats(ed,~three,f8,g), five_fold_nine=stats(cert_delta,~three&ec,f8,g),
            five_new_order_nine=stats(cert_delta,~three&ec&new_orders,f8,g),
            historical_new_vs_original=dict(fixed=int(((hist['original']!=y[rows])&(hist[name]==y[rows])&hfree).sum()),
                                            harmed=int(((hist['original']==y[rows])&(hist[name]!=y[rows])&hfree).sum())))
        print(name, report[name]['three_fold_bounds'],report[name]['five_fold_nine']['fixed'],
              report[name]['five_fold_nine']['harmed'],flush=True)
    DEST.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    with threadpool_limits(3):
        run()
