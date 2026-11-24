"""Measure the denominators behind repeated reversals by the failure-weighted forest."""
import json

import numpy as np

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,BILLS,banks,data,latest_reference


def load(rows,fold,g,y,tag):
    scores = np.empty(len(rows))
    for f in np.unique(fold):
        q = rows[fold==f]
        t = rows[(fold!=f) & ~np.isin(g[rows],g[q])]
        assert not np.isin(g[t],g[q]).any()
        with np.load(ROOT/f'artifacts/nov24_honest_failure_partitions/{tag}_{f}.npz') as z:
            for k,v in [('q',q),('t',t),('truth',y[t])]:
                np.testing.assert_array_equal(z[k],v)
            scores[fold==f] = z['p'][:,2]
    return scores>=.5


def count(mask,pred,baseline,y):
    return dict(n=int(mask.sum()),fixed=int((mask & (pred==y) & (baseline!=y)).sum()),
                harmed=int((mask & (pred!=y) & (baseline==y)).sum()))


def run():
    raw,y,g,fold = data()
    _,oy,og,rows,of,_,champ,ref,*_ = aligned()
    names,h,full = banks(raw,y,fold)
    base = latest_reference(y,g,fold,names,h,full)
    np.testing.assert_array_equal(base[rows],ref)
    now = load(np.arange(len(y)),fold,g,y,'full')
    past = load(rows,of,g,y,'historical')
    both = (past!=champ)&(now[rows]!=ref)
    same = both & (champ==ref)
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as z:
        for k,v in [('passenger',raw.PassengerId),('fold',fold),('y',y)]:
            np.testing.assert_array_equal(z[k],v)
        hard = z['remaining']
    repeated = rows[same & hard[rows] & (past==oy)]
    child = (raw.Age.lt(13) & raw.CryoSleep.eq(False) & raw[BILLS].eq(0).all(axis=1)).to_numpy()
    multi = raw.groupby(g).PassengerId.transform('size').to_numpy()>1
    cabin = raw.Cabin.str.split('/',expand=True)
    route = (raw.HomePlanet.eq('Earth') & raw.Destination.eq('55 Cancri e') &
             cabin[0].eq('G') & cabin[2].eq('S')).to_numpy()
    report = dict(overlap=len(rows), same_direction=int(same.sum()),correct_both=int((same & (past==oy)).sum()),
                  harmed_both=int((same & (past!=oy)).sum()), both_flipped=int(both.sum()),
                  overlapping_hard=int(hard[rows].sum()), repeated=raw.PassengerId.iloc[repeated].tolist(),
                  repeated_ages=raw.Age.iloc[repeated].tolist(),repeated_targets=y[repeated].tolist())
    for name,mask in [('all_awake_zero_children',child),('with_companion',child & multi),
                      ('retrospective_exact_route',child & multi & route)]:
        report[name] = dict(full=count(mask,now,base,y),historical=count(mask[rows],past,champ,oy),
                            already_seen_orders_removed=count(mask & ~np.isin(g,og),now,base,y),
                            transported=int((mask & y).sum()), matched_reversals=count(mask[rows]&same,past,champ,oy))
    path = ROOT/'reports/nov23_matched_learning_curve/repeated_child_errors.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(report['repeated'],report['same_direction'],report['retrospective_exact_route'])


if __name__=='__main__':
    run()
