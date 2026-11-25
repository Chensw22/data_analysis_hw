"""Test archive families only on passengers whose original decision survives the 214-edit union."""
import json

import numpy as np
from sklearn.model_selection import GroupKFold

from disjoint_tree_fields_n25 import ART,DEST,stats
from family_identity_transfer_n26 import panels_and_reference
from full_roster_booking_validation_n23 import data
from joint_candidate_coverage_n26 import branch_decisions


def observed(rows,fold,panel,names,base,prior,low,high,awake,kind):
    pred=branch_decisions(rows,fold,panel,names,base,prior,low,high,awake,kind,891)
    anchor=pred['all_three'][0]
    categories=np.array([name.split('/')[0] for name in names])
    families=list(dict.fromkeys(categories))
    family_votes=np.column_stack([panel[:,categories==name].mean(1)>=.5 for name in families])
    count=(panel>=.5).sum(1)
    minority=np.minimum(count,68-count)
    unchanged=(anchor==pred['nine'][0])&(prior>=0)
    # Fixed bands are shared with the full vote audit. Only charge a new edit if
    # the combined candidate kept the original decision.
    bins=np.digitize(minority,[1,4,17])
    state=2*bins+anchor
    return dict(votes=family_votes,anchor=anchor,eligible=unchanged,minority=minority,state=state,known=prior>=0,cohort=families)


def rewards(v,y):
    return (v['votes']==y[:,None]).astype(int)-(v['anchor']==y)[:,None]


def train_gain(view,reward,t,cell):
    selected=t[view['eligible'][t]&(view['state'][t]==cell)]
    return reward[selected].sum(0)


def execute(test_view,test,test_groups,train_sets,paired_groups):
    assert not np.isin(paired_groups,test_groups).any()
    gains=[]
    rng=np.random.default_rng(6241)
    groups=rng.permutation(np.unique(paired_groups))
    first=np.isin(paired_groups,groups[::2])
    for cell in range(8):
        test_pos=(test_view['state'][test]==cell)&test_view['eligible'][test]
        if not test_pos.any():
            continue
        halves=[]
        for train_view,reward,t in train_sets:
            values=np.stack([train_gain(train_view,reward,t[m],cell) for m in (first,~first)])
            halves.append(values)
        by_regime=np.stack(halves)
        best=np.min(by_regime.sum(1),axis=0)
        winner=best.argmax()
        supported=by_regime[:,:,winner].min()>0
        delta=test_view['votes'][test[test_pos],winner]!=test_view['anchor'][test[test_pos]]
        gains.append((test_pos,delta,bool(best[winner]>0),bool(supported and best[winner]>0)))
    loose=np.zeros(len(test),bool)
    strict=loose.copy()
    for m,edit,accept,repeat in gains:
        if accept:loose[m]=edit
        if repeat:strict[m]=edit
    return loose,strict


def run():
    raw,y,g,f=data()
    _,_,_,rows,names,p,h,native,old_native,r=panels_and_reference()
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        old=z['nine'];old_parent=z['v131'];hf=z['fold']
    n=np.arange(len(y))
    awake=raw.CryoSleep.eq(False).to_numpy()
    full=observed(n,f,p,names,native,*r,awake,'full')
    hist=observed(rows,hf,h,names,old_native,old_parent,old,old,awake,'historical')
    fresh=~np.isin(g,g[rows])
    gain_full=rewards(full,y)
    gain_hist=rewards(hist,y[rows])
    out={}
    for k in range(8):
        ix=n[f==k]
        for a,b in GroupKFold(4,shuffle=True,random_state=2653).split(ix,groups=g[ix]):
            t,q=ix[a],ix[b]
            switches=execute(full,q,g[q],[(full,gain_full,t)],g[t])
            out.setdefault('all_current',[]).append((q,*switches))
        pos=np.flatnonzero(f[rows]==k)
        q=ix[fresh[ix]]
        t=rows[pos]
        for name,sets in [('current',[(full,gain_full,t)]),('older',[(hist,gain_hist,pos)]),
                          ('repeat',[(full,gain_full,t),(hist,gain_hist,pos)])]:
            switches=execute(full,q,g[q],sets,g[t])
            out.setdefault('new_'+name,[]).append((q,*switches))
    result={}
    for name,parts in out.items():
        q,loose,strict=(np.concatenate([v[j] for v in parts]) for j in range(3))
        np.testing.assert_array_equal(np.sort(q),n if name=='all_current' else n[fresh])
        result[name]={}
        for action,flip in [('positive',loose),('both_booking_halves',strict)]:
            delta=((full['anchor'][q]^flip)==y[q]).astype(int)-(full['anchor'][q]==y[q])
            result[name][action]=stats(delta,full['known'][q],f[q],g[q])
            print(name,action,result[name][action]['fixed'],result[name][action]['harmed'])
    opportunity={}
    for name,view,label in [('full',full,y),('historical',hist,y[rows])]:
        dis=(view['votes']!=view['anchor'][:,None]).any(1)&view['eligible']
        wrong=view['anchor']!=label
        opportunity[name]=dict(known_remaining_errors=int((wrong&view['known']).sum()),
            remaining_errors_where_nine_unchanged=int((wrong&view['eligible']).sum()),
            reachable=int((dis&wrong).sum()),exposed_correct=int((dis&~wrong).sum()))
    print(opportunity)
    DEST.with_name('outstanding_error_families.json').write_text(json.dumps(dict(crossfit=result,upper_coverage=opportunity),indent=2)+'\n')


if __name__=='__main__':
    run()
