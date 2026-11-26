"""Cross-check a surprise-trained foundation model using specialized training orders and manifest neighbors."""
import json
import numpy as np
import pandas as pd

from exhaustive_family_quorums_n26 import summary
from full_state_context_n26 import OUT as FULL
from latent_passenger_difficulty_n26 import OUT as PANEL,ROOT
from matched_state_context_n26 import OUT as SMALL,inputs
from surprise_sampled_foundation_n26 import OUT as SURPRISE

REPORT=ROOT/'reports/nov23_matched_learning_curve/corroborate_surprise_context.json'


def run():
    frame,y,g,f,masks=inputs()
    ids=frame.PassengerId.to_numpy(str)
    n=len(y)
    with np.load(PANEL/'panel.npz') as z:
        np.testing.assert_array_equal(ids,z['passenger'])
        base,known,common,fresh=z['anchor'],z['known'],z['common'],z['fresh']
    def read(path,names):
        with np.load(path/'aligned.npz') as z:
            for key,value in [('passenger',ids),('fold',f),('y',y)]:
                np.testing.assert_array_equal(z[key],value)
            return [z[k] for k in names]
    adjusted,uniform,unadjusted,entire=read(SURPRISE,['adjusted','uniform','unadjusted','full'])
    large,=read(FULL,['p'])
    small,=read(SMALL,['specific'])
    support,rate=read(ROOT/'artifacts/nov26_relax_blind_matches',['none_support','none_rate'])
    available=np.isfinite(large)
    a=(adjusted>=.5)!=base
    plain=(uniform>=.5)!=base
    ordinary=(entire>=.5)!=base
    s=available&((large>=.5)!=base)
    s2=available&((small>=.5)!=base)
    matching=(support>=3)&np.where(base,rate<.5,rate>.5)
    relevant=(masks['awake_paid']&~base)|(masks['sleep_unpaid']&base)
    votes=dict(specialist_and_adjusted=a&s, both_specialists_and_adjusted=a&s&s2,
               matched_passengers_and_adjusted=a&matching,
               adjusted_with_other_support=a&(matching|s), control_with_other_support=plain&(matching|s),
               unique_adjusted_with_other_support=a&~plain&~ordinary&(matching|s),
               state_targeted=a&(matching|s)&relevant, state_targeted_control=plain&(matching|s)&relevant)
    scopes=dict(all=known,common=common,unseen=known&fresh,awake_negative=known&masks['awake_paid'][:n]&~base[:n],
                sleeping_positive=known&masks['sleep_unpaid'][:n]&base[:n])
    unknown=pd.read_csv(ROOT/'artifacts/nov26_clean_fare_unknown_branches/paired_rows.csv')
    np.testing.assert_array_equal(unknown.PassengerId,ids[:n][~known])
    out={}
    for name,act in votes.items():
        out[name]=dict(test_changes=int(act[n:].sum()),scopes={k:summary(act[:n],base[:n]!=y,m,f) for k,m in scopes.items()})
    # A direction-only two-source rule can be re-evaluated for either feasible incumbent.
    for label,state in [('awake',masks['awake_paid']),('sleep',masks['sleep_unpaid'])]:
        target=0 if label=='awake' else 1
        deltas=[]
        for branch in ['161_low','161_high']:
            ref=base[:n].copy()
            ref[~known]=unknown[branch].to_numpy(bool)
            context=available[:n]&((large[:n]>=.5)!=ref)
            neighbor=(support[:n]>=3)&np.where(ref,rate[:n]<.5,rate[:n]>.5)
            action=state[:n]&(ref==target)&((adjusted[:n]>=.5)!=ref)&(context|neighbor)
            deltas.append(np.where(action,np.where(y!=ref,1,-1),0))
        out['state_targeted'][label+'_all_row_net_bounds']=[int(np.min(deltas,axis=0).sum()),int(np.max(deltas,axis=0).sum())]
    REPORT.write_text(json.dumps(out,indent=2)+'\n')
    print([(k,r['scopes']['all']['fixed'],r['scopes']['all']['harmed'],r['test_changes']) for k,r in out.items()])


if __name__=='__main__':
    run()
