"""Remove one family of matching constraints at a time before voting with independent bookings."""
import json
import numpy as np

from compare_blind_manifest_cases_n26 import match
from exhaustive_family_quorums_n26 import summary
from latent_passenger_difficulty_n26 import OUT as PANEL,ROOT
from matched_state_context_n26 import inputs

REPORT=ROOT/'reports/nov23_matched_learning_curve/relax_blind_matches.json'


def run():
    frame,y,g,f,masks=inputs()
    n=len(y)
    fold=np.r_[f,np.full(len(g)-n,-1)]
    with np.load(PANEL/'panel.npz') as z:
        np.testing.assert_array_equal(z['passenger'],frame.PassengerId)
        base,known,common,fresh=[z[k] for k in ('anchor','known','common','fresh')]
    with np.load(ROOT/'artifacts/nov26_retained_order_mechanisms/aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],frame.PassengerId.iloc[:n])
        group=z['eligible']
        blind=group&((z['pairs']>=.5)==base[:n,None,None]).all(axis=(1,2))&((z['original_panel']>=.5)==base[:n,None]).all(axis=1)
    wrong=base[:n]!=y
    scopes=dict(all=known,common=common,unseen=known&fresh,blind=blind,
                awake_blind=blind&masks['awake_paid'][:n]&~base[:n],sleep_blind=blind&masks['sleep_unpaid'][:n]&base[:n])
    assert (scopes['awake_blind']&wrong).sum()==150
    out={}
    acts={}
    cache={}
    for omission in (None,'registration','cabin','bill','size'):
        counts,rates,_,_=match(frame,y,g,fold,omit=omission)
        name=omission or 'none'
        n_ref,p=counts['close'],rates['close']
        active=(n_ref>=3)&np.where(base,p<.5,p>.5)
        cache[name+'_support']=n_ref
        cache[name+'_rate']=p
        acts[name]=active
        out[name]=dict(test_changes=int(active[n:].sum()),scopes={s:{
            **summary(active[:n],wrong,m,f),
            'errors_with_any_match':int((m&wrong&(n_ref[:n]>0)).sum()),
            'errors_with_three_bookings':int((m&wrong&(n_ref[:n]>=3)).sum()),
            'originally_correct_with_three':int((m&~wrong&(n_ref[:n]>=3)).sum())}
            for s,m in scopes.items()})
    # Agreement between relaxed votes uses only fields and votes available at prediction time.
    relaxed=np.column_stack([a for k,a in acts.items() if k!='none'])
    two=(relaxed.sum(axis=1)>=2)&~acts['none']
    out['two_relaxations_only']=dict(test_changes=int(two[n:].sum()),
                    scopes={s:summary(two[:n],wrong,m,f) for s,m in scopes.items()})
    folder=ROOT/'artifacts/nov26_relax_blind_matches'
    folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/'aligned.npz',passenger=frame.PassengerId.to_numpy(str),fold=f,y=y,**cache)
    REPORT.write_text(json.dumps(out,indent=2)+'\n')
    for k,row in out.items():
        print(k,row['scopes']['all']['net'],row['scopes']['awake_blind']['fixed'],row['scopes']['awake_blind']['harmed'],flush=True)


if __name__=='__main__':
    run()
