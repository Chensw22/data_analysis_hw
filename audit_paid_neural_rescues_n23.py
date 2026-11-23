"""Check repeatability and decision precision of neural corrections on complete multiple-service bills."""
import json
import numpy as np
from threadpoolctl import threadpool_limits

from broad_bill_completion_states_n23 import REPORT
from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS, load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, SEEDS, write


def select(score, eligible, count, row):
    ids=np.flatnonzero(eligible)
    order=np.lexsort((row[ids],-score[ids]))
    mask=np.zeros(len(row),bool)
    mask[ids[order[:count]]]=True
    assert mask.sum()==count
    return mask


def main():
    row,y,frame,g,_=load()
    wrong,right,hard=cohorts(row,y)
    earlier=json.loads((REPORT/'novel_errors.json').read_text())['cases']
    four=np.isin(row,[c['row'] for c in earlier if any(s.startswith('partner_specific_fields/') for s in c['experts'])])
    assert four.sum()==4 and np.all(y[four])
    scopes=frame[BILLS].notna().all(axis=1).to_numpy()&(frame[BILLS].gt(0).sum(axis=1).to_numpy()>=2)
    assert np.all(scopes[four])
    plans,history={},{}
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_partner_specific_fields/{seed}.npz') as z:
            np.testing.assert_array_equal(row,z['row'])
            np.testing.assert_array_equal(y,z['y'])
            base,ref,fold,first,shared=[z[k] for k in ['base','ref','fold','partner_specific','shared']]
        with np.load(ART/f'nov23_repeat_partner_specific_fields/{seed}.npz') as z:
            for k,v in [('row',row),('y',y),('base',base),('ref',ref),('fold',fold)]:
                np.testing.assert_array_equal(z[k],v)
            second=z['replication']
        avg=.5*(first+second)
        active=ref==(base>=.5)
        before=y==ref
        votes={'reference':ref, 'seed_one_raw':first>=.5,'seed_two_raw':second>=.5,'mean_raw':avg>=.5,
               'quarter_all':np.where(active,.75*base+.25*avg>=.5,ref)}
        both=(first>=.5)==(second>=.5)
        confirmed=np.where(both,first>=.5,ref)
        for value in [False,True]:
            segment=active&scopes&(ref==value)
            name='to_negative' if value else 'to_positive'
            sgn=-1 if value else 1
            nominate=segment&(confirmed!=ref)
            votes[name+'_agreed']=ref^nominate
            budget=int((nominate & ((.75*base+.25*avg>=.5)!=ref)).sum())
            ranking={'baseline':sgn*base, 'mean':sgn*(.75*base+.25*avg),
                     'new_interactions':sgn*(avg-shared), 'conservative':sgn*base+np.minimum(sgn*first,sgn*second)}
            for method,score in ranking.items():
                votes[name+'_'+method]=ref^select(score,nominate,budget,row)
        hits={k:(v==y) for k,v in votes.items()}
        plans[seed]={k:dict(**paired(h,before),four_hits=int((h&four).sum()),hard_hits=int((hard&h).sum()))
                     for k,h in hits.items()}
        if seed!='v131':
            for k,h in hits.items():
                history.setdefault(k,[]).append(h)
        print(seed,{k:(r['fixed']-r['harmed'],r['four_hits']) for k,r in plans[seed].items()},flush=True)
    prev=np.stack(history['reference'])
    stable={}
    for k,seq in history.items():
        h=np.stack(seq)
        stable[k]=dict(hard_stable=int((hard&h.all(0)).sum()),hard_ever=int((hard&h.any(0)).sum()),
                       four_stable=int((four&h.all(0)).sum()),stable_harms=int((right&~h.any(0)).sum()),
                       common_stable=int((wrong&h.all(0)).sum()), gain95_pp=interval((h.astype(float)-prev).mean(0),g))
    budget={k:int(v) for k,v in [('all',scopes.sum()),('hard',(hard&scopes).sum())]}
    write(REPORT/'paid_neural_rescues.json',dict(scope=budget,plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(1):
        main()
