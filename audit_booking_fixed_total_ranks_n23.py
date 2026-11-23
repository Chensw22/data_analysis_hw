"""Measure within-booking ranking with a diagnostic known total, and transferable incremental coverage."""
import json
import numpy as np
from scipy.special import logit
from threadpoolctl import threadpool_limits

from booking_fixed_total_ranks_n23 import REPORT
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, SEEDS, write
from broad_service_pairs_n23 import REPORT as PAIR_REPORT
from service_pair_change_budget_n23 import changed
from zero_donor_pair_overlay_n23 import additional


def within_order(groups, size, fold, y, bank):
    found={k:dict(bookings=0,people=0,correct=0,concordance_credit=0.,opposite_pairs=0) for k in bank}
    for g in np.unique(groups):
        ix=np.flatnonzero(groups==g)
        if len(ix)!=size[ix[0]] or len(np.unique(fold[ix]))!=1:
            continue
        label=y[ix]
        k=label.sum()
        if not 0<k<len(ix):
            continue
        for name,p in bank.items():
            score=p[ix]
            # Fractional credit for tied ordering at the cutoff avoids using row order as a predictor.
            edge=np.sort(score)[-k]
            top=score>edge
            ties=score==edge
            membership=top.astype(float)+ties*((k-top.sum())/ties.sum())
            correct=np.where(label,membership,1-membership)
            delta=score[label,None]-score[None,~label]
            s=found[name]
            s['bookings']+=1
            s['people']+=len(ix)
            s['correct']+=float(correct.sum())
            s['concordance_credit']+=float(((delta>0)+.5*(delta==0)).sum())
            s['opposite_pairs']+=delta.size
    return found


def main():
    row,y,frame,g,_=load()
    _,_,hard=cohorts(row,y)
    weight=json.loads((PAIR_REPORT/'test_budget.json').read_text())['test']['change_150']['weight']
    history={}
    plans={}
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_booking_fixed_total_ranks/{seed}.npz') as rank, \
             np.load(ART/f'nov23_service_pairs/{seed}.npz') as pairs, \
             np.load(ART/f'nov23_zero_spend_analogues/{seed}.npz') as donor:
            for field in ['row','y','fold','base','ref']:
                for a in [pairs,donor]:
                    np.testing.assert_array_equal(rank[field],a[field])
            base,ref,fold=rank['base'],rank['ref'],rank['fold']
            pair=changed(pairs['same_passenger_both'],base,ref,weight)
            prior=pair | additional(base,ref,pair,donor['matched'],row,.10)
            old_hit=prior==y
            alloc={k:rank[k] for k in ['independent','count','order','both','shuffled_order']}
            order=within_order(g,frame.BookingSize.to_numpy(),fold,y,alloc)
            metrics={}
            for name,p in alloc.items():
                shift=logit(p.clip(.001,.999))-logit(alloc['independent'].clip(.001,.999))
                flip=(ref==(base>=.5))&(ref!=(logit(base.clip(.001,.999))+shift>=0))
                added=flip&(ref==prior)
                hit=old_hit^added
                metrics[name]=dict(additional=paired(hit,old_hit),total=paired(hit,ref==y),
                                   new=int(added.sum()))
                if seed!='v131':
                    history.setdefault(name,[]).append(hit)
            plans[seed]=dict(oracle_count=order,joined=metrics)
            print(seed,metrics['both'],flush=True)
    write(REPORT/'diagnostic.json',dict(plans=plans,hard_stable={k:int((hard&np.stack(h).all(0)).sum()) for k,h in history.items()}))


if __name__=='__main__':
    with threadpool_limits(1):
        main()
