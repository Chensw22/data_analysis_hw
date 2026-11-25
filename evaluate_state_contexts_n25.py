"""Compare the queued awake release with full-roster and payment/sleep conditional contexts."""
import argparse
import json
import numpy as np

from awake_context_ranks_n25 import BILLS,OUT as AWAKE
from disjoint_tree_fields_n25 import ART,DEST,stats
from evaluate_awake_demonstrations_n25 import deployed
from full_roster_booking_validation_n23 import data
from full_roster_six_failure_archive_n24 import prediction_sets
from grouped_peer_teacher_n25 import OUT as REF
from passenger_state_contexts_n25 import OUT


def scores_for(f,awake,asleep,paid,fold):
    q=np.flatnonzero(fold==f)
    missing=~(awake|asleep)
    def read(folder,name,eligible):
        with np.load(folder/f'{f}_{name}.npz') as z:
            np.testing.assert_array_equal(z['q'],q[eligible[q]])
            return z['p']
    full=read(OUT,'global',np.ones(len(fold),bool))
    settled=np.where(awake[q]|asleep[q],True,False)
    sleep=np.full(len(q),np.nan)
    p_asleep=read(OUT,'asleep',asleep|missing)
    sleep[~awake[q]]=p_asleep
    sleep[awake[q]]=read(AWAKE,'personal',awake)
    p_awake=read(OUT,'awake_missing',missing)
    with np.load(OUT/f'{f}_awake_missing.npz') as z:
        np.testing.assert_array_equal(z['missing_q'],q[missing[q]])
        prior=z['prior']
    sleep[~settled]=prior*sleep[~settled]+(1-prior)*p_awake
    payment=np.full(len(q),np.nan)
    for key,select in [('paid',paid),('no_payment',~paid)]:
        payment[select[q]]=read(OUT,key,select)
    assert np.isfinite(np.column_stack([full,sleep,payment])).all()
    return full,sleep,payment


def run(folds):
    raw,y,g,fold=data()
    awake=raw.CryoSleep.eq(False).to_numpy()
    asleep=raw.CryoSleep.eq(True).to_numpy()
    paid=raw[list(BILLS)].gt(0).any(axis=1).to_numpy()
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        rows,hf=z['rows'],z['fold']
    _,panels=prediction_sets(raw,y,g,fold,rows,hf)
    votes=(np.column_stack(panels[0])>=.5).sum(1)
    majority=votes>34
    unanimous=(votes==0)|(votes==68)
    near=(votes<=3)|(votes>=65)
    index=[]
    effects={}
    for f in folds:
        full,sleep,payment=scores_for(f,awake,asleep,paid,fold)
        q=np.flatnonzero(fold==f)
        with np.load(REF/f'release{f}.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            native,prior,nine,lo,hi=(z[k] for k in ('native','v131','nine','nine_on_low_branch','nine_on_high_branch'))
        base=native>=.5
        queued=np.where(awake[q],sleep>=.5,base)
        known=prior>=0
        decisions={'queued_awake':queued,'global_all':full>=.5,'global_awake_only':np.where(awake[q],full>=.5,base),
                   'sleep_context':sleep>=.5,'payment_context':payment>=.5,'both_mean':(payment+sleep)/2>=.5,
                   'both_agree':np.where((sleep>=.5)==(payment>=.5),sleep>=.5,base),
                   'sleep_quarter':.75*native+.25*sleep>=.5,'payment_quarter':.75*native+.25*payment>=.5}
        index.append((q,known))
        for name,d in decisions.items():
            pairs=[]
            for sign,n in [(False,lo),(True,hi)]:
                parent=np.where(known,prior,sign)
                correct=deployed(d,parent,n,base)==y[q]
                pairs.extend([correct.astype(int)-(n==y[q]),correct.astype(int)-(deployed(queued,parent,n,base)==y[q])])
            # An awake-only policy retains the archive reference where inactive.
            local=np.where(awake[q],d,majority[q]) if name in ('queued_awake','global_awake_only') else d
            arc=(local==y[q]).astype(int)-(majority[q]==y[q])
            effects.setdefault(name,[]).append((*pairs,arc))
    ix=np.concatenate([q for q,_ in index])
    known=np.concatenate([v for _,v in index])
    report={}
    for name,blocks in effects.items():
        a,da,b,db,arc=(np.concatenate([v[i] for v in blocks]) for i in range(5))
        np.testing.assert_array_equal(a[known],b[known])
        np.testing.assert_array_equal(da[known],db[known])
        def bounds(a,b):return [int(np.minimum(a,b).sum()),int(np.maximum(a,b).sum())]
        def summary(delta,mask):return stats(delta,mask,fold[ix],g[ix])
        report[name]=dict(net_bounds=bounds(a,b),vs_queued_bounds=bounds(da,db),reference=summary(a,known),
            vs_queued=summary(da,known),awake=summary(a,known&awake[ix]),asleep=summary(a,known&asleep[ix]),
            missing=summary(a,known&~(awake[ix]|asleep[ix])),fresh=summary(a,known&~np.isin(g[ix],g[rows])),
            consensus=summary(arc,unanimous[ix]),near=summary(arc,near[ix]))
        r=report[name]
        print(name,r['net_bounds'],r['vs_queued_bounds'],r['consensus']['fixed'],r['consensus']['harmed'],flush=True)
    path=DEST.with_name('passenger_state_contexts.json') if len(folds)==8 else OUT/'interim.json'
    path.write_text(json.dumps(dict(folds=folds,passengers=len(ix),known=int(known.sum()),results=report),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--folds',type=int,nargs='+',choices=range(8),default=list(range(8)))
    run(parser.parse_args().folds)
