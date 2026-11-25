"""Measure which nine-vote corrections survive changing training bookings, including shared mistakes."""
import argparse
import json
import numpy as np
from scipy.special import expit,logit

from awake_error_demonstrations_n25 import OUT,MODES
from awake_context_ranks_n25 import OUT as FULL
from disjoint_tree_fields_n25 import ART,DEST,stats
from full_roster_booking_validation_n23 import data
from full_roster_six_failure_archive_n24 import prediction_sets
from grouped_peer_teacher_n25 import OUT as REFERENCE


def deployed(placement,parent,nine,base):
    return np.where((parent==base)&(nine==parent),placement,nine)


def report(folds):
    raw,y,group,fold=data()
    awake=raw.CryoSleep.eq(False).to_numpy()
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        rows,hf=z['rows'],z['fold']
    _,panels=prediction_sets(raw,y,group,fold,rows,hf)
    votes=(np.column_stack(panels[0])>=.5).sum(axis=1)
    majority=votes>34
    unanimous=(votes==0)|(votes==68)
    near=(votes<=3)|(votes>=65)
    quantities={}
    masks=[]
    retention=[]
    for k in folds:
        ix=np.flatnonzero(fold==k)
        q=ix[awake[ix]]
        with np.load(REFERENCE/f'release{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],ix)
            np.testing.assert_array_equal(z['y'],y[ix])
            p,parent,nine,low,high=(z[j] for j in ('native','v131','nine','nine_on_low_branch','nine_on_high_branch'))
        base=p>=.5
        pred={}
        for mode in (*MODES,'full'):
            path=FULL/f'{k}_personal.npz' if mode=='full' else OUT/f'{k}_{mode}.npz'
            with np.load(path) as z:
                np.testing.assert_array_equal(z['q'],q)
                prob=z['p']
                if mode in ('error','balanced_error'):
                    # Restore the observed class priors after sampling only correct examples.
                    if mode=='balanced_error':
                        f=z['training_correct_fraction'][base[awake[ix]].astype(int)]
                        prob=expit(logit(prob.clip(.00001,.99999))+np.log(f))
                    prob=np.where(base[awake[ix]],1-prob,prob)
                pred[mode]=p.astype(float).copy()
                pred[mode][awake[ix]]=prob
        first,second,full=(pred[j]>=.5 for j in ('first_half','second_half','full'))
        agree=first==second
        masks.append((ix,parent>=0))
        decisions={k:p>=.5 for k,p in pred.items()}
        decisions['mean_halves']=(pred['first_half']+pred['second_half'])/2>=.5
        decisions['both_halves']=np.where(agree,first,base)
        decisions['all_three']=np.where(agree&(first==full),full,base)
        # Test error prediction only as a veto on the original candidate as well.
        for key in ('error','balanced_error'):
            decisions['full_confirmed_by_'+key]=np.where(decisions[key]==full,full,base)
        exact=parent>=0
        placed=deployed(full,parent,nine,base)
        changed=exact&(placed!=nine)
        supporters=(first==full).astype(int)+(second==full).astype(int)
        retention.append([[int((changed&(supporters==i)&(placed==y[ix])).sum()),
                           int((changed&(supporters==i)&(placed!=y[ix])).sum())] for i in range(3)])
        for mode,d in decisions.items():
            against_nine=[]
            against_full=[]
            for sign,branch in [(False,low),(True,high)]:
                prior=np.where(exact,parent,sign)
                outcome=deployed(d,prior,branch,base)==y[ix]
                against_nine.append(outcome.astype(int)-(branch==y[ix]))
                against_full.append(outcome.astype(int)-(deployed(full,prior,branch,base)==y[ix]))
            # Inspect every archive-unanimous passenger, without requiring a verified nine-vote replay.
            viewed=np.where(awake[ix],d,majority[ix])
            vs_archive=(viewed==y[ix]).astype(int)-(majority[ix]==y[ix])
            quantities.setdefault(mode,[]).append((*against_nine,*against_full,vs_archive))
    ix=np.concatenate([v[0] for v in masks])
    known=np.concatenate([v[1] for v in masks])
    unseen=~np.isin(group[ix],group[rows])
    summary={}
    for name,parts in quantities.items():
        left,right,dl,dr,shared=(np.concatenate([p[j] for p in parts]) for j in range(5))
        np.testing.assert_array_equal(left[known],right[known])
        np.testing.assert_array_equal(dl[known],dr[known])
        def pair_bounds(a,b):
            return [int(np.minimum(a,b).sum()),int(np.maximum(a,b).sum())]
        summary[name]=dict(vs_nine_bounds=pair_bounds(left,right),vs_full_bounds=pair_bounds(dl,dr),
            vs_nine=stats(left,known,fold[ix],group[ix]),vs_full=stats(dl,known,fold[ix],group[ix]),
            unseen_vs_nine=stats(left,known&unseen,fold[ix],group[ix]),
            consensus=stats(shared,unanimous[ix],fold[ix],group[ix]),within_three=stats(shared,near[ix],fold[ix],group[ix]))
        r=summary[name]
        print(name,r['vs_nine_bounds'],r['vs_full_bounds'],r['consensus']['fixed'],r['consensus']['harmed'],flush=True)
    replication=np.sum(retention,axis=0).tolist()
    print('full candidate corrections and harms by zero, one, two supporting half contexts',replication,flush=True)
    result=dict(folds=list(folds),passengers=len(ix),known=int(known.sum()),retention=replication,results=summary)
    path=DEST.with_name('awake_error_demonstrations.json') if len(folds)==8 else OUT/'interim.json'
    path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--folds',type=int,nargs='+',choices=range(8),default=list(range(8)))
    report(parser.parse_args().folds)
