"""Evaluate the three awake-only TabICL views against the nine-vote release and the 68-model archive."""
import argparse
import json

import numpy as np
from scipy.special import expit,logit

from awake_context_ranks_n25 import OUT as SCORES, MODES
from disjoint_tree_fields_n25 import ART,DEST,stats
from full_roster_booking_validation_n23 import data
from full_roster_six_failure_archive_n24 import prediction_sets
from grouped_peer_teacher_n25 import OUT
from grouped_peer_release_replay_n25 import edit_effect_bounds


def evaluate(folds):
    raw,y,g,f=data()
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        rows,hf=z['rows'],z['fold']
    _,panels=prediction_sets(raw,y,g,f,rows,hf)
    votes=(np.column_stack(panels[0])>=.5).sum(1)
    awake=raw.CryoSleep.eq(False).to_numpy()
    new_order=~np.isin(g,g[rows])
    scores=np.zeros((len(y),len(MODES)))
    native=np.empty(len(y))
    low,high=np.zeros(len(y),bool),np.zeros(len(y),bool)
    parent,nine=np.full(len(y),-1),np.full(len(y),-1)
    first=SCORES/'0_personal.npz'
    with np.load(first) as z:
        digest=str(z['checkpoint'])
    for k in folds:
        q,t=np.flatnonzero((f==k)&awake),np.flatnonzero((f!=k)&awake)
        for i,key in enumerate(MODES):
            with np.load(SCORES/f'{k}_{key}.npz') as z:
                for name,val in [('t',t),('q',q),('train_y',y[t]),('passenger',raw.PassengerId.iloc[q]),('checkpoint',digest)]:
                    np.testing.assert_array_equal(z[name],val)
                scores[q,i]=z['p']
        with np.load(OUT/f'release{k}.npz') as z:
            at=np.flatnonzero(f==k)
            np.testing.assert_array_equal(z['q'],at)
            np.testing.assert_array_equal(z['y'],y[at])
            native[at],parent[at],nine[at],low[at],high[at]=(z[key] for key in ('native','v131','nine','nine_on_low_branch','nine_on_high_branch'))
    ix=np.flatnonzero(np.isin(f,folds))
    native=native[ix]
    known=parent[ix]>=0
    yq=y[ix]
    base=native>=.5
    majority=votes[ix]>34
    unanimous=(votes[ix]==0)|(votes[ix]==68)
    almost=(votes[ix]<=3)|(votes[ix]>=65)
    p_all=scores[ix].clip(.001,.999)
    result={}
    for i,key in enumerate(MODES):
        p=p_all[:,i]
        choices={'direct':p,'quarter':.75*native+.25*p}
        if i:
            choices['contrast']=expit(logit(native.clip(.001,.999))+.5*(logit(p)-logit(p_all[:,0])))
        for rule,prob in choices.items():
            decision=np.where(awake[ix],prob>=.5,base)
            lo,hi=edit_effect_bounds(decision,base,parent[ix],low[ix],high[ix],yq)
            assert (lo[known]==hi[known]).all()
            direct=np.where(awake[ix],decision,majority)
            delta=(direct==yq).astype(int)-(majority==yq)
            scopes={'all':np.ones(len(ix),bool),'unseen_orders':new_order[ix],
                    'remaining_five':f[ix]>=3,'unanimous':unanimous&(nine[ix]==majority)}
            result[key+'/'+rule]=dict(net_bounds=[int(lo.sum()),int(hi.sum())],
                vs_nine={n:stats(lo,m&known,f[ix],g[ix]) for n,m in scopes.items() if (m&known).any()},
                vs_archive_unanimous=stats(delta,unanimous,f[ix],g[ix]),vs_archive_near=stats(delta,almost,f[ix],g[ix]))
    path=DEST.with_name('awake_context_ranks.json') if len(folds)==8 else SCORES/'interim.json'
    path.write_text(json.dumps(dict(folds=folds,rows=len(ix),awake=int(awake[ix].sum()),known=int(known.sum()),results=result),indent=2)+'\n')
    for name,r in result.items():
        a=r['vs_nine']['all']
        c=r['vs_archive_unanimous']
        print(name,r['net_bounds'],a['fixed'],a['harmed'],c['fixed'],c['harmed'])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--folds',type=int,nargs='+',choices=range(8),default=list(range(8)))
    evaluate(p.parse_args().folds)
