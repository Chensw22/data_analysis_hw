"""Test how much of the sampling adjustment transfers before taking a high-confidence alternative."""
import json

import numpy as np
from scipy.special import logit
from sklearn.model_selection import GroupKFold

from exhaustive_family_quorums_n26 import summary
from latent_passenger_difficulty_n26 import OUT as PANEL,ROOT
from matched_state_context_n26 import inputs
from surprise_sampled_foundation_n26 import OUT as SAMPLE

REPORT=ROOT/'reports/nov23_matched_learning_curve/calibrate_surprise_sampling.json'
STRENGTH=np.linspace(0,1,11)


def choose(values,wrong,fit):
    # Include abstention; evaluate only after the last equal score at each possible cutoff.
    best=(0,0.,0,np.inf)
    for j,column in enumerate(values.T):
        idx=fit[column[fit]>0]
        if not len(idx): continue
        idx=idx[np.argsort(-column[idx],kind='stable')]
        last=np.r_[column[idx][1:]!=column[idx][:-1],True]
        net=np.cumsum(np.where(wrong[idx],1,-1))
        for pos in np.flatnonzero(last&(net>0)):
            # When tied, prefer fewer edits, then the theoretical correction.
            choice=(int(net[pos]),-int(pos)-1,j,float(column[idx[pos]]))
            if choice[:3]>best[:3]: best=choice
    return best[2],best[3],best[0]


def run():
    frame,y,g,f,masks=inputs()
    n=len(y)
    with np.load(PANEL/'panel.npz') as z:
        np.testing.assert_array_equal(z['passenger'],frame.PassengerId)
        base,known,common,fresh=[z[k] for k in ('anchor','known','common','fresh')]
        panel_agrees=((z['scores']>=.5)==base[:,None]).all(axis=1)
    with np.load(SAMPLE/'aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],frame.PassengerId)
        np.testing.assert_array_equal(z['fold'],f)
        raw,adjusted,uniform=[logit(z[k].clip(1e-8,1-1e-8)) for k in ('unadjusted','adjusted','uniform')]
    sign=np.where(base,-1.,1.)
    axes=(raw[:,None]+(adjusted-raw)[:,None]*STRENGTH)*sign[:,None]
    ordinary=(uniform*sign)[:,None]
    with np.load(ROOT/'artifacts/nov26_retained_order_mechanisms/aligned.npz') as z:
        blind=z['eligible']&((z['pairs']>=.5)==base[:n,None,None]).all(axis=(1,2))&panel_agrees[:n]
    wrong=y!=base[:n]
    scopes=dict(all=known,common=common,unseen=known&fresh,already_unanimous=blind,
                hard_awake=common&masks['awake_paid'][:n]&~base[:n],hard_sleep=common&masks['sleep_unpaid'][:n]&base[:n])
    target=np.where(masks['awake_paid']&~base,0,np.where(masks['sleep_unpaid']&base,1,2))
    schemes={'all':np.zeros(len(g),int),'behavior_direction':target}
    out={}
    for name,cell in schemes.items():
        for model,values in [('ordinary',ordinary),('adjustment',axes)]:
            states={}
            for ordered in (False,True):
                changes=np.zeros(n,bool)
                test_votes=np.zeros(len(g)-n,int)
                folds=[]
                for k in range(8):
                    ids=np.flatnonzero((f==k)&known)
                    if ordered:
                        blocks=np.array_split(np.unique(g[ids]),3)
                        inner=[(ids[~np.isin(g[ids],b)],ids[np.isin(g[ids],b)]) for b in blocks]
                    else:
                        inner=[(ids[a],ids[b]) for a,b in GroupKFold(3,shuffle=True,random_state=844+k).split(ids,groups=g[ids])]
                    for t,q in [*inner,(ids,np.arange(n,len(g)))]:
                        assert not np.isin(g[t],g[q]).any()
                        for level in np.unique(cell[q]):
                            train=t[cell[t]==level]
                            query=q[cell[q]==level]
                            j,cut,gain=choose(values,wrong,train)
                            edit=values[query,j]>=cut
                            if q[0]>=n:
                                test_votes[query-n]+=edit
                                folds.append(dict(fold=k,cell=int(level),axis=int(j),cutoff=float(cut) if np.isfinite(cut) else None,training_gain=gain))
                            else: changes[query]=edit
                key='ordered' if ordered else 'random'
                states[key]=dict(test_changes=int((test_votes>4).sum()),scopes={s:summary(changes,wrong,m,f) for s,m in scopes.items()},test_fit=folds)
            oracle={}
            for s,m in scopes.items():
                full=np.flatnonzero(m)
                oracle[s]=sum(choose(values,wrong,full[cell[full]==v])[2] for v in np.unique(cell[full]))
            states['in_sample_net_ceiling']=oracle
            out[name+'_'+model]=states
    REPORT.write_text(json.dumps(out,indent=2)+'\n')
    for key,r in out.items():
        print(key,[(s,r[s]['scopes']['all']['net'],r[s]['scopes']['already_unanimous']['net']) for s in ('random','ordered')],r['in_sample_net_ceiling']['all'])


if __name__=='__main__':
    run()
