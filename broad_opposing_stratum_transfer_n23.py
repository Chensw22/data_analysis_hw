"""Adjust inferred membership in the hard-pair stratum while anchoring original outcome odds."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import cohorts
from broad_opposing_passengers_n23 import REPORT
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART,SEEDS,write


def components(scores,key,base):
    total,neutral,conditional=(scores[key+s] for s in ('','_neutral','_conditional'))
    # The four-class fit recorded P(Y=1), P(unpaired,Y=1)+.5*P(paired), and
    # P(Y=1|paired). Together they determine all four cells except when conditional=.5.
    denominator=conditional-.5
    assert np.min(np.abs(denominator))>1e-10
    mass=(total-neutral)/denominator
    assert ((mass>0)&(mass<1)).all()
    ordinary=(total-mass*conditional)/(1-mass)
    assert ((ordinary>0)&(ordinary<1)).all()
    np.testing.assert_allclose(mass*conditional+(1-mass)*ordinary,total)
    # A common log-odds offset makes the weighted average reproduce the strong
    # probability exactly before applying any change to the estimated stratum mass.
    c,u=logit(conditional),logit(ordinary)
    lo,hi=np.full(len(base),-60.),np.full(len(base),60.)
    target=base.clip(1e-10,1-1e-10)
    for _ in range(55):
        mid=(lo+hi)/2
        above=mass*expit(c+mid)+(1-mass)*expit(u+mid)>target
        lo=np.where(above,lo,mid)
        hi=np.where(above,mid,hi)
    c,u=expit(c+mid),expit(u+mid)
    np.testing.assert_allclose(mass*c+(1-mass)*u,target,atol=1e-12)
    return mass,c,u


def fit(x,y,offset,inside,outside):
    def loss(beta):
        mass=expit(offset+x@beta)
        gap=inside-outside
        p=(outside+mass*gap).clip(1e-10,1-1e-10)
        nll=-np.where(y,np.log(p),np.log1p(-p)).sum()
        gradient=x.T@((p-y)/(p*(1-p))*gap*mass*(1-mass))
        return nll+3*(beta@beta),gradient+6*beta
    fitted=minimize(loss,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B')
    if not fitted.success:
        raise RuntimeError(fitted.message)
    return fitted.x


def scores_for_seed(data,frame,y,g):
    fold,base,ref=data['fold'],data['base'],data['ref']
    active=ref==(base>=.5)
    scores={'reference':ref.astype(float)}
    for source in ('matched','shuffled'):
        mass,inside,outside=components(data,source,base)
        offset=logit(mass)
        cols=np.column_stack([offset,logit(base.clip(.002,.998)),inside-outside,np.log(frame.BookingSize),
                              frame.CryoSleep.eq('True'),frame[['Spa','VRDeck','RoomService']].gt(0).any(axis=1)])
        def mix(shift,q):
            return outside[q]+expit(offset[q]+shift)*(inside[q]-outside[q])
        scores[source+'_double_odds']=np.where(active,mix(np.log(2.),np.arange(len(y))),ref)
        for f in np.unique(fold):
            ids=np.flatnonzero(fold==f)
            for a,b in GroupKFold(3,shuffle=True,random_state=523).split(ids,groups=g[ids]):
                t,q=ids[a],ids[b]
                assert not np.intersect1d(g[t],g[q]).size
                t=t[active[t]]
                scaled=(cols-cols[t].mean(0))/np.maximum(cols[t].std(0),.1)
                design=np.column_stack([np.ones(len(y)),scaled])
                for name,n in [('global',1),('context',design.shape[1])]:
                    beta=fit(design[t,:n],y[t],offset[t],inside[t],outside[t])
                    out=scores.setdefault(source+'_'+name,np.full(len(y),np.nan))
                    out[q]=np.where(active[q],mix(design[q,:n]@beta,q),ref[q])
    return scores


def main():
    row,y,frame,g,_=load()
    wrong,right,hard=cohorts(row,y)
    plans,history={},{}
    directory=ART/'nov23_opposing_stratum_transfer'
    directory.mkdir(exist_ok=True)
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_opposing_passengers/{seed}.npz') as z:
            for k,v in [('row',row),('y',y)]:
                np.testing.assert_array_equal(z[k],v)
            fold,ref,base=z['fold'],z['ref'],z['base']
            score=scores_for_seed(z,frame,y,g)
        assert all(np.isfinite(p).all() for p in score.values())
        hit={k:(v>=.5)==y for k,v in score.items()}
        plans[seed]={k:paired(v,ref==y) for k,v in hit.items()}
        if seed!='v131':
            for k,v in hit.items():
                history.setdefault(k,[]).append(v)
        np.savez_compressed(directory/f'{seed}.npz',row=row,y=y,fold=fold,base=base,ref=ref,**score)
        print(seed,{k:v['fixed']-v['harmed'] for k,v in plans[seed].items()},flush=True)
    baseline=np.stack(history['reference'])
    stable={}
    for k,h in history.items():
        h=np.stack(h)
        stable[k]=dict(hard_stable=int((hard&h.all(0)).sum()),hard_ever=int((hard&h.any(0)).sum()),
                       common_stable=int((wrong&h.all(0)).sum()),stable_harms=int((right&~h.any(0)).sum()),
                       gain95_pp=interval((h.astype(float)-baseline).mean(0),g))
    write(REPORT/'stratum_transfer.json',dict(plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(1):
        main()
