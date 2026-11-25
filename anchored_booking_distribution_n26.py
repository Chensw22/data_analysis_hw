"""Transport a learned booking joint distribution onto the strong individual passenger odds."""
import json

import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from confidence_ordered_booking_n26 import OUT,MODES,simulate
from disjoint_tree_fields_n25 import ART,DEST,stats
from family_identity_transfer_n26 import panels_and_reference
from full_roster_booking_validation_n23 import data
from outstanding_error_families_n26 import observed

CACHE=ART/'nov26_anchored_booking_distribution'


def archive(raw,y,g,tag,k,t,q):
    path=CACHE/f'{tag}_{k}.npz'
    if not path.exists():
        pred,_,dist=simulate(raw,y,g,t,q,keep_joint=True)
        with np.load(OUT/f'{tag}_{k}.npz') as previous:
            for n in MODES:
                np.testing.assert_allclose(pred[n],previous[n],atol=1e-12,rtol=0)
        members=sorted(dist['conditional'])
        ptr=np.r_[0,np.cumsum([len(group) for group in members])]
        state_ptr=np.r_[0,np.cumsum([1<<len(group) for group in members])]
        with path.with_suffix('.tmp').open('wb') as buf:
            np.savez_compressed(buf,q=q,t=t,train_y=y[t],members=np.concatenate(members),ptr=ptr,state_ptr=state_ptr,
                                **{n:np.concatenate([dist[n][party] for party in members]) for n in MODES})
        path.with_suffix('.tmp').replace(path)
    with np.load(path) as z:
        for name,value in [('q',q),('t',t),('train_y',y[t])]:
            np.testing.assert_array_equal(z[name],value)
        passengers,ptr,state_ptr=z['members'],z['ptr'],z['state_ptr']
        vectors={name:z[name] for name in MODES}
    for j,(lo,hi) in enumerate(zip(ptr[:-1],ptr[1:])):
        first,last=state_ptr[j:j+2]
        yield passengers[lo:hi],{name:v[first:last] for name,v in vectors.items()}


def independent_log(bits,margins):
    margin=np.asarray(margins).clip(1e-8,1-1e-8)
    return bits@np.log(margin)+(1-bits)@np.log1p(-margin)


def update(bits,prior,ratio,strength):
    unnormalized=independent_log(bits,prior)+strength*ratio
    return np.exp(unnormalized-logsumexp(unnormalized))@bits


def run():
    raw,y,g,f=data()
    _,_,_,rows,names,p,h,native,old_native,ref=panels_and_reference()
    with np.load(ART/'nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['original_prior'],ref[0]);ref[0]=z['prior']
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        hf,old,par=z['fold'],z['nine'],z['v131']
    awake=raw.CryoSleep.eq(False).to_numpy()
    cur=observed(np.arange(len(y)),f,p,names,native,*ref,awake,'full')
    hist=observed(rows,hf,h,names,old_native,par,old,old,awake,'historical')
    result={}
    CACHE.mkdir(exist_ok=True)
    for tag,ids,fold,base,view in [('historical',rows,hf,old_native,hist),('full',np.arange(len(y)),f,native,cur)]:
        fresh=~np.isin(g[ids],g[rows])
        preds={}
        for k in np.unique(fold):
            loc=np.flatnonzero(fold==k);q=ids[loc];other=ids[fold!=k];t=other[~np.isin(g[other],g[q])]
            if tag=='historical':
                fresh[loc]=~np.isin(g[q],g[other])
            lookup=np.full(len(y),-1,int)
            lookup[q]=loc
            for group,mass in archive(raw,y,g,tag,k,t,q):
                pos=lookup[group]
                has=pos>=0
                target=pos[has]
                bits=(np.arange(1<<len(group))[:,None]>>np.arange(len(group)))&1
                before=mass['independent']@bits
                neutral=independent_log(bits,before)
                # Historical held slices cover only a subset of some bookings.
                # Such unscored members retain the independent model's probability.
                prior=before.copy()
                prior[has]=base[target]
                np.testing.assert_allclose(update(bits,prior,neutral,0),prior,atol=1e-8)
                for mode in ['conditional','shuffled']:
                    m=mass[mode]
                    marginal=independent_log(bits,m@bits)
                    joint=np.log(m.clip(1e-15))
                    for kind,ratio in [('dependence',joint-marginal),('marginal',marginal-neutral),('joint',joint-neutral)]:
                        for strength in [.5,1.]:
                            value=update(bits,prior,ratio,strength)
                            key=f'{mode}_{kind}_{strength}'
                            preds.setdefault(key,base.copy())[target]=value[has]
            print(tag,k,flush=True)
        wrong=view['anchor']!=y[ids]
        eligible=view['known']&(view['anchor']==(base>=.5))
        family=view['eligible']&(view['votes']==view['anchor'][:,None]).all(1)
        metrics={}
        for name,prob in preds.items():
            flag=((prob>=.5)!=(base>=.5))&eligible
            delta=flag*np.where(wrong,1,-1)
            metrics[name]={n:stats(delta,mask,fold,g[ids]) for n,mask in [('all',view['known']),('new',view['known']&fresh),('families',family)]}
        result[tag]=metrics
        print(tag,[(n,s['all']['fixed'],s['all']['harmed']) for n,s in metrics.items()],flush=True)
    DEST.with_name('anchored_booking_distribution.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
