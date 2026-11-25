"""Can outcomes of independent bookings above and below a bill basket overturn shared errors?"""
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from disjoint_tree_fields_n25 import ART, DEST, stats
from solo_archive_recovery_n26 import archive
from solo_selection_likelihood_n26 import regimes

FEES=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
POLARITY=np.array([-1,1,1,-1,-1])


def donor_prob(candidates, distance, groups, y):
    # Use one witness from each independent order; outcomes do not enter selection.
    order=np.argsort(distance[candidates],kind='stable')
    ordered=candidates[order]
    _,first=np.unique(groups[ordered],return_index=True)
    use=ordered[np.sort(first)[:32]]
    weights=np.exp(-distance[use]/1.5)
    effective=weights.sum()**2/max(weights@weights,1e-9)
    return (1.5+weights@y[use])/(3+weights.sum()), effective


def projections(raw,y,g,t,q):
    v=raw[FEES].to_numpy(float)
    signed=np.log1p(np.nan_to_num(v))*POLARITY
    pattern=(v>0) @ (1<<np.arange(5))
    total=raw[FEES].sum(axis=1).to_numpy()
    mask=np.isnan(v) @ (1<<np.arange(5))
    cab=raw.Cabin.str.split('/',expand=True)
    planet=raw.HomePlanet.fillna('?').to_numpy()
    sleep=raw.CryoSleep.astype('string').fillna('?').to_numpy(str)
    age=raw.Age.to_numpy(float)
    deck=cab[0].fillna('?').to_numpy()
    dest=raw.Destination.fillna('?').to_numpy()
    solo=~pd.Series(g).duplicated(keep=False).to_numpy()
    keys=pd.DataFrame(dict(planet=planet,sleep=sleep,missing=mask)).groupby(['planet','sleep','missing']).ngroup().to_numpy()
    scores=np.full((len(q),3),.5)
    counts=np.zeros_like(scores)
    for key in np.unique(keys[q]):
        pool=t[(keys[t]==key)&(total[t]>0)]
        for pos in np.flatnonzero((keys[q]==key)&(total[q]>0)):
            i=q[pos]
            d=signed[pool]-signed[i]
            lower=(d<=0).all(1)&(d<0).any(1)
            upper=(d>=0).all(1)&(d>0).any(1)
            # Missingness is matched exactly; a missing amount never decides dominance.
            # Registration and solo status affect distance, without narrowing to one small stratum.
            cost=(np.abs(d).sum(1)/5 + np.where(np.isfinite(age[pool]+age[i]),abs(age[pool]-age[i])/32,.6)
                  + .8*(deck[pool]!=deck[i]) + .45*(dest[pool]!=dest[i]) + .5*(solo[pool]!=solo[i])
                  + .55*(pattern[pool]!=pattern[i]))
            fullcost=np.empty(len(raw))
            fullcost[pool]=cost
            for col,subset in enumerate([pool,pool[lower],pool[upper]]):
                if len(subset):
                    scores[pos,col],counts[pos,col]=donor_prob(subset,fullcost,g,y)
    return scores,counts


def run():
    raw,y,g,views=regimes()
    solo=~pd.Series(g).duplicated(keep=False).to_numpy()
    report={}
    folder=ART/'nov26_pareto_bill_witnesses'
    folder.mkdir(exist_ok=True)
    for tag,ids,fold,native,view in views:
        scores=np.zeros((len(ids),3))
        counts=np.zeros_like(scores)
        fresh=~np.isin(g[ids],g[views[0][1]])
        for k in np.unique(fold):
            loc=fold==k
            q=ids[loc]
            other=ids[~loc]
            t=other[~np.isin(g[other],g[q])]
            assert not np.isin(g[q],g[t]).any()
            if tag=='historical':
                fresh[loc]=~np.isin(g[q],g[other])
            path=folder/f'{tag}_{k}.npz'
            if not path.exists():
                p,c=projections(raw,y,g,t,q)
                with path.with_suffix('.tmp').open('wb') as h:
                    np.savez_compressed(h,t=t,q=q,y=y[t],scores=p,counts=c)
                path.with_suffix('.tmp').replace(path)
            with np.load(path) as z:
                for name,v in [('t',t),('q',q),('y',y[t])]:
                    np.testing.assert_array_equal(z[name],v)
                scores[loc],counts[loc]=z['scores'],z['counts']
        blind,lower,upper=scores.T
        anchor=view['anchor']
        wrong=anchor!=y[ids]
        common=view['eligible']&(view['votes']==anchor[:,None]).all(1)
        _,arch=archive(tag,ids,fold,g,y,ART/'nov24_remaining_cases/archive_scores.npz')
        unanimous=common&(arch==anchor[:,None]).all(1)&solo[ids]
        # Lower witnesses should be less likely to transport than focal, upper more likely.
        # Swapping the directions is a control with the same two eligible donor pools.
        def contrary(lo,hi,lo_n,hi_n):
            return ((~anchor)&(lo_n>=8)&(lo>.5)) | (anchor&(hi_n>=8)&(hi<.5))
        directional=contrary(lower,upper,counts[:,1],counts[:,2])
        reversed_rule=contrary(upper,lower,counts[:,2],counts[:,1])
        nearby=(((blind>=.5)!=anchor)&(counts[:,0]>=8))
        support=(counts[:,1:]>=8).all(1)
        bracket=support & (lower<=upper)
        diagnostic=dict(with_both_sides=int(support.sum()),ordered_means=int(bracket.sum()),
                        mean_upper_minus_lower=float((upper-lower)[support].mean()))
        actions={'ordinary':nearby,'directional':directional,'reversed':reversed_rule,
                 'both_confirm':directional&nearby,'direction_only':directional&~nearby,
                 'strict_bound':directional&bracket}
        known=view['known']
        scopes={'all':known,'solo':known&solo[ids],'common_solo':common&solo[ids],
                'unreached_solo':unanimous,'unreached_false_negative_scope':unanimous&~anchor,
                'fresh_solo':known&solo[ids]&fresh}
        if tag=='full':
            assert (unanimous&wrong).sum()==179
        table={name:{c:stats(accept*np.where(wrong,1,-1),m,fold,g[ids])
                     for c,m in scopes.items()} for name,accept in actions.items()}
        report[tag]=dict(diagnostic=diagnostic,results=table)
        print(tag,diagnostic,[(n,v['all']['fixed'],v['all']['harmed'],v['unreached_solo']['fixed'],v['unreached_solo']['harmed'])
                             for n,v in table.items()],flush=True)
    DEST.with_name('pareto_bill_witnesses.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
