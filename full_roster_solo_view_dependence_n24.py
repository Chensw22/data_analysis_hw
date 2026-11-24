"""Estimate outcome-specific dependence between solo passengers' registration and spending records."""
import json

import numpy as np
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_class_neighborhood_mass_n24 import protocol
from full_roster_conditional_pool_n23 import ROOT,banks,data,latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_gate_probability_n23 import gate_probabilities
from full_roster_two_view_reliability_n24 import design,tree

MODES = ('joint','independent','marginal_splits','interaction_splits','shuffled_interactions')
TAG = 'solo_view_dependence_products'


def shuffled(ids,rng):
    order = rng.permutation(len(ids))
    other = np.empty_like(ids)
    other[order] = ids[np.roll(order,1)]
    assert not np.any(ids==other)
    return other


def real_vs_product(a,b,products,ids,q,seed,sham):
    rng = np.random.default_rng(seed)
    left,true,right = [],[],[]
    for _ in range(3):
        left.append(ids)
        paired = shuffled(ids,rng)
        true.append(paired if sham else ids)
        right.append(shuffled(ids,rng))
    left = np.concatenate(left)
    def field_pairs(i,j):
        columns = [a[i],b[j]]
        if products is not None:
            av,bv = products
            columns.append((av[i,:,None]*bv[j,None,:]).reshape(len(i),-1))
        return np.column_stack(columns)
    real = field_pairs(left,np.concatenate(true))
    product = field_pairs(left,np.concatenate(right))
    x = np.vstack([real,product])
    labels = np.repeat([True,False],len(left))
    model = tree(13).fit(x,labels)
    return model.decision_function(field_pairs(q,q))


def score(views,y,g,t,q):
    assert len(np.unique(g[t]))==len(t) and not np.isin(g[t],g[q]).any()
    # Companion-only attributes are always missing for solo travelers.
    a,b = [np.nan_to_num(v,nan=-99.) for v in views[:2]]
    joint = np.column_stack([a,b])
    direct = tree(15).fit(joint[t],y[t]).predict_proba(joint[q])[:,1]
    independent = (tree(15).fit(a[t],y[t]).decision_function(a[q])+
                   tree(15).fit(b[t],y[t]).decision_function(b[q])-logit(y[t].mean()))
    out = [direct, expit(independent)]
    av,bv = views[2:]
    def centered(v):
        return ((v-v[t].mean(0))/v[t].std(0).clip(.2)).clip(-6,6)
    products = centered(av),centered(bv)
    for pairs,sham in ((None,False),(products,False),(products,True)):
        contrasts = [real_vs_product(a,b,pairs,t[y[t]==side],q,851+int(side),sham) for side in (False,True)]
        out.append(expit(independent+contrasts[1]-contrasts[0]))
    return np.column_stack(out)


def run():
    raw,y,g,fold = data()
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    gate,_ = gate_probabilities(names,h,full,y,g,fold)
    np.testing.assert_array_equal(gate>=.5,ref)
    solo = raw.groupby(g).PassengerId.transform('size').to_numpy()==1
    hard_path = ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz'
    with np.load(hard_path) as z:
        for k,v in [('passenger',raw.PassengerId),('y',y),('fold',fold)]:
            np.testing.assert_array_equal(z[k],v)
        hard = z['remaining']
    assert (solo & hard).sum()==102 and (solo & (ref!=y)).sum()==945
    a,b = design(raw,g)
    deck = raw.Cabin.str.split('/').str[0]
    av = np.column_stack([raw.Age.fillna(-1),raw.Age.isna(),raw.VIP.eq(True),raw.HomePlanet.eq('Earth'),
         raw.HomePlanet.eq('Europa'),raw.Destination.eq('55 Cancri e'),deck.eq('F'),deck.eq('G')])
    bv = np.column_stack([np.log1p(raw[['Spa','FoodCourt','VRDeck','RoomService','ShoppingMall']].fillna(0)),
                          raw.CryoSleep.eq(True),raw.CryoSleep.isna()])
    views = a,b,av,bv
    source_rows = rows[solo[rows]]
    a = protocol(views,y,g,source_rows,of[solo[rows]],'historical',score,MODES,TAG)
    b = protocol(views,y,g,np.flatnonzero(solo),fold[solo],'full',score,MODES,TAG)
    unseen = ~np.isin(g,og)
    report = {}
    def assess(name,new,prev):
        p,past = ref.copy(),champ.copy()
        p[solo] = new
        past[solo[rows]] = prev
        rescued = hard & solo & (p==y)
        r = dict(full=stats(p,ref,y,g,fold),solo=stats(p[solo],ref[solo],y[solo],g[solo],fold[solo]),
                 historical=stats(past,champ,oy,og,of),unseen=stats(p[unseen],ref[unseen],y[unseen],g[unseen],fold[unseen]),
                 rescued_of_102=int(rescued.sum()),repeated=int((rescued[rows]&(champ!=oy)&(past==oy)).sum()),
                 ids=raw.loc[rescued,'PassengerId'].tolist())
        report[name] = r
        print(name,r['full']['corrected'],r['full']['harmed'],r['historical']['net'],r['unseen']['net'],r['rescued_of_102'],flush=True)
    os,cs = old[solo[rows],0],champ[solo[rows]]
    for j,mode in enumerate(MODES):
        assess(mode,b[:,j]>=.5,a[:,j]>=.5)
        assess(mode+'_quarter',.75*gate[solo]+.25*b[:,j]>=.5,
               np.where((os>=.5)==cs,.75*os+.25*a[:,j]>=.5,cs))
    def increment(base,m):
        z = logit(m.clip(.002,.998))
        return logit(base.clip(.002,.998))+z[:,3]-z[:,1]>=0
    assess('dependence_increment',increment(gate[solo],b),np.where((os>=.5)==cs,increment(os,a),cs))
    path = ROOT/'reports/nov23_matched_learning_curve/solo_view_dependence.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
