"""Test if labelled passengers in nearby cabins improve a held passenger's prediction."""
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from exhaustive_family_quorums_n26 import inputs, summary
from full_roster_booking_validation_n23 import OUT, ROOT, data
from matched_bill_pair_states_n26 import vectors


def neighbors(bank, queries, cabin, zone, group):
    edges = []
    for region in np.unique(zone[queries]):
        if region == 'unknown':
            continue
        donors = bank[zone[bank]==region]
        donors = donors[np.argsort(cabin[donors],kind='stable')]
        for i in queries[zone[queries]==region]:
            center = np.searchsorted(cabin[donors],cabin[i])
            # Grow until all distance ties at the third distinct booking are included.
            reach = 8
            while True:
                lo, hi = max(0,center-reach), min(len(donors),center+reach)
                options = donors[lo:hi]
                options = options[group[options]!=group[i]]
                distance = abs(cabin[options]-cabin[i])
                order = np.lexsort((options,distance))
                rows = options[order]
                _, pos = np.unique(group[rows],return_index=True)
                nearest = rows[np.sort(pos)[:3]]
                exhausted = lo==0 and hi==len(donors)
                if len(nearest)==3:
                    last = abs(cabin[nearest[-1]]-cabin[i])
                    enclosed = (lo==0 or cabin[i]-cabin[donors[lo-1]]>last) and (
                                 hi==len(donors) or cabin[donors[hi]]-cabin[i]>last)
                    if enclosed or exhausted:
                        edges.extend((i,j) for j in nearest)
                        break
                if exhausted:
                    break
                reach *= 2
    left,right = np.asarray(edges,int).T
    assert (group[left]!=group[right]).all()
    return left,right


def feature(x, cabin, i, j):
    return np.column_stack([x[i],x[j],x[i]-x[j],np.log1p(np.abs(cabin[i]-cabin[j]))])


def donor_messages(prob, label):
    p0,p1,p2,p3 = prob.T
    # Compare the conditional target odds to the target odds without the donor label.
    return np.where(label, np.log(p3/p1), np.log(p2/p0)) - logit(p2+p3)


def aggregate(i, score, length):
    return np.bincount(i,weights=score,minlength=length)/np.bincount(i,minlength=length).clip(1)


def predict(raw,y,g,f):
    x = np.column_stack(vectors(raw))
    split = raw.Cabin.str.split('/',expand=True)
    cabin = pd.to_numeric(split[1]).to_numpy()
    zone = (split[0]+'_'+split[2]).fillna('unknown').to_numpy(str)
    with np.load(OUT/'mitra96.npz') as z:
        for key,value in [('y',y),('fold',f),('passenger',raw.PassengerId.to_numpy(str))]:
            np.testing.assert_array_equal(z[key],value)
        base = z['p'].clip(.003,.997)
    num_shams = 31
    score = np.zeros((len(y),1+num_shams))
    usable = np.zeros(len(y),bool)
    for k in np.unique(f):
        t,q = np.flatnonzero(f!=k),np.flatnonzero(f==k)
        assert not np.isin(g[t],g[q]).any()
        a,b = neighbors(t,t,cabin,zone,g)
        i,j = neighbors(t,q,cabin,zone,g)
        assert np.isin(j,t).all() and not np.isin(i,t).any()
        # Reversing each training pair imposes symmetric outcome coding.
        a,b = np.r_[a,b],np.r_[b,a]
        model = HistGradientBoostingClassifier(max_iter=65,max_leaf_nodes=13,l2_regularization=6.,
                    random_state=3126,early_stopping=False)
        model.fit(feature(x,cabin,a,b),2*y[a].astype(int)+y[b])
        np.testing.assert_array_equal(model.classes_,np.arange(4))
        forward = model.predict_proba(feature(x,cabin,i,j))
        reverse = model.predict_proba(feature(x,cabin,j,i))[:,[0,2,1,3]]
        prob = (forward+reverse)/2
        usable[i] = True
        np.testing.assert_array_equal(np.unique(i,return_counts=True)[1],3)
        # Keep each query's cabin neighbours fixed. Exchange observed donor labels among
        # edges with similar predicted donor risk in the same deck and side as a control.
        donor_prob = prob[:,1]+prob[:,3]
        bins = []
        for region in np.unique(zone[j]):
            ix = np.flatnonzero(zone[j]==region)
            ix = ix[np.argsort(donor_prob[ix],kind='stable')]
            bins.extend(np.array_split(ix,max(1,len(ix)//20)))
        rng = np.random.default_rng(703+k)
        for c in range(1+num_shams):
            label = y[j].copy()
            if c:
                for ix in bins:
                    label[ix] = rng.permutation(label[ix])
            score[q,c] = aggregate(i,donor_messages(prob,label),len(y))[q]
        print('held fold',int(k),'passengers',len(np.unique(i)),flush=True)
    return base,score,usable


def run():
    raw,y,g,f = data()
    base,score,usable = predict(raw,y,g,f)
    _, cur, _ = inputs()
    parent, known = cur['parent'],cur['known']
    single = raw.groupby(g).PassengerId.transform('size').eq(1).to_numpy()
    probs = expit(logit(base)[:,None]+score)
    flips = ((probs>=.5)!=(base>=.5)[:,None]) & ((probs>=.5)!=parent[:,None])
    report = {}
    for name,mask in [('all_known',known&usable),('eleven_agree',cur['common']&usable),
                      ('solo',known&single&usable),('new_orders',known&cur['new']&usable)]:
        result = summary(flips[:,0],y!=parent,mask,f)
        shams = [summary(v,y!=parent,mask,f)['net'] for v in flips[:,1:].T]
        original = np.where(y[mask],base[mask],1-base[mask])
        updated = np.where(y[mask,None],probs[mask],1-probs[mask])
        result.update(sham_net_min_median_max=np.quantile(shams,[0,.5,1]).tolist(),
                      observed_minus_base_logloss=float(np.log(original/updated[:,0]).mean()),
                      sham_minus_base_mean_logloss=float(np.log(original[:,None]/updated[:,1:]).mean()))
        report[name] = result
    report['scope'] = ('Each donor label is from a different training booking; three distinct training bookings, '
                       'same deck and side, closest cabin numbers. Secondary model and strong anchor both exclude the '
                       'target booking. All 31 within-risk-bin shuffles are descriptive controls, not independent trials.')
    (ROOT/'reports/nov23_matched_learning_curve/neighboring_booking_labels.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    with threadpool_limits(1):
        run()
