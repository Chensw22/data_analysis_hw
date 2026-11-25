"""Measure joint rescue coverage and independently fit agreement between booking evidence sources."""
import json

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from booking_tree_messages_n26 import train_and_predict
from exhaustive_family_quorums_n26 import inputs,summary
from full_roster_booking_validation_n23 import OUT, ROOT, data
from neighboring_booking_labels_n26 import predict
from price_implied_deck_n26 import FEES
from unlabeled_order_prior_shift_n26 import crossfit


def learn(individual,effects,paid,asleep,has_peer,has_neighbor,y,g,fold):
    x = np.column_stack([individual,paid,asleep,has_peer,has_neighbor])
    z = np.column_stack([x,effects,effects*individual[:,None],effects*paid[:,None],
                        effects*asleep[:,None],effects.prod(1)])
    estimates = np.zeros((len(y),2))
    for k in np.unique(fold):
        rows = np.flatnonzero(fold==k)
        for a,b in GroupKFold(4,shuffle=True,random_state=1506).split(rows,groups=g[rows]):
            t,q = rows[a],rows[b]
            assert not np.isin(g[t],g[q]).any()
            for j,fields in enumerate([x,z]):
                model = LogisticRegression(C=.5,max_iter=400).fit(fields[t],y[t])
                estimates[q,j] = model.decision_function(fields[q])
    return estimates


def run():
    raw,y,g,f = data()
    p,peer_prob,has_peer,_ = train_and_predict(raw,y,g,f)
    base,neighbor,has_neighbor = predict(raw,y,g,f)
    np.testing.assert_allclose(base,p)
    individual = logit(base)
    effects = np.column_stack([logit(peer_prob.clip(1e-8,1-1e-8))-individual,neighbor[:,0]])
    paid = raw[FEES].gt(0).any(axis=1).to_numpy()
    asleep = raw.CryoSleep.eq(True).to_numpy()
    learned = learn(individual,effects,paid,asleep,has_peer,has_neighbor,y,g,f)
    values = [individual]
    for model in ('tabicl','cat'):
        with np.load(OUT/(model+'.npz')) as z:
            for key,value in [('passenger',raw.PassengerId.to_numpy(str)),('y',y),('fold',f)]:
                np.testing.assert_array_equal(z[key],value)
            values.append(logit(z['p'].clip(.003,.997)))
    # This nested estimator uses labels from within base folds. Only fixed voting
    # rules combine it with the first two sources, so their readout cannot see those labels.
    before,after,_ = crossfit(np.column_stack(values),y,g,f,2*asleep.astype(int)+paid,False)
    batch = np.median(logit(after)-logit(before),axis=1)
    changes = np.column_stack([effects,batch])
    _,cur,_ = inputs()
    parent = cur['parent']
    wrong = y!=parent
    votes = ((individual[:,None]+changes>=0)!=parent[:,None]) & ((individual>=0)==parent)[:,None]
    count = votes.sum(1)
    actions = dict(same_booking=votes[:,0],nearby_booking=votes[:,1],unlabelled_batch=votes[:,2],
                   any=count>0,two=count>=2,three=count==3,
                   sum_messages=((individual+changes.sum(1)>=0)!=parent)&((individual>=0)==parent),
                   learned_and_control_disagree=((learned[:,1]>=0)!=parent)&((learned[:,0]>=0)==parent))
    report = {}
    for scope,mask in [('all',cur['known']),('eleven_agree',cur['common']),('new_orders',cur['known']&cur['new'])]:
        r = {name:summary(flag,wrong,mask,f) for name,flag in actions.items()}
        delta = (learned[:,1]>=0)!=(learned[:,0]>=0)
        r['learned_vs_control'] = summary(delta,(learned[:,0]>=0)!=y,mask,f)
        r['exact_vote_patterns'] = {}
        patterns = votes.astype(int) @ np.array([1,2,4])
        for key in range(1,8):
            r['exact_vote_patterns'][str(key)] = summary(patterns==key,wrong,mask,f)
        r['oracle_on_any_new_challenge'] = int((mask&wrong&(count>0)).sum())
        r['learned_minus_control_logloss'] = float((np.logaddexp(0,learned[mask,1])-y[mask]*learned[mask,1]
                             -np.logaddexp(0,learned[mask,0])+y[mask]*learned[mask,0]).mean())
        report[scope] = r
        print(scope,[(k,r[k]['fixed'],r[k]['harmed']) for k in actions],flush=True)
    # Independently recover the exact corrections from each of the two earlier experiments.
    for path,key,reference in [('booking_tree_messages','same_booking',('multi','only_transferred_evidence')),
                               ('neighboring_booking_labels','nearby_booking',('all_known',))]:
        old = json.loads((ROOT/f'reports/nov23_matched_learning_curve/{path}.json').read_text())
        if path=='booking_tree_messages':
            old = old['grouped_results']
        for part in reference:
            old = old[part]
        assert (report['all'][key]['fixed'],report['all'][key]['harmed'])==(old['fixed'],old['harmed'])
    report['scope'] = ('A new challenge requires Mitra to agree with the frozen parent before a source changes its '
        'decision. Learned control and augmented models share booking-held training/calibration splits. Only '
        'out-of-main-fold pair models enter the learned readout. The batch source is used in fixed combinations.')
    (ROOT/'reports/nov23_matched_learning_curve/combine_booking_messages.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
