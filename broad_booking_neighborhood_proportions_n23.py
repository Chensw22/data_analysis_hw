"""Infer passenger probabilities from predicted outcome proportions in overlapping booking neighborhoods."""
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import coordinates, cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from broad_service_bag_trees_n23 import fit

REPORT = ROOT / 'reports/nov23_booking_neighborhood_proportions'
OUT = ART / 'nov23_booking_neighborhood_proportions'
PASSES = 12


def neighborhoods(features, groups, seed):
    unique, inverse = np.unique(groups, return_inverse=True)
    n = len(groups)
    counts = np.bincount(inverse)
    group_features = np.zeros((len(unique), features.shape[1]))
    np.add.at(group_features, inverse, features)
    group_features /= counts[:, None]
    rng = np.random.default_rng(seed)
    rows = []
    start = 0
    # Split entire bookings. Different randomized projection trees give overlapping
    # neighborhoods, but every person belongs to exactly one leaf per partition.
    for _ in range(PASSES):
        ids = np.arange(len(unique))
        bins = np.empty(len(unique), int)
        leaves = [ids]
        while leaves:
            members = leaves.pop()
            if len(members) <= 20:
                bins[members] = start
                start += 1
                continue
            directions = rng.normal(size=(features.shape[1], 4))
            projected = group_features[members] @ directions
            axis = np.argmax(np.ptp(projected, axis=0))
            order = np.lexsort((rng.random(len(members)), projected[:, axis]))
            mid = len(order) // 2
            leaves.extend((members[order[:mid]], members[order[mid:]]))
        rows.append(bins[inverse])
    membership = csr_matrix((np.ones(n*PASSES), (np.concatenate(rows), np.tile(np.arange(n), PASSES))),
                            shape=(start, n))
    size = np.asarray(membership.sum(1)).ravel()
    np.testing.assert_array_equal(np.asarray(membership.sum(0)).ravel(), PASSES)
    assert np.all(size>0)
    return diags(1/size) @ membership, size


def summaries(a, size, x, native):
    b = np.column_stack([x, native, native >= .5, np.minimum(native, 1-native)])
    mean = a @ b
    sd = np.sqrt(np.maximum(a @ (b*b) - mean*mean, 0.))
    return np.column_stack([mean, sd, np.log(size)])


def infer(a, size, predicted, anchor):
    penalty = .18
    error = predicted-a@anchor
    rhs = np.asarray(a.T@(size*error)).ravel()/PASSES
    # Each independent partition has closed-form correction residual/(1+penalty)
    # for every member. Averaging the PASSES solutions gives rhs/(1+penalty).
    independent = rhs/(1+penalty)
    def multiply(v):
        return a.T@(size*(a@v))/PASSES+penalty*v
    joint, status = cg(LinearOperator((len(anchor),len(anchor)),matvec=multiply),rhs,rtol=1e-7,atol=1e-9)
    assert status==0 and np.max(abs(multiply(joint)-rhs))<1e-5
    return independent, joint


def fit_fold(frame, groups, y, t, q, base):
    assert not np.intersect1d(groups[t],groups[q]).size
    native_t = np.zeros(len(t))
    for left,right in GroupKFold(3,shuffle=True,random_state=481).split(t,groups=groups[t]):
        a,b=t[left],t[right]
        assert not np.intersect1d(groups[a],groups[b]).size
        native_t[right]=fit(frame.iloc[a],y[a],np.ones(len(a))).predict_proba(frame.iloc[b])[:,1]
    native_q=fit(frame.iloc[t],y[t],np.ones(len(t))).predict_proba(frame.iloc[q])[:,1]
    x,_=coordinates(frame,t)
    tr,nt=neighborhoods(x[t],groups[t],301)
    qr,nq=neighborhoods(x[q],groups[q],829)
    train_features=summaries(tr,nt,x[t],native_t)
    query_features=summaries(qr,nq,x[q],native_q)
    actual=tr@y[t]
    result={'direct':native_q, 'quarter_direct':.75*base+.25*native_q}
    details={'training_neighborhoods':len(nt),'query_neighborhoods':len(nq)}
    q_true=qr@y[q]
    for name in ('proportion','residual'):
        response=actual if name=='proportion' else actual-tr@native_t
        model=HistGradientBoostingRegressor(max_iter=105,learning_rate=.06,max_leaf_nodes=9,
                                            min_samples_leaf=24,l2_regularization=6.,
                                            early_stopping=False,random_state=490)
        model.fit(train_features,response,sample_weight=nt/nt.mean())
        predicted=model.predict(query_features)
        if name=='residual':
            predicted+=qr@native_q
        predicted=predicted.clip(0,1)
        # Truth is used only for this diagnostic, never for the reconstructed votes.
        details[name+'_brier']=float(np.average((q_true-predicted)**2,weights=nq))
        independent,joint=infer(qr,nq,predicted,native_q)
        for mode,difference in [('independent',independent),('joint',joint)]:
            result[name+'_'+mode]=(native_q+difference).clip(0,1)
            result[name+'_transfer_'+mode]=(base+difference).clip(0,1)
            result[name+'_quarter_'+mode]=.75*base+.25*result[name+'_'+mode]
    details['direct_brier']=float(np.average((q_true-qr@native_q)**2,weights=nq))
    return result,details


def main():
    row,y,frame,groups,_=load()
    wrong,right,hard=cohorts(row,y)
    plans,history={},{}
    OUT.mkdir(exist_ok=True)
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as z:
            np.testing.assert_array_equal(z['row'],row)
            np.testing.assert_array_equal(z['y'],y)
            base,ref,fold=z['base'],z['ref'],z['fold']
        notes,bank=[],{}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(groups,groups[q]))
            pred,details=fit_fold(frame,groups,y,t,q,base[q])
            notes.append(details)
            for key,p in pred.items():bank.setdefault(key,np.zeros(len(y)))[q]=p
        active=ref==(base>=.5)
        hits={k:np.where(active,p>=.5,ref)==y for k,p in bank.items()}
        hits['reference']=ref==y
        m={k:paired(v,hits['reference']) for k,v in hits.items()}
        plans[seed]=dict(metrics=m,folds=notes)
        if seed!='v131':
            for k,h in hits.items():history.setdefault(k,[]).append(h)
        np.savez_compressed(OUT/f'{seed}.npz',row=row,y=y,ref=ref,base=base,fold=fold,**bank)
        print(seed,{k:v['fixed']-v['harmed'] for k,v in m.items()},flush=True)
    prev=np.stack(history['reference'])
    stable={}
    for k,h in history.items():
        h=np.stack(h)
        stable[k]=dict(hard_stable=int((hard&h.all(0)).sum()),hard_ever=int((hard&h.any(0)).sum()),
                       common_stable=int((wrong&h.all(0)).sum()),stable_harms=int((right&~h.any(0)).sum()),
                       gain95_pp=interval((h.astype(float)-prev).mean(0),groups))
    write(REPORT/'proportions.json',dict(passes=PASSES,plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(2):
        main()
