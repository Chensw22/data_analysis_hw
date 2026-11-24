"""Fit weighted PLS-DA separately for each query and audit disagreement-region errors."""
import json

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cross_decomposition import PLSRegression
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_class_neighborhood_mass_n24 import protocol
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_mistake_influence_n24 import design
from full_roster_six_failure_archive_n24 import prediction_sets

MODES = ('global_four','random_four','local_rate','local_one','local_four')
TAG = 'broad_error_transport_local_pls'


def regress(train,truth,ask,weight):
    """Simultaneous weighted one-response PLS on batches of independent neighborhoods."""
    weight = weight / weight.sum(1,keepdims=True)
    mean = (weight[:,:,None]*train).sum(1)
    rate = (weight*truth).sum(1)
    root = np.sqrt(weight)
    x = (train - mean[:,None,:])*root[:,:,None]
    residual = (truth - rate[:,None])*root
    point = ask-mean
    estimate = rate.copy()
    out = [rate]
    for k in range(4):
        direction = (x*residual[:,:,None]).sum(1)
        norm = np.linalg.norm(direction,axis=1)
        direction /= np.maximum(norm[:,None],1e-12)
        comp = np.einsum('bnd,bd->bn',x,direction)
        length = (comp*comp).sum(1)
        safe = np.where(length>1e-12,length,1.)
        slope = (comp*residual).sum(1)/safe
        loading = (x*comp[:,:,None]).sum(1)/safe[:,None]
        held = (point*direction).sum(1)
        estimate += held*slope
        point -= held[:,None]*loading
        x -= comp[:,:,None]*loading[:,None,:]
        residual -= comp*slope[:,None]
        if k in (0,3):
            out.append(estimate.copy())
    return np.column_stack(out).clip(.001,.999)


def check_numerics():
    rng = np.random.default_rng(380)
    x = rng.normal(size=(70,9))
    y = x[:,0]+.4*x[:,3]+rng.normal(size=70) > .2
    q = rng.normal(size=(6,9))
    fitted = PLSRegression(n_components=4,scale=False).fit(x,y)
    observed = regress(np.tile(x[None,:,:],(len(q),1,1)),np.tile(y,(len(q),1)),q,
                       np.ones((len(q),len(y))))[:,-1]
    np.testing.assert_allclose(observed,fitted.predict(q).ravel().clip(.001,.999),atol=1e-9)
    # Integer weights are equivalent to replicating the respective rows.
    weights = rng.integers(1,4,len(y))
    repeated = np.repeat(np.arange(len(y)),weights)
    one = regress(x[None,:,:],y[None,:],q[:1],weights[None,:])
    two = regress(x[None,repeated,:],y[None,repeated],q[:1],np.ones((1,len(repeated))))
    np.testing.assert_allclose(one,two,atol=1e-9)


def predict(x,y,g,t,q):
    assert not np.isin(g[t],g[q]).any()
    z = ((x-x[t].mean(0))/x[t].std(0).clip(.3)).clip(-6.,6.)
    model = PLSRegression(n_components=4,scale=False).fit(z[t],y[t])
    global_score = model.predict(z[q]).ravel().clip(.001,.999)
    distance,idx = cKDTree(z[t]).query(z[q],k=256)
    ids = t[idx]
    weight = np.exp(-2*(distance/distance[:,-1:].clip(.01))**2)
    rng = np.random.default_rng(631)
    local, random = [],[]
    for start in range(0,len(q),48):
        chunk = slice(start,start+48)
        donor = ids[chunk]
        qr = z[q[chunk]]
        local.append(regress(z[donor],y[donor],qr,weight[chunk]))
        control = np.stack([rng.choice(t,size=256,replace=False) for _ in qr])
        random.append(regress(z[control],y[control],qr,weight[chunk])[:,-1])
    return np.column_stack([global_score,np.concatenate(random),np.vstack(local)])


def run():
    check_numerics()
    raw,y,g,fold = data()
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    names,h,b = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,b)
    _,panels = prediction_sets(raw,y,g,fold,rows,of)
    x = design(raw)
    result = {}
    for (tag,ids,f,base,anchor),(six,archive) in zip([
             ('full',np.arange(len(y)),fold,ref,b[:,names.index('mitra96')]),
             ('historical',rows,of,champ,old[:,0])],panels,strict=True):
        truth = y[ids]
        probabilities = protocol(x,y,g,ids,f,tag,predict,MODES,TAG)
        same = ((six>=.5)==base[:,None]).all(1) & ((archive>=.5)==base[:,None]).all(1)
        eligible = ~same
        errors = truth!=base
        if tag=='full':
            assert (sum(same&errors),sum(eligible&errors))==(240,1335)
        summary = {}
        for j,kind in enumerate(MODES):
            p = probabilities[:,j]
            for mode,pred in [('direct',p>=.5),('quarter',np.where((anchor>=.5)==base,.75*anchor+.25*p>=.5,base))]:
                def evaluate(mask):
                    return stats(pred[mask],base[mask],truth[mask],g[ids][mask],f[mask])
                r = dict(full=evaluate(np.ones(len(truth),bool)),disagreement=evaluate(eligible),
                         unanimity=evaluate(same))
                if tag=='full':
                    r['new_orders'] = evaluate(~np.isin(g[ids],og))
                summary[kind+'_'+mode] = r
                print(tag,kind,mode,r['full']['corrected'],r['full']['harmed'],r['disagreement']['corrected'],
                      r['disagreement']['harmed'],r['unanimity']['corrected'],flush=True)
        local_right = (probabilities[:,-1]>=.5)==truth
        other_right = ((probabilities[:,:-1]>=.5)==truth[:,None]).any(1)
        result[tag] = dict(eligible=int(eligible.sum()),disagreement_errors=int((errors&eligible).sum()),
                           unanimity_errors=int((errors&same).sum()),models=summary,
                           local_four_exclusive=int((errors&local_right&~other_right).sum()))
    path = ROOT/'reports/nov23_matched_learning_curve'/f'{TAG}.json'
    path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
