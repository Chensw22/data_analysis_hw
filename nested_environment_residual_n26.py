"""Maximin aggregation of manifest residual functions trained on disjoint departure populations."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits
from full_roster_booking_validation_n23 import ROOT, data
from tabnet_attention_transfer_n26 import collect
from target_mix_expert_routing_n26 import ticker

CACHE = ROOT / 'artifacts/nov26_booking_block_teacher/nested'
OUT = ROOT / 'artifacts/nov26_nested_environment_residual'
REPORT = ROOT / 'reports/nov23_matched_learning_curve/nested_environment_residual.json'
FEES = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']


def fields(raw, size):
    cabin = raw.Cabin.str.split('/',expand=True)
    cols = [np.log(size)]
    for s,levels in [(cabin[0],list('ABCDEFGT')),(cabin[2],['S','P']),
                     (raw.Destination,['TRAPPIST-1e','55 Cancri e','PSO J318.5-22']),
                     (raw.CryoSleep,[True,False]),(raw.VIP,[True,False])]:
        cols.extend([s.eq(v).fillna(False).to_numpy() for v in levels])
        cols.append(s.isna())
    numeric = [(raw.Age/80,[.15,.25,.45,.65]),
               (pd.to_numeric(cabin[1])/1800,[.2,.4,.6,.8])]
    numeric.extend((np.log1p(raw[n])/9.,[.35,.55,.7,.85]) for n in FEES)
    for series,knots in numeric:
        a = series.fillna(0).to_numpy()
        cols.extend([series.isna(),series.eq(0),a,*[np.maximum(a-v,0.) for v in knots]])
    return np.column_stack(cols).astype(float)


def linear_offset(x,offset,y):
    def objective(beta):
        log_odds = offset+x@beta
        loss = np.mean(np.logaddexp(0.,log_odds)-y*log_odds)+.004*(beta@beta)/2
        grad = x.T@(expit(log_odds)-y)/len(y)+.004*beta
        return loss,grad
    opt = minimize(objective,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',
                   options=dict(maxiter=350,ftol=1.e-10))
    if not opt.success:
        raise ValueError(opt.message)
    return opt.x


def teacher_fields(p):
    return np.column_stack([np.ones(len(p)),logit(p.clip(.002,.998))])


def modify(x, p):
    # A manifest effect is allowed to differ between low- and high-teacher-probability passengers.
    prior = p.mean(1)-.5
    return np.column_stack([np.ones(len(x)),x,x*prior[:,None]])


def common_weights(x,betas):
    # Magging finds the point in the convex hull of residual functions closest to zero.
    pred = x@betas
    gram = pred.T@pred/len(x)
    start = np.full(betas.shape[1],1/betas.shape[1])
    opt = minimize(lambda w: (w@gram@w,2*gram@w),start,jac=True,bounds=[(0,1)]*len(start),
                   constraints=[dict(type='eq',fun=lambda w:w.sum()-1)],method='SLSQP',
                   options=dict(ftol=1.e-12))
    if not opt.success:
        raise ValueError(opt.message)
    return opt.x


def random_environments(g, observed, seed):
    # Permute booking labels among equally sized observed departure bookings.
    groups, first, inverse, num = np.unique(g,return_index=True,return_inverse=True,return_counts=True)
    orig = observed[first]
    np.testing.assert_array_equal(orig[inverse],observed)
    new = orig.copy()
    rng = np.random.default_rng(seed)
    for n in np.unique(num):
        ids = np.flatnonzero((num==n)&(orig>=0))
        new[ids] = rng.permutation(orig[ids])
    return new[inverse]


def trial(x,t,q,g,env,p,held,truth,seed):
    use = env[t]>=0
    mu = x[t][use].mean(0)
    sd = x[t][use].std(0).clip(.1)
    x0 = np.clip((x[t]-mu)/sd,-6,6)
    x1 = np.clip((x[q]-mu)/sd,-6,6)
    # Fit the same opinion-only reference in every arm.
    priors = teacher_fields(p)
    head = linear_offset(priors[use],np.zeros(use.sum()),truth[t][use])
    offset = priors@head
    design = modify(x0,p)
    initial = linear_offset(design[use],offset[use],truth[t][use])
    query_features = [modify(x1,score) for score in held]
    query_prior = np.stack([teacher_fields(score)@head for score in held])

    def predict(beta):
        return np.stack([expit(base+part@beta) for base,part in zip(query_prior,query_features,strict=True)])

    scores = dict(prior=expit(query_prior),pooled=predict(initial))
    regime_weights = {}
    for name,region in [('departure',env[t]),('shuffled',random_environments(g[t],env[t],seed))]:
        bank = np.column_stack([linear_offset(design[region==k],offset[region==k],truth[t][region==k])
                                for k in range(3)])
        weight = common_weights(design[use],bank)
        regime_weights[name] = weight.tolist()
        scores[name+'_equal'] = predict(bank.mean(1))
        scores[name+'_maximin'] = predict(bank@weight)
        # Unanimous support uses all departure models separately on each held teacher panel.
        scores[name+'_all'] = np.concatenate([predict(b) for b in bank.T],axis=0)
    return scores,regime_weights


def run():
    raw,y,g,fold = data()
    # For bookings with several declarations, first known departure assigns the whole order one environment.
    departure = raw.HomePlanet.groupby(g).transform('first').map({'Earth':0,'Europa':1,'Mars':2}).fillna(-1).to_numpy(int)
    size = raw.PassengerId.groupby(g).transform('size').to_numpy()
    x = fields(raw,size)
    results = {}
    weights = []
    OUT.mkdir(exist_ok=True)
    for k in range(8):
        t,q = np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        with np.load(CACHE/f'fold{k}.npz') as z:
            for key,expected in [('t',t),('q',q),('train_y',y[t])]:
                np.testing.assert_array_equal(z[key],expected)
            assert not np.isin(g[t],g[q]).any()
            for j in range(3):
                assert not np.isin(g[t[z['inner_fold']==j]],g[t[z['inner_fold']!=j]]).any()
            score,w = trial(x,t,q,g,departure,z['soft'],z['outer_single'],y,407+k)
        weights.append(w)
        for name,p in score.items():
            results.setdefault(name,np.full((p.shape[0],len(y)),np.nan))[:,q] = p
    assert all(np.isfinite(v).all() for v in results.values())
    np.savez_compressed(OUT/'scores.npz',ids=raw.PassengerId.to_numpy(str),y=y,fold=fold,**results)
    archive,truth,groups,f,_ = collect()
    for a,b in [(y,truth),(g,groups),(f,fold)]:
        np.testing.assert_array_equal(a,b)
    report = dict(protocol='Three nested teachers, eight unseen booking folds. The base calibrates the three logits on outer-training orders. Manifest features, interactions with mean teacher probability, and L2 coefficient are identical across the pooled and regional fits. Each departure-specific model learns a log-odds correction; maximin combines them with the smallest training-population L2 norm. Shuffled assigns departure at booking level within booking size. The environmental models exclude unknown-departure training bookings; every passenger is scored. No names.',weights=weights, views={})
    for tag,rows in archive.items():
        ids,same,parent = (np.concatenate([r[j] for r in rows]) for j in range(3))
        scope = same & ~np.concatenate([r[3]['actual_and_scores'] for r in rows])
        wrong = y[ids]!=parent
        report['views'][tag] = {}
        for key,v in results.items():
            posterior = v[:,ids]
            if key.endswith('_all'):
                act = ((posterior>=.5)!=parent[None,:]).all(0)
            else:
                act = (posterior.mean(0)>=.5)!=parent
            prior_agrees = (results['prior'][:,ids].mean(0)>=.5)==parent
            actions = {key:act}
            if key!='prior':
                actions[key+'_field_required'] = act & prior_agrees
            for action_name,edit in actions.items():
                summary = {}
                for name,mask in [('all',scope),('solo',scope&(size[ids]==1)),('multiple',scope&(size[ids]>1))]:
                    correction = ticker(mask,wrong,edit)
                    correction['fold_fixed_harmed'] = [[int((mask&edit&wrong&(fold[ids]==k)).sum()),
                                                        int((mask&edit&~wrong&(fold[ids]==k)).sum())] for k in range(8)]
                    summary[name] = correction
                report['views'][tag][action_name] = summary
        print(tag,{key:(v['solo']['fixed'],v['solo']['harmed']) for key,v in report['views'][tag].items()},flush=True)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
