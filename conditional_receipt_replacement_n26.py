"""Integrate actual held receipt predictions over matched training-order baskets at the passenger's exact total."""
import json
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits
from full_roster_booking_validation_n23 import ROOT,data
from solo_remaining_countervotes_n26 import context
from arbitration_against_scored_nine_n26 import champions

FEES=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
OUT=ROOT/'artifacts/nov26_conditional_receipt_replacement'
REPORT=ROOT/'reports/nov23_matched_learning_curve/conditional_receipt_replacement.json'


def inputs(raw,group):
    fees=raw[FEES].to_numpy()
    missing=np.isnan(fees)
    b=np.nan_to_num(fees).astype(int)
    total=b.sum(1)
    cabin=raw.Cabin.str.split('/',expand=True)
    size=raw.PassengerId.groupby(group).transform('size').to_numpy()
    cat=pd.DataFrame(dict(planet=raw.HomePlanet,deck=cabin[0],side=cabin[2],to=raw.Destination,
                          sleep=raw.CryoSleep,vip=raw.VIP)).astype('string').fillna('?')
    base=np.column_stack([pd.get_dummies(cat,dtype=float),raw.Age,pd.to_numeric(cabin[1]),size,missing,
                          np.log1p(total)])
    stratum=cat[['planet','deck','to']].to_numpy()
    return b,total,missing.astype(int)@(2**np.arange(5)),stratum,base


def features(base,b,mode):
    if mode=='budget':
        return base
    if mode=='active':
        return np.column_stack([base,b>0])
    return np.column_stack([base,b>0,np.log1p(b),np.sqrt(b/np.maximum(1,b.sum(1,keepdims=True)))])


def fit(x,y):
    m=HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,min_samples_leaf=20,
            learning_rate=.06,l2_regularization=4.,early_stopping=False,random_state=9702)
    return m.fit(x,y)


def allocate(donor,amount):
    # Leave one currency unit to every positive service, then apportion the remainder.
    active=donor>0
    n=active.sum(1)
    assert (amount>=n).all()
    remainder=np.maximum(donor-1,0)
    sums=remainder.sum(1)
    prop=np.where(sums[:,None]>0,remainder/np.maximum(1,sums)[:,None],active/n[:,None])
    values=prop*(amount-n)[:,None]
    floor=np.floor(values).astype(int)
    to_go=amount-n-floor.sum(1)
    # Mask inactive services even if every fractional component rounds to zero.
    ranks=np.argsort(np.argsort(-np.where(active,values-floor,-2),axis=1,kind='stable'),axis=1)
    result=active.astype(int)+floor+(ranks<to_go[:,None])
    np.testing.assert_array_equal(result.sum(1),amount)
    np.testing.assert_array_equal(result>0,active)
    return result


def swapped(t,q,g,b,tot,missing,strata,base,model,exact,seed):
    log_total=np.log1p(tot)
    active=b>0
    face=active.astype(int)@(2**np.arange(5))
    code=missing*32+face if exact else missing
    banks={k:t[(code[t]==k)&(tot[t]>0)] for k in np.unique(code[t])}
    values=[]
    targets=[]
    gaps=[]
    for s,idx in enumerate(q):
        if tot[idx]==0 or (exact and active[idx].sum()==1):
            continue
        bank=banks.get(code[idx],np.empty(0,int))
        amount_gap=np.abs(log_total[bank]-log_total[idx])
        valid=(amount_gap<=np.log(1.5))&(g[bank]!=g[idx])&(active[bank].sum(1)<=tot[idx])
        bank,amount_gap=bank[valid],amount_gap[valid]
        if not len(bank):
            continue
        distance=amount_gap+(strata[bank]!=strata[idx])@np.array([.20,.12,.08])
        best=np.argsort(distance,kind='stable')
        # Keep each contributing order once regardless of its number of passengers.
        _,first=np.unique(g[bank[best]],return_index=True)
        sel=best[np.sort(first)[:32]]
        values.extend(bank[sel])
        targets.extend([s]*len(sel))
        gaps.extend(amount_gap[sel])
    targets=np.asarray(targets,int)
    values=np.asarray(values,int)
    donor_probs=np.empty(0)
    if len(values):
        ids=q[targets]
        repl=allocate(b[values],tot[ids])
        assert not ((repl>0)&((missing[ids,None]&(2**np.arange(5)))!=0)).any()
        if exact:
            np.testing.assert_array_equal(repl>0,active[ids])
        donor_probs=model.predict_proba(features(base[ids],repl,'full'))[:,1]
    support=np.bincount(targets,minlength=len(q))
    total_logit=np.bincount(targets,weights=logit(donor_probs.clip(.001,.999)),minlength=len(q))
    summed=np.bincount(targets,weights=donor_probs,minlength=len(q))
    # Unsupported, zero-spend and single-service exact-face rows contribute exactly zero contrast.
    native=model.predict_proba(features(base[q],b[q],'full'))[:,1]
    average=np.where(support>0,summed/support.clip(1),native)
    center=np.where(support>0,total_logit/support.clip(1),logit(native.clip(.001,.999)))
    rng=np.random.default_rng(seed)
    draw=np.floor(rng.random(len(q))*support).astype(int)+np.cumsum(support)-support
    random=native.copy()
    random[support>0]=donor_probs[draw[support>0]]
    return average,center,random,support,np.asarray(gaps)


def compare(mask,pred,ref,y,fold):
    wrong=ref!=y
    edit=pred!=ref
    return dict(rows=int(mask.sum()),errors=int((mask&wrong).sum()),fixed=int((mask&edit&wrong).sum()),
                harmed=int((mask&edit&~wrong).sum()),fold_fixed_harmed=[
                [int((mask&edit&wrong&(fold==k)).sum()),int((mask&edit&~wrong&(fold==k)).sum())] for k in range(8)])


def run():
    raw,y,g,f=data()
    b,total,m,condition,base=inputs(raw,g)
    scores={}
    support={}
    diagnostics={}
    for k in range(8):
        t,q=np.flatnonzero(f!=k),np.flatnonzero(f==k)
        assert not np.isin(g[t],g[q]).any()
        direct={}
        for mode in ['budget','active','full']:
            x=features(base,b,mode)
            model=fit(x[t],y[t])
            direct[mode]=model.predict_proba(x[q])[:,1]
        real=logit(direct['full'].clip(.001,.999))
        for label,exact,prior in [('budget',False,'budget'),('face',True,'active')]:
            avg,center,random,count,gaps=swapped(t,q,g,b,total,m,condition,base,model,exact,5071+k)
            raw_prior=logit(direct[prior].clip(.001,.999))
            direct[label+'_averaged']=avg
            direct[label+'_random']=random
            direct[label+'_centered']=expit(raw_prior+real-center)
            direct[label+'_sham_centered']=expit(raw_prior+logit(random.clip(.001,.999))-center)
            support.setdefault(label,np.zeros(len(y),int))[q]=count
            diagnostics.setdefault(label,[]).append({'fold':k,'pairs':len(gaps),'median_budget_ratio':float(np.exp(np.median(gaps))),
                    'p90_budget_ratio':float(np.exp(np.quantile(gaps,.9)))})
        for name,v in direct.items():
            scores.setdefault(name,np.full(len(y),np.nan))[q]=v
    assert all(np.isfinite(v).all() for v in scores.values())
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'scores.npz',passenger=raw.PassengerId.to_numpy(str),fold=f,
                        **scores,**{'support_'+k:v for k,v in support.items()})
    _,truth,groups,fold,parent,_,_,wrong,masks=context()
    for a,v in [(y,truth),(g,groups),(f,fold),(wrong,parent!=y)]:
        np.testing.assert_array_equal(a,v)
    _,known,_,_=champions()
    assert (masks['full']&~wrong).sum()==1924 and (masks['full']&wrong).sum()==524
    subsets={'remaining':masks['full'],'remaining_bookingmix':masks['bookingmix'],
            'remaining_new':masks['historical_to_new'],'all_known':known,'remaining_paid':masks['full']&(total>0)}
    report={'protocol':'Eight whole-order folds. Three gradient boosting models share one specification and receive registration plus observed budget and missing indicators, then active service identities, then full charges. For held paying passengers, up to 32 distinct training orders within 1.5x observed total and identical missing-field pattern donate a service basket. Ranking uses distance in budget, origin, deck, destination. Integer apportionment preserves the query bill and the donor active services. The face control also holds all five service presence indicators constant. A seeded matched donor is the sham contrast. All training omits names, outcomes of query orders, and archive decisions.',
            'donors':diagnostics,'scopes':{}}
    for n,mask in subsets.items():
        results={}
        for key,values in scores.items():
            pred=values>=.5
            results[key]=compare(mask,pred,parent,y,f)
        for label,basekey in [('budget','budget'),('face','active')]:
            agrees=(scores[basekey]>=.5)==parent
            for key in ['centered','sham_centered']:
                altered=scores[label+'_'+key]>=.5
                pred=np.where(agrees,altered,parent)
                results[label+'_'+key+'_field_required']=compare(mask,pred,parent,y,f)
        report['scopes'][n]={'models_against_parent':results,
            'with_donors':{n:int((mask&(v>0)).sum()) for n,v in support.items()}}
    report['model_comparisons']={}
    for n,mask in {'all_8693':np.ones(len(y),bool),'paid':total>0,'remaining':masks['full'],
                   'remaining_paid':subsets['remaining_paid']}.items():
        report['model_comparisons'][n]={}
        for lhs,rhs in [('full','budget'),('full','active'),('full','budget_averaged'),('full','face_averaged'),
                ('budget_centered','budget'),('face_centered','active'),('budget_centered','budget_sham_centered'),
                ('face_centered','face_sham_centered')]:
            result=compare(mask,scores[lhs]>=.5,scores[rhs]>=.5,y,f)
            result['log_loss_difference']=float(log_loss(y[mask],scores[lhs][mask])-log_loss(y[mask],scores[rhs][mask]))
            report['model_comparisons'][n][lhs+'_vs_'+rhs]=result
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    for name in ['remaining','all_known']:
        r=report['scopes'][name]['models_against_parent']
        print(name,{k:(r[k]['fixed'],r[k]['harmed']) for k in ('budget','active','full','budget_centered_field_required',
               'budget_sham_centered_field_required','face_centered_field_required','face_sham_centered_field_required')},flush=True)


if __name__=='__main__':
    with threadpool_limits(1):
        run()
