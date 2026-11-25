"""Learn a sparse additive correction to receipt-tree decisions from genuinely nested weak-model margins."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from full_roster_booking_validation_n23 import data,ROOT
from tabnet_attention_transfer_n26 import collect
from tabnet_history_test_n26 import historical_indices

OUT=ROOT/'artifacts/nov26_historical_margin_stumps'
REPORT=ROOT/'reports/nov23_matched_learning_curve/historical_margin_stumps.json'
FEES=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
TEACHERS=['total_tree','active_tree','amount_tree','amount_additive']


def matrices(raw,g):
    cabin=raw.Cabin.str.split('/',expand=True)
    cols=[np.log(raw.PassengerId.groupby(g).transform('size'))]
    for field,levels in [(raw.HomePlanet,['Earth','Mars','Europa']), (raw.CryoSleep,[True,False]),
            (raw.Destination,['TRAPPIST-1e','55 Cancri e','PSO J318.5-22']), (raw.VIP,[True,False]),
            (cabin[0],list('ABCDEFGT')),(cabin[2],['S','P'])]:
        cols.extend([field.eq(v).fillna(False).to_numpy() for v in levels])
        cols.append(field.isna())
    bills=raw[FEES].to_numpy()
    for arr in [raw.Age.to_numpy()/80,pd.to_numeric(cabin[1]).to_numpy()/1800,
                np.log1p(np.nansum(bills,axis=1))/10]:
        x=np.nan_to_num(arr)
        cols.extend([~np.isfinite(arr),x,*[np.maximum(x-v,0) for v in [.25,.5,.75]]])
    budget=np.column_stack([*cols,np.isnan(bills)])
    active=np.column_stack([budget,bills>0])
    log=np.log1p(np.nan_to_num(bills))/9
    full=np.column_stack([active,log,*[np.maximum(log-k,0) for k in [.35,.6,.8]]])
    return [budget,active,full,full]


def weak(x,xt,t,outer_train,g,train_labels,seed):
    n=len(outer_train)
    soft=np.full((n,4),np.nan)
    panels=[]
    inner=GroupKFold(3,shuffle=True,random_state=seed)
    for a,b in inner.split(outer_train,groups=g[outer_train]):
        train,verify=outer_train[a],outer_train[b]
        assert not np.isin(g[train],g[verify]).any()
        query=np.zeros((len(t),4))
        for j,xx in enumerate(x):
            model=(HistGradientBoostingClassifier(max_iter=130,max_leaf_nodes=13,l2_regularization=5.,
                    learning_rate=.07,early_stopping=False,random_state=656)
                   if j<3 else LogisticRegression(C=3,max_iter=500))
            model.fit(xx[train],train_labels[train])
            soft[b,j]=model.predict_proba(xx[verify])[:,1]
            query[:,j]=model.predict_proba(xt[j][t])[:,1]
        panels.append(query)
    assert np.isfinite(soft).all()
    return soft,np.stack(panels)


def design(p):
    base=p[:,2]>=.5
    direction=np.where(base,-1,1)
    evidence=(2*p-1)*direction[:,None]
    labels=['amount_tree_predicts_positive',*[n+'_opposition' for n in TEACHERS]]
    cols=[base,*evidence.T]
    for j in range(4):
        for k in range(j):
            cols.append(evidence[:,j]-evidence[:,k])
            labels.append(f'{TEACHERS[j]}_minus_{TEACHERS[k]}')
    cols.append((evidence>0).sum(1)/3)
    labels.append('fraction_opposing')
    return np.column_stack(cols),base,labels


def stumps(x):
    return (x[:,:,None]>np.array([-.75,-.4,-.15,0.,.15,.4,.75])[None,None,:]).reshape(len(x),-1).astype(float)


def train_meta(p,truth):
    x,base,names=design(p)
    wrong=(base!=truth).astype(float)
    off=logit(np.minimum(p[:,2],1-p[:,2]).clip(.005,.995))
    xx=np.column_stack([np.ones(len(x)),x])
    def loss(w):
        s=off+xx@w
        return np.mean(np.logaddexp(0,s)-wrong*s)+.004*(w@w)/2, xx.T@(expit(s)-wrong)/len(x)+.004*w
    opt=minimize(loss,np.zeros(xx.shape[1]),jac=True,method='L-BFGS-B')
    assert opt.success
    z=stumps(x)
    means=z.mean(0)
    z-=means
    useful=(means>.025)&(means<.975)
    value=off.copy()
    intercept=0.
    coef=np.zeros(z.shape[1])
    # Forward stagewise additive logistic loss. All 32 steps operate on training orders only.
    for _ in range(32):
        prob=expit(value)
        residual=wrong-prob
        hessian=prob*(1-prob)
        jump=.3*residual.sum()/(hessian.sum()+6.)
        intercept+=jump
        value+=jump
        prob=expit(value)
        gradient=z.T@(wrong-prob)
        curvature=(z*z).T@(prob*(1-prob))+12
        idx=int(np.argmax(np.where(useful,gradient**2/curvature,-np.inf)))
        step=.3*gradient[idx]/curvature[idx]
        value+=step*z[:,idx]
        coef[idx]+=step
    terms=[dict(feature=names[i//7],threshold=[-.75,-.4,-.15,0.,.15,.4,.75][i%7],
                coefficient=float(v),training_prevalence=float(means[i])) for i,v in enumerate(coef) if v]
    return opt.x,means,coef,intercept,terms


def apply_meta(p,fit):
    x,base,_=design(p)
    w,means,coef,intercept,_=fit
    off=logit(np.minimum(p[:,2],1-p[:,2]).clip(.005,.995))
    linear=expit(off+np.column_stack([np.ones(len(x)),x])@w)
    sparse=expit(off+intercept+(stumps(x)-means)@coef)
    return np.column_stack([p[:,2],np.where(base,1-linear,linear),np.where(base,1-sparse,sparse)])


def measures(ids,scope,ref,y,fold,p):
    truth=y[ids]
    direct,linear,sparse=(p.mean(0)>=.5).T
    wrong=ref!=truth
    result={}
    for name,edit in dict(no_correction=direct!=ref,linear=linear!=ref,sparse=sparse!=ref,
           linear_required=(direct==ref)&(linear!=ref),sparse_required=(direct==ref)&(sparse!=ref),
           repeated_sparse_required=np.all((p[:,:,0]>=.5)==ref,axis=0)&np.all((p[:,:,2]>=.5)!=ref,axis=0)).items():
        used=edit&scope
        result[name]=dict(rows=int(scope.sum()),errors=int((scope&wrong).sum()),fixed=int((used&wrong).sum()),
             harmed=int((used&~wrong).sum()),fold_fixed_harmed=[[int((used&wrong&(fold[ids]==k)).sum()),
             int((used&~wrong&(fold[ids]==k)).sum())] for k in range(8)])
    return result


def run():
    raw,y,g,fold=data()
    h=np.sort(np.concatenate([historical_indices(raw,y,g,k)[1] for k in range(3)]))
    assert len(np.unique(h))==len(h)==6954
    new=np.flatnonzero(~np.isin(g,g[h]))
    x=matrices(raw.iloc[h],g[h])
    xt=matrices(raw,g)
    OUT.mkdir(exist_ok=True)
    cross=np.full((3,len(y),3),np.nan)
    rules={}
    for k in range(9):
        tag=f'fold{k}' if k<8 else 'all_history_to_new'
        q=np.flatnonzero(fold==k) if k<8 else new
        t=np.flatnonzero(~np.isin(g[h],g[q]))
        assert not np.isin(g[h[t]],g[q]).any()
        cache=OUT/f'{tag}.npz'
        if cache.exists():
            with np.load(cache) as z:
                for key,want in [('t',h[t]),('q',q),('train_y',y[h[t]])]:
                    np.testing.assert_array_equal(z[key],want)
                soft,weak_query=z['soft'],z['weak_query']
        else:
            soft,weak_query=weak(x,xt,q,t,g[h],y[h],3217+k)
            np.savez_compressed(cache,t=h[t],q=q,train_y=y[h[t]],soft=soft,weak_query=weak_query)
        fitted=train_meta(soft,y[h[t]])
        scores=np.stack([apply_meta(v,fitted) for v in weak_query])
        if k<8:
            cross[:,q,:]=scores
        else:
            new_scores=scores
        rules[tag]=dict(train_rows=len(t),train_bookings=len(np.unique(g[h[t]])),
                        intercept=float(fitted[3]),terms=fitted[-1])
        print(tag,len(q),len(fitted[-1]),flush=True)
    assert np.isfinite(cross).all()
    np.savez_compressed(OUT/'scores.npz',fold=fold,ids=raw.PassengerId.to_numpy(str),p=cross,new=new,new_p=new_scores)
    runs,truth,groups,f,versions=collect()
    for a,b in [(y,truth),(g,groups),(f,fold)]:
        np.testing.assert_array_equal(a,b)
    report=dict(protocol='Every outer training passenger and their three-fold-out probabilities come from the 6954 historic passengers. Both levels purge whole query bookings. The dedicated transfer retunes nothing and trains on all historic orders before scoring the 1044 entirely new passengers. No names. Four models give probabilities for budget, active services, exact amounts and additive exact amounts. The correction predicts errors of the exact-amount tree from each signed opinion, six pair gaps and proportion opposing, with that tree confidence as log-odds offset. Threshold candidates and 32 forward steps fixed. Linear control sees the same continuous inputs and offset. Raw feature coverage is identical in both corrections.',
            rules=rules,scopes={},independent_of_parent={})
    for tag,children in runs.items():
        ids,same,ref=(np.concatenate([r[j] for r in children]) for j in range(3))
        keep=same & ~np.concatenate([r[3]['actual_and_scores'] for r in children])
        if tag=='historical_to_new':
            assert np.isin(ids,new).all()
            s=new_scores[:,np.searchsorted(new,ids),:]
        else:
            s=cross[:,ids,:]
        report['scopes'][tag]={'all_known':measures(ids,np.ones(len(ids),bool),ref,y,fold,s),
                               'remaining':measures(ids,keep,ref,y,fold,s)}
        if tag=='full':
            solo=(raw.PassengerId.groupby(g).transform('size').to_numpy()[ids]==1)
            for tier,m in [('remaining_solo',keep&solo),('remaining_multiple',keep&~solo)]:
                report['scopes'][tag][tier]=measures(ids,m,ref,y,fold,s)
    for tag,ids,values in [('all_8693',np.arange(len(y)),cross),('all_new_1044',new,new_scores)]:
        base=values[:,:,0].mean(0)>=.5
        result=measures(ids,np.ones(len(ids),bool),base,y,fold,values)
        report['independent_of_parent'][tag]=dict(accuracy={
            nm:float(((values[:,:,i].mean(0)>=.5)==y[ids]).mean()) for i,nm in enumerate(['amount_tree','linear','sparse'])},
            changes_vs_amount_tree=result)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    for tag,scopes in report['scopes'].items():
        print(tag,{key:(m['fixed'],m['harmed']) for key,m in scopes['remaining'].items()},flush=True)


if __name__=='__main__':
    with threadpool_limits(1):
        run()
