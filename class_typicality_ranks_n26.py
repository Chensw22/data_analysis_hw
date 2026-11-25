"""Class-conditional conformal ranks of local density, scored on previously unseen bookings."""
import json
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import QuantileTransformer
from threadpoolctl import threadpool_limits
from exaone_validation_data import CATEGORICAL,compact_features
from held_expert_portfolios_n26 import FEES,ROOT,sources,outcome_scores,panel_predictions,ticker
from outside_panel_coverage_n26 import projection
from within_fold_outside_evidence_n26 import fit_probability


def matrix(frame,t):
    cat=pd.get_dummies(frame[CATEGORICAL].astype(str),dtype=float)
    num=frame[['Age','CabinNumber','party_size',*FEES]]
    quant=QuantileTransformer(n_quantiles=64,output_distribution='uniform',random_state=717).fit(num.iloc[t])
    return np.column_stack([cat,np.nan_to_num(quant.transform(num),nan=.5),num.isna()])


def typicalities(raw,y,g,fold):
    x=compact_features(raw)
    x['party_size']=raw.groupby(g).PassengerId.transform('size')
    paid=raw[FEES].gt(0).any(axis=1)
    strata=(x.Deck+'|'+x.CryoSleep+'|'+paid.astype(str)).to_numpy()
    out=np.full((len(y),2,2),np.nan)
    for k in np.unique(fold):
        t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        ranks=[]
        for j,(a,b) in enumerate(GroupKFold(2,shuffle=True,random_state=729).split(t,groups=g[t])):
            source,cal=t[a],t[b]
            assert not np.isin(g[q],g[t]).any() and not np.isin(g[source],g[cal]).any()
            feat=matrix(x,source)
            rng=np.random.default_rng(117+k*2+j)
            sham=y.copy()
            for subset in (source,cal):
                for key in np.unique(strata[subset]):
                    loc=subset[strata[subset]==key]
                    sham[loc]=rng.permutation(y[loc])
            p=np.zeros((len(q),2,2))
            for h,labels in enumerate((y,sham)):
                for c in (0,1):
                    train=source[labels[source]==c]
                    held=cal[labels[cal]==c]
                    lof=LocalOutlierFactor(n_neighbors=30,novelty=True,algorithm='brute',n_jobs=1).fit(feat[train])
                    ref=np.sort(lof.score_samples(feat[held]))
                    val=lof.score_samples(feat[q])
                    p[:,h,c]=(1+np.searchsorted(ref,val,side='right'))/(len(ref)+1)
            ranks.append(p)
        out[q]=np.mean(ranks,axis=0)
    assert np.isfinite(out).all()
    return out


def evaluate():
    raw,y,g,f,views,votes=projection(include_full=True)
    p=typicalities(raw,y,g,f)
    v=sources()[-1]['full']
    probs=outcome_scores(v,panel_predictions(v,'full'))
    dry=~raw[FEES].gt(0).any(axis=1).to_numpy()
    summary=np.quantile(probs,[.1,.5,.9],axis=1).T
    summary[dry]=np.quantile(probs[dry,:-2],[.1,.5,.9],axis=1).T
    basic=np.column_stack([logit(summary.clip(.004,.996)),dry,v['off']*np.where(v['anchor'],-1,1)])
    designs=[basic]
    for j in (0,1):
        z=logit(p[:,j,:].clip(.003,.997))
        delta=z[:,1]-z[:,0]
        designs.append(np.column_stack([basic,z,delta,delta*basic[:,1],delta*dry]))
    report={}
    for tag in ('full','bookingmix','historical_to_new'):
        valid,parent=views[tag]
        pred=np.full((len(y),3),np.nan)
        for k in np.unique(f):
            bank=np.flatnonzero(f==k)
            if tag=='historical_to_new':
                parts=[(bank[~np.isin(g[bank],g[valid])],bank[valid[bank]])]
            else:
                parts=[(bank[t],bank[q][valid[bank[q]]]) for t,q in GroupKFold(3,shuffle=True,random_state=618).split(bank,groups=g[bank])]
            for t,q in parts:
                if not len(q):
                    continue
                assert not np.isin(g[t],g[q]).any()
                for j,design in enumerate(designs):
                    pred[q,j]=fit_probability(design,t,q,y)
        assert np.isfinite(pred[valid]).all()
        original=(votes==parent[:,None]).all(1)
        new=~original&dry&(votes[:,:-2]==parent[:,None]).all(1)
        wrong=parent!=y
        flip=(pred>=.5)!=parent[:,None]
        real_rank=np.where(parent,p[:,0,0]-p[:,0,1],p[:,0,1]-p[:,0,0])
        fake_rank=np.where(parent,p[:,1,0]-p[:,1,1],p[:,1,1]-p[:,1,0])
        rules={'base':flip[:,0],'real':flip[:,1],'shuffled':flip[:,2],
               'only_real':flip[:,1]&~flip[:,0],'only_shuffled':flip[:,2]&~flip[:,0],
               'withdrawn_by_real':~flip[:,1]&flip[:,0],
               'raw_other_class_more_typical':real_rank>0,'raw_shuffled':fake_rank>0}
        masks={'all':valid,'original':valid&original,'added':valid&new,'disputed':valid&~original&~new,
               'parent_positive':valid&parent,'parent_negative':valid&~parent}
        report[tag]={n:{s:ticker(m,wrong,a) for s,m in masks.items()} for n,a in rules.items()}
        print(tag,{n:(vals['all']['fixed'],vals['all']['harmed']) for n,vals in report[tag].items()},flush=True)
        # Compare an error with a correct prediction of the same direction and similar base risk.
        risk=np.where(parent,1-pred[:,0],pred[:,0])
        cell=40*parent+np.digitize(risk,np.arange(.05,1,.05))
        comparisons={}
        for name,mask in masks.items():
            wins=np.zeros(2)
            pairs=0
            for key in np.unique(cell[mask]):
                ids=np.flatnonzero(mask&(cell==key))
                e=wrong[ids]
                n=int(e.sum())
                pairs+=n*(len(e)-n)
                for j,priority in enumerate((real_rank,fake_rank)):
                    wins[j]+=rankdata(priority[ids])[e].sum()-n*(n+1)/2
            comparisons[name]=dict(pairs=pairs,real=float(wins[0]/pairs) if pairs else None,
                                   shuffled=float(wins[1]/pairs) if pairs else None)
        report[tag]['within_direction_risk_band_auc']=comparisons
    (ROOT/'reports/nov23_matched_learning_curve/class_typicality_ranks.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        evaluate()
