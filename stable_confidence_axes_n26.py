"""Align continuous confidence patterns across two training populations, then predict held-out errors."""
import json

import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from disjoint_tree_fields_n25 import ART, DEST
from family_identity_transfer_n26 import fit, panels_and_reference
from solo_archive_recovery_n26 import archive, ticker
from solo_selection_likelihood_n26 import regimes


def panel(tag, ids, fold, base, view, group, truth, original, names):
    bank_names, other = archive(tag,ids,fold,group,truth,ART/'nov24_remaining_cases/archive_scores.npz',probabilities=True)
    categories=np.array([n.split('/')[0] for n in names])
    family=np.column_stack([original[:,categories==n].mean(1) for n in view['cohort']])
    np.testing.assert_array_equal(family>=.5,view['votes'])
    p=np.column_stack([family,other])
    anchor=view['anchor']
    against=np.where(anchor,-1,1)
    cols=against[:,None]*logit(p.clip(.002,.998))
    off=against*logit(base.clip(.002,.998))
    quant=np.quantile(cols,[.1,.25,.5,.75,.9],axis=1).T
    center=quant[:,2]
    # Leave the empirical confidence distribution available even to the anonymous control.
    basic=np.column_stack([np.ones(len(ids)),against,off,center-off,quant[:,0]-center,
                           quant[:,1]-center,quant[:,3]-center,quant[:,4]-center])
    basic=np.column_stack([basic,basic[:,3:]*against[:,None]])
    residual=cols-center[:,None]
    votes=p>=.5
    common=view['eligible']&(votes==anchor[:,None]).all(1)
    shared=set(names)&set(bank_names)
    for n in shared:
        np.testing.assert_array_equal(original[:,names.index(n)],other[:,bank_names.index(n)])
    return dict(basic=basic,relative=residual,off=off,known=view['known'],common=common,anchor=anchor,votes=votes,shared_configs=len(shared),
                wrong=anchor!=truth[ids],ids=ids,fold=fold,group=group[ids],names=[*view['cohort'],*bank_names])


def standardized(v, ix):
    # Remove the model-specific confidence expected from the anonymous spread and the original probability.
    a=v['basic'][ix]
    coef=np.linalg.solve(a.T@a+4*np.eye(a.shape[1]),a.T@v['relative'][ix])
    residual=v['relative']-v['basic']@coef
    return ((residual-residual[ix].mean(0))/residual[ix].std(0).clip(.15)).clip(-7,7)


def whiten(x):
    eig,vec=np.linalg.eigh(x.T@x/len(x))
    return (vec*(eig.clip(0)+.3)**-.5)@vec.T


def matched_donors(ix,v,rng):
    # Disrupt passenger identity while preserving direction and coarse confidence.
    bands=np.searchsorted(np.quantile(v['off'][ix],[.2,.4,.6,.8]),v['off'][ix])
    code=5*v['anchor'][ix]+bands
    out=ix.copy()
    for label in np.unique(code):
        where=code==label
        out[where]=rng.permutation(ix[where])
    return out


def calibrated(source,target,train,q,source_axes=None,target_axes=None):
    a,b=source['basic'][train],target['basic'][q]
    if source_axes is not None:
        a=np.column_stack([a,source_axes[train]])
        b=np.column_stack([b,target_axes[q]])
    coef=fit(a,source['off'][train],source['wrong'][train])
    return target['off'][q]+b@coef


def experiment(old,cur,oi,ci,train,q,seed):
    assert len(oi)==len(ci)
    np.testing.assert_array_equal(old['ids'][oi],cur['ids'][ci])
    assert not np.isin(cur['group'][q],old['group'][train]).any()
    old_x=standardized(old,oi)
    new_x=standardized(cur,ci)
    h,f=old_x[oi],new_x[ci]
    hwhite,fwhite=whiten(h),whiten(f)
    pred={'native':cur['off'][q],'anonymous':calibrated(old,cur,train,q),
          'all_named':calibrated(old,cur,train,q,old_x,new_x)}
    diagnostics={}
    for name,pair in [('matched',oi),('shuffled',matched_donors(oi,old,np.random.default_rng(seed)))]:
        cross=hwhite@(old_x[pair].T@f/len(oi))@fwhite
        left,_,right=np.linalg.svd(cross,full_matrices=False)
        a=old_x@(hwhite@left[:,:8])
        b=new_x@(fwhite@right[:8].T)
        a/=a[oi].std(0).clip(.1)
        b/=b[ci].std(0).clip(.1)
        pred[name]=calibrated(old,cur,train,q,a,b)
        if 'old_query' in cur:
            oq=cur['old_query'][q]
            correlations=[np.corrcoef(a[oq,j],b[q,j])[0,1] for j in range(8)]
            diagnostics[name]=float(np.mean(correlations))
    return pred,diagnostics


def metrics(batches,v):
    ids=np.concatenate([q for q,_ in batches])
    assert len(np.unique(ids))==len(ids)
    result={}
    for name in batches[0][1]:
        s=np.concatenate([p[name] for _,p in batches])
        table={}
        for c,mask in [('all',v['known'][ids]),('unanimous',v['common'][ids])]:
            wrong=v['wrong'][ids]
            count=ticker(mask,wrong,s>0)
            top=np.flatnonzero(mask)
            top=top[np.argsort(-s[top],kind='stable')[:int(np.ceil(len(top)*.1))]]
            table[c]=dict(**count,auc=float(roc_auc_score(wrong[mask],s[mask])),
                          nll=float(log_loss(wrong[mask],expit(s[mask]))),top_tenth=len(top),
                          top_tenth_errors=int(wrong[top].sum()))
        result[name]=table
    return result


def run():
    raw,y,g,views=regimes()
    _,_,_,rows,names,p,h,*_=panels_and_reference()
    old,cur=[panel(*v,g,y,a,names) for v,a in zip(views,[h,p],strict=True)]
    assert old['names']==cur['names'] and len(cur['names'])==89
    np.testing.assert_array_equal(old['ids'],rows)
    assert (cur['common']&cur['wrong']).sum()==245
    lookup=np.full(len(y),-1)
    lookup[rows]=np.arange(len(rows))
    fresh=~np.isin(g,g[rows])&cur['known']
    paired=[];unseen=[];local=[];corr=[]
    for k in np.unique(cur['fold']):
        pool=np.flatnonzero(cur['known']&(cur['fold']==k))
        for t,q in GroupKFold(3,shuffle=True,random_state=1871).split(pool,groups=g[pool]):
            t,q=pool[t],pool[q]
            assert not np.isin(g[t],g[q]).any()
            unit=standardized(cur,t)
            local.append((q,{'native':cur['off'][q],'anonymous':calibrated(cur,cur,t,q),
                             'all_named':calibrated(cur,cur,t,q,unit,unit)}))
        oi=np.flatnonzero(old['known']&cur['known'][rows]&(cur['fold'][rows]==k))
        ci=rows[oi]
        q=pool[fresh[pool]]
        assert not np.isin(g[q],g[ci]).any()
        pred,_=experiment(old,cur,oi,ci,np.flatnonzero(old['known']),q,801+k)
        unseen.append((q,pred))
        for j in np.unique(old['fold']):
            intersect=oi[old['fold'][oi]==j]
            for a,b in GroupKFold(2,shuffle=True,random_state=3276).split(intersect,groups=g[rows[intersect]]):
                t,held=intersect[a],intersect[b]
                q=rows[held]
                pred,diagnostic=experiment(old,{**cur,'old_query':lookup},t,rows[t],t,q,924+j+8*k)
                paired.append((q,pred))
                corr.append(diagnostic)
    output={'columns':cur['names'],'shared_individual_configurations':cur['shared_configs'],'all_current':metrics(local,cur),'matched_passengers':metrics(paired,cur),
            'new_orders':metrics(unseen,cur),'held_passenger_mean_axis_correlation':{
                n:float(np.mean([entry[n] for entry in corr])) for n in corr[0]}}
    print(output['held_passenger_mean_axis_correlation'])
    for task in ['all_current','matched_passengers','new_orders']:
        print(task,[(n,v['all']['fixed'],v['all']['harmed'],v['unanimous']['top_tenth_errors']) for n,v in output[task].items()])
    DEST.with_name('stable_confidence_axes.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
