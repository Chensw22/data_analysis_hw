"""Test whether independent flanks of an unseen booking corroborate an unusual run of transport outcomes."""
import json

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from family_identity_transfer_n26 import panels_and_reference
from full_roster_booking_validation_n23 import ROOT, data
from grouped_peer_teacher_n25 import OUT

RESULT=ROOT/'reports/nov23_matched_learning_curve/bilateral_residual_runs.json'


def scan_channel(donor,query,coordinate,group,key,teacher,labels,shuffle,seed):
    # Collapse to one mark per booking within the specified deck/side and activity class.
    donors=pd.DataFrame(dict(row=donor,group=group[donor],key=key[donor],coord=coordinate[donor],
                            resid=labels-teacher,var=teacher*(1-teacher)))
    donors=donors.dropna(subset=['coord']).groupby(['key','group'],as_index=False).agg(
        coord=('coord','median'),resid=('resid','mean'),var=('var','mean'))
    out=np.zeros((len(query),4))
    rng=np.random.default_rng(seed)
    for k,records in donors.groupby('key',sort=True):
        ix=np.flatnonzero((key[query]==k)&np.isfinite(coordinate[query]))
        if not len(ix) or len(records)<4:
            continue
        records=records.sort_values(['coord','group'])
        position=records.coord.to_numpy()
        marks=records[['resid','var']].to_numpy()
        if shuffle:
            marks=marks[rng.permutation(len(marks))]
        prefix=np.vstack([np.zeros(2),np.cumsum(marks,axis=0)])
        v=coordinate[query[ix]]
        left=np.searchsorted(position,v,side='left')
        right=np.searchsorted(position,v,side='right')
        for j,span in enumerate((6,24)):
            a,b=np.maximum(0,left-span),np.minimum(len(position),right+span)
            left_sum=prefix[left]-prefix[a]
            right_sum=prefix[b]-prefix[right]
            l=left_sum[:,0]/np.sqrt(1+left_sum[:,1])
            r=right_sum[:,0]/np.sqrt(1+right_sum[:,1])
            out[ix,2*j]=(l+r)/2
            # Corroboration vanishes if one flank supports higher and the other lower odds.
            out[ix,2*j+1]=np.where(l*r>0,np.sign(l)*np.minimum(abs(l),abs(r)),0)
    return out


def run():
    raw,y,g,fold=data()
    _,_,_,rows,_,p,h,native,_,ref=panels_and_reference()
    minority=np.minimum((p>=.5).sum(1),(p<.5).sum(1))
    unanimous=minority==0
    consensus=p[:,0]>=.5
    repeat=np.zeros(len(y),bool)
    repeat[rows]=unanimous[rows]&(((h>=.5)==consensus[rows,None]).all(1))
    fresh=~np.isin(g,g[rows])
    berth=raw.Cabin.str.split('/',expand=True)
    key=(berth[0]+'/'+berth[2]).fillna('unknown').to_numpy(str)
    num=pd.to_numeric(berth[1]).to_numpy(float)
    state=np.where(raw.CryoSleep.eq(True),'sleep',np.where(raw.CryoSleep.eq(False),'awake','missing'))
    side_state=np.char.add(np.char.add(key,'/'),state)
    order=g.astype(float)
    feature_sets={}
    random_runs=np.zeros((len(y),31))
    for k in np.unique(fold):
        t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        with np.load(OUT/f'outer{k}.npz') as z:
            np.testing.assert_array_equal(t,z['t'])
            np.testing.assert_array_equal(y[t],z['labels'])
            teacher=z['p']
        assert not np.isin(g[q],g[t]).any()
        base=np.column_stack([logit(native[q].clip(.002,.998)),np.median(logit(p[q].clip(.005,.995)),axis=1)])
        feature_sets.setdefault('anchor',np.zeros((len(y),base.shape[1])))[q]=base
        for shuffled in (False,True):
            blocks=[]
            for j,(coordinate,strata) in enumerate([(num,key),(num,side_state),(order,state)]):
                block=scan_channel(t,q,coordinate,g,strata,teacher,y[t],shuffled,2691+91*k+j)
                blocks.append(block)
            signals=np.column_stack(blocks)
            for tag,feat in [('means',signals[:,::2]),('both_flanks',signals),('flanks_only',signals[:,1::2])]:
                name=tag+('_permuted' if shuffled else '')
                matrix=np.column_stack([base,feat])
                feature_sets.setdefault(name,np.zeros((len(y),matrix.shape[1])))[q]=matrix
        for replicate in range(random_runs.shape[1]):
            blocks=[]
            for j,(coordinate,strata) in enumerate([(num,key),(num,side_state),(order,state)]):
                v=scan_channel(t,q,coordinate,g,strata,teacher,y[t],True,70331+1009*replicate+91*k+j)
                blocks.append(v[:,1::2])
            random_runs[q,replicate]=np.column_stack(blocks).mean(1)
    prior,lo,hi=ref
    known=prior>=0
    nine=np.where(prior.astype(bool),hi,lo)
    report={}
    for protocol in ('all','fresh'):
        splits=[]
        for k in np.unique(fold):
            ids=np.flatnonzero(fold==k)
            if protocol=='fresh':
                splits.append((ids[~fresh[ids]],ids[fresh[ids]]))
            else:
                for a,b in GroupKFold(4,shuffle=True,random_state=2781).split(ids,groups=g[ids]):
                    splits.append((ids[a],ids[b]))
        by_method={}
        for name,features in feature_sets.items():
            ix=[]; probs=[]
            for a,b in splits:
                assert not np.isin(g[a],g[b]).any() and np.unique(fold[np.r_[a,b]]).size==1
                model=LogisticRegression(C=.08,max_iter=500).fit(features[a],y[a])
                probs.append(model.predict_proba(features[b])[:,1]);ix.extend(b)
            ix=np.asarray(ix);pred=np.concatenate(probs)
            np.testing.assert_array_equal(np.sort(ix),np.flatnonzero(fresh) if protocol=='fresh' else np.arange(len(y)))
            good=known[ix]
            d=(pred>=.5)==y[ix]
            base=nine[ix]==y[ix]
            def impact(mask):
                return dict(rows=int(mask.sum()),fixed=int((mask&d&~base).sum()),harmed=int((mask&~d&base).sum()))
            r=dict(known=impact(good),common=impact(good&unanimous[ix]),repeat=impact(good&repeat[ix]),
                   four_or_more=impact(good&(minority[ix]>=4)))
            for side in (False,True):
                mask=unanimous[ix]&(consensus[ix]==side)
                risk=1-pred if side else pred
                r['unanimous_'+str(side)]=dict(n=int(mask.sum()),wrong=int(np.sum(mask&(y[ix]!=side))),
                    auc=float(roc_auc_score(y[ix][mask]!=side,risk[mask])))
            r['fold_net']=[int(np.sum((d[good&(fold[ix]==k)]).astype(int)-base[good&(fold[ix]==k)])) for k in range(8)]
            by_method[name]=r
            print(protocol,name,r['known'],r['common'],round(r['unanimous_False']['auc'],3),flush=True)
        report[protocol]=by_method
    # In the prespecified hard cohort each order has exactly one person. Randomize
    # outcomes only inside outer fold, planet, and native-confidence quartile.
    solo=~pd.Series(g).duplicated(keep=False).to_numpy()
    hard=solo&raw.CryoSleep.eq(False).to_numpy()&unanimous&~consensus
    assert hard.sum()==1284 and y[hard].sum()==143
    quartile=np.searchsorted(np.quantile(native[hard],[.25,.50,.75]),native)
    planet=raw.HomePlanet.fillna('?').to_numpy()
    def randomization(mask,signal):
        ix=np.flatnonzero(mask)
        scores=rankdata(signal[ix])
        strata=pd.DataFrame(dict(fold=fold[ix],planet=planet[ix],confidence=quartile[ix]))
        blocks=[np.asarray(b) for b in strata.groupby(list(strata.columns)).indices.values()]
        truth=y[ix]
        centered=scores.copy()
        for ids in blocks:
            centered[ids]-=scores[ids].mean()
        obs=float(np.mean(centered[truth]))
        rng=np.random.default_rng(4703)
        null=np.zeros(999)
        for ids in blocks:
            count=truth[ids].sum()
            if not count:
                continue
            # Every random assignment preserves the number of transported in each stratum.
            take=np.argsort(rng.random((len(null),len(ids))),axis=1)[:,:count]
            null+=centered[ids][take].sum(1)/truth.sum()
        strongest=ix[np.argsort(-scores,kind='stable')[:int(np.ceil(len(ix)*.1))]]
        return dict(rows=len(ix),wrong=int(truth.sum()),top_tenth_wrong=int(y[strongest].sum()),
                    top_tenth_correct=int((~y[strongest]).sum()),raw_auc=float(roc_auc_score(truth,scores)),
                    conditional_rank_difference=obs,p_for_elevated_risk=float((1+np.sum(null>=obs))/(len(null)+1)))
    report['direct_hard_case_test']={}
    for region,mask in [('all',hard),('fresh',hard&fresh)]:
        r={}
        for name in ('flanks_only','flanks_only_permuted','means','means_permuted'):
            signal=feature_sets[name][:,2:].mean(1)
            r[name]=randomization(mask,signal)
        null_auc=[roc_auc_score(y[mask],scores[mask]) for scores in random_runs.T]
        observed=r['flanks_only']['raw_auc']
        r['shuffled_order_control']=dict(permutations=len(null_auc),auc_quantiles=np.quantile(null_auc,[0,.1,.5,.9,1]).tolist(),
            fraction_at_least_observed=float((1+sum(a>=observed for a in null_auc))/(len(null_auc)+1)))
        report['direct_hard_case_test'][region]=r
        print(region,'targeted',r['flanks_only'],r['flanks_only_permuted'],flush=True)
    RESULT.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    run()
