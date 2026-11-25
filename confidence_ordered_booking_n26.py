"""Predict all outcome configurations in an unseen booking with confidence-ordered conditional trees."""
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from disjoint_tree_fields_n25 import ART,DEST,stats
from exaone_validation_data import CATEGORICAL,compact_features
from family_identity_transfer_n26 import panels_and_reference
from full_roster_booking_validation_n23 import data
from outstanding_error_families_n26 import observed

OUT=ART/'nov26_confidence_ordered_booking'
MODES=('independent','conditional','shuffled')


def classifier(iterations,depth=5):
    return CatBoostClassifier(iterations=iterations,depth=depth,learning_rate=.055,l2_leaf_reg=6,
                              thread_count=2,random_seed=3486,verbose=False,allow_writing_files=False)


def parties(ids,g):
    return [ids[ix] for ix in pd.Series(g[ids]).groupby(g[ids],sort=True).indices.values() if len(ix)>1]


def person_scores(frame,y,g,t,whole_query):
    result=np.full(len(y),np.nan)
    for a,b in GroupKFold(3,shuffle=True,random_state=7281).split(t,groups=g[t]):
        tr,te=t[a],t[b]
        assert not np.isin(g[tr],g[te]).any()
        model=classifier(390,6).fit(frame.iloc[tr],y[tr],cat_features=CATEGORICAL)
        result[te]=model.predict_proba(frame.iloc[te])[:,1]
    model=classifier(390,6).fit(frame.iloc[t],y[t],cat_features=CATEGORICAL)
    result[whole_query]=model.predict_proba(frame.iloc[whole_query])[:,1]
    assert not np.isin(g[t],g[whole_query]).any()
    return result


def sequence(rows,p,reverse):
    # Break confidence ties by the manifest's registration order.
    asc=np.lexsort((rows,-abs(p[rows]-.5)))
    return rows[asc[::-1] if reverse else asc]


def examples(groups,p,x,cabin,sleep,teacher=None):
    matrices=[]; states=[]; targets=[]; descriptions=[]
    for reverse in (False,True):
        for group in groups:
            order=sequence(group,p,reverse)
            full=p[order]
            widths=[]
            for j,i in enumerate(order):
                earlier=order[:j]
                peer=np.delete(full,j)
                fixed=np.r_[x[i],p[i],len(order),j,float(reverse),peer.min(),peer.max(),peer.mean(),
                            full[:j].sum(),full[j+1:].sum()]
                if teacher is None:
                    patterns=(np.arange(2**j)[:,None]>>np.arange(j))&1
                else:
                    patterns=teacher[earlier].reshape(1,-1)
                history=patterns-p[earlier]
                same_cabin=(cabin[i]!='?')&(cabin[earlier]==cabin[i])
                same_sleep=(sleep[i]!='?')&(sleep[earlier]==sleep[i])
                last=history[:,-1] if j else np.zeros(len(patterns))
                features=np.column_stack([history.sum(1),history@same_cabin,history@same_sleep,last])
                matrices.append(np.broadcast_to(fixed,(len(patterns),len(fixed))))
                states.append(features)
                targets.extend([i]*len(patterns))
                widths.append(len(patterns))
            descriptions.append((order,widths))
    return np.vstack(matrices),np.vstack(states),np.array(targets),descriptions


def integrate(probability,description,rows):
    total=np.zeros(rows)
    hit=np.zeros(rows,int)
    cursor=0
    joint=[]
    for order,widths in description:
        mass=np.ones(1)
        for i,width in zip(order,widths):
            assert width==len(mass)
            cond=probability[cursor:cursor+width]
            total[i]+=mass@cond
            hit[i]+=1
            mass=np.r_[mass*(1-cond),mass*cond]
            cursor+=width
        np.testing.assert_allclose(mass.sum(),1.,atol=1e-10)
        joint.append(mass)
    assert cursor==len(probability) and (hit[hit>0]==2).all()
    return np.divide(total,hit,out=np.full(rows,np.nan),where=hit>0),joint


def simulate(raw,y,g,t,q,keep_joint=False):
    frame=compact_features(raw)
    x=pd.get_dummies(frame,dtype=float).fillna(-1).to_numpy()
    whole_query=np.flatnonzero(np.isin(g,g[q]))
    p=person_scores(frame,y,g,t,whole_query)
    train_groups=parties(t,g)
    test_groups=parties(whole_query,g)
    cabin=raw.Cabin.fillna('?').to_numpy()
    sleep=raw.CryoSleep.astype('string').fillna('?').to_numpy()
    fixed,actual,train_rows,_=examples(train_groups,p,x,cabin,sleep,y)
    rng=np.random.default_rng(9826)
    permuted=y.copy()
    bins=np.digitize(p[t],np.arange(.1,1,.1))
    for b in np.unique(bins):
        use=t[bins==b]
        permuted[use]=rng.permutation(y[use])
    _,sham,same,_=examples(train_groups,p,x,cabin,sleep,permuted)
    np.testing.assert_array_equal(same,train_rows)
    xf,states,_,desc=examples(test_groups,p,x,cabin,sleep)
    results={'raw':p[q]}
    logs={}
    distributions={}
    for mode in MODES:
        model=classifier(310)
        if mode=='independent':
            model.fit(fixed,y[train_rows])
            prob=model.predict_proba(xf)[:,1]
        else:
            hist=actual if mode=='conditional' else sham
            model.fit(np.column_stack([fixed,hist]),y[train_rows])
            prob=model.predict_proba(np.column_stack([xf,states]))[:,1]
        scores,joint=integrate(prob,desc,len(y))
        results[mode]=np.where(np.isfinite(scores[q]),scores[q],p[q])
        if keep_joint:
            canonical={}
            for (order,_),outcomes in zip(desc,joint):
                members=tuple(sorted(order.tolist()))
                ranks=np.searchsorted(members,order)
                index=np.arange(len(outcomes))
                remap=((index[:,None]>>ranks)&1)@(1<<np.arange(len(members)))
                canonical[members]=canonical.get(members,0)+outcomes[remap]/2
            for members,mass in canonical.items():
                bits=(np.arange(len(mass))[:,None]>>np.arange(len(members)))&1
                np.testing.assert_allclose(mass@bits,scores[list(members)],atol=1e-12)
            distributions[mode]=canonical
        # Evaluate joint likelihood only for bookings whose entire truth belongs
        # to this query. Combine reversed orderings in the same passenger order.
        known=np.zeros(len(y),bool);known[q]=True
        by_group={}
        for (order,_),outcomes in zip(desc,joint):
            if known[order].all():
                code=int(y[order]@(1<<np.arange(len(order))))
                by_group.setdefault(g[order[0]],[]).append(outcomes[code])
        logs[mode]=np.array([-np.log(np.mean(v).clip(1e-12)) for v in by_group.values()])
    if keep_joint:
        return results,logs,distributions
    return results,logs


def run():
    raw,y,g,f=data()
    size=pd.Series(g).map(pd.Series(g).value_counts()).to_numpy()
    _,_,_,rows,names,p,h,native,old_native,ref=panels_and_reference()
    with np.load(ART/'nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['original_prior'],ref[0])
        ref[0]=z['prior']
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        hf,old,par=z['fold'],z['nine'],z['v131']
    awake=raw.CryoSleep.eq(False).to_numpy()
    current=observed(np.arange(len(y)),f,p,names,native,*ref,awake,'full')
    hist=observed(rows,hf,h,names,old_native,par,old,old,awake,'historical')
    OUT.mkdir(exist_ok=True)
    result={}
    for tag,ids,fold,base,view in [('historical',rows,hf,old_native,hist),('full',np.arange(len(y)),f,native,current)]:
        clean=~np.isin(g[ids],g[rows]); votes={}; losses={m:[] for m in MODES}
        for k in np.unique(fold):
            loc=fold==k; q=ids[loc];other=ids[~loc];t=other[~np.isin(g[other],g[q])]
            if tag=='historical':
                clean[loc]=~np.isin(g[q],g[other])
            path=OUT/f'{tag}_{k}.npz'
            if not path.exists():
                values,logs=simulate(raw,y,g,t,q)
                with path.with_suffix('.tmp').open('wb') as buf:
                    np.savez_compressed(buf,t=t,q=q,train_y=y[t],**values,**{name+'_nll':v for name,v in logs.items()})
                path.with_suffix('.tmp').replace(path)
            with np.load(path) as z:
                for key,val in [('t',t),('q',q),('train_y',y[t])]:
                    np.testing.assert_array_equal(z[key],val)
                for m in MODES:
                    losses[m].extend(z[m+'_nll'])
                    votes.setdefault(m,np.zeros(len(ids)))[loc]=z[m]
            print(tag,k,flush=True)
        anchor=view['anchor']; wrong=anchor!=y[ids]; multi=size[ids]>1
        fam=view['eligible']&(view['votes']==anchor[:,None]).all(1)
        scores={}
        for name,v in votes.items():
            binary=v>=.5
            rules={'direct':binary!=anchor,'induced':(binary!=(votes['independent']>=.5))&(binary!=anchor),
                   'quarter':(anchor==(base>=.5))&((.75*base+.25*v>=.5)!=anchor)}
            for method,flag in rules.items():
                delta=(flag&multi)*np.where(wrong,1,-1)
                scores[name+'_'+method]={n:stats(delta,mask,fold,g[ids]) for n,mask in
                    [('all',view['known']),('parties',view['known']&multi),('new',view['known']&clean),('families',fam)]}
        result[tag]=dict(complete_booking_count=len(losses['conditional']),joint_nll={k:float(np.mean(v)) for k,v in losses.items()},scores=scores)
        print(tag,result[tag]['joint_nll'],[(n,s['all']['fixed'],s['all']['harmed']) for n,s in scores.items()],flush=True)
    DEST.with_name('confidence_ordered_booking.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
