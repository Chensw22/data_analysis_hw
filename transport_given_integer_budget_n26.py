"""Test whether integer budget peaks explain residual transport outcomes after holding the coarse bill fixed."""
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit,logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from disjoint_tree_fields_n25 import ART,DEST
from exaone_validation_data import CATEGORICAL,compact_features
from family_identity_transfer_n26 import fit,panels_and_reference
from full_roster_booking_validation_n23 import data
from latent_bill_rounding_n26 import BILLS,OUT,REAL,learn,score
from outstanding_error_families_n26 import observed


def exact_evidence(total,count,eligible,y,planet,t,q,kernel,length,sham):
    out=np.zeros(len(q))
    rng=np.random.default_rng(1476)
    for name in np.unique(planet):
        tr=t[eligible[t]&(planet[t]==name)]
        pos=np.flatnonzero(eligible[q]&(planet[q]==name))
        if not len(pos):
            continue
        label=y[tr].copy()
        if sham:
            # Preserve coarse amount and origin effects in the negative control.
            bins=total[tr]//64
            for level in np.unique(bins):
                ix=np.flatnonzero(bins==level)
                label[ix]=rng.permutation(label[ix])
        pooled=learn(total,count,tr,kernel,length)
        scores=[]
        for side in [False,True]:
            train=tr[label==side]
            mass=learn(total,count,train,kernel,length)
            # Shrink both conditional integer densities toward the shared price spectrum.
            strength=len(train)/(len(train)+350.)
            mass=strength*mass+(1-strength)*pooled
            scores.append(score(mass,kernel,total[q[pos]],count[q[pos]])[:,2])
        out[pos]=scores[1]-scores[0]
    return out


def run():
    raw,y,g,f=data()
    x=compact_features(raw)
    with np.load(OUT/'features.npz') as z:
        n=len(raw)
        np.testing.assert_array_equal(z['passenger'][:n],raw.PassengerId)
        total,count,eligible=(z[key][:n] for key in ['total','count','eligible'])
        all_features={k:z[k][:n] for k in ['floor','neutral','reverse']}
    coarse=x.copy()
    for bill in BILLS:
        amount=x[bill]
        coarse[bill]=np.where(amount.isna(),np.nan,np.where(amount>0,1+amount//64,0))
    coarse['sum_band']=total//64
    numeric=x.copy()
    numeric['sum_exact']=total
    def augmented(key):
        frame=numeric.copy()
        for j,col in enumerate(['absent_units','posterior_peak','local_log_density','budget_peak']):
            frame[col]=all_features[key][:,j]
        frame['budget_code']=all_features[key][:,4].astype(int).astype(str)
        return frame
    models={'coarse':coarse,'direct':numeric,'floor_features':augmented('floor'),'neutral_features':augmented('neutral')}
    planet=x.HomePlanet.to_numpy()
    length=int(total.max())+5
    discrete=np.tile(np.array([1.,0,0,0,0]),(6,1))
    _,_,_,rows,names,p,h,native,old_native,ref=panels_and_reference()
    with np.load(ART/'nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['original_prior'],ref[0])
        ref[0]=z['prior']
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        hf,old,par=z['fold'],z['nine'],z['v131']
    awake=raw.CryoSleep.eq(False).to_numpy()
    hist=observed(rows,hf,h,names,old_native,par,old,old,awake,'historical')
    full=observed(np.arange(len(y)),f,p,names,native,*ref,awake,'full')
    output={}
    for tag,ids,fold,nat,view in [('historical',rows,hf,old_native,hist),('full',np.arange(len(y)),f,native,full)]:
        clean=~np.isin(g[ids],g[rows])
        edits={}
        for k in np.unique(fold):
            loc=np.flatnonzero(fold==k)
            q=ids[loc]; other=ids[fold!=k]; t=other[~np.isin(g[other],g[q])]
            assert not np.isin(g[t],g[q]).any()
            if tag=='historical':
                clean[loc]=~np.isin(g[q],g[other])
            path=OUT/f'outcome_{tag}_{k}.npz'
            if not path.exists():
                scores={}
                for name,frame in models.items():
                    cats=CATEGORICAL+(['budget_code'] if 'budget_code' in frame else [])
                    model=CatBoostClassifier(iterations=430,depth=6,learning_rate=.06,l2_leaf_reg=5,
                                             random_seed=4287,thread_count=2,verbose=False,allow_writing_files=False)
                    model.fit(frame.iloc[t],y[t],cat_features=cats)
                    scores[name]=model.predict(frame.iloc[q],prediction_type='RawFormulaVal')
                for method,ker,sham in [('floor_ratio',REAL,False),('sham_ratio',REAL,True),('atomic_ratio',discrete,False)]:
                    scores[method]=exact_evidence(total,count,eligible,y,planet,t,q,ker,length,sham)
                with path.with_suffix('.tmp').open('wb') as file:
                    np.savez_compressed(file,t=t,q=q,train_y=y[t],**scores)
                path.with_suffix('.tmp').replace(path)
            with np.load(path) as z:
                for key,val in [('t',t),('q',q),('train_y',y[t])]:
                    np.testing.assert_array_equal(z[key],val)
                s={n:z[n] for n in [*models,'floor_ratio','atomic_ratio','sham_ratio']}
            anchor=view['anchor'][loc]
            def add(key,flag):
                edits.setdefault(key,np.zeros(len(ids),bool))[loc]=flag
            for name,base in [('coarse','coarse'),('direct','direct'),('floor_features','direct'),('neutral_features','direct')]:
                after=s[name]>=0
                add(name,after!=anchor)
                if name!=base:
                    add(name+'_induced',(after!=(s[base]>=0))&(after!=anchor))
            off=logit(nat[loc].clip(.002,.998))
            for name in ['floor_ratio','atomic_ratio','sham_ratio']:
                adjusted=s['coarse']+s[name]>=0
                add(name+'_induced',(adjusted!=(s['coarse']>=0))&(adjusted!=anchor))
                add(name+'_native',(anchor==(off>=0))&((off+s[name]>=0)!=anchor))
            base=np.column_stack([np.ones(len(q)),off,s['direct']-off,s['coarse']-off])
            designs={'basic':base,'floor':np.column_stack([base,s['floor_features']-s['direct'],s['floor_ratio']]),
                     'control':np.column_stack([base,s['neutral_features']-s['direct'],s['sham_ratio']])}
            for mode in ['all','new']:
                if mode=='new' and tag=='historical':
                    continue
                sets=([(np.flatnonzero(~clean[loc]),np.flatnonzero(clean[loc]))] if mode=='new' else
                      GroupKFold(4,shuffle=True,random_state=4732).split(q,groups=g[q]))
                for tr,val in sets:
                    assert not np.isin(g[q[tr]],g[q[val]]).any()
                    for name,design in designs.items():
                        beta=fit(design[tr],off[tr],y[q[tr]])
                        decision=off[val]+design[val]@beta>=0
                        edits.setdefault(mode+'_'+name,np.zeros(len(ids),bool))[loc[val]]=(
                            (anchor[val]==(off[val]>=0))&(decision!=anchor[val]))
            print(tag,k,flush=True)
        table={}
        wrong=view['anchor']!=y[ids]
        fam=view['eligible']&(view['votes']==view['anchor'][:,None]).all(1)
        for method,flag in edits.items():
            sig={}
            for name,mask in [('all',view['known']),('billed',view['known']&eligible[ids]),('new',view['known']&clean),
                              ('families',fam),('new_families',fam&clean)]:
                if method.startswith('new_') and not name.startswith('new'):
                    continue
                change=flag&mask
                sig[name]=dict(rows=int(mask.sum()),fixed=int((change&wrong).sum()),harmed=int((change&~wrong).sum()),
                               by_fold=[int((change&wrong&(fold==k)).sum())-int((change&~wrong&(fold==k)).sum()) for k in np.unique(fold)])
            table[method]=sig
        output[tag]=table
        print([(k,s.get('all',s['new'])['fixed'],s.get('all',s['new'])['harmed']) for k,s in table.items()],flush=True)
    DEST.with_name('transport_given_integer_budget.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
