"""Check registration gaps on both deck sides, accounting for nearby unassigned bookings."""
import json
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from matched_state_context_n26 import inputs,ROOT
from exaone_validation_data import compact_features,CATEGORICAL
from repeated_archive_misses_n26 import ART,cohort,count

OUT=ART/'nov27_orphan_berth_gaps'
REPORT=ROOT/'reports/nov23_matched_learning_curve/orphan_berth_gaps.json'


def design(frame,g):
    info=frame.Cabin.str.split('/',expand=True)
    tag=(info[0]+info[2]).fillna('?').to_numpy(str)
    lane=np.unique(tag[tag!='?'])
    number=pd.to_numeric(info[1]).to_numpy()
    groups=g.astype(int)
    registered=set(groups[tag!='?'])
    orphan=np.array(sorted(set(groups)-registered))
    maps={}
    for name in lane:
        ix=np.flatnonzero(tag==name)
        pts=pd.Series(groups[ix],index=number[ix].astype(int)).groupby(level=0).median().sort_index()
        nums=pts.index.to_numpy()
        length=nums[-1]+1
        occupied=np.zeros(length,bool)
        occupied[nums]=True
        inside=(np.arange(length)>=nums[0]) & (np.arange(length)<=nums[-1])
        missing=np.flatnonzero(inside&~occupied)
        assigned=np.zeros(length,bool)
        for b in missing:
            j=np.searchsorted(nums,b)
            # A completely unregistered booking between the two flanking bookings can explain this gap.
            lower,upper=sorted([pts.iloc[j-1],pts.iloc[j]])
            assigned[b]=np.searchsorted(orphan,upper,side='right')>np.searchsorted(orphan,lower)
        maps[name]=(missing,assigned,length)
    matrix=np.zeros((len(frame),12))
    fake=np.zeros_like(matrix)
    true_gaps=explained=paired=0
    rng=np.random.default_rng(425730)
    for name in lane:
        selected=np.flatnonzero(tag==name)
        z=number[selected].astype(int)
        holes,registered_missing,length=maps[name]
        peer,_,opplength=maps[name[0]+('P' if name[-1]=='S' else 'S')]
        coords=np.arange(length)
        values=[]
        def nearest(vals):
            if not len(vals):return np.full(length,9999)
            j=np.searchsorted(vals,coords)
            return np.minimum(abs(coords-vals[np.maximum(0,j-1)]),abs(coords-vals[np.minimum(j,len(vals)-1)]))
        for h in (holes,holes[registered_missing[holes]],holes[~registered_missing[holes]]):
            d=nearest(h)
            values.extend([d==1,(d>1)&(d<=3),np.exp(-d/9)])
        opposed=nearest(peer)
        covered=coords<opplength
        values.extend([(opposed==0)&covered, (opposed<=3)&(nearest(holes)<=3)&covered,
                       (np.exp(-opposed/9)*np.exp(-nearest(holes)/9))*covered])
        features=np.column_stack(values)
        matrix[selected]=features[z]
        seen=np.unique(z)
        # Rotate complete feature vectors over occupied focal positions; preserve all cross-side dependence.
        rotated=np.roll(seen,rng.integers(max(1,len(seen)//5),max(2,4*len(seen)//5)))
        fake[selected]=features[rotated[np.searchsorted(seen,z)]]
        true_gaps+=len(holes)
        explained+=int(registered_missing[holes].sum())
        if name.endswith('P'):paired+=len(np.intersect1d(holes,peer))
    sleep=frame.CryoSleep.eq(True).to_numpy()
    solo=pd.Series(g).map(pd.Series(g).value_counts()).to_numpy()==1
    earth=frame.HomePlanet.eq('Earth').to_numpy()
    def terms(mat):
        return np.column_stack([mat,mat*sleep[:,None],mat*(sleep&solo)[:,None],mat*(earth&sleep)[:,None]])
    summary=dict(interior_missing_berths=int(true_gaps),potentially_explained_by_missing_cabin_order=explained,
                 both_sides_same_number=paired,unregistered_bookings=len(orphan))
    return terms(matrix),terms(fake),matrix,summary


def shift(X,base,y,t):
    prior=logit(np.clip(base,.005,.995))
    def loss(w):
        z=prior+X[t]@w
        return (np.logaddexp(0,z)-y[t]*z).sum()+12.*(w@w),X[t].T@(expit(z)-y[t])+24.*w
    return minimize(loss,np.zeros(X.shape[1]),jac=True,method='L-BFGS-B').x


def run():
    frame,y,g,f,scope=inputs(); n=len(y); ids=frame.PassengerId.to_numpy(str)
    _,blind,known,fresh=cohort(frame.iloc[:n],y,f)
    x=pd.get_dummies(compact_features(frame),columns=CATEGORICAL,dtype=float).to_numpy()
    z,placebo,raw,summary=design(frame,g)
    order=g[:n]
    cont=np.zeros(n,int)
    for j,gr in enumerate(np.array_split(np.unique(order),8)):
        cont[np.isin(order,gr)]=j
    with np.load(ART/'nov27_sleeper_companion_tabicl/aligned.npz') as a:
        np.testing.assert_array_equal(a['passenger'],ids)
        np.testing.assert_array_equal(a['fold'],f)
        anchor,shared,party=a['anchor'],a['shared'],a['party']
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as a:
        np.testing.assert_array_equal(a['passenger'],ids)
        native=np.median(a['scores'][:,-3:],axis=1)
    odds=lambda p:logit(np.clip(p,.005,.995))
    prior=np.where(anchor,1,-1)*abs(odds(native))
    separate=scope['sleep_unpaid']&~party
    seven_solo=shared&~party[:n]
    assert (seven_solo&(anchor[:n]!=y)).sum()==177
    scopes={'all':known,'new':known&fresh,'common_solo':seven_solo,'solo_sleep':known&separate[:n],'older_common':blind}
    domains={'all':np.ones(n,bool),'solo_sleep':separate[:n],'seven_solo':seven_solo}
    OUT.mkdir(exist_ok=True)
    summary['common_solo_with_cabin']=int((seven_solo&(anchor[:n]!=y)&frame.Cabin.notna().to_numpy()[:n]).sum())
    summary['common_solo_near_hole']=int((seven_solo&(anchor[:n]!=y)&np.any(raw[:n,:2]>0,axis=1)).sum())
    summary['seven_solo_total']=int(seven_solo.sum())
    summary['seven_solo_near_hole']=int((seven_solo&np.any(raw[:n,:2]>0,axis=1)).sum())
    results={}; saved={}
    def predict(t,q):
        model=HistGradientBoostingClassifier(max_iter=105,max_leaf_nodes=18,learning_rate=.10,l2_regularization=5.,early_stopping=False,random_state=913)
        return model.fit(x[t],y[t]).predict_proba(x[q])[:,1]
    for mode,fold in [('random',f),('contiguous',cont)]:
        bank=np.empty((len(ids),3))
        for j in range(9):
            tr=np.flatnonzero(fold!=j)
            q=np.flatnonzero(fold==j) if j<8 else np.arange(n,len(ids))
            assert not np.isin(g[tr],g[q]).any()
            path=OUT/('final.npz' if j==8 else f'{mode}_{j}.npz')
            if not path.exists():
                base=predict(tr,q)
                local=np.empty(len(tr))
                for a,b in GroupKFold(3,shuffle=True,random_state=579).split(tr,groups=g[tr]):
                    local[b]=predict(tr[a],tr[b])
                moved=np.column_stack([X[q]@shift(X,local,y,tr) for X in (z,placebo)])
                pred=np.column_stack([base,expit(odds(base)[:,None]+moved)])
                np.savez_compressed(path,t=tr,q=q,ids=ids[q],labels=y[tr],p=pred)
            with np.load(path) as a:
                for name,v in [('t',tr),('q',q),('labels',y[tr]),('ids',ids[q])]:
                    np.testing.assert_array_equal(a[name],v)
                bank[q]=a['p']
            print(mode,j,flush=True)
        base,real,sham=bank.T
        out=(real>=.5)!=anchor
        gain=odds(real)-odds(base)
        noise=odds(sham)-odds(base)
        added=((prior+gain>=0)!=anchor)&(np.abs(gain)>1e-8)
        placebo_added=((prior+noise>=0)!=anchor)&(np.abs(noise)>1e-8)
        actions={'base':(base>=.5)!=anchor,'real':out,'placebo':(sham>=.5)!=anchor,
                 'real_only':out&((base>=.5)==anchor)&((sham>=.5)==anchor),
                 'anchor_shift':added,'placebo_shift':placebo_added,'real_shift_only':added&~placebo_added}
        for key in ('real_only','anchor_shift','real_shift_only'):
            actions[key+'_solo_sleep']=actions[key]&separate
        table={}
        for name,act in actions.items():
            d=act[:n]*np.where(anchor[:n]!=y,1,-1)
            table[name]=dict(test=int(act[n:].sum()),scopes={k:count(act[:n],m,anchor[:n]!=y) for k,m in scopes.items()},
                             fold_net=[int(d[(fold==j)&known].sum()) for j in range(8)])
        results[mode]={'base_real_placebo_accuracy':{k:((bank[:n][m]>=.5)==y[m,None]).mean(0).tolist() for k,m in domains.items()},
                       'base_real_placebo_logloss':{k:np.mean(np.logaddexp(0,odds(bank[:n][m]))-y[m,None]*odds(bank[:n][m]),axis=0).tolist() for k,m in domains.items()},
                       'rules':table}
        saved[mode]=bank
    np.testing.assert_array_equal(saved['random'][n:],saved['contiguous'][n:])
    np.savez_compressed(OUT/'aligned.npz',passenger=ids,anchor=anchor,y=y,random_fold=f,contiguous_fold=cont,
                        seven_solo=seven_solo,**saved)
    current=pd.read_csv(ART/'proposal_n27_directional_foundation/test_probabilities.csv',float_precision='round_trip')
    np.testing.assert_array_equal(current.PassengerId,ids[n:]);np.testing.assert_array_equal(current.probability>=.5,anchor[n:])
    test=pd.DataFrame(saved['random'][n:],columns=['baseline','gap','rotated'])
    test.insert(0,'PassengerId',ids[n:]);test['scored_parent']=current.probability
    test.to_csv(OUT/'test_probabilities.csv',index=False)
    REPORT.write_text(json.dumps({'gap_counts':summary,'previous_work':['cabin_change_points.py','synchronized_cabin_segments_n23.py',
            'unlabeled_cabin_activity_segments_n26.py','cross_deck_booking_clock_n27.py'],'results':results},indent=2)+'\n')
    print(summary)
    for name,r in results.items():
        print(name,r['base_real_placebo_accuracy'])
        for key in ['real_only','anchor_shift','real_shift_only','real_only_solo_sleep','real_shift_only_solo_sleep']:
            t=r['rules'][key];print(key,t['scopes'],t['test'])

if __name__=='__main__':
    with threadpool_limits(1):run()
