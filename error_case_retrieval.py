"""Nested error-case retrieval, two views, calibrated confidence and deployable union."""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','FALSE')
import json
from pathlib import Path
import time
import signal
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.model_selection import StratifiedKFold, GroupShuffleSplit
from threadpoolctl import threadpool_limits
import error_overlap_diagnostic as overlap
import relational_error_corrector as r
ROOT=Path(__file__).resolve().parent
REPORT=ROOT/'reports/error_case_retrieval'
OUT=ROOT/'artifacts/error_case_retrieval'
BILLS=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
CATS=['HomePlanet','CryoSleep','Destination','VIP','Deck','Side']
LEVELS={'low':.25,'medium':.5,'high':.75}
K=32


class View:
    def __init__(self,name):self.name=name
    def fit(self,x):
        self.numeric=BILLS if self.name=='consumption' else ['Age','CabinNumber']
        self.cats=[] if self.name=='consumption' else CATS
        a=x[self.numeric].to_numpy(float)
        if self.name=='consumption':
            if np.any(a[np.isfinite(a)]<0):raise ValueError('Negative bill')
            a=np.log1p(a)
        self.center=np.nanmedian(a,axis=0);self.center[~np.isfinite(self.center)]=0.
        self.scale=np.nanquantile(a,.75,axis=0)-np.nanquantile(a,.25,axis=0);self.scale[~np.isfinite(self.scale)|(self.scale==0)]=1.
        self.vocab={c:sorted(x[c].fillna('__MISSING__').astype(str).unique()) for c in self.cats}
        return self
    def transform(self,x):
        a=x[self.numeric].to_numpy(float);miss=~np.isfinite(a);zero=(a==0).astype(float)
        if self.name=='consumption':a=np.log1p(a)
        z=np.clip((np.where(miss,self.center,a)-self.center)/self.scale,-8,8)
        parts=[z,miss.astype(float)]
        if self.name=='consumption':parts.append(zero)
        for c in self.cats:
            v=x[c].fillna('__MISSING__').astype(str).to_numpy();parts.append(v[:,None]==np.asarray(self.vocab[c])[None,:])
        return np.column_stack(parts).astype(float)


def retrieve(lx,ly,lp,lids,lg,qx,qp,qids,qg,view):
    """No query targets: neighbor errors depend ONLY on library targets/OOF p."""
    if len(np.unique(lids))!=len(lids):raise ValueError('Library IDs not unique')
    prep=View(view).fit(lx);a=prep.transform(lx);b=prep.transform(qx)
    errors=((lp>=.5)!=ly).astype(float);direction=lp>=.5
    scores=np.zeros(len(qx));neighbors=np.full((len(qx),K),-1,dtype=np.int64);available=np.zeros(len(qx),int)
    for start in range(0,len(qx),256):
        stop=min(start+256,len(qx));dist=cdist(b[start:stop],a,metric='sqeuclidean')
        allowed=((qp[start:stop,None]>=.5)==direction[None,:])&(qg[start:stop,None]!=lg[None,:])&(qids[start:stop,None]!=lids[None,:])
        dist[~allowed]=np.inf
        for row in range(stop-start):
            n=int(allowed[row].sum());available[start+row]=n
            if not n:continue  # No eligible library: abstain, no same-group prior.
            # Stable library order makes tied distances deterministic.
            ix=np.argsort(dist[row],kind='stable')[:min(K,n)]
            w=np.exp(-dist[row,ix]/max(float(dist[row,ix[-1]]),1e-12))
            prior=float(errors[allowed[row]].mean())
            scores[start+row]=(w@errors[ix]+2*prior)/(w.sum()+2)
            neighbors[start+row,:len(ix)]=lids[ix]
            assert not np.any(lg[ix]==qg[start+row]) and not np.any(lids[ix]==qids[start+row])
    return scores,neighbors,available,prep


def thresholds(score,error,direction):
    result={k:[] for k in [*LEVELS,'deploy']}
    detail=[]
    for d in [0,1]:
        rows=direction==d;s=score[rows];e=error[rows]
        for level,target in LEVELS.items():
            # Desired error recall, NOT target precision; ties may increase recall.
            threshold=float(np.quantile(s[e],1-target,method='higher')) if e.any() else 2.
            result[level].append(threshold)
        # Select max calibration net-correct, including abstention; ties choose fewer flips.
        choices=np.r_[2.,np.unique(s)]
        options=[]
        for threshold in choices:
            flip=s>=threshold;repair=int((flip&e).sum());broken=int((flip&~e).sum())
            options.append((repair-broken,-int(flip.sum()),float(threshold)))
        result['deploy'].append(max(options)[2])
        detail.append(dict(direction=d,rows=int(rows.sum()),errors=int(e.sum()),correct=int((~e).sum())))
    return result,detail


def decide(score,p,threshold):return score>=np.asarray(threshold)[(p>=.5).astype(int)]


def effect(flip,base,y,mask):
    original=(base>=.5)==y;repair=int((flip&~original&mask).sum());broken=int((flip&original&mask).sum())
    errors=int((~original&mask).sum());correct=int((original&mask).sum())
    return dict(rows=int(mask.sum()),baseline_errors=errors,baseline_correct=correct,repairs=repair,breaks=broken,net=repair-broken,
        error_recall=repair/errors if errors else None,correct_harm_rate=broken/correct if correct else None,
        flip_precision=repair/(repair+broken) if repair+broken else None,flips=repair+broken,
        candidate_accuracy=(correct+repair-broken)/int(mask.sum()) if mask.any() else None)


def checks():
    x=pd.DataFrame({c:[0.,10.,20.,30.] for c in BILLS});ids=np.arange(4);groups=np.array(['a','a','b','c'])
    s,ix,n,_=retrieve(x,np.array([0,1,0,1]),np.full(4,.8),ids,groups,x.iloc[[0]],np.array([.8]),np.array([0]),np.array(['a']),'consumption')
    assert set(ix[0][ix[0]>=0])=={2,3} and n[0]==2 and np.isfinite(s).all()
    # No neighbor => score zero abstention, without consuming forbidden same-group labels.
    s,ix,n,_=retrieve(x.iloc[:2],np.array([0,1]),np.full(2,.8),ids[:2],groups[:2],x.iloc[[0]],np.array([.8]),np.array([0]),np.array(['a']),'consumption')
    assert s[0]==0 and n[0]==0 and (ix==-1).all()
    return dict(same_row_and_same_group_excluded=True,no_allowed_neighbor_abstains=True,no_query_y_argument=True)


def run():
    if REPORT.exists() or OUT.exists():raise FileExistsError('Fixed experiment no overwrite')
    check=checks();data,splits,folds,bases,evidence=overlap.load_authenticated();dev,y=data['dev'],data['y'];x=data['features'];groups=np.asarray(data['groups'],dtype=str);raw=bases['rawMitra']
    oldfreeze=r.nested.read(ROOT/'artifacts/planet_mixture_20261117/freeze.json');r.nested.verify_hashes(oldfreeze['sources']);evidence.update(oldfreeze['sources'])
    inner=[r.load_inner(i,a,b,dev,y,evidence) for i,(a,b) in enumerate(splits)]
    hist=json.loads((ROOT/'reports/v96_oof_completion/result.json').read_text());path=ROOT/'artifacts/v96_oof_completion/oof.npz';assert r.sha(path)==hist['oof_sha256']
    assert hist==json.loads((ROOT/'artifacts/v96_oof_completion/result.json').read_text()) and hist['status']=='complete_v96_fixed15_original3fold_oof'
    with np.load(path) as z:
        for key,w in [('dev_ids',dev),('fold_id',folds),('passenger_ids',data['passengers'])]:np.testing.assert_array_equal(z[key],w)
        v96=z['prob'].copy()
    names=[Path(__file__).name,'error_overlap_diagnostic.py','relational_error_corrector.py','reports/v96_oof_completion/result.json','artifacts/v96_oof_completion/result.json','artifacts/v96_oof_completion/oof.npz','artifacts/error_rescue_20261117/catalog_rows.csv']
    evidence.update({n:r.sha(ROOT/n) for n in names});REPORT.mkdir(parents=True);OUT.mkdir(parents=True)
    plan=dict(status='fixed_before_cases_or_threshold_selection',base='Authenticated rawMitra inner OOF for all case labels. No authenticated v96 innerOOF available; v96 is reported only as direction-compatible selector transfer, not v96-specific case training.',
        views=dict(consumption='5log1p bills,5zero bits,5missing bits',passenger_cabin='Age,CabinNumber robust scaled+missing; HomePlanet,CryoSleep,Destination,VIP,Deck,Side onehot; no Name/ID feature'),
        scaling='Median/IQR fit only case library; fill numeric missing with median; clip standardized to[-8,8].',
        neighbors='K32 or all available, same baseline predicted class; exclude same row and sameGroup in both neighbors and prior. Stable ascending distance/library-order ties. No available => abstain.',
        score='Neighbor weights exp(-squared_distance/max(kth_squared_distance,1e-12)); (weighted error sum+2*allowed-pool-error-rate)/(weight sum+2).',
        calibration='Only innerquery packet0 of each outerfold: GroupShuffleSplit50/50 seed20261132+outer yields case and independent calibration parts. That teacher support excludes BOTH sets. Do not aggregate calibration across inner packets whose teachers may have seen other calibration labels.',
        final_library='All4636 outertrain with strict3innerOOF raw p/error labels after thresholds frozen. Query2318 true targets never used.',
        thresholds='Separate predicted0/FN and predicted1/FP per view. Low/medium/high target calibration error recalls25/50/75%; quantile higher with >= threshold. deploy max calibration(repairs-breaks), tie fewer flips, includes abstention.',
        joint='Continuous max(two view scores), calibrated independently by same procedure. Also report OR of two separately calibrated flip decisions. Neither uses querylabels.',
        oracle='Union of rows correctly flipped by either individual expert uses query truth only for diagnostic upper bound; cannot deploy without errors.',
        v96_transfer='Apply fixed raw-case selector only where v96/raw predicted classes agree; other rows abstain, not training v96 case labels.',
        threshold_levels=list(LEVELS),CPU_threads=4,budget_seconds=600,no_GPU=True,
        limitations=['Adaptive same6954 development labels;546mask reporting only.','Calibration library about772 rows vs final4636; neighbor density/score distribution shift remains.','One calibration subset per outerfold limits threshold reliability; no seed/K/view/weight scan.','High-recall experts retained even if overall worse; reported with all errors and all correct denominators.'])
    r.write(REPORT/'protocol.json',plan);evidence[str((REPORT/'protocol.json').relative_to(ROOT))]=r.sha(REPORT/'protocol.json');r.write(REPORT/'freeze.json',evidence);r.write(REPORT/'checks.json',check)
    scores={k:np.full(len(y),np.nan) for k in ['consumption','passenger_cabin','joint_max']}
    flips={f'{view}_{level}':np.zeros(len(y),bool) for view in [*scores,'union_or'] for level in [*LEVELS,'deploy']};records=[]
    started=time.monotonic();old=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Case retrieval600s')));signal.setitimer(signal.ITIMER_REAL,600)
    try:
        with threadpool_limits(limits=4):
            for outer,(a,b) in enumerate(splits):
                _,ib=next(StratifiedKFold(3,shuffle=True,random_state=20261117+outer).split(a,y[a]))
                local_case,local_cal=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=20261132+outer).split(ib,groups=groups[a[ib]]))
                ca,va=ib[local_case],ib[local_cal]
                assert not np.intersect1d(groups[a[ca]],groups[a[va]]).size
                cal_scores={};thresholds_by_view={};views={};neighbors_saved={};calerror=(inner[outer][va]>=.5)!=y[a[va]];caldir=(inner[outer][va]>=.5).astype(int)
                for view in ['consumption','passenger_cabin']:
                    cs,ci,cn,_=retrieve(x.loc[dev[a[ca]]],y[a[ca]],inner[outer][ca],dev[a[ca]],groups[a[ca]],x.loc[dev[a[va]]],inner[outer][va],dev[a[va]],groups[a[va]],view)
                    fs,fi,fn,prep=retrieve(x.loc[dev[a]],y[a],inner[outer],dev[a],groups[a],x.loc[dev[b]],raw[b],dev[b],groups[b],view)
                    cal_scores[view]=cs;scores[view][b]=fs;neighbors_saved[view]=fi
                    views[view]=dict(scaler=prep,minimum_eligible_neighbors=int(fn.min()),no_neighbor_queries=int((fn==0).sum()))
                cal_scores['joint_max']=np.maximum(cal_scores['consumption'],cal_scores['passenger_cabin']);scores['joint_max'][b]=np.maximum(scores['consumption'][b],scores['passenger_cabin'][b])
                cal_details={}
                for view in scores:
                    ts,counts=thresholds(cal_scores[view],calerror,caldir);thresholds_by_view[view]=ts
                    cal_details[view]=dict(directions=counts,thresholds=ts,operating_points={level:effect(decide(cal_scores[view],inner[outer][va],ts[level]),inner[outer][va],y[a[va]],np.ones(len(va),bool)) for level in ts})
                    for level in ts:flips[f'{view}_{level}'][b]=decide(scores[view][b],raw[b],ts[level])
                for level in [*LEVELS,'deploy']:flips[f'union_or_{level}'][b]=flips[f'consumption_{level}'][b]|flips[f'passenger_cabin_{level}'][b]
                path=OUT/f'fold{outer}_selector.joblib';joblib.dump(dict(views=views,thresholds=thresholds_by_view,library_ids=dev[a],library_group=groups[a],library_raw=inner[outer],library_y=y[a],library_attributes=x.loc[dev[a]],recipe=plan),path)
                saved=joblib.load(path)
                for view in scores:
                    for level in [*LEVELS,'deploy']:np.testing.assert_array_equal(flips[f'{view}_{level}'][b],decide(scores[view][b],raw[b],saved['thresholds'][view][level]))
                np.savez_compressed(OUT/f'fold{outer}.npz',train_ids=dev[a],query_ids=dev[b],calibration_case_ids=dev[a[ca]],calibration_query_ids=dev[a[va]],calibration_case_p=inner[outer][ca],calibration_query_p=inner[outer][va],**{k+'_neighbors':v for k,v in neighbors_saved.items()},**{k+'_calibration_score':v for k,v in cal_scores.items()})
                record=dict(fold=outer,case_rows=len(ca),calibration_rows=len(va),final_library_rows=len(a),query_rows=len(b),calibration_group_disjoint=True,calibration=cal_details,model_sha256=r.sha(path),rows_sha256=r.sha(OUT/f'fold{outer}.npz'),metrics={k:effect(v[b],raw[b],y[b],np.ones(len(b),bool)) for k,v in flips.items()})
                records.append(record);r.write(REPORT/f'fold{outer}.json',record);print(json.dumps(dict(fold=outer,deploy=record['metrics']['joint_max_deploy'],high=record['metrics']['joint_max_high'])),flush=True)
        rows=pd.read_csv(ROOT/'artifacts/error_rescue_20261117/catalog_rows.csv',dtype={'PassengerId':str})
        for key,w in [('row_id',dev),('y',y),('fold',folds),('PassengerId',data['passengers'])]:np.testing.assert_array_equal(rows[key],w)
        hard=rows.all_recipes_still_wrong.to_numpy(bool);assert hard.sum()==546
        compatible=(raw>=.5)==(v96>=.5);err=(raw>=.5)!=y
        subsets={'all':np.ones(len(y),bool),'hard546':hard,'raw_false_positive':(raw>=.5)&(y==0),'raw_false_negative':(raw<.5)&(y==1),'raw_original_correct':~err}
        summaries={name:{key:effect(v,raw,y,mask) for key,v in flips.items()} for name,mask in subsets.items()}
        unions={}
        for level in [*LEVELS,'deploy']:
            ar=flips[f'consumption_{level}']&err;br=flips[f'passenger_cabin_{level}']&err
            unions[level]=dict(consumption_unique=int((ar&~br).sum()),passenger_unique=int((br&~ar).sum()),both=int((ar&br).sum()),oracle_union_repairs=int((ar|br).sum()),oracle_hard546_union=int(((ar|br)&hard).sum()),oracle_not_deployable=True,deployable_or=summaries['all'][f'union_or_{level}'],deployable_max=summaries['all'][f'joint_max_{level}'])
        transfer={name:{key:effect(v&compatible,v96,y,mask) for key,v in flips.items()} for name,mask in {'all':np.ones(len(y),bool),'hard546':hard}.items()}
        for i,record in enumerate(records):record['hard546']={key:effect(v,raw,y,hard&(folds==i)) for key,v in flips.items()};r.write(REPORT/f'fold{i}.json',record)
        path=OUT/'oof.npz';np.savez_compressed(path,dev_ids=dev,fold_id=folds,passenger_ids=np.asarray(data['passengers'],dtype=str),raw=raw,v96=v96,**{k+'_score':v for k,v in scores.items()},**{k+'_flip':v for k,v in flips.items()})
        table=[]
        for subset,vals in summaries.items():
            for name,stats in vals.items():table.append(dict(subset=subset,selector=name,**stats))
        pd.DataFrame(table).to_csv(REPORT/'operating_points.csv',index=False)
        for name,digest in evidence.items():assert r.sha(ROOT/name)==digest,name
        result=dict(status='complete',raw_accuracy=float(np.mean((raw>=.5)==y)),v96_accuracy=float(np.mean((v96>=.5)==y)),subsets=summaries,unique_and_union=unions,v96_direction_compatible_transfer=transfer,compatible_rows=int(compatible.sum()),folds=records,seconds=time.monotonic()-started,oof_sha256=r.sha(path),source_hashes=evidence,no_upload=True,no_gpu=True)
        r.write(REPORT/'result.json',result);print(json.dumps(dict(seconds=result['seconds'],deploy=summaries['all']['joint_max_deploy'],union=unions)),flush=True)
    except BaseException as exc:r.write(REPORT/'failure.json',dict(type=type(exc).__name__,error=str(exc)));raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
View.__module__='error_case_retrieval'
if __name__=='__main__':
    import sys
    sys.modules['error_case_retrieval']=sys.modules[__name__]
    run()
