"""Reproduce v96 fold models, apply query-only v106 patches and strict v108 offsets."""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='FALSE'
import argparse
import gc
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
import audit_top5_references as audit
import error_overlap_diagnostic as overlap
import peer_bill_correction as peer
import relational_error_corrector as ref
import mitra_candidate as base
import mitra_counts_fixed15_online_first as counts
import v96_oof_completion as completion

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/v108_exact_oof'
REPORT=ROOT/'reports/v108_exact_oof'
SEED=20261112
BUDGET=5400
OWN=['v108_exact_oof.py','v96_oof_completion.py','mitra_counts_fixed15_online_first.py',
     'mitra_full_episode.py','mitra_full_episode_probe.py','mitra_candidate.py',
     'peer_cabin_impute.py','peer_planet_impute.py','peer_bill_correction.py','export_party_bill.py',
     'relational_error_corrector.py','error_overlap_diagnostic.py','exaone_validation_data.py',
     'exception_error_detector_nov18.py']
sha=audit.file_hash
write=audit.json_write
read=lambda p:json.loads(Path(p).read_text())


def cabin_updates():
    """Original peer_cabin_impute numerical rule, without its hard-case statistics read."""
    frame=pd.concat([pd.read_csv(ROOT/'data/raw'/f,usecols=['PassengerId','Cabin']) for f in ['train.csv','test.csv']],ignore_index=True)
    group=frame.PassengerId.str.split('_').str[0].astype(int)
    location=frame.Cabin.str.split('/',expand=True);location.columns=['Deck','CabinNumber','Side']
    location.CabinNumber=pd.to_numeric(location.CabinNumber);location['group']=group
    knots={key:rows.groupby('group').CabinNumber.median().sort_index() for key,rows in location.groupby(['Deck','Side'])}
    filled={}
    for indices in frame.groupby(group).indices.values():
        known=location.iloc[indices].dropna(subset=['CabinNumber'])
        for i in indices:
            if pd.notna(frame.Cabin.iat[i]):continue
            peers=known.drop(index=i,errors='ignore')
            if peers.empty or peers.Deck.nunique()!=1 or peers.Side.nunique()!=1:continue
            key=(peers.Deck.iloc[0],peers.Side.iloc[0]);curve=knots[key];train=curve.index!=group.iat[i]
            if not train.any():continue
            number=float(np.rint(np.interp(group.iat[i],curve.index[train],curve.to_numpy()[train])))
            filled[int(i)]=dict(Deck=str(key[0]),Side=str(key[1]),CabinNumber=number)
    return frame,filled


def check():
    p=read(REPORT/'protocol.json')
    for name,h in p['source_hashes'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('Source changed: '+name)
    return p


def patch_query(model,updates):
    idx=np.asarray(sorted(int(i) for i in updates),dtype=np.int64)
    lookup=model.preprocessor.roster
    for col in ['Deck','Side','CabinNumber']:
        old=lookup.plain.loc[idx,col]
        if not (old.isna()|old.eq('__MISSING__')).all():raise RuntimeError('Only missing query cabins may change.')
        lookup.plain.loc[idx,col]=[updates[str(i)][col] for i in idx]
    return idx,lookup.plain.loc[idx]


def worker(fold):
    started=time.monotonic();log=dict(status='running',fold=fold)
    model=loaded=None
    try:
        check();packet=read(OUT/f'fold{fold}_packet.json')
        if set(packet)!={'train_ids','query_ids','train_y','cabin_updates','peer_eligible','peer_shift'}:raise RuntimeError('Unexpected packet fields.')
        ids=np.asarray(packet['train_ids'],dtype=np.int64);qids=np.asarray(packet['query_ids'],dtype=np.int64);y=np.asarray(packet['train_y'],dtype=np.int64)
        if len(ids)!=4636 or len(qids)!=2318 or np.intersect1d(ids,qids).size:raise RuntimeError('Invalid fold partition.')
        if not set(map(int,packet['cabin_updates']))<=set(qids):raise RuntimeError('Cabin patch outside query.')
        if not torch.backends.mps.is_available():raise RuntimeError('Original MPS required.')
        torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.65);base.DEADLINE=time.monotonic()+BUDGET
        counts.verify_reference();roster=counts.load_roster()
        # Capture the portable predictor returned internally by the exact original fit.
        # Its original finally still releases the engine; training and full query call are untouched.
        captured=[];original=counts.CountsPredictor;old_out=completion.OUT
        def capture(*args,**kwargs):
            result=original(*args,**kwargs);captured.append(result);return result
        try:
            counts.CountsPredictor=capture;completion.OUT=OUT
            p96,training_log=completion.fit_fold(fold,roster.plain,ids,y,qids,roster)
        finally:
            counts.CountsPredictor=original;completion.OUT=old_out
        if len(captured)!=1:raise RuntimeError('Expected one native predictor.')
        model=captured.pop();model.metadata.update(training_log)
        with np.load(OUT/'expected_v96.npz',allow_pickle=False) as z:
            positions=np.searchsorted(z['dev_ids'],qids);np.testing.assert_array_equal(z['dev_ids'][positions],qids)
            expected=z['prob'][positions]
        np.testing.assert_array_equal(p96,expected)
        log.update(v96_bitwise_equal=True,v96_max_abs_diff=0.,training=training_log)
        model.release();model_path=OUT/f'fold{fold}_v96_model.joblib'
        joblib.dump(model,model_path)
        support_hash=hashlib.sha256(model.support.tobytes()).hexdigest()
        original_counts=model.preprocessor.roster.counts.copy(deep=True)
        p106=p96.copy();patches=packet['cabin_updates']
        if patches:
            base.seed_everything(SEED,'mps');model.device='mps';idx,query=patch_query(model,patches)
            repaired=np.asarray(model.predict_proba(query))[:,1]
            positions=np.searchsorted(qids,idx);np.testing.assert_array_equal(qids[positions],idx)
            p106[positions]=repaired
            if hashlib.sha256(model.support.tobytes()).hexdigest()!=support_hash:raise RuntimeError('Training support changed.')
            pd.testing.assert_frame_equal(model.preprocessor.roster.counts,original_counts,check_exact=True)
            model.release();model=None;query=None;base.release_memory()
            base.seed_everything(SEED,'mps');loaded=joblib.load(model_path);loaded.device='mps'
            if hashlib.sha256(loaded.support.tobytes()).hexdigest()!=support_hash:raise RuntimeError('Reload support changed.')
            idx2,query2=patch_query(loaded,patches);np.testing.assert_array_equal(idx,idx2)
            again=np.asarray(loaded.predict_proba(query2))[:,1];np.testing.assert_array_equal(repaired,again)
            loaded.release();loaded=None;query2=None;base.release_memory()
        else:
            idx=np.array([],dtype=np.int64);model.release();model=None;base.release_memory()
            loaded=joblib.load(model_path)
            if hashlib.sha256(loaded.support.tobytes()).hexdigest()!=support_hash:raise RuntimeError('Reload support changed.')
            loaded=None
        eligible=np.asarray(packet['peer_eligible'],dtype=bool);shift=np.asarray(packet['peer_shift'],dtype=float)
        if eligible.shape!=(len(qids),) or shift.shape!=eligible.shape or not np.isfinite(shift).all() or np.any(shift[~eligible]!=0):raise RuntimeError('Invalid peer offsets.')
        p108=p106.astype(float).copy();p108[eligible]=expit(ref.logit(p108[eligible])+shift[eligible])
        if not np.isfinite(p108).all():raise RuntimeError('Invalid final probabilities.')
        unchanged=~np.isin(qids,idx);np.testing.assert_array_equal(p106[unchanged],p96[unchanged])
        np.testing.assert_array_equal(p108[~eligible],p106[~eligible])
        np.savez(OUT/f'fold{fold}.npz',v96=p96,v106=p106,v108=p108,train_ids=ids,query_ids=qids,
            repaired_query_ids=idx,peer_shift=shift,peer_eligible=eligible)
        check();log.update(status='complete',model_path=str(model_path.relative_to(ROOT)),model_sha256=sha(model_path),
            support_sha256=support_hash,cabin_repaired_rows=len(idx),repaired_subset_reload_max_abs_diff=0.,
            no_redundant_full_query_reload=True,query_only_patch=True,training_support_and_count_features_unchanged=True,
            peer_eligible_rows=int(eligible.sum()),prediction_sha256=sha(OUT/f'fold{fold}.npz'))
    except BaseException as e:
        log.update(status='failed',error_type=type(e).__name__,error=str(e));raise
    finally:
        if model is not None:model.release()
        if loaded is not None:loaded.release()
        model=loaded=None;base.release_memory();log['seconds']=time.monotonic()-started
        write(REPORT/f'fold{fold}.json',log)


def metrics(probabilities,y,masks,frame):
    result={}
    for name,p in probabilities.items():
        ok=(p>=.5)==y
        row=dict(**ref.score(p,y),rows=len(y),correct=int(ok.sum()),errors=int((~ok).sum()),
            versus={n:ref.pair(p,q,y) for n,q in probabilities.items()},
            historical_subsets={n:dict(rows=int(m.sum()),correct=int((ok&m).sum()),errors=int((~ok&m).sum())) for n,m in masks.items()})
        row['Earth']={}
        for state in ['True','False']:
            m=(frame.HomePlanet.astype(str).to_numpy()=='Earth')&(frame.CryoSleep.astype(str).to_numpy()==state)
            pr=p[m]>=.5;t=y[m].astype(bool)
            row['Earth'][state]=dict(rows=int(m.sum()),tn=int((~pr&~t).sum()),fp=int((pr&~t).sum()),fn=int((~pr&t).sum()),tp=int((pr&t).sum()))
        result[name]=row
    return result


def run():
    if OUT.exists() or (REPORT/'protocol.json').exists():raise FileExistsError('No overwrite/resume.')
    OUT.mkdir(parents=True);REPORT.mkdir(parents=True,exist_ok=True);start=time.monotonic();records=[]
    try:
        previous,environment=counts.verify_reference()
        inputs,splits,folds,bases,evidence=overlap.load_authenticated();dev,y=inputs['dev'],inputs['y']
        cached_v96=peer.load_v96(dev,folds)
        vp=ROOT/'artifacts/v96_oof_completion/oof.npz';vr=ROOT/'reports/v96_oof_completion/result.json'
        evidence.update({str(p.relative_to(ROOT)):sha(p) for p in [vp,vr]})
        roster=counts.load_roster();pd.testing.assert_frame_equal(inputs['features'],roster.plain,check_exact=True)
        passengers,known,_,design=peer.context();np.testing.assert_array_equal(passengers.iloc[dev],inputs['passengers'])
        old_offsets=ROOT/'artifacts/peer_bill_offsets.npz';evidence[str(old_offsets.relative_to(ROOT))]=sha(old_offsets)
        with np.load(old_offsets,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['dev'],dev);cached_shift=z['peer_bills'].copy()
        shifts=np.zeros(len(dev));packs=[]
        for f,(a,b) in enumerate(splits):
            teacher=ref.load_inner(f,a,b,dev,y,evidence);eligible=known[dev[a]]
            fitted=peer.fit_offset(design[dev[a[eligible]]],teacher[eligible],y[a[eligible]])
            q=b[known[dev[b]]];shifts[q]=peer.apply(fitted,design[dev[q]])
            np.testing.assert_array_equal(shifts[b],cached_shift[b])
            path=OUT/f'fold{f}_peer_coefficients.joblib';joblib.dump(fitted,path)
            np.testing.assert_array_equal(peer.apply(joblib.load(path),design[dev[q]]),shifts[q]);packs.append(path)
        cabins,updates=cabin_updates();np.testing.assert_array_equal(cabins.PassengerId.iloc[dev],inputs['passengers'])
        np.savez(OUT/'expected_v96.npz',dev_ids=dev,prob=cached_v96)
        write(OUT/'cabin_updates_all_roster.json',updates)
        source_names=set(OWN)|{n for n in previous['source_hashes'] if n.startswith('third_party/mitra/')}
        plan=dict(status='frozen_before_three_native_fits',seed_each_fold=SEED,steps=15,folds=3,development_rows=6954,
            pipeline='Original v96_oof_completion.fit_fold called unchanged. Exact prior v96 OOF match required. Save portable predictor; repair only missing outerquery cabin columns with original v106 rule. Original count features and training support untouched. Add original v108 peer_bill shift only on eligible rows.',
            predictor_capture='Temporary process-local counts.CountsPredictor constructor capture only; original finally/release still executes. Old code files unchanged.',
            peer_offsets='Each outertrain fit on authenticated strict inner rawMitra OOF, same original features/objective. Full recomputed shift array bitwise equals artifacts/peer_bill_offsets.npz before any GPU fit.',
            no_holdout_labels=True,no_query_labels_worker=True,roster='Full unlabeled roster, same deployed transductive definition.',
            support_classification='All15 updates and native prediction use original samefold training labels only.',
            budget_seconds=BUDGET,device='mps',threads=4,memory_fraction=.65,
            source_hashes={n:sha(ROOT/n) for n in sorted(source_names)},evidence_sha256=evidence,
            environment_sha256=hashlib.sha256(environment.encode()).hexdigest(),
            cabin_fillable_dev=int(sum(int(i) in set(dev) for i in updates)),
            peer_offsets_bitwise_reproduced=True,
            limitations=['Adaptive repeated development folds, not independent validation.','Teacher for correction is rawMitra as deployed, not adaptedv96.','Query-only cabin repair never changes training attributes/counts.','OOF reconstructs deployed recipe at fold training size, not the full8693 trained weights.'])
        write(REPORT/'protocol.json',plan);write(OUT/'environment.lock.json',{'lock':environment})
        all_probs={n:np.full(len(y),np.nan,dtype=np.float32 if n!='v108' else np.float64) for n in ['v96','v106','v108']}
        for f,(a,b) in enumerate(splits):
            patches={str(int(i)):updates[int(i)] for i in dev[b] if int(i) in updates}
            packet=dict(train_ids=dev[a].tolist(),query_ids=dev[b].tolist(),train_y=y[a].tolist(),
                cabin_updates=patches,peer_eligible=known[dev[b]].tolist(),peer_shift=shifts[b].tolist())
            write(OUT/f'fold{f}_packet.json',packet)
            remain=BUDGET-(time.monotonic()-start)
            if remain<=0:raise RuntimeError('Total5400seconds exhausted.')
            print('Starting exact v96/v106/v108 fold',f,'cabin patches',len(patches),flush=True)
            with (OUT/f'fold{f}.log').open('w') as log:
                child=subprocess.Popen([sys.executable,__file__,'--worker',str(f)],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=child.wait(timeout=remain)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait();raise RuntimeError('Total5400seconds exhausted.')
            if code:raise RuntimeError(f'Fold{f} failed, see preserved log.')
            record=read(REPORT/f'fold{f}.json');path=OUT/f'fold{f}.npz'
            if record['status']!='complete' or sha(path)!=record['prediction_sha256']:raise RuntimeError('Incomplete fold.')
            with np.load(path,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['query_ids'],dev[b]);np.testing.assert_array_equal(z['train_ids'],dev[a])
                for n in all_probs:all_probs[n][b]=z[n]
            np.testing.assert_array_equal(all_probs['v96'][b],cached_v96[b]);records.append(record)
            from exception_error_detector_nov18 import reporting_masks
            masks=reporting_masks(dev,folds,y,inputs,bases,evidence)
            summary=metrics({n:p[b] for n,p in all_probs.items()},y[b],{n:m[b] for n,m in masks.items()},inputs['features'].iloc[dev[b]])
            write(REPORT/f'fold{f}_result.json',dict(fold=f,metrics=summary,model=record))
            print(json.dumps({'fold':f,'metrics':{n:dict(accuracy=m['accuracy'],errors=m['errors'],vs_v96=m['versus']['v96'],hard546=m['historical_subsets']['all_recipes_common_wrong546']['correct']) for n,m in summary.items()}}),flush=True)
        check();np.testing.assert_array_equal(all_probs['v96'],cached_v96)
        if any(not np.isfinite(p).all() for p in all_probs.values()):raise RuntimeError('Incomplete OOF.')
        np.savez(OUT/'oof.npz',dev_id=dev,dev_ids=dev,fold=folds,fold_id=folds,PassengerId=np.asarray(inputs['passengers'],dtype=str),
            true=y,**all_probs,peer_shift=shifts,peer_eligible=known[dev])
        result=dict(status='complete_exact_v108_oof_and_reusable_fold_models',metrics=metrics(all_probs,y,masks,inputs['features'].iloc[dev]),
            folds=records,seconds=time.monotonic()-start,oof_path=str((OUT/'oof.npz').relative_to(ROOT)),oof_sha256=sha(OUT/'oof.npz'),
            source_hashes=plan['source_hashes'],evidence_sha256=evidence,protocol_sha256=sha(REPORT/'protocol.json'),
            model_components=[dict(fold=f,base_model=str((OUT/f'fold{f}_v96_model.joblib').relative_to(ROOT)),base_sha256=records[f]['model_sha256'],
                peer_coefficients=str(p.relative_to(ROOT)),peer_sha256=sha(p),query_patch_packet=str((OUT/f'fold{f}_packet.json').relative_to(ROOT)),query_patch_packet_sha256=sha(OUT/f'fold{f}_packet.json')) for f,p in enumerate(packs)],
            no_upload=True,limitations=plan['limitations'])
        write(REPORT/'result.json',result);print(json.dumps({'status':result['status'],'metrics':{n:dict(accuracy=m['accuracy'],errors=m['errors'],vs_v96=m['versus']['v96'],hard546=m['historical_subsets']['all_recipes_common_wrong546']['correct']) for n,m in result['metrics'].items()}},indent=2))
    except BaseException as e:
        write(REPORT/'failure.json',dict(status='incomplete',error_type=type(e).__name__,error=str(e),completed_folds=len(records),seconds=time.monotonic()-start));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int);args=parser.parse_args()
    worker(args.worker) if args.worker is not None else run()
