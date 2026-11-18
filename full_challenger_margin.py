"""Full8693 independent CPU challengers and fixed margin-only selector component."""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','FALSE')
from pathlib import Path
import json
import gc
import time
import signal
import subprocess
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold,GroupShuffleSplit
from threadpoolctl import threadpool_limits
import error_evidence_challengers as original
import peer_bill_correction as peer
import iterate
r=original.r
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/full_challenger_margin'
REPORT=ROOT/'reports/full_challenger_margin'
TEACHER=ROOT/'artifacts/proposal_ready_nonlinear_residual_v2'
CHAMPION=ROOT/'artifacts/proposal_v108_party_bills'
SOURCE=ROOT/'artifacts/proposal_nov18_peer_cabin'


class FullMarginSelector:
    def __init__(self,packets,columns,query_ids,metadata):
        self.packets=packets;self.columns=list(columns);self.query_ids=np.asarray(query_ids).copy();self.metadata=metadata
    def predict_components(self,frame,base):
        if list(frame.columns)!=self.columns:raise ValueError('Original30feature order required')
        np.testing.assert_array_equal(frame.index.to_numpy(),self.query_ids)
        base=np.asarray(base,dtype=float)
        if base.shape!=(len(frame),) or not np.isfinite(base).all():raise ValueError('Aligned baseline required')
        members=np.empty((len(frame),3,2));risks=np.empty((len(frame),3));active=np.empty((len(frame),3),bool)
        for j,pack in enumerate(self.packets):
            models=pack['challengers']
            members[:,j,0]=models['forest'].predict_proba(frame)[:,1]
            members[:,j,1]=models['latent'].predict(models['latent_encoder'].transform(frame[original.latent.COLS]))
            active[:,j]=np.any((members[:,j]>=.5)!=(base[:,None]>=.5),axis=1)
            risks[:,j]=original.risk(pack['gate'],original.gate_features(base,members[:,j]))*active[:,j]
        if not np.isfinite(members).all() or not np.isfinite(risks).all():raise RuntimeError('Nonfinite full components')
        mean=risks.mean(1)
        return dict(members=members,member_risk=risks,member_challenged=active,mean_risk=mean,margin_flip=mean>=.5)
FullMarginSelector.__module__='full_challenger_margin'


def load_reference(testids):
    manifest=json.loads((CHAMPION/'manifest.json').read_text());selection=json.loads((CHAMPION/'selection.json').read_text())
    assert manifest['valid'] and manifest['rows']==4277 and manifest['version']==selection['version']=='v108_peer_bills_v106'
    assert manifest['reload_difference']==0 and r.sha(CHAMPION/'submission.csv')==manifest['sha256']
    p=pd.read_csv(CHAMPION/'test_probabilities.csv',dtype={'PassengerId':str},float_precision='round_trip')
    submission=pd.read_csv(CHAMPION/'submission.csv',dtype={'PassengerId':str})
    np.testing.assert_array_equal(p.PassengerId,testids);np.testing.assert_array_equal(submission.PassengerId,testids)
    np.testing.assert_array_equal(p.probability,p.reload_probability);np.testing.assert_array_equal(p.probability>=.5,submission.Transported)
    # Rebuild deployed v108 arithmetic from its frozen v106 cached probabilities.
    assert r.sha(SOURCE/'submission.csv')==manifest['source_csv']
    assert r.sha(CHAMPION/'peer_coefficients.joblib')==manifest['correction_sha256']
    source=pd.read_csv(SOURCE/'test_probabilities.csv',dtype={'PassengerId':str},float_precision='round_trip')
    np.testing.assert_array_equal(source.PassengerId,testids);np.testing.assert_array_equal(source.probability,source.reload_probability)
    ids,known,_,features=peer.context();np.testing.assert_array_equal(ids.iloc[8693:].to_numpy(),testids)
    q=source.probability.to_numpy().copy();mask=known[8693:];coef=joblib.load(CHAMPION/'peer_coefficients.joblib')
    q[mask]=expit(r.logit(q[mask])+peer.apply(coef,features[8693:][mask]))
    np.testing.assert_array_equal(q,p.probability.to_numpy())
    return q,manifest


def run():
    if OUT.exists() or REPORT.exists():raise FileExistsError('No overwrite/resume')
    sources=[Path(__file__).name,'error_evidence_challengers.py','mixed_type_forest_imputation.py','latent_class_em_probe.py','relational_error_corrector.py','iterate.py','experiment.py','peer_bill_correction.py','export_party_bill.py','prepare_residual_ready.py']
    inputs=[TEACHER/'manifest.json',TEACHER/'raw_probabilities.npz',TEACHER/'raw_metadata.json',CHAMPION/'manifest.json',CHAMPION/'selection.json',CHAMPION/'test_probabilities.csv',CHAMPION/'submission.csv',CHAMPION/'peer_coefficients.joblib',SOURCE/'test_probabilities.csv',SOURCE/'submission.csv',ROOT/'data/raw/train.csv',ROOT/'data/raw/test.csv']
    for j in range(3):inputs.extend([TEACHER/f'raw_oof_fold{j}.json',TEACHER/f'raw_oof_fold{j}_rows.npz'])
    bindings={n:r.sha(ROOT/n) for n in sources};bindings.update({str(p.relative_to(ROOT)):r.sha(p) for p in inputs})
    OUT.mkdir(parents=True);REPORT.mkdir(parents=True)
    plan=dict(status='frozen_before_full_training_labels_read',training_rows=8693,query_rows=4277,
        teacher='Existing v101 rawMitra full8693 3foldOOF, originalSKFseed20261112; T=other2folds, teacher querypacket=D+C. Verify original split IDs and fitmetadata.',
        calibration='PerpacketGroupShuffleSplit .33,seed20261118+packet; identical D/C recipe. Challengers fit ONLY T; gate fits challenged C only. D unused because margin-only needs no retrieval.',
        challenger='Original mixed forest2round50trees/depth10+Cat700 and supervised latentK24/EM80, seed20261118+packet; same parameters as development experiment.',
        gate='Original margin-only4features and StandardScaler+LogisticRegression C=.15;3 independentpacket gate risks zero when neither challenger disagrees, average3 and >=.5. No v114 or other voters involved.',
        baseline='Actual saved full-data v108 test probabilities, exact arithmetic reconstruction from v106+peer correction.',
        outputs='Reusable CPUmodel + all3member probabilities/risks/challenge flags and margin_flip, not final upload CSV.',
        full_labels_authorized=True,no_CV_scoring=True,no_GPU=True,no_upload=True,budget_seconds=1200,CPU_threads=4,
        limitations=['Full-data component, no new validation score.','Gate calibrated against raw teacher margin transferred to actualv108.','No new weighting,threshold or candidate selection.'])
    r.write(REPORT/'protocol.json',plan);bindings[str((REPORT/'protocol.json').relative_to(ROOT))]=r.sha(REPORT/'protocol.json');r.write(OUT/'protocol.json',plan)
    freeze=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_hashes=bindings)
    r.write(REPORT/'freeze.json',freeze);r.write(OUT/'freeze.json',freeze)
    started=time.monotonic();old=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Full marginCPU1200s')));signal.setitimer(signal.ITIMER_REAL,1200)
    try:
        teacher=json.loads((TEACHER/'manifest.json').read_text());meta=json.loads((TEACHER/'raw_metadata.json').read_text())
        assert teacher['status']=='complete' and teacher['training_rows']==8693 and teacher['raw_component']['raw_reload_max_abs_diff']==0
        for f in ['raw_probabilities.npz','raw_metadata.json']:assert r.sha(TEACHER/f)==teacher['raw_component']['files'][f]
        assert meta['recipe']=='zero-shot' and meta['selected_steps']==0 and meta['training_rows']==8693
        train=pd.read_csv(ROOT/'data/raw/train.csv',dtype={'PassengerId':str});test=pd.read_csv(ROOT/'data/raw/test.csv',dtype={'PassengerId':str},usecols=lambda c:c!='Transported')
        if len(train)!=8693 or len(test)!=4277:raise ValueError('Exact competition row counts required')
        y=train.Transported.to_numpy(int);attrs=pd.concat([train.drop(columns=['Transported','Name']),test.drop(columns=['Name'])],ignore_index=True)
        x,_=iterate.build_features(attrs,False);assert x.shape==(12970,30)
        ids=np.arange(8693);qids=np.arange(8693,12970);groups=train.PassengerId.str.split('_').str[0].to_numpy();testids=test.PassengerId.to_numpy(dtype=str)
        with np.load(TEACHER/'raw_probabilities.npz') as z:
            np.testing.assert_array_equal(z['train_ids'],ids);np.testing.assert_array_equal(z['train_passenger_ids'],train.PassengerId);np.testing.assert_array_equal(z['test_ids'],testids)
            teacher_p=z['oof_raw'].copy();folds=z['fold_ids'].copy()
        if teacher_p.shape!=(8693,) or not np.isfinite(teacher_p).all():raise ValueError('Incomplete full teacherOOF')
        base,champion_manifest=load_reference(testids)
        assert champion_manifest['teacher_oof_sha256']==teacher['raw_component']['files']['raw_probabilities.npz']
        packets=[];records=[];members=np.empty((4277,3,2));risks=np.empty((4277,3));active=np.empty((4277,3),bool)
        with threadpool_limits(limits=4):
            for j,(t,packet) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261112).split(ids,y)):
                tick=time.monotonic();seed=20261118+j
                np.testing.assert_array_equal(np.flatnonzero(folds==j),packet)
                with np.load(TEACHER/f'raw_oof_fold{j}_rows.npz') as z:np.testing.assert_array_equal(z['train_ids'],t);np.testing.assert_array_equal(z['query_ids'],packet)
                log=json.loads((TEACHER/f'raw_oof_fold{j}.json').read_text());assert log['training_ids_sha256']==r.nested.ids_hash(t) and log['training_rows']==len(t) and log['seed']==20261112+j and log['selected_steps']==0 and log['recipe']=='zero-shot'
                ld,lc=next(GroupShuffleSplit(n_splits=1,test_size=.33,random_state=seed).split(packet,groups=groups[packet]));d,c=packet[ld],packet[lc]
                assert not np.intersect1d(groups[d],groups[c]).size and not np.intersect1d(t,np.r_[d,c]).size
                query=np.r_[c,qids];prob,challengers=original.fit_challengers(x.loc[t],y[t].copy(),x.loc[query],seed)
                cal=prob[:len(c)];members[:,j]=prob[len(c):]
                gate,gate_meta=original.fit_gate(original.gate_features(teacher_p[c],cal),teacher_p[c],cal,y[c],seed)
                active[:,j]=np.any((members[:,j]>=.5)!=(base[:,None]>=.5),axis=1)
                risks[:,j]=original.risk(gate,original.gate_features(base,members[:,j]))*active[:,j]
                packets.append(dict(challengers=challengers,gate=gate,teacher_packet=j,seed=seed))
                packetpath=OUT/f'packet{j}.npz';np.savez_compressed(packetpath,T_ids=t,D_ids=d,C_ids=c,C_teacher=teacher_p[c],C_challengers=cal,test_ids=qids)
                record=dict(packet=j,seed=seed,T_rows=len(t),D_rows=len(d),C_rows=len(c),fit=gate_meta,teacher_training_log_sha256=r.sha(TEACHER/f'raw_oof_fold{j}.json'),packet_sha256=r.sha(packetpath),seconds=time.monotonic()-tick)
                records.append(record);r.write(OUT/f'packet{j}.json',record);print(json.dumps(dict(packet=j,seconds=record['seconds'],challenged_C=gate_meta['training_rows'])),flush=True)
            first=dict(members=members,member_risk=risks,member_challenged=active,mean_risk=risks.mean(1),margin_flip=risks.mean(1)>=.5)
            model=FullMarginSelector(packets,x.columns,qids,dict(training_rows=8693,query_rows=4277,recipe=plan))
            path=OUT/'model.joblib';joblib.dump(model,path);model=None;packets=None;challengers=None;gate=None;gc.collect()
            loaded=joblib.load(path);second=loaded.predict_components(x.loc[qids],base)
            for key in first:np.testing.assert_array_equal(first[key],second[key])
        csv=OUT/'test_decisions.csv';pd.DataFrame(dict(PassengerId=testids,margin_flip=first['margin_flip'],reload_margin_flip=second['margin_flip'])).to_csv(csv,index=False)
        back=pd.read_csv(csv,dtype={'PassengerId':str});np.testing.assert_array_equal(back.PassengerId,testids);np.testing.assert_array_equal(back.margin_flip,first['margin_flip']);np.testing.assert_array_equal(back.reload_margin_flip,first['margin_flip'])
        components=OUT/'test_components.npz';np.savez_compressed(components,test_ids=qids,passenger_ids=np.asarray(testids,dtype=str),v108=base,**first,**{'reload_'+k:v for k,v in second.items()})
        for n,digest in bindings.items():assert r.sha(ROOT/n)==digest,n
        files={p.name:r.sha(p) for p in [path,csv,components]}
        manifest=dict(status='complete',artifact_kind='reusable_cpu_margin_selector',version='v115_margin_selector_component',training_rows=8693,rows=4277,
            test_decisions_sha256=files['test_decisions.csv'],model_sha256=files['model.joblib'],test_components_sha256=files['test_components.npz'],files=files,
            reload_max_abs_difference=0.,reload_margin_flip_equal=True,all4277_components_exact=True,flip_count=int(first['margin_flip'].sum()),
            baseline_version='v108_peer_bills_v106',baseline_submission_sha256=champion_manifest['sha256'],baseline_probabilities_sha256=r.sha(CHAMPION/'test_probabilities.csv'),
            packets=records,seconds=time.monotonic()-started,source_commit=freeze['source_commit'],source_hashes=bindings,protocol_sha256=r.sha(REPORT/'protocol.json'),no_upload=True,no_GPU=True)
        for folder in [OUT,REPORT]:r.write(folder/'manifest.json',manifest)
        print(json.dumps({k:manifest[k] for k in ['status','rows','flip_count','seconds','reload_max_abs_difference','model_sha256','test_decisions_sha256']}),flush=True)
    except BaseException as exc:
        failure=dict(status='incomplete',type=type(exc).__name__,error=str(exc))
        for folder in [OUT,REPORT]:r.write(folder/'failure.json',failure)
        raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
if __name__=='__main__':
    import sys
    sys.modules['full_challenger_margin']=sys.modules[__name__]
    run()
