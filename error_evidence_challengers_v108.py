"""Authenticate exact champion OOF, audit packet boundaries, replay frozen arbitration."""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','FALSE')
from pathlib import Path
import json
import time
import hashlib
import subprocess
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold,GroupShuffleSplit
from threadpoolctl import threadpool_limits
import error_evidence_challengers as m
r=m.r
ROOT=Path(__file__).resolve().parent
OUT=m.OUT
REPORT=m.REPORT


def run():
    if (REPORT/'v108_result.json').exists():raise FileExistsError('Do not overwrite exact-reference evaluation')
    start=time.monotonic();source=json.loads((REPORT/'result.json').read_text())
    refpath=ROOT/'reports/v108_exact_oof/result.json';reference=json.loads(refpath.read_text());refcache=ROOT/'artifacts/v108_exact_oof/oof.npz'
    assert source['status']=='complete' and reference['status']=='complete_exact_v108_oof_and_reusable_fold_models'
    assert r.sha(refcache)==reference['oof_sha256'];assert r.sha(ROOT/'reports/v108_exact_oof/protocol.json')==reference['protocol_sha256']
    historical_source={}
    for n,d in {**source['source_hashes'],**reference['source_hashes']}.items():
        if n=='error_trained_experts.py' and r.sha(ROOT/n)!=d:
            # Historical partition-design source only: not imported by our numerical pipeline.
            original=subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)
            assert hashlib.sha256(original).hexdigest()==d
            snapshot=OUT/'historical_error_trained_experts.py';snapshot.write_bytes(original)
            historical_source=dict(path=str(snapshot.relative_to(ROOT)),sha256=d,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),reason='Root added exactv108 reporting after our completed run; original D/C design verified in committed historical bytes, not a runtime dependency.')
        else:assert r.sha(ROOT/n)==d,n
    assert r.sha(OUT/'oof.npz')==source['oof_sha256']
    data,splits,folds,bases,e=m.overlap.load_authenticated();dev,y=data['dev'],data['y'];groups=np.asarray(data['groups'],dtype=str)
    with np.load(refcache) as z:
        for k,w in [('dev_ids',dev),('fold_id',folds),('PassengerId',data['passengers']),('true',y)]:np.testing.assert_array_equal(z[k],w)
        base=z['v108'].copy()
    assert int(np.sum((base>=.5)!=y))==reference['metrics']['v108']['errors']==1291
    with np.load(OUT/'oof.npz') as z:
        for k,w in [('dev_ids',dev),('fold_id',folds),('passenger_ids',data['passengers'])]:np.testing.assert_array_equal(z[k],w)
        members=z['members'].copy();ev=z['evidence_by_direction'].copy();raw=z['raw'].copy();previous={k:{f:z[k+'_'+f].copy() for f in ['flip','mean_risk','member_risk']} for k in ['margin','case']}
    binding={str(p.relative_to(ROOT)):r.sha(p) for p in [ROOT/Path(__file__).name,refpath,refcache,ROOT/'reports/v108_exact_oof/protocol.json',REPORT/'result.json',OUT/'oof.npz']}
    if historical_source:binding[historical_source['path']]=historical_source['sha256']
    models={};audit_packets=[]
    with threadpool_limits(limits=4):
        for outer,(a,b) in enumerate(splits):
            inside=r.load_inner(outer,a,b,dev,y,e)
            foldpath=ROOT/f'artifacts/v108_exact_oof/fold{outer}.npz';assert r.sha(foldpath)==reference['folds'][outer]['prediction_sha256']
            with np.load(foldpath) as z:
                np.testing.assert_array_equal(z['train_ids'],dev[a]);np.testing.assert_array_equal(z['query_ids'],dev[b]);np.testing.assert_array_equal(z['v108'],base[b])
            binding[str(foldpath.relative_to(ROOT))]=r.sha(foldpath)
            for j,(t,packet) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261117+outer).split(a,y[a])):
                seed=20261118+100*outer+j;ld,lc=next(GroupShuffleSplit(n_splits=1,test_size=.33,random_state=seed).split(packet,groups=groups[a[packet]]));d,c=packet[ld],packet[lc]
                rec=source['packets'][outer*3+j];assert (rec['outer'],rec['packet'])==(outer,j)
                pth=OUT/f'outer{outer}_packet{j}.npz';assert r.sha(pth)==rec['packet_sha256'];gpath=OUT/f'outer{outer}_packet{j}_gates.joblib';assert r.sha(gpath)==rec['gate_model_sha256']
                assert r.sha(OUT/f'outer{outer}_packet{j}_challengers.joblib')==rec['challenger_model_sha256']
                models[(outer,j)]=joblib.load(gpath)
                for p in [pth,gpath]:binding[str(p.relative_to(ROOT))]=r.sha(p)
                with np.load(pth) as z:
                    for k,w in [('teacher_support_ids',dev[a[t]]),('D_ids',dev[a[d]]),('C_ids',dev[a[c]]),('outer_query_ids',dev[b]),('C_base',inside[c]),('D_base',inside[d])]:np.testing.assert_array_equal(z[k],w)
                    assert not np.intersect1d(dev[a[t]],np.r_[dev[a[d]],dev[a[c]],dev[b]]).size
                    assert not np.intersect1d(groups[a[d]],groups[a[c]]).size
                    lookup={int(v):idx for idx,v in enumerate(dev[a[d]])}
                    for view in ['consumption','passenger_cabin']:
                        for direction in [0,1]:
                            neighbors=z[view+f'_outer{direction}_neighbors'];ii=np.array([[lookup[int(v)] for v in row] for row in neighbors])
                            assert not np.any(groups[a[d]][ii]==groups[b,None]);assert not np.any(neighbors==dev[b,None]);assert np.all((inside[d][ii]>=.5)==direction)
                    active=np.any((z['C_challengers']>=.5)!=(z['C_base'][:,None]>=.5),axis=1)
                    for key in ['margin','case']:assert int(active.sum())==rec['fits'][key]['training_rows']
                audit_packets.append(dict(outer=outer,packet=j,T_D_C_query_rows_disjoint=True,D_C_groups_disjoint=True,case_neighbors_only_D_not_self_or_group=True))
        # Exact replay of prior raw output first, then change ONLY reference base input.
        replayed=m.replay(raw,members,ev,folds,models)
        for k in previous:
            for f in previous[k]:np.testing.assert_array_equal(previous[k][f],replayed[k][f])
        outputs=m.replay(base,members,ev,folds,models)
    rows=pd.read_csv(ROOT/'artifacts/error_rescue_20261117/catalog_rows.csv',dtype={'PassengerId':str});np.testing.assert_array_equal(rows.row_id,dev);np.testing.assert_array_equal(rows.y,y)
    hard=rows.all_recipes_still_wrong.to_numpy(bool);assert hard.sum()==546
    reference_labels=base>=.5;expert_prob=members.mean(1)
    labels={name:expert_prob[:,j]>=.5 for j,name in enumerate(m.FAMILIES)};labels.update({k:reference_labels^v['flip'] for k,v in outputs.items()});labels['v108']=reference_labels
    def summary(mask):
        out={}
        for name,p in labels.items():
            original=reference_labels==y;correct=p==y;repaired=correct&~original;broken=~correct&original
            errors=int((~original&mask).sum());right=int((original&mask).sum());rr=int((repaired&mask).sum());bb=int((broken&mask).sum())
            out[name]=dict(rows=int(mask.sum()),baseline_errors=errors,baseline_correct=right,repairs=rr,breaks=bb,net=rr-bb,error_recall=rr/errors if errors else None,correct_harm_rate=bb/right if right else None,correct=int((correct&mask).sum()),accuracy=float(correct[mask].mean()) if mask.any() else None)
        return out
    frame=data['features'].loc[dev];earth=frame.HomePlanet.astype(str).eq('Earth').to_numpy();sleep=frame.CryoSleep.astype(str).str.lower().eq('true').to_numpy();awake=frame.CryoSleep.astype(str).str.lower().eq('false').to_numpy()
    masks=dict(all=np.ones(len(y),bool),hard546=hard,base_predicts_positive=reference_labels,base_predicts_negative=~reference_labels,base_false_positive=reference_labels&(y==0),base_false_negative=~reference_labels&(y==1),Earth_sleep=earth&sleep,Earth_awake=earth&awake)
    wrong=reference_labels!=y;mean_rescue=((expert_prob>=.5)==y[:,None])&wrong[:,None];member_rescue=((members>=.5)==y[:,None,None])&wrong[:,None,None]
    path=OUT/'v108_replay.npz';np.savez_compressed(path,dev_ids=dev,fold_id=folds,passenger_ids=np.asarray(data['passengers'],dtype=str),v108=base,**{k+'_'+field:value for k,items in outputs.items() for field,value in items.items()})
    result=dict(status='complete_exact_v108_replay_no_refit',primary_reference='v108_exact_oof, not proxy',subsets={k:summary(mask) for k,mask in masks.items()},
        case_increment_over_margin=r.pair(labels['case'].astype(float),labels['margin'].astype(float),y),case_increment_hard546=r.pair(labels['case'].astype(float),labels['margin'].astype(float),y,hard),
        oracle_not_deployable=dict(two_mean_challengers=int(mean_rescue.any(1).sum()),two_mean_hard546_new=int((mean_rescue.any(1)&hard).sum()),any_of_six_members=int(member_rescue.any((1,2)).sum()),any_member_hard546_new=int((member_rescue.any((1,2))&hard).sum())),
        fold_results=[dict(fold=i,all=summary(folds==i),hard546=summary(hard&(folds==i))) for i in range(3)],
        audit=dict(all9_packet_boundaries_verified=True,all_neighbor_ID_direction_group_checks_passed=True,raw_saved_replay_exact=True,v108_threefold_prediction_hash_and_identity_verified=True,packets=audit_packets),
        historical_design_source=historical_source,seconds=time.monotonic()-start,replay_sha256=r.sha(path),bound_files=binding,limitations=['Fixed raw-trained arbitration transferred to exactv108; no v108-specific calibration fit.','All thresholds/coefficients unchanged; same6954 labels only evaluate.','Full challenger/prospective boundaries remain as frozen original protocol.'],no_refit=True,no_upload=True)
    for n,d in binding.items():assert r.sha(ROOT/n)==d,n
    r.write(REPORT/'v108_result.json',result);print(json.dumps(dict(all=result['subsets']['all'],hard=result['subsets']['hard546'],increment=result['case_increment_over_margin'],oracle=result['oracle_not_deployable']),indent=2))
if __name__=='__main__':run()
