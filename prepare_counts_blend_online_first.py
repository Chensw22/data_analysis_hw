"""One fixed cache-only half mean of the previous and new online best. No fit, model import, GPU, CV or upload."""
import csv
from datetime import datetime, timezone
from pathlib import Path
import struct
import subprocess

import prepare_v84_online_first as common

ROOT = Path(__file__).resolve().parent
POLICY = 'reports/online_iteration_20261117/online_first_counts_blend_policy.json'
COMPONENTS = {
    'adapt': ('v81_Mitra_full_episode50', 'proposal_nov16_full_episode', 'float32'),
    'counts': ('v96_Mitra_counts_fixed15_online_first', 'proposal_online_first_counts15', 'float32'),
}
RECIPES = [
    ('v98_v81_v96_equal2_online_first', 'proposal_online_first_counts_half', ('adapt','counts')),
]
sha, read, write = common.sha, common.read, common.write


def inspect():
    ids = [r['PassengerId'] for r in common.table(ROOT/'data/raw/test.csv')]
    if len(ids)!=4277 or len(set(ids))!=4277:
        raise RuntimeError('Complete unique test row order required.')
    evidence = {'data/raw/test.csv':sha(ROOT/'data/raw/test.csv')}
    components, values = {}, {}
    for key, (version,proposal,dtype) in COMPONENTS.items():
        directory=ROOT/'artifacts'/proposal
        manifest=common.pair(proposal,'manifest.json');selection=common.pair(proposal,'selection.json')
        if (manifest.get('status')!='complete' or manifest.get('export_reload_inference_verified') is not True
                or manifest.get('fit_rows',manifest.get('training_rows'))!=8693
                or manifest.get('rows')!=4277 or manifest.get('threshold')!=.5
                or selection['version']!=version or manifest['selection']!=selection
                or sha(directory/'model.joblib')!=manifest['model_sha256']
                or sha(directory/'submission.csv')!=manifest['sha256']):
            raise RuntimeError('Complete component identity/reload missing: '+proposal)
        common.reload_check(manifest)
        p=common.vector(directory/'test_probabilities.csv',ids)
        q=common.vector(directory/'test_probabilities.csv',ids,'reload_probability')
        serialized_difference=common.compare(p,q)
        # Recover original native FP32 results from the historical CSV's shortest
        # roundtrip decimal spelling. EXAONE's native CPU aggregation remains FP64.
        if dtype=='float32':
            fp32=lambda value:struct.unpack('f',struct.pack('f',value))[0]
            p=[fp32(v) for v in p];q=[fp32(v) for v in q]
        difference=common.compare(p,q)
        if abs(difference-manifest['reload_max_absolute_difference'])>1e-12:
            raise RuntimeError('Full cached reload differs from recorded component proof.')
        rows=common.table(directory/'submission.csv')
        if [r['PassengerId'] for r in rows]!=ids or any(r['Transported']!=str(v>=.5) for r,v in zip(rows,p)):
            raise RuntimeError('Component probabilities disagree with submitted CSV.')
        for name in ('manifest.json','selection.json','model.joblib','test_probabilities.csv','submission.csv','reload_metadata.json'):
            path=f'artifacts/{proposal}/{name}';evidence[path]=sha(ROOT/path)
        for name in ('manifest.json','selection.json'):
            path=f'reports/{proposal}/{name}';evidence[path]=sha(ROOT/path)
        # Keep exact source/context metadata evidence without loading any model.
        components[key]=dict(version=version,proposal=proposal,model_sha256=manifest['model_sha256'],
            probability_csv_sha256=sha(directory/'test_probabilities.csv'),training_rows=8693,query_rows=4277,
            native_probability_dtype=dtype,serialized_decimal_reload_difference=serialized_difference,
            native_probability_reload_difference=difference,reload_evidence=f'reports/{proposal}/manifest.json')
        values[key]=(p,q)
    return ids,values,components,evidence


def prepare():
    paths=[ROOT/base/proposal for _,proposal,_ in RECIPES for base in ('artifacts','reports')]
    if (ROOT/POLICY).exists() or any(p.exists() for p in paths):
        raise RuntimeError('No overwrite or implicit resume of fixed proposal/policy.')
    ids,values,components,evidence=inspect()
    code={Path(__file__).name:sha(Path(__file__)), 'prepare_v84_online_first.py':sha(ROOT/'prepare_v84_online_first.py')}
    policy=dict(competition='spaceship-titanic',batch_id='online_iteration_20261117',
        user_instruction='Continued online-first iteration: fixed half mean of v81 and new v96, no weight search; shared final daily slot.',
        allow_full_export=True,allow_direct_submit=True,versions=[v for v,_,_ in RECIPES],
        maximum_new_upload_attempts_current_batch=9,maximum_upload_attempts_for_each_file=1,
        independent_additional_budget=False,local_cv_or_group_gate_required=False,
        actual_daily_limit_required=True,threshold=.5,
        recipes=[dict(version=v,proposal=p,components=list(keys),weights=[.5,.5]) for v,p,keys in RECIPES],
        components=components,evidence_files=evidence,source_hashes=code,
        reuse_complete_component_reload_evidence=True,no_new_model_inference=True,
        no_weight_seed_threshold_search=True,original_failed_gates_unchanged=True,
        recorded_at_utc=datetime.now(timezone.utc).isoformat())
    write(ROOT/POLICY,policy)
    sources={**evidence,**code,POLICY:sha(ROOT/POLICY)}
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    completed=[]
    for version,proposal,keys in RECIPES:
        out,report=ROOT/'artifacts'/proposal,ROOT/'reports'/proposal
        out.mkdir(parents=True);report.mkdir(parents=True)
        try:
            left,right=(values[key] for key in keys)
            probability=[.5*a+.5*b for a,b in zip(left[0],right[0])]
            reload_probability=[.5*a+.5*b for a,b in zip(left[1],right[1])]
            difference=common.compare(probability,reload_probability)
            with (out/'test_probabilities.csv').open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['PassengerId','probability','reload_probability'])
                writer.writerows(zip(ids,probability,reload_probability))
            with (out/'submission.csv').open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['PassengerId','Transported'])
                writer.writerows(zip(ids,[p>=.5 for p in probability]))
            saved_difference=common.compare(probability,common.vector(out/'test_probabilities.csv',ids))
            if saved_difference!=0:raise RuntimeError('Saved probability CSV not an exact roundtrip.')
            rows=common.table(out/'submission.csv')
            if [r['PassengerId'] for r in rows]!=ids or any(r['Transported']!=str(v>=.5) for r,v in zip(rows,probability)):
                raise RuntimeError('Output IDs/order/boolean values differ.')
            digest=sha(out/'submission.csv')
            for path in (ROOT/'reports').rglob('online_attempts/*.json'):
                if read(path).get('file_sha256')==digest:raise RuntimeError('Identical CSV already has an upload attempt.')
            for path,expected in sources.items():
                if sha(ROOT/path)!=expected:raise RuntimeError('Frozen component/source changed: '+path)
            selection=dict(version=version,components=[components[k]['version'] for k in keys],weights=[.5,.5],
                threshold=.5,stage='explicit_online_first_cached_components',A_eligible=None,S_supported=False)
            manifest=dict(status='complete',version=version,proposal=proposal,rows=4277,training_rows=8693,
                sha256=digest,valid=True,threshold=.5,selection=selection,
                artifact_kind='cached_probability_ensemble',cached_ensemble_verified=True,
                test_probabilities_sha256=sha(out/'test_probabilities.csv'),
                saved_probability_roundtrip_max_difference=saved_difference,
                components=[dict(components[k],weight=.5) for k in keys],component_evidence=evidence,
                component_full_export_reload_verified=True,new_fit_performed=False,new_model_inference_performed=False,
                new_combined_model_reload_performed=False,reload_rows=4277,reload_max_absolute_difference=difference,
                reload_threshold_decisions_equal=True,
                reload_verification_method='Reuse exact complete component reload proofs; mean complete cached original/reload probabilities; save and reread all4277. No new combined GPU reload.',
                original_failed_gates_unchanged=True,source_commit=commit,source_hashes=sources,
                policy_path=POLICY,policy_sha256=sources[POLICY],
                source_commit_note='Root must commit new code/policy/results before upload.',
                completed_at_utc=datetime.now(timezone.utc).isoformat())
            for name,value in [('selection.json',selection),('manifest.json',manifest)]:
                write(out/name,value);write(report/name,value)
            completed.append(dict(version=version,proposal=proposal,sha256=digest,rows=4277,reload_difference=difference))
        except BaseException as error:
            for directory in (out,report):
                for name in ('manifest.json','selection.json','submission.csv'):(directory/name).unlink(missing_ok=True)
                write(directory/'failure.json',dict(status='failed_not_upload_eligible',version=version,error_type=type(error).__name__,error=str(error)))
            raise
    print(common.json.dumps(completed,indent=2))
    return completed


if __name__=='__main__':
    prepare()
