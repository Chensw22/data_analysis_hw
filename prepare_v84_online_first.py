"""Cache-only export of the predeclared v71/v82/v83 equal probability mean.

No numerical/model library imports, training, inference, CV or upload. Existing
full component reload evidence is reused; a new combined GPU reload is NOT claimed.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import struct

ROOT = Path(__file__).resolve().parent
VERSION = 'v84_Mitra_TabDPT_Kumo_equal3'
PROPOSAL = 'proposal_online_first_v84'
POLICY = 'reports/online_iteration_20261117/online_first_v84_policy.json'
PLAN = 'reports/online_iteration_20261116/fixed_three_family_plan.json'
COMPONENTS = [('v71_Mitra2_zeroshot', 'proposal_batch_mitra2'),
              ('v82_TabDPT13_full8', 'proposal_nov16_tabdpt'),
              ('v83_Kumo_medium8', 'proposal_nov16_kumo')]
SUPPLEMENT = 'proposal_nov16_prior_adapt'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n'); tmp.replace(path)


def table(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def pair(proposal, name):
    artifact, report = ROOT/'artifacts'/proposal/name, ROOT/'reports'/proposal/name
    if sha(artifact) != sha(report):
        raise RuntimeError('Artifact/report identity mismatch: ' + proposal + '/' + name)
    # Reports establish the historical evidence chain, not a claim that today's
    # uncommitted exporter has already been reviewed/committed.
    old = subprocess.check_output(['git', 'show', 'HEAD:' + str(report.relative_to(ROOT))], cwd=ROOT)
    if hashlib.sha256(old).hexdigest() != sha(report):
        raise RuntimeError('Historical report differs from committed evidence.')
    return read(artifact)


def reload_check(record):
    difference = record.get('reload_max_absolute_difference')
    if (record.get('reload_rows') != 4277 or record.get('reload_threshold_decisions_equal') is not True
            or not isinstance(difference, (float, int)) or not math.isfinite(difference)
            or not 0 <= difference <= 2e-6):
        raise RuntimeError('Complete component reload evidence required.')


def vector(path, ids, field='probability'):
    rows = table(path)
    if [r.get('PassengerId') for r in rows] != ids:
        raise RuntimeError('Probability row identity/order mismatch: ' + str(path))
    values = [float(r[field]) for r in rows]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise RuntimeError('Invalid full probability vector.')
    return values


def compare(first, second):
    if len(first) != 4277 or len(second) != 4277:
        raise RuntimeError('All 4277 predictions required.')
    difference = max(abs(a-b) for a,b in zip(first,second))
    if difference > 2e-6 or any((a >= .5) != (b >= .5) for a,b in zip(first,second)):
        raise RuntimeError('Complete probability/threshold equivalence failed.')
    return difference


def inspect():
    plan = read(ROOT/PLAN)
    if plan['candidate'] != VERSION or [(r['version'],r['weight']) for r in plan['components']] != [(v,1/3) for v,_ in COMPONENTS]:
        raise RuntimeError('Original fixed three-family recipe changed.')
    ids = [r['PassengerId'] for r in table(ROOT/'data/raw/test.csv')]
    if len(ids) != 4277 or len(set(ids)) != 4277:
        raise RuntimeError('Expected complete unique test PassengerIds.')
    evidence = {PLAN: sha(ROOT/PLAN), 'three_family_mean_screen.py': sha(ROOT/'three_family_mean_screen.py'),
                'data/raw/test.csv': sha(ROOT/'data/raw/test.csv')}
    supplement = pair(SUPPLEMENT, 'manifest.json')
    logs = pair(SUPPLEMENT, 'component_reload_log.json')
    raw_check = [r for r in logs if r['version'] == COMPONENTS[0][0]]
    if len(raw_check) != 1 or raw_check[0] != supplement['component_reload_checks'][0]:
        raise RuntimeError('Raw Mitra supplementary reload identity differs.')
    if supplement.get('status') != 'complete' or supplement.get('export_reload_inference_verified') is not True:
        raise RuntimeError('Incomplete supplementary raw reload evidence.')
    reload_check(raw_check[0])
    for name in ('manifest.json', 'component_reload_log.json'):
        for prefix in ('reports', 'artifacts'):
            path = f'{prefix}/{SUPPLEMENT}/{name}'; evidence[path] = sha(ROOT/path)
    vectors, reloaded, provenance = [], [], []
    for version, proposal in COMPONENTS:
        folder = ROOT/'artifacts'/proposal
        manifest = pair(proposal, 'manifest.json'); selection = pair(proposal, 'selection.json')
        if (selection['version'] != version or manifest['selection'] != selection
                or manifest.get('fit_rows') != 8693 or manifest.get('rows') != 4277
                or manifest.get('threshold') != .5 or sha(folder/'model.joblib') != manifest['model_sha256']
                or sha(folder/'submission.csv') != manifest['sha256']):
            raise RuntimeError('Component full-data artifact identity differs: ' + proposal)
        p = vector(folder/'test_probabilities.csv', ids)
        submission = table(folder/'submission.csv')
        if ([r['PassengerId'] for r in submission] != ids
                or any(r['Transported'] != str(v >= .5) for r,v in zip(submission,p))):
            raise RuntimeError('Cached probabilities disagree with original submission.')
        if version == COMPONENTS[0][0]:
            check = raw_check[0]
            if (check['source_model_sha256'] != manifest['model_sha256'] or check['support_rows'] != 8693
                    or check['proposal'] != proposal or check['reload_max_absolute_difference'] != 0):
                raise RuntimeError('Raw reload does not bind the exact complete cached model.')
            for name in ('manifest.json', 'model.joblib', 'selection.json', 'test_probabilities.csv'):
                path = f'artifacts/{proposal}/{name}'
                if sha(ROOT/path) != supplement['provenance']['evidence_files'][path]:
                    raise RuntimeError('Raw cache changed since full supplementary reload.')
            q = p.copy()  # Supplementary full reload proves exact difference zero.
            reload_source = f'reports/{SUPPLEMENT}/component_reload_log.json'
        else:
            if manifest.get('status') != 'complete' or manifest.get('export_reload_inference_verified') is not True:
                raise RuntimeError('Component original reload incomplete.')
            reload_check(manifest)
            serialized_q = vector(folder/'test_probabilities.csv', ids, 'reload_probability')
            compare(p, serialized_q)
            # Historical exporter wrote first as float64 and reload as float32.
            # Recover the original float32 values from their shortest-roundtrip decimals.
            q = [struct.unpack('f', struct.pack('f', v))[0] for v in serialized_q]
            difference = compare(p,q)
            if abs(difference-manifest['reload_max_absolute_difference']) > 1e-12:
                raise RuntimeError('Stored component reload difference does not match full cache.')
            metadata = read(folder/'reload_metadata.json')
            if metadata.get('training_rows') != 8693 or metadata.get('prediction_rows') != 4277 or metadata.get('full_training_context') is not True:
                raise RuntimeError('Reload metadata lacks complete training/query context.')
            evidence[str((folder/'reload_metadata.json').relative_to(ROOT))] = sha(folder/'reload_metadata.json')
            reload_source = f'reports/{proposal}/manifest.json'
        for name in ('manifest.json','selection.json','model.joblib','submission.csv','test_probabilities.csv'):
            path = f'artifacts/{proposal}/{name}'; evidence[path] = sha(ROOT/path)
        for name in ('manifest.json','selection.json'):
            path = f'reports/{proposal}/{name}'; evidence[path] = sha(ROOT/path)
        vectors.append(p); reloaded.append(q)
        provenance.append(dict(version=version, proposal=proposal, weight=1/3, model_sha256=manifest['model_sha256'],
            probability_csv_sha256=sha(folder/'test_probabilities.csv'), reload_evidence=reload_source,
            historical_original_export_reload=manifest.get('export_reload_inference_verified'),
            training_rows=8693, query_rows=4277))
    return ids, vectors, reloaded, provenance, evidence


def freeze_policy():
    if (ROOT/POLICY).exists():
        raise RuntimeError('Policy already exists; no overwrite.')
    _,_,_,components,evidence = inspect()
    write(ROOT/POLICY, dict(competition='spaceship-titanic', batch_id='online_iteration_20261117',
        user_instruction='用户要求线上优先，利用每日额度直接尝试已有模型；root明确授权准备原v84固定三结构均值作为第三个线上候选。',
        version=VERSION, versions=[VERSION], proposal=PROPOSAL, allow_full_export=True, allow_direct_submit=True,
        maximum_upload_attempts_for_this_file=1, maximum_new_upload_attempts_current_batch=9, shared_batch_maximum_new_attempts=9,
        independent_additional_budget=False, actual_daily_limit_required=True,
        local_cv_or_group_gate_required=False, weights=[1/3]*3, threshold=.5,
        components=components, evidence_files=evidence, exporter_sha256=sha(Path(__file__)),
        original_failed_gates_unchanged=True, no_new_model_inference=True,
        reuse_complete_component_reload_evidence=True, new_combined_model_reload_performed=False,
        recorded_at_utc=datetime.now(timezone.utc).isoformat()))


def prepare():
    out, report = ROOT/'artifacts'/PROPOSAL, ROOT/'reports'/PROPOSAL
    if out.exists() or report.exists():
        raise RuntimeError('No overwrite or implicit resume.')
    policy = read(ROOT/POLICY)
    if (policy.get('version') != VERSION or policy.get('allow_direct_submit') is not True
            or policy.get('exporter_sha256') != sha(Path(__file__))):
        raise RuntimeError('Fixed exporter/policy identity changed.')
    ids, vectors, reloads, components, evidence = inspect()
    if evidence != policy['evidence_files'] or components != policy['components']:
        raise RuntimeError('Frozen component evidence changed.')
    out.mkdir(parents=True); report.mkdir(parents=True)
    try:
        # Same float64 summation order as fixed_mean over the original three rows.
        mean = [(a+b+c)/3 for a,b,c in zip(*vectors)]
        reload_mean = [(a+b+c)/3 for a,b,c in zip(*reloads)]
        difference = compare(mean,reload_mean)
        with (out/'test_probabilities.csv').open('w',newline='') as stream:
            w=csv.writer(stream); w.writerow(['PassengerId','probability','reload_probability'])
            w.writerows(zip(ids,mean,reload_mean))
        with (out/'submission.csv').open('w',newline='') as stream:
            w=csv.writer(stream);w.writerow(['PassengerId','Transported']);w.writerows(zip(ids,[v>=.5 for v in mean]))
        saved_difference = compare(mean,vector(out/'test_probabilities.csv',ids))
        if saved_difference != 0: raise RuntimeError('Combined CSV roundtrip must be exact.')
        rows=table(out/'submission.csv')
        if [r['PassengerId'] for r in rows]!=ids or any(r['Transported']!=str(p>=.5) for r,p in zip(rows,mean)):
            raise RuntimeError('Export CSV identity or boolean values differ.')
        digest=sha(out/'submission.csv')
        for path in (ROOT/'reports').rglob('online_attempts/*.json'):
            if read(path).get('file_sha256')==digest:
                raise RuntimeError('Identical submission already attempted.')
        for path,expected in evidence.items():
            if sha(ROOT/path)!=expected: raise RuntimeError('Evidence changed while preparing: '+path)
        selection=dict(version=VERSION,components=[v for v,_ in COMPONENTS],weights=[1/3]*3,
            threshold=.5,stage='explicit_online_first_cached_components',A_eligible=None,S_supported=False)
        manifest=dict(status='complete',version=VERSION,proposal=PROPOSAL,rows=4277,training_rows=8693,
            artifact_kind='cached_probability_ensemble',cached_ensemble_verified=True,
            test_probabilities_sha256=sha(out/'test_probabilities.csv'),saved_probability_roundtrip_max_difference=saved_difference,
            component_evidence=evidence,
            sha256=digest,valid=True,threshold=.5,selection=selection,components=components,
            probabilities_sha256=sha(out/'test_probabilities.csv'),new_fit_performed=False,
            new_model_inference_performed=False,new_combined_model_reload_performed=False,
            component_full_export_reload_verified=True,reload_rows=4277,
            reload_max_absolute_difference=difference,reload_threshold_decisions_equal=True,
            reload_verification_method='Reuse exact complete component model reload evidence; average their complete cached reload probabilities. No new combined neural model/artifact.',
            original_raw_manifest_false_unchanged=True,original_failed_gates_unchanged=True,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            source_hashes={**evidence,Path(__file__).name:sha(Path(__file__)),POLICY:sha(ROOT/POLICY)},policy_path=POLICY,policy_sha256=sha(ROOT/POLICY),
            evidence_files=evidence,source_commit_note='New exporter/policy/results must be committed by root before upload.',
            completed_at_utc=datetime.now(timezone.utc).isoformat())
        for name,obj in [('selection.json',selection),('manifest.json',manifest)]:
            write(out/name,obj);write(report/name,obj)
        print(json.dumps(dict(status='complete',proposal=PROPOSAL,rows=4277,sha256=digest,combined_cached_reload_difference=difference)))
        return manifest
    except BaseException as error:
        for directory in (out,report):
            for name in ('manifest.json','selection.json','submission.csv'):(directory/name).unlink(missing_ok=True)
            write(directory/'failure.json',dict(status='failed_not_upload_eligible',error_type=type(error).__name__,error=str(error)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--freeze-policy',action='store_true')
    args=parser.parse_args()
    freeze_policy() if args.freeze_policy else prepare()
