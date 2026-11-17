"""Fixed half mean of adapted v81 and v44 trees; cached predictions only."""
import csv
from datetime import datetime, timezone
from pathlib import Path
import struct
import subprocess

import prepare_v84_online_first as common

ROOT = Path(__file__).resolve().parent
VERSION = 'v97_v81_v44_equal2_online_first'
PROPOSAL = 'proposal_online_first_adapt_tree_half'
POLICY = 'reports/online_iteration_20261117/online_first_adapt_tree_policy.json'
ADAPT = 'artifacts/proposal_nov16_full_episode'
TREE = 'artifacts/proposal_batch_v44'
RELOAD = 'artifacts/proposal_crossfamily_v80/v44'


def prepare():
    out, report = ROOT/'artifacts'/PROPOSAL, ROOT/'reports'/PROPOSAL
    if out.exists() or report.exists() or (ROOT/POLICY).exists():
        raise RuntimeError('Existing export is immutable.')
    ids = [r['PassengerId'] for r in common.table(ROOT/'data/raw/test.csv')]
    a = common.pair('proposal_nov16_full_episode', 'manifest.json')
    t = common.pair('proposal_batch_v44', 'manifest.json')
    old = common.pair('proposal_crossfamily_v80', 'manifest.json')
    reload = common.read(ROOT/RELOAD/'complete.json')
    common.reload_check(a)
    if (a['selection']['version'] != 'v81_Mitra_full_episode50'
            or t['selection']['version'] != 'v44_seed_ensemble'
            or reload != old['components']['v44']
            or reload['full_reload_verified'] is not True
            or reload['max_difference'] != 0 or reload['full_fit_rows'] != 8693
            or reload['rows'] != 4277
            or common.sha(ROOT/RELOAD/'probabilities.csv') != reload['probabilities_sha256']
            or common.sha(ROOT/TREE/'manifest.json') != reload['source_manifest_sha256']
            or t['model_sha256'] != reload['source_model_sha256']):
        raise RuntimeError('Complete original tree reload evidence required.')
    sources = {}
    for directory, manifest in [(ADAPT, a), (TREE, t)]:
        if (manifest['fit_rows'] != 8693 or manifest['rows'] != 4277
                or common.sha(ROOT/directory/'model.joblib') != manifest['model_sha256']):
            raise RuntimeError('Full component identity changed.')
        for name in ['manifest.json', 'model.joblib', 'test_probabilities.csv', 'submission.csv']:
            path = directory+'/'+name
            sources[path] = common.sha(ROOT/path)
    for path in [RELOAD+'/complete.json', RELOAD+'/probabilities.csv',
                 'reports/proposal_crossfamily_v80/manifest.json',
                 'reports/proposal_nov16_full_episode/manifest.json',
                 'reports/proposal_batch_v44/manifest.json',
                 'prepare_v84_online_first.py', Path(__file__).name, 'data/raw/test.csv']:
        sources[path] = common.sha(ROOT/path)
    f32 = lambda x: struct.unpack('f', struct.pack('f', x))[0]
    p = [f32(x) for x in common.vector(ROOT/ADAPT/'test_probabilities.csv', ids)]
    r = [f32(x) for x in common.vector(ROOT/ADAPT/'test_probabilities.csv', ids, 'reload_probability')]
    q = common.vector(ROOT/TREE/'test_probabilities.csv', ids)
    s = common.vector(ROOT/RELOAD/'probabilities.csv', ids)
    common.compare(p, r); common.compare(q, s)
    mean = [(x+y)/2 for x, y in zip(p, q)]
    reload_mean = [(x+y)/2 for x, y in zip(r, s)]
    difference = common.compare(mean, reload_mean)
    policy = dict(competition='spaceship-titanic', batch_id='online_iteration_20261117',
        allow_full_export=True, allow_direct_submit=True, versions=[VERSION],
        maximum_new_upload_attempts_current_batch=9, independent_additional_budget=False,
        maximum_upload_attempts_for_this_file=1, local_cv_or_group_gate_required=False,
        weights=[.5,.5], threshold=.5, components=['v81_Mitra_full_episode50','v44_seed_ensemble'],
        reason='User requested continued direct online iteration. Earlier v80 used raw v71; this uses adapted v81.',
        no_weight_search=True, evidence_files=sources)
    common.write(ROOT/POLICY, policy)
    sources[POLICY] = common.sha(ROOT/POLICY)
    out.mkdir(parents=True); report.mkdir(parents=True)
    with (out/'test_probabilities.csv').open('w', newline='') as stream:
        w = csv.writer(stream); w.writerow(['PassengerId','probability','reload_probability'])
        w.writerows(zip(ids, mean, reload_mean))
    with (out/'submission.csv').open('w', newline='') as stream:
        w = csv.writer(stream); w.writerow(['PassengerId','Transported'])
        w.writerows(zip(ids, [v >= .5 for v in mean]))
    if common.compare(mean, common.vector(out/'test_probabilities.csv', ids)) != 0:
        raise RuntimeError('CSV roundtrip changed probabilities.')
    digest = common.sha(out/'submission.csv')
    if any(common.read(path).get('file_sha256') == digest
           for path in (ROOT/'reports').rglob('online_attempts/*.json')):
        raise RuntimeError('Duplicate submission.')
    selection = dict(version=VERSION, components=policy['components'], weights=[.5,.5],
                     threshold=.5, stage='explicit_online_first_cached_components', S_supported=False)
    manifest = dict(status='complete', version=VERSION, proposal=PROPOSAL, rows=4277,
        training_rows=8693, sha256=digest, selection=selection,
        artifact_kind='cached_probability_ensemble', cached_ensemble_verified=True,
        test_probabilities_sha256=common.sha(out/'test_probabilities.csv'),
        component_full_export_reload_verified=True, new_model_inference_performed=False,
        new_combined_model_reload_performed=False, reload_rows=4277,
        reload_max_absolute_difference=difference, reload_threshold_decisions_equal=True,
        source_hashes=sources, policy_path=POLICY, policy_sha256=sources[POLICY],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_commit_note='New exporter and result must be committed before upload.',
        completed_at_utc=datetime.now(timezone.utc).isoformat())
    for name, value in [('manifest.json',manifest),('selection.json',selection)]:
        common.write(out/name,value); common.write(report/name,value)
    common.write(report/'authorization.json',dict(batch_id=policy['batch_id'], proposal=PROPOSAL,
        version=VERSION,file_sha256=digest,policy_sha256=sources[POLICY],maximum_upload_attempts_for_this_file=1))
    print(common.json.dumps(dict(proposal=PROPOSAL,sha256=digest,reload_difference=difference)))


if __name__ == '__main__':
    prepare()
