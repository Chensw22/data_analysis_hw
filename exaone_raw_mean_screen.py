"""One committed, fixed 50/50 EXAONE/rawMitra cached-OOF experiment in two partitions."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

import exaone_validation as validation

data=validation.data
core=validation.core
ROOT=Path(__file__).resolve().parent
NAME='EXAONE_v92_rawMitra_half'
RAW='v71_Mitra2_zeroshot'
PLAN='reports/exaone_raw_mean_review.json'
PLAN_SHA='8b8dd68c29515fbc5061f06872f763eb6eb35ffe1464a89b5934614ea7c1b271'
OWN=['exaone_raw_mean_screen.py','test_exaone_raw_mean_screen.py',PLAN]
RUNS={'random':'run_055','group':'exaone_mean_group_20261117'}
DISCLOSURE='Adaptively selected after seeing single-model results; reused development folds, not independent validation. No old holdout targets.'


def fixed_mean(exaone,raw):
    a,b=np.asarray(exaone,dtype=np.float64),np.asarray(raw,dtype=np.float64)
    if (a.ndim!=1 or b.shape!=a.shape or len(a)==0 or not np.isfinite([a,b]).all()
            or np.any((a<0)|(a>1)) or np.any((b<0)|(b>1))):
        raise ValueError('Exactly two complete aligned probability vectors required.')
    return .5*a+.5*b


def compare_metrics(actual,recorded):
    for key in ['accuracy','log_loss','fold_accuracy','fold_log_loss']:
        a,b=np.asarray(actual[key]),np.asarray(recorded[key])
        if a.shape!=b.shape or not np.isfinite(b).all() or not np.allclose(a,b,rtol=0,atol=1e-12):
            raise RuntimeError('Authenticated single-model metrics differ: '+key)


def expected_fold_ids(dev,splits):
    result=np.full(len(dev),-1,dtype=np.int64)
    for fold,(a,b) in enumerate(splits):
        if (len(np.unique(b))!=len(b) or np.any(result[b]>=0)
                or not np.array_equal(np.sort(np.r_[a,b]),np.arange(len(dev)))):
            raise RuntimeError('Partitions are not complete disjoint folds.')
        result[b]=fold
    if len(splits)!=3 or np.any(result<0):raise RuntimeError('Three complete folds required.')
    return result


def aligned_cache(cache,inputs,splits):
    for key,expected in [('dev_ids',inputs['dev']),('fold_id',expected_fold_ids(inputs['dev'],splits)),
            ('passenger_ids',inputs['passengers']),('group_ids',inputs['groups'])]:
        if not np.array_equal(cache[key],expected):raise RuntimeError('EXAONE saved OOF alignment differs: '+key)
    return cache['prob'].copy(),cache['fold_accuracy'].copy()


def authenticate_exaone(phase,inputs,splits,raw,evidence):
    directory=validation.DIRECTORIES[phase];folder=ROOT/'artifacts'/directory
    csv='screen_results.csv' if phase=='random' else 'group_metrics.csv'
    for name in ['summary.json','protocol.json','assessment.json','environment.lock.txt',csv]:
        data.report_pair(directory,name,evidence)
    summary=json.loads((folder/'summary.json').read_text());protocol=json.loads((folder/'protocol.json').read_text())
    if (summary.get('status')!='complete' or summary.get('completed_folds')!=3 or summary.get('candidate')!=core.NAME
            or summary.get('phase')!=phase or summary.get('no_old_holdout_targets_used') is not True
            or protocol.get('phase')!=phase or protocol.get('candidate')!=core.NAME
            or protocol.get('plan_sha256')!=validation.PLAN_SHA or protocol.get('settings')!=core.SETTINGS
            or [protocol.get(k) for k in ['seed','folds','development_rows','threshold','ensemble_seed_every_fold']]!=[20261112,3,6954,.5,0]):
        raise RuntimeError('Complete original EXAONE screen required.')
    validation.assert_frozen(protocol['frozen'])
    for key in ['sources','evidence']:evidence.update(protocol['frozen'][key])
    evidence.update(protocol['frozen']['resource']['sources'])
    cache_name=f'{core.NAME}_seed20261112.npz'
    expected={cache_name,csv,'assessment.json',*[f'fold{k}.json' for k in range(3)],*[f'fold{k}_rows.npz' for k in range(3)]}
    if set(summary['result_files_sha256'])!=expected:raise RuntimeError('Complete EXAONE result manifest required.')
    for name,digest in summary['result_files_sha256'].items():
        path=folder/name
        if core.sha(path)!=digest:raise RuntimeError('EXAONE result artifact changed: '+name)
        data.remember(path,evidence)
    dev,y=inputs['dev'],inputs['y']
    for key,expected_value in [('development_ids_sha256',data.audit.row_hash(dev)),
            ('development_labels_sha256',core.array_hash(y)),('passenger_ids_sha256',data.audit.string_hash(inputs['passengers'])),
            ('group_ids_sha256',data.audit.string_hash(inputs['groups']))]:
        if protocol.get(key)!=expected_value:raise RuntimeError('EXAONE original roster differs: '+key)
    # Existing self-generated v91 passenger IDs use object dtype. Deserialize only
    # after the exact committed summary digest above authenticates this local file.
    with np.load(folder/cache_name,allow_pickle=True) as cache:
        p,stored=aligned_cache(cache,inputs,splits)
    actual=data.metrics(p,y,splits);compare_metrics(actual,summary['metrics'])
    data.common.cached_fold_scores(p,stored,y,splits)
    compare_metrics(data.metrics(raw,y,splits),summary['references'][RAW])
    table=pd.read_csv(folder/csv);row=table[table.version==core.NAME]
    if (len(row)!=1 or abs(float(row.iloc[0].accuracy)-actual['accuracy'])>1e-12
            or abs(float(row.iloc[0].log_loss)-actual['log_loss'])>1e-12):raise RuntimeError('EXAONE CSV differs from cache.')
    for fold,(a,b) in enumerate(splits):
        log=json.loads(data.report_pair(directory,f'fold{fold}.json',evidence).read_text())
        with np.load(folder/f'fold{fold}_rows.npz',allow_pickle=False) as rows:
            if not np.array_equal(rows['train_ids'],dev[a]) or not np.array_equal(rows['validation_ids'],dev[b]):
                raise RuntimeError('EXAONE original fold row arrays differ.')
        expected_membership=dict(train_ids_sha256=data.audit.row_hash(dev[a]),validation_ids_sha256=data.audit.row_hash(dev[b]))
        if protocol['fold_memberships'][fold]!=expected_membership or any(log.get(k)!=v for k,v in expected_membership.items()):
            raise RuntimeError('EXAONE protocol/log fold hashes differ.')
        if (log.get('status')!='complete' or log.get('fold')!=fold or log.get('ensemble_seed')!=0
                or log.get('split_seed')!=20261112 or log.get('train_rows')!=len(a) or log.get('validation_rows')!=len(b)
                or abs(log['accuracy']-actual['fold_accuracy'][fold])>1e-12
                or abs(log['log_loss']-actual['fold_log_loss'][fold])>1e-12
                or (phase=='group' and log.get('group_overlap_count')!=0)):
            raise RuntimeError('EXAONE fold completion/metrics differ.')
        metadata=log['metadata'];last=metadata['last_prediction']
        if (metadata.get('candidate')!=core.NAME or metadata.get('settings')!=core.SETTINGS
                or metadata.get('checkpoint_sha256')!=core.WEIGHT_SHA or metadata.get('training_rows')!=len(a)
                or metadata.get('training_ids_sha256')!=core.array_hash(dev[a]) or metadata.get('labels_sha256')!=core.array_hash(y[a].astype(np.int64))
                or metadata['preprocessing'].get('support_rows')!=len(a) or metadata['preprocessing'].get('views')!=8
                or last.get('full_support_rows')!=len(a) or last.get('query_rows')!=len(b) or last.get('views')!=8):
            raise RuntimeError('EXAONE actual context recipe differs.')
        validation.coverage(last['forward_records'],len(a),len(b))
    return p,actual


def prepare_inputs():
    # Read-only authentication; this function never computes a mean.
    frozen,lock=validation.resource_gate();inputs=data.load_data();frozen['evidence'].update(inputs['evidence'])
    phases={}
    for phase,reader in [('random',data.random_references),('group',data.group_references)]:
        splits,references,evidence=reader(inputs);raw=references[RAW]
        exaone,exaone_metrics=authenticate_exaone(phase,inputs,splits,raw,evidence)
        frozen['evidence'].update(evidence)
        phases[phase]=dict(splits=splits,raw=raw,exaone=exaone,raw_metrics=data.metrics(raw,inputs['y'],splits),exaone_metrics=exaone_metrics)
    data.common.require_committed([p for p in frozen['evidence'] if p.startswith('reports/')])
    validation.assert_frozen(frozen)
    return inputs,phases,frozen,lock


def gate(candidate,raw):
    unchanged=np.asarray(candidate['fold_accuracy'])+1e-12>=np.asarray(raw['fold_accuracy'])
    if unchanged.shape!=(3,):raise RuntimeError('Three-fold metrics required.')
    checks=dict(pooled_accuracy=candidate['accuracy']+1e-12>=raw['accuracy'],
        pooled_log_loss=candidate['log_loss']<=raw['log_loss']+1e-12,at_least_two_nonlower_folds=int(unchanged.sum())>=2)
    return dict(passed=all(checks.values()),checks=checks,nonlower_folds=int(unchanged.sum()),tolerance=1e-12)


def evaluate_phase(inputs,component):
    probability=fixed_mean(component['exaone'],component['raw'])
    metrics=data.metrics(probability,inputs['y'],component['splits'])
    result=dict(candidate=NAME,completed_folds=3,metrics=metrics,
        components={core.NAME:component['exaone_metrics'],RAW:component['raw_metrics']},
        gate=gate(metrics,component['raw_metrics']),
        paired={key:validation.paired(probability,values,inputs['y'],component['splits'])
                for key,values in [(RAW,component['raw']),(core.NAME,component['exaone'])]},
        disclosure=DISCLOSURE,no_old_holdout_targets_used=True,S_supported=False,automatic_upload=False)
    return probability,result


def assert_own(sources):
    for name,digest in sources.items():
        if core.sha(ROOT/name)!=digest:raise RuntimeError('Mean plan/source changed: '+name)


def execute(inputs,phases,frozen,lock,sources,commit):
    folders={phase:(ROOT/'artifacts'/run,ROOT/'reports'/run) for phase,run in RUNS.items()}
    if any(p.exists() for pair in folders.values() for p in pair):raise RuntimeError('Mean destination exists; no overwrite or implicit resume.')
    protocol=dict(candidate=NAME,seed=20261112,folds=3,development_rows=6954,threshold=.5,
        weights={core.NAME:.5,RAW:.5},source_commit=commit,plan_sha256=PLAN_SHA,sources=sources,frozen=frozen,
        development_ids_sha256=data.audit.row_hash(inputs['dev']),development_labels_sha256=core.array_hash(inputs['y']),
        disclosure=DISCLOSURE,old_holdout_labels_read=False,model_training=False,automatic_upload=False)
    results={};probabilities={};started=time.monotonic()
    def write(phase,name,value):
        for folder in folders[phase]:validation.resource.write(folder/name,value)
    try:
        for phase,pair in folders.items():
            for folder in pair:folder.mkdir(parents=True)
            write(phase,'protocol.json',dict(protocol,phase=phase,fold_memberships=[dict(
                train_ids_sha256=data.audit.row_hash(inputs['dev'][a]),validation_ids_sha256=data.audit.row_hash(inputs['dev'][b]))
                for a,b in phases[phase]['splits']]))
            for folder in pair:(folder/'environment.lock.txt').write_text(lock)
            for name in sources:
                target=pair[0]/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        for phase in ['random','group']:
            assert_own(sources);validation.assert_frozen(frozen)
            probabilities[phase],results[phase]=evaluate_phase(inputs,phases[phase])
            write(phase,'partial_assessment.json',dict(results[phase],status='incomplete_pair_not_for_tiers'))
        passed=all(results[phase]['gate']['passed'] for phase in ['random','group'])
        assert_own(sources);validation.assert_frozen(frozen)
        for phase,(out,report) in folders.items():
            score=results[phase]['metrics'];cache=out/f'{NAME}_seed20261112.npz'
            np.savez(cache,prob=probabilities[phase],fold_accuracy=score['fold_accuracy'],dev_ids=inputs['dev'],
                fold_id=expected_fold_ids(inputs['dev'],phases[phase]['splits']),
                passenger_ids=np.asarray(inputs['passengers'],dtype=str),group_ids=np.asarray(inputs['groups'],dtype=str))
            csv='screen_results.csv' if phase=='random' else 'group_metrics.csv'
            frame=pd.DataFrame([dict(version=NAME,accuracy=score['accuracy'],log_loss=score['log_loss'],
                fold_std=float(np.std(score['fold_accuracy'])),seconds=time.monotonic()-started)])
            for folder in [out,report]:frame.to_csv(folder/csv,index=False)
            assessment=dict(status='complete',candidate=NAME,both_phases_complete=True,passed=passed,
                phases={k:r['gate'] for k,r in results.items()},S_supported=False,automatic_upload=False,
                next_action='separate_root_review_of_one_mean_export' if passed else 'archive_no_weight_search_no_solo_fallback')
            write(phase,'assessment.json',assessment)
            results[phase].update(status='complete',phase=phase,both_phases_complete=True,continue_to_root_decision=passed,
                seconds=time.monotonic()-started,result_files_sha256={p.name:core.sha(p) for p in [cache,out/csv,out/'assessment.json']})
            write(phase,'summary.json',results[phase])
        assert_own(sources);validation.assert_frozen(frozen)
        for phase,result in results.items():
            print(json.dumps(dict(phase=phase,accuracy=result['metrics']['accuracy'],log_loss=result['metrics']['log_loss'],
                fold_accuracy=result['metrics']['fold_accuracy'],gate=result['gate'],both_phases_passed=passed)),flush=True)
        return results
    except BaseException as error:
        for phase,pair in folders.items():
            for folder in pair:
                if folder.exists():
                    for name in [f'{NAME}_seed20261112.npz','screen_results.csv','group_metrics.csv','assessment.json','summary.json']:
                        (folder/name).unlink(missing_ok=True)
                    validation.resource.write(folder/'summary.json',dict(status='failed_inconclusive',phase=phase,candidate=NAME,
                        error_type=type(error).__name__,error=str(error),partial_results_not_for_tiers=True,
                        both_phases_complete=False,S_supported=False,automatic_upload=False))
        raise


def run(session_file):
    validation.resource.check_quota(session_file)
    if core.sha(ROOT/PLAN)!=PLAN_SHA:raise RuntimeError('Fixed mean prospective plan changed.')
    commit=data.common.require_committed(OWN)
    if any((ROOT/k/run).exists() for run in RUNS.values() for k in ['artifacts','reports']):
        raise RuntimeError('Mean destination exists; no overwrite or implicit resume.')
    sources={name:core.sha(ROOT/name) for name in OWN}
    inputs,phases,frozen,lock=prepare_inputs()
    # No averaging occurs before both component phases and all committed sources are authenticated.
    assert_own(sources);validation.assert_frozen(frozen)
    return execute(inputs,phases,frozen,lock,sources,commit)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--session-file',type=Path,required=True)
    args=parser.parse_args();run(args.session_file)
