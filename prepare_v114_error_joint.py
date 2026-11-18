"""Prepare v114 unanimous error-specialist vote over immutable v108. No upload."""
import os
if os.environ.get('KMP_DUPLICATE_LIB_OK') not in (None, 'FALSE'):
    raise RuntimeError('Do not enable duplicate OpenMP runtimes')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'FALSE'
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import signal
import subprocess
import time
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold
from threadpoolctl import threadpool_limits
import iterate
import nonlinear_error_corrector as residual
import error_trained_experts as specialists
import mixed_type_forest_imputation as forest
import latent_class_em_probe as latent
import peer_bill_correction as peer
from relational_error_corrector import logit
import kaggle_workflow as workflow

ROOT = Path(__file__).resolve().parent
NAME = 'proposal_v114_error_joint'
VERSION = 'v114_v108_three_expert_unanimous_flip'
OUT = ROOT/'artifacts'/NAME
REPORT = ROOT/'reports'/NAME
BASE = ROOT/'artifacts/proposal_v108_party_bills'
CABIN = ROOT/'artifacts/proposal_nov18_peer_cabin'
TEACHER = ROOT/'artifacts/proposal_ready_nonlinear_residual_v2'
RULE = 'matched_party_and_two_challengers'
SEED = 20261112


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
def check_prob(value, n=4277):
    value = np.asarray(value)
    if value.shape != (n,) or not np.isfinite(value).all() or ((value < 0)|(value > 1)).any():
        raise ValueError('Invalid probability vector')
    return value


def record(evidence, path):
    path = Path(path); evidence[str(path.relative_to(ROOT))] = sha(path)


def authenticate_manifest(folder, evidence):
    public = ROOT/'reports'/folder.name/'manifest.json'
    if public.read_bytes() != (folder/'manifest.json').read_bytes():
        raise RuntimeError('Public/artifact manifest mismatch')
    for path in [public, folder/'manifest.json']: record(evidence, path)
    return read(public)


def load_inputs():
    evidence = {}
    # Full training labels are authorized for final fitting only; no test targets or Name.
    train = pd.read_csv(ROOT/'data/raw/train.csv', usecols=lambda c: c != 'Name')
    test = pd.read_csv(ROOT/'data/raw/test.csv', usecols=lambda c: c != 'Name')
    assert len(train) == 8693 and len(test) == 4277 and 'Transported' not in test
    ids = train.PassengerId.to_numpy(dtype=str); query_ids = test.PassengerId.to_numpy(dtype=str)
    y = train.Transported.to_numpy(int)
    roster = pd.concat([train.drop(columns='Transported'), test], ignore_index=True)
    x, cats = iterate.build_features(roster, False)
    assert list(x.columns) == residual.COLUMNS and cats == residual.CATS and x.shape == (12970, 30)
    peer_ids, known, _, party = peer.context()
    np.testing.assert_array_equal(peer_ids.to_numpy(dtype=str), roster.PassengerId.to_numpy(dtype=str))
    assert party.shape == (12970, 33)
    enriched = pd.concat([x, pd.DataFrame(party, columns=[f'peer_{i}' for i in range(33)])], axis=1)
    teacher = authenticate_manifest(TEACHER, evidence)
    assert teacher['status'] == 'complete'
    for name, digest in teacher['raw_component']['files'].items():
        if sha(TEACHER/name) != digest: raise RuntimeError('Teacher file changed: '+name)
        record(evidence, TEACHER/name)
    # Historical source attestations must still identify the exact OOF producer.
    for name in ['prepare_residual_ready.py', 'mitra_candidate.py', 'iterate.py', 'experiment.py']:
        if sha(ROOT/name) != teacher['source_hashes'][name]: raise RuntimeError('Teacher source changed: '+name)
        record(evidence, ROOT/name)
    with np.load(TEACHER/'raw_probabilities.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(z['train_ids'], np.arange(8693))
        np.testing.assert_array_equal(z['train_passenger_ids'], ids)
        np.testing.assert_array_equal(z['test_ids'], query_ids)
        raw = check_prob(z['oof_raw'].copy(), 8693); folds = z['fold_ids'].copy()
    metadata = read(TEACHER/'raw_metadata.json')
    assert metadata['seed'] == SEED and metadata['recipe'] == 'zero-shot' and metadata['selected_steps'] == 0
    np.testing.assert_array_equal(metadata['passenger_ids'], ids)
    for j, (a, b) in enumerate(StratifiedKFold(3, shuffle=True, random_state=SEED).split(ids, y)):
        np.testing.assert_array_equal(np.flatnonzero(folds == j), b)
        rows = TEACHER/f'raw_oof_fold{j}_rows.npz'
        with np.load(rows, allow_pickle=False) as z:
            np.testing.assert_array_equal(z['train_ids'], a); np.testing.assert_array_equal(z['query_ids'], b)
        info = TEACHER/f'raw_oof_fold{j}.json'; m = read(info)
        assert m['seed'] == SEED+j and m['training_rows'] == len(a) and m['selected_steps'] == 0
        assert m['training_ids_sha256'] == hashlib.sha256(np.asarray(a, dtype=np.int64).tobytes()).hexdigest()
        record(evidence, rows); record(evidence, info)
    base_manifest = authenticate_manifest(BASE, evidence)
    cabin_manifest = authenticate_manifest(CABIN, evidence)
    assert base_manifest['version'] == 'v108_peer_bills_v106' and base_manifest['reload_difference'] == 0
    assert cabin_manifest['version'] == 'v106_v96_peer_cabin' and cabin_manifest['reload_max_difference'] == 0
    assert sha(BASE/'submission.csv') == base_manifest['sha256']
    assert sha(CABIN/'submission.csv') == cabin_manifest['sha256'] == base_manifest['source_csv']
    assert sha(BASE/'peer_coefficients.joblib') == base_manifest['correction_sha256']
    assert base_manifest['teacher_oof_sha256'] == sha(TEACHER/'raw_probabilities.npz')
    source = pd.read_csv(CABIN/'test_probabilities.csv', float_precision='round_trip')
    cached = pd.read_csv(BASE/'test_probabilities.csv', float_precision='round_trip')
    for frame, folder in [(source, CABIN), (cached, BASE)]:
        np.testing.assert_array_equal(frame.PassengerId, query_ids)
        np.testing.assert_array_equal(frame.probability, frame.reload_probability)
        submission = pd.read_csv(folder/'submission.csv')
        np.testing.assert_array_equal(submission.PassengerId, query_ids)
        np.testing.assert_array_equal(frame.probability >= .5, submission.Transported)
        for filename in ['submission.csv', 'test_probabilities.csv']: record(evidence, folder/filename)
    coefficient = joblib.load(BASE/'peer_coefficients.joblib')
    base = source.probability.to_numpy().copy()
    mask = known[8693:]
    base[mask] = expit(logit(base[mask])+peer.apply(coefficient, party[8693:][mask]))
    np.testing.assert_array_equal(base, cached.probability.to_numpy())
    check_prob(base); record(evidence, BASE/'peer_coefficients.joblib')
    for path in [ROOT/'data/raw/train.csv', ROOT/'data/raw/test.csv', ROOT/'reports/champion_error_portfolio/result.json']:
        record(evidence, path)
    return dict(ids=ids, query_ids=query_ids, y=y, x=x, enriched=enriched, cats=cats,
                groups=np.asarray([v.split('_')[0] for v in ids]), raw=raw, folds=folds,
                base=base, coefficient=coefficient, v106=source.probability.to_numpy(),
                eligible=mask, party_query=party[8693:], evidence=evidence)


def predict(bundle, x, enriched, query_ids):
    np.testing.assert_array_equal(query_ids, bundle['query_ids'])
    assert list(x.columns) == bundle['columns'] and list(enriched.columns) == bundle['party_columns']
    members = np.column_stack([m.predict_proba(enriched)[:, 1] for m in bundle['matched_party']])
    party = check_prob(members.mean(axis=1))
    chain = check_prob(bundle['forest_chain'].predict_proba(x)[:, 1])
    em = bundle['supervised_em']
    latent_p = check_prob(em['model'].predict(em['encoder'].transform(x[latent.COLS])))
    base = bundle['v106'].copy(); mask = bundle['peer_eligible']
    base[mask] = expit(logit(base[mask])+peer.apply(bundle['peer_coefficients'], bundle['party_query'][mask]))
    np.testing.assert_array_equal(base, bundle['v108'])
    votes = np.column_stack([party, chain, latent_p]) >= .5
    flip = np.all(votes != (base >= .5)[:, None], axis=1)
    decision = (base >= .5)^flip
    p = base.copy(); p[flip] = decision[flip].astype(float)
    np.testing.assert_array_equal(p[~flip], base[~flip])
    np.testing.assert_array_equal(p >= .5, decision)
    return dict(base=base, matched_members=members, matched_party=party, forest_chain=chain,
                supervised_em=latent_p, votes=votes, flip=flip, decision=decision, probability=p)


def run():
    if OUT.exists() or REPORT.exists(): raise FileExistsError('No overwrite/resume')
    OUT.mkdir(parents=True); REPORT.mkdir(parents=True)
    start = time.monotonic()
    try:
        inputs = load_inputs()
        sources = ['prepare_v114_error_joint.py', 'error_trained_experts.py', 'teacher_distillation.py',
                   'mixed_type_forest_imputation.py', 'latent_class_em_probe.py', 'peer_bill_correction.py',
                   'export_party_bill.py', 'iterate.py', 'experiment.py', 'nonlinear_error_corrector.py',
                   'relational_error_corrector.py', 'kaggle_workflow.py', 'submit_committed_export.py']
        evidence = inputs['evidence']
        for name in sources: record(evidence, ROOT/name)
        prior = read(ROOT/'reports/champion_error_portfolio/result.json')['joint'][RULE]
        assert (prior['repairs'], prior['breaks'], prior['hard546_new']) == (62, 56, 2)
        protocol = dict(version=VERSION, status='fixed_before_fit', base='v108 exact cached v106 plus saved peer offset',
            rule=RULE, threshold=.5, full_training_rows=8693, test_rows=4277, no_Name=True,
            matched_party=dict(packets='authenticated v101 full raw OOF SKF3 seed20261112; do not resplit packets',
                discovery='GroupShuffleSplit(test_size=.33), seed20261118+packet; reserve other groups',
                controls='Original select_correct_controls within discovery; matched rawclass×planet×sleep',
                model='Original Cat180/depth3/lr.05/L2=10; seed=20261118+packet+3; mean three probabilities'),
            forest_chain=dict(seed=SEED, rounds=2, forest=forest.FOREST, classifier=forest.CATPARAMS),
            supervised_em=dict(seed=SEED, K=latent.K, iterations=latent.ITERATIONS, original_encoder=True),
            vote='Flip only if all three >=.5 classes oppose base. Member averaging occurs BEFORE matched_party vote.',
            probability_semantics='Unchanged rows retain v108 probability; changed rows encode unanimous decision as0/1, NOT calibrated confidence.',
            scope='Final fitting on all labeled training rows authorized. No test targets, no new CV/accuracy, no upload.',
            prior_development_result=prior, limitations=['Adaptive rule selected on reused6954dev; +6 is not stable superiority.',
                'Full teacher is3fold OOF, not historical nested development packets; full-fit extrapolation disclosed.',
                'Peer features use target-free full roster; no train/test group target propagation.',
                'Fixed-roster composite reuses authenticated v108 cached base; it is not an arbitrary-new-row predictor.'],
            evidence=evidence, environment={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','catboost','joblib']},
            recipe_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
        write(REPORT/'protocol.json', protocol); write(OUT/'protocol.json', protocol)
        x=inputs['x']; enriched=inputs['enriched']; y=inputs['y']; groups=inputs['groups']
        members=[]; discovery_records=[]
        with threadpool_limits(limits=4):
            for inner in range(3):
                packet=np.flatnonzero(inputs['folds']==inner); seed=20261118+inner
                dlocal,clocal=next(GroupShuffleSplit(n_splits=1,test_size=.33,random_state=seed).split(packet,groups=groups[packet]))
                d,c=packet[dlocal],packet[clocal]
                assert not set(groups[d]).intersection(groups[c])
                selected=specialists.select_correct_controls(d,inputs['raw'][d],y[d],enriched,seed)
                rows=d[selected]
                if len(np.unique(y[rows]))!=2: raise RuntimeError('Both training classes required')
                model=CatBoostClassifier(iterations=180,depth=3,learning_rate=.05,l2_leaf_reg=10,
                    random_seed=seed+3,thread_count=2,verbose=False,allow_writing_files=False,cat_features=inputs['cats'])
                model.fit(enriched.iloc[rows], y[rows]); members.append(model)
                np.savez_compressed(OUT/f'matched_packet{inner}.npz',packet=packet,discovery=d,reserved=c,selected=rows)
                rec=dict(packet=inner,rows=len(packet),discovery=len(d),reserved=len(c),selected=len(rows),
                    selected_errors=int(((inputs['raw'][rows]>=.5)!=y[rows]).sum()),
                    split_seed=seed,model_seed=seed+3,group_overlap=0,selected_class_counts=np.bincount(y[rows],minlength=2).tolist())
                discovery_records.append(rec); print(json.dumps(rec),flush=True)
            del model
            print('Fitting original full-data forest chain',flush=True)
            chain=forest.ForestChain(); filled=chain.fit_transform(forest.semantic(x.iloc[:8693]),SEED)
            completed=forest.repaired_features(x.iloc[:8693],filled)
            tree=CatBoostClassifier(**forest.CATPARAMS,cat_features=inputs['cats'],random_seed=SEED)
            tree.fit(completed,y); forest_model=forest.Predictor(tree,chain)
            print('Fitting original supervised EM',flush=True)
            enc=latent.Encoder().fit(x.iloc[:8693][latent.COLS]); z=enc.transform(x.iloc[:8693][latent.COLS])
            em=latent.LatentClasses(enc.cards).fit(z,y,latent.initialize(z,enc.cards,SEED),supervised=True)
            bundle=dict(version=VERSION,kind='fixed_roster_CPU_composite',matched_party=members,forest_chain=forest_model,
                supervised_em=dict(encoder=enc,model=em),columns=list(x.columns),party_columns=list(enriched.columns),
                query_ids=inputs['query_ids'],v108=inputs['base'],v106=inputs['v106'],peer_coefficients=inputs['coefficient'],
                peer_eligible=inputs['eligible'],party_query=inputs['party_query'],source_evidence=evidence)
            p=predict(bundle,x.iloc[8693:],enriched.iloc[8693:],inputs['query_ids'])
            joblib.dump(bundle,OUT/'model.joblib')
            # No live references to any original trained estimator survive this boundary.
            del bundle,members,forest_model,tree,chain,em,enc,filled,completed,z
            gc.collect()
            restored=joblib.load(OUT/'model.joblib')
            q=predict(restored,x.iloc[8693:],enriched.iloc[8693:],inputs['query_ids'])
            for key in p: np.testing.assert_array_equal(p[key],q[key],err_msg=key)
            np.savez_compressed(OUT/'probabilities.npz',PassengerId=inputs['query_ids'],**p)
            np.savez_compressed(OUT/'reload_probabilities.npz',PassengerId=inputs['query_ids'],**q)
        opinions=pd.DataFrame(dict(PassengerId=inputs['query_ids'],v108=p['base'],matched_party=p['matched_party'],
            forest_chain=p['forest_chain'],supervised_em=p['supervised_em'],flip=p['flip'],Transported=p['decision']))
        for j in range(3):
            opinions[f'matched_member{j}']=p['matched_members'][:,j]
            opinions[f'expert{j}_vote']=p['votes'][:,j]
        opinions.to_csv(OUT/'expert_opinions.csv',index=False)
        pd.DataFrame(dict(PassengerId=inputs['query_ids'],probability=p['probability'],reload_probability=q['probability'])).to_csv(OUT/'test_probabilities.csv',index=False)
        pd.DataFrame(dict(PassengerId=inputs['query_ids'],Transported=p['decision'])).to_csv(OUT/'submission.csv',index=False)
        check=workflow.validate_submission(OUT/'submission.csv',ROOT/'data/raw/test.csv')
        for attempt in (ROOT/'reports').glob('*/online_attempts/*.json'):
            if read(attempt).get('file_sha256')==check['sha256']: raise RuntimeError('Previously submitted identical CSV')
        csv=pd.read_csv(OUT/'submission.csv'); probs=pd.read_csv(OUT/'test_probabilities.csv')
        assert probs.PassengerId.equals(csv.PassengerId) and probs.probability.equals(probs.reload_probability)
        assert (probs.probability>=.5).equals(csv.Transported)
        for name,digest in evidence.items():
            if sha(ROOT/name)!=digest: raise RuntimeError('Frozen evidence changed: '+name)
        assert os.environ['KMP_DUPLICATE_LIB_OK']=='FALSE'
        files={f.name:sha(f) for f in OUT.iterdir() if f.is_file()}
        manifest=dict(**check,status='complete_ready_not_uploaded',version=VERSION,model_sha256=files['model.joblib'],
            reload_difference=0,all_three_expert_reload_bitwise=True,base4277_reconstructed_bitwise=True,
            changes=int(p['flip'].sum()),false_to_true=int((p['flip']&p['decision']).sum()),
            true_to_false=int((p['flip']&~p['decision']).sum()),unchanged=int((~p['flip']).sum()),
            expert_disagreements_with_v108={key:int(((p[key]>=.5)!=(p['base']>=.5)).sum()) for key in ['matched_party','forest_chain','supervised_em']},
            artifact_kind='fixed_roster_CPU_composite',component_files={'model.joblib':files['model.joblib']},
            files=files,source_hashes={n:evidence[n] for n in sources},evidence=evidence,
            matched_packets=discovery_records,seconds=time.monotonic()-start,
            probability_semantics=protocol['probability_semantics'],prior_development_result=prior,no_upload=True,
            protocol_sha256=sha(REPORT/'protocol.json'))
        for folder in [OUT,REPORT]:
            write(folder/'manifest.json',manifest);write(folder/'selection.json',dict(version=VERSION,threshold=.5,rule=RULE))
        print(json.dumps({k:manifest[k] for k in ['status','changes','false_to_true','true_to_false','reload_difference','seconds','sha256']}),flush=True)
    except BaseException as exc:
        for folder in [OUT,REPORT]:
            for name in ['manifest.json','selection.json']:(folder/name).unlink(missing_ok=True)
            write(folder/'failure.json',dict(type=type(exc).__name__,error=str(exc),seconds=time.monotonic()-start))
        raise


if __name__=='__main__':
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Fixed CPU export1800s')))
    signal.setitimer(signal.ITIMER_REAL,1800)
    try: run()
    finally: signal.setitimer(signal.ITIMER_REAL,0)
