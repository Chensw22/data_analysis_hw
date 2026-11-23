"""Freeze the adaptive balanced-bill proposal with the verified v131 overrides."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audit_flip_capacity_n23 import correct_count
from broad_geographic_transfer_n23 import BILLS
from exaone_validation_data import compact_features
from export_budget_quarter_n23 import source
from full_roster_adaptive_bill_prior_n23 import EVIDENCE, proposal_design
from full_roster_bill_prior_balance_n23 import predict_priors
from full_roster_booking_validation_n23 import FAMILIES, OUT, ROOT, data
from full_roster_consensus_ranking_n23 import half_bank
from full_roster_metric_adjudication_n23 import load_scores
from kaggle_workflow import sha256, validate_submission
from relational_error_corrector import write

VERSION = 'n23_adaptive_bill_prior'
NAME = 'proposal_'+VERSION
QUEUED = 'proposal_n23_half_roster_disagreement'


def export():
    raw, y, g, fold = data()
    test_path = ROOT/'data/raw/test.csv'
    test = pd.read_csv(test_path, usecols=lambda k: k != 'Name')
    joined = pd.concat([raw,test],ignore_index=True)
    groups = joined.PassengerId.str.split('_').str[0].to_numpy()
    np.testing.assert_array_equal(groups[:len(y)],g)
    art = ROOT/'artifacts'
    out = art/NAME
    out.mkdir(exist_ok=True)
    parents = {}
    mitra, _, parents['v96'] = source('proposal_online_first_counts15',test)
    incumbent, champion, parents['v131'] = source('proposal_v131_party_consensus',test)
    protected = champion != (mitra >= .5)
    assert protected.sum() == 61
    prior_score_path = art/QUEUED/'query_scores.npz'
    reference_manifest = json.loads((ROOT/'reports'/QUEUED/'manifest.json').read_text())
    assert sha256(prior_score_path) == reference_manifest['score_file_sha256']
    assert sha256(art/QUEUED/'submission.csv') == reference_manifest['sha256']
    with np.load(prior_score_path) as z:
        np.testing.assert_array_equal(z['passenger'],test.PassengerId)
        np.testing.assert_array_equal(z['scores'][:,0],mitra)
        ordinary = dict(cat=z['scores'][:,1], budget=z['scores'][:,2], mitra96=mitra)
    for name in ['v44_without_name','tabicl']:
        with np.load(OUT/f'original_{name}.npz') as z:
            np.testing.assert_array_equal(z['test_ids'],test.PassengerId)
            np.testing.assert_array_equal(y[z['row']],z['y'])
            ordinary[name] = z['test']
    base = np.column_stack([ordinary[k] for k in (*FAMILIES,'mitra96')])
    half = half_bank(raw,y,fold)
    h = load_scores(raw,y,fold%2,art/'nov23_bill_prior_balance/half.npz',EVIDENCE)
    use, hx = proposal_design(half,h)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=.1,max_iter=500))
    clf.fit(hx[use], (y != (half[:,-1] >= .5))[use])
    model_file = out/'gate.joblib'
    joblib.dump(clf,model_file)
    x = compact_features(joined)
    cue = ['CryoSleep', *BILLS]
    fields = x.drop(columns=cue)
    t, q = np.arange(len(y)),np.arange(len(y),len(x))
    def score(t,q):
        parts = predict_priors(x,cue,fields,y,groups,t,q)
        return np.column_stack([parts[k] for k in EVIDENCE])
    # Validate the shared training routine on an existing fold before full fitting.
    outer = load_scores(raw,y,fold,art/'nov23_bill_prior_balance/full.npz',EVIDENCE)
    q0 = np.flatnonzero(fold == 0)
    np.testing.assert_array_equal(score(np.flatnonzero(fold != 0),q0),outer[q0])
    evidence = score(t,q)
    replay_evidence = score(t,q)
    np.testing.assert_array_equal(evidence,replay_evidence)
    def decide(model, scores):
        conflict, xx = proposal_design(base,scores)
        eligible = conflict & ~protected
        change = eligible & (model.predict_proba(xx)[:,1] >= .5)
        return np.where(change,scores[:,1],incumbent), change, eligible
    p, change, eligible = decide(clf,evidence)
    repeat, _, _ = decide(joblib.load(model_file),replay_evidence)
    np.testing.assert_array_equal(p,repeat)
    np.testing.assert_array_equal((p >= .5) != champion,change)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all() and change.any()
    csv = out/'submission.csv'
    pd.DataFrame(dict(PassengerId=test.PassengerId,Transported=p >= .5)).to_csv(csv,index=False)
    pd.DataFrame(dict(PassengerId=test.PassengerId,probability=p,reload_probability=repeat)).to_csv(
        out/'test_probabilities.csv',index=False)
    check = validate_submission(csv,test_path)
    distances, bounds = [], []
    for path in (ROOT/'reports').glob('*/online_attempts/*.json'):
        receipt = json.loads(path.read_text())
        assert receipt.get('file_sha256') != check['sha256']
        if receipt.get('evaluation',{}).get('Status') != 'COMPLETE':
            continue
        old = art/path.parents[1].name/'submission.csv'
        assert sha256(old) == receipt['file_sha256']
        table = pd.read_csv(old)
        np.testing.assert_array_equal(table.PassengerId,test.PassengerId)
        distance = int(((p >= .5) != table.Transported).sum())
        distances.append(distance)
        bounds.append(correct_count(str(receipt['evaluation']['Public Score']),len(q))+distance)
    queued = pd.read_csv(art/QUEUED/'submission.csv')
    np.testing.assert_array_equal(queued.PassengerId,test.PassengerId)
    manifest = dict(**check, version=VERSION, model_sha256=sha256(model_file), reload_difference=0, parents=parents,
        training_rows=int(use.sum()), full_roster_rows=len(y), locked_v131=int(protected.sum()), eligible_test=int(eligible.sum()),
        v131_changes=int(change.sum()), changed_to_true=int((~champion & change).sum()), changed_to_false=int((champion & change).sum()),
        different_from_109=int(((p >= .5) != queued.Transported).sum()), receipts_checked=len(bounds),
        closest_submitted_changes=min(distances), public_correct_upper_bound=min(min(bounds),len(q)),
        crossfamily_query_hashes={k:sha256(OUT/f'original_{k}.npz') for k in ['tabicl','v44_without_name']},
        ordinary_query_sha256=sha256(prior_score_path))
    for directory in (out,ROOT/'reports'/NAME):
        write(directory/'manifest.json',manifest)
        write(directory/'selection.json',dict(version=VERSION,threshold=.5,gate='general',folds=8))
    print(manifest)


if __name__ == '__main__':
    export()
