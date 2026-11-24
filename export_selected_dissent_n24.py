"""Export the nine-vote selected-tree correction evaluated on held-out bookings."""
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import BILLS
from exaone_validation_data import compact_features
from export_adaptive_bill_prior_n23 import EVIDENCE, FAMILIES, NAME as PRIOR, OUT, QUEUED, ROOT, data, predict_priors
from export_budget_quarter_n23 import source
from full_roster_permutation_selection_n24 import ensemble_probability, feature_bank, fit_ensemble, select
from kaggle_workflow import sha256, validate_submission
from relational_error_corrector import write

NAME = 'proposal_n24_selected_dissent'


def nine_votes(train, test, y, groups, mitra):
    manifest = json.loads((ROOT/'reports'/PRIOR/'manifest.json').read_text())
    path = ROOT/'artifacts'/QUEUED/'query_scores.npz'
    assert sha256(path) == manifest['ordinary_query_sha256']
    with np.load(path) as z:
        np.testing.assert_array_equal(z['passenger'], test.PassengerId)
        np.testing.assert_array_equal(z['scores'][:,0], mitra)
        ordinary = dict(cat=z['scores'][:,1], budget=z['scores'][:,2], mitra96=mitra)
    for key in ('tabicl','v44_without_name'):
        path = OUT/f'original_{key}.npz'
        assert sha256(path) == manifest['crossfamily_query_hashes'][key]
        with np.load(path) as z:
            np.testing.assert_array_equal(z['test_ids'], test.PassengerId)
            ordinary[key] = z['test']
    joined = pd.concat([train,test], ignore_index=True)
    roster = joined.PassengerId.str.split('_').str[0].to_numpy()
    np.testing.assert_array_equal(groups, roster[:len(y)])
    assert not np.isin(groups, roster[len(y):]).any()
    x = compact_features(joined)
    cue = ['CryoSleep',*BILLS]
    priors = predict_priors(x, cue, x.drop(columns=cue), y, roster, np.arange(len(y)), np.arange(len(y),len(x)))
    votes = np.column_stack([ordinary[k] for k in (*FAMILIES,'mitra96')] + [priors[k] for k in EVIDENCE])
    assert votes.shape == (len(test),9) and np.isfinite(votes).all()
    return votes


def export():
    train, y, groups, _ = data()
    test_file = ROOT/'data/raw/test.csv'
    test = pd.read_csv(test_file, usecols=lambda col: col != 'Name')
    mitra, _, _ = source('proposal_online_first_counts15',test)
    incumbent, champion, parent = source('proposal_v131_party_consensus',test)
    votes = nine_votes(train,test,y,groups,mitra)
    eligible = ((votes >= .5) == champion[:,None]).all(axis=1)
    x, fields, core = feature_bank(train)
    columns = select(x,y,np.arange(len(y)),groups,core,497)[1]
    models = fit_ensemble(x[:,columns],y,497)
    chosen = [fields[i] for i in columns]
    test_x, test_fields, _ = feature_bank(test)
    query = pd.DataFrame(test_x,columns=test_fields).reindex(columns=chosen,fill_value=0).to_numpy()
    out = ROOT/'artifacts'/NAME
    out.mkdir(exist_ok=True)
    model_file = out/'ensemble.joblib'
    joblib.dump(dict(fields=chosen,models=models),model_file)
    restored = joblib.load(model_file)
    assert restored['fields'] == chosen
    challenger = ensemble_probability(models,query)
    reloaded = ensemble_probability(restored['models'],query)
    np.testing.assert_array_equal(challenger,reloaded)
    assert np.isfinite(challenger).all() and ((challenger >= 0) & (challenger <= 1)).all()
    pred = np.where(eligible,challenger,incumbent)
    hard = pred >= .5
    changed = hard != champion
    assert not changed[~eligible].any()
    csv = out/'submission.csv'
    pd.DataFrame(dict(PassengerId=test.PassengerId,Transported=hard)).to_csv(csv,index=False)
    check = validate_submission(csv,test_file)
    distances = []
    for receipt in (ROOT/'reports').glob('*/online_attempts/*.json'):
        record = json.loads(receipt.read_text())
        old = ROOT/'artifacts'/receipt.parents[1].name/'submission.csv'
        if record.get('evaluation',{}).get('Status') == 'COMPLETE':
            assert sha256(old) == record['file_sha256']
            df = pd.read_csv(old)
            np.testing.assert_array_equal(df.PassengerId,test.PassengerId)
            distances.append(int(np.count_nonzero(hard != df.Transported)))
    assert distances
    report = dict(**check, version=NAME.removeprefix('proposal_'), source_v131=parent, features=chosen, model_sha256=sha256(model_file),
                  seed=497, train_rows=len(y), eligible=int(eligible.sum()), reload_difference=0,
                  changed_vs_v131=int(changed.sum()), changed_to_true=int((changed & hard).sum()),
                  changed_to_false=int((changed & ~hard).sum()), closest_evaluated_changes=min(distances),
                  completed_receipts=len(distances), zero_filled_fields=[c for c in chosen if c not in test_fields])
    pd.DataFrame(dict(PassengerId=test.PassengerId, probability=pred,
                      reload_probability=np.where(eligible,reloaded,incumbent))
                 ).to_csv(out/'test_probabilities.csv',index=False)
    for directory in (out,ROOT/'reports'/NAME):
        write(directory/'manifest.json',report)
        write(directory/'selection.json',dict(version=report['version'], fields=chosen, votes=[*FAMILIES,'mitra96',*EVIDENCE],
             threshold=.5, rule='Use the selected tree when all nine votes agree with v131.',
             local_evidence='reports/nov23_matched_learning_curve/selected_tree_replay.json'))
    print({k:v for k,v in report.items() if k not in ('features','source_v131')},flush=True)


if __name__ == '__main__':
    with threadpool_limits(3):
        export()
