"""Build the fixed full-archive disagreement candidate from 68 refitted scores and the scored nine-vote release."""
import json

import numpy as np
import pandas as pd

from dispute_band_stability_n26 import gate
from export_budget_quarter_n23 import source
from family_identity_transfer_n26 import panels_and_reference
from full_roster_booking_validation_n23 import ROOT,data
from kaggle_workflow import sha256,validate_submission

NAME='proposal_n26_dispute_band'
ART=ROOT/'artifacts'
ARCHIVE=ART/'nov26_archive_release'
NINE='proposal_n24_selected_dissent'


def read_archive(train,test,y,names):
    columns={}
    part_digests={}
    replay_error={}
    for part in 'abc':
        path=ARCHIVE/f'part_{part}.npz'
        part_digests[part]=sha256(path)
        with np.load(path) as z:
            np.testing.assert_array_equal(z['train_passenger'],train.PassengerId)
            np.testing.assert_array_equal(z['passenger'],test.PassengerId)
            np.testing.assert_array_equal(z['train_y' if 'train_y' in z else 'labels'],y)
            ids=z['names'].tolist()
            p=z['p']
            assert p.shape==(len(test),len(ids)) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            if 'replay_maxdiff' in z:
                replay_error[part]=float(np.max(z['replay_maxdiff']))
            else:
                audit=ROOT/'reports/nov23_matched_learning_curve'/f'archive_release_part_{part}.json'
                records=json.loads(audit.read_text())['results' if part=='b' else 'replay']
                errors=[]
                checked=[]
                for family,row in records.items():
                    checked.extend(n if '/' in n else family+'/'+n for n in row['columns'])
                    errors.extend(row['maximum_replay_error' if part=='b' else 'max_absolute_error'])
                    assert not any(row['changed_replay_decisions' if part=='b' else 'hard_label_differences'])
                assert set(checked)==set(ids) and len(errors)==len(ids)
                replay_error[part]=max(errors)
            assert replay_error[part]<1e-5
            for key,col in zip(ids,p.T,strict=True):
                if key in columns:
                    raise ValueError('Duplicate configuration '+key)
                columns[key]=col
    assert set(columns)==set(names) and len(columns)==68
    return np.column_stack([columns[key] for key in names]),part_digests,replay_error


def apply(names,p,anchor):
    tree=[j for j,n in enumerate(names) if n.startswith('permutation_selection/')]
    assert len(tree)==5
    source_p=p[:,tree].mean(1)
    band=gate(p)
    return np.where(band,source_p,anchor),band


def run():
    train,y,_,_=data()
    test_path=ROOT/'data/raw/test.csv'
    test=pd.read_csv(test_path,usecols=train.columns.tolist())
    _,_,_,_,names,_,_,_,_,_=panels_and_reference()
    anchor,hard,digest=source(NINE,test)
    p,digests,error=read_archive(train,test,y,names)
    out=ART/NAME
    out.mkdir(exist_ok=True)
    report=ROOT/'reports'/NAME
    report.mkdir(exist_ok=True)
    # Serialize the fitted voters' scores on the fixed competition roster and the identified anchor.
    file=out/'frozen_votes.npz'
    with file.with_suffix('.tmp').open('wb') as f:
        np.savez_compressed(f,names=names,p=p,anchor=anchor,passenger=test.PassengerId.to_numpy(str))
    file.with_suffix('.tmp').replace(file)
    prob,eligible=apply(names,p,anchor)
    with np.load(file) as z:
        np.testing.assert_array_equal(z['passenger'],test.PassengerId)
        np.testing.assert_array_equal(z['anchor'],anchor)
        repeat,_=apply(z['names'].tolist(),z['p'],z['anchor'])
    np.testing.assert_array_equal(repeat,prob)
    changed=(prob>=.5)!=hard
    assert not changed[~eligible].any()
    pd.DataFrame(dict(PassengerId=test.PassengerId,Transported=prob>=.5)).to_csv(out/'submission.csv',index=False)
    pd.DataFrame(dict(PassengerId=test.PassengerId,probability=prob,reload_probability=repeat)).to_csv(out/'test_probabilities.csv',index=False)
    check=validate_submission(out/'submission.csv',test_path)
    distances={}
    for receipt in (ROOT/'reports').glob('*/online_attempts/*.json'):
        record=json.loads(receipt.read_text())
        if record.get('evaluation',{}).get('Status')!='COMPLETE':
            continue
        previous=ART/receipt.parents[1].name/'submission.csv'
        assert sha256(previous)==record['file_sha256']
        old=pd.read_csv(previous)
        np.testing.assert_array_equal(old.PassengerId,test.PassengerId)
        distances[receipt.parents[1].name]=int(np.sum(old.Transported!=(prob>=.5)))
    pending={}
    for candidate in ['proposal_n25_confirmed_random_fields','proposal_n25_awake_personal_tabicl',
                      'proposal_n25_confirmed_awake_union','proposal_n25_awake_error_confirmation']:
        old=pd.read_csv(ART/candidate/'submission.csv')
        np.testing.assert_array_equal(old.PassengerId,test.PassengerId)
        pending[candidate]=int((old.Transported!=(prob>=.5)).sum())
    nearest=min(distances.values())
    assert nearest>0 and min(pending.values())>0
    manifest=dict(**check,version=NAME.removeprefix('proposal_'),parents={'nine':digest},
        training_rows=len(y),model_sha256=sha256(file),reload_difference=0,
        training_sha256=sha256(ROOT/'data/raw/train.csv'),query_sha256=sha256(test_path),
        archive_parts=digests,reported_replay_maxdiff=error,band_rows=int(eligible.sum()),
        nine_changes=int(changed.sum()),to_true=int((changed&~hard).sum()),to_false=int((changed&hard).sum()),
        closest_submitted_changes=nearest,closest_submitted=[key for key,d in distances.items() if d==nearest],
        distances_to_pending=pending)
    selection=dict(version=manifest['version'],archive=names,
        gate='4 through 16 minority votes among 68 configurations, inclusive; label threshold >= 0.5',
        candidate='arithmetic mean of the five permutation_selection probabilities; retain official nine outside the band',
        sources=['archive_release_part_a_n26.py','archive_release_part_b_n26.py','archive_release_part_c_n26.py'],
        local_evaluation='reports/nov23_matched_learning_curve/dispute_band_stability.json')
    for folder in (out,report):
        for name,content in [('manifest',manifest),('selection',selection)]:
            (folder/(name+'.json')).write_text(json.dumps(content,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    run()
