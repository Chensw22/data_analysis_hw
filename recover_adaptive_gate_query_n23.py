"""Recover the 124-candidate gate probability for future experiments without changing its CSV."""
import json

import joblib
import numpy as np
import pandas as pd

from broad_geographic_transfer_n23 import BILLS
from exaone_validation_data import compact_features
from export_adaptive_bill_prior_n23 import EVIDENCE, FAMILIES, NAME, OUT, QUEUED, ROOT, data, predict_priors, proposal_design
from export_budget_quarter_n23 import source
from kaggle_workflow import sha256


def run():
    raw, y, group, fold = data()
    test = pd.read_csv(ROOT/'data/raw/test.csv', usecols=lambda k:k!='Name')
    all_raw = pd.concat([raw,test], ignore_index=True)
    groups = all_raw.PassengerId.str.split('_').str[0].to_numpy()
    np.testing.assert_array_equal(groups[:len(y)],group)
    art = ROOT/'artifacts'
    directory = art/NAME
    manifest = json.loads((ROOT/'reports'/NAME/'manifest.json').read_text())
    model_path = directory/'gate.joblib'
    locked = directory/'submission.csv'
    assert sha256(model_path) == manifest['model_sha256'] and sha256(locked) == manifest['sha256']
    truth = pd.read_csv(locked)
    np.testing.assert_array_equal(truth.PassengerId,test.PassengerId)
    probs = pd.read_csv(directory/'test_probabilities.csv')
    np.testing.assert_array_equal(probs.PassengerId,test.PassengerId)
    np.testing.assert_array_equal(probs.probability>=.5,truth.Transported)
    mitra,_,parent = source('proposal_online_first_counts15',test)
    assert parent==manifest['parents']['v96']
    _,champion,parent = source('proposal_v131_party_consensus',test)
    assert parent==manifest['parents']['v131']
    protected = champion!=(mitra>=.5)
    query_score = art/QUEUED/'query_scores.npz'
    assert sha256(query_score) == manifest['ordinary_query_sha256']
    with np.load(query_score) as z:
        np.testing.assert_array_equal(z['passenger'],test.PassengerId)
        np.testing.assert_array_equal(z['scores'][:,0],mitra)
        ordinary=dict(cat=z['scores'][:,1],budget=z['scores'][:,2],mitra96=mitra)
    for name, expected in manifest['crossfamily_query_hashes'].items():
        path = OUT/f'original_{name}.npz'
        assert sha256(path)==expected
        with np.load(path) as z:
            np.testing.assert_array_equal(z['test_ids'],test.PassengerId)
            ordinary[name]=z['test']
    base=np.column_stack([ordinary[k] for k in (*FAMILIES,'mitra96')])
    x=compact_features(all_raw)
    t,q=np.arange(len(y)),np.arange(len(y),len(x))
    cue=['CryoSleep',*BILLS]
    parts=predict_priors(x,cue,x.drop(columns=cue),y,groups,t,q)
    scores=np.column_stack([parts[k] for k in EVIDENCE])
    conflict, design=proposal_design(base,scores)
    conflict &= ~protected
    gate=joblib.load(model_path)
    def recover():
        flip=gate.predict_proba(design[conflict])[:,1]
        result=mitra.copy()
        result[conflict]=np.where(mitra[conflict]>=.5,1-flip,flip)
        result[protected]=np.nan
        return result
    probability=recover()
    eligible=~protected
    np.testing.assert_array_equal(np.where(protected,champion,probability>=.5),truth.Transported)
    changed=eligible & ((mitra>=.5)!=truth.Transported.to_numpy())
    assert protected.sum()==61 and changed.sum()==124
    result_path=directory/'recovered_gate_scores.npz'
    np.savez_compressed(result_path,passenger=test.PassengerId.to_numpy(str),gate_probability=probability,
                        protected=protected,conflict=conflict,ordinary_mean=base[:,:4].mean(axis=1),
                        restored_full=parts['restored_full'],export_score=probs.probability.to_numpy())
    with np.load(result_path) as z:
        np.testing.assert_array_equal(z['gate_probability'],recover())
    report = dict(rows=len(test), direct_gate_or_mitra_rows=int(eligible.sum()), inherited_v131_rows=int(protected.sum()),
                  conflict_rows=int(conflict.sum()), changed=int(changed.sum()), classification_mismatches=0,
                  changed_export_at_most_30=int((probs.probability[changed]<=.3).sum()),
                  changed_gate_at_most_30=int((probability[changed]<=.3).sum()),
                  csv_sha256=sha256(locked), model_sha256=sha256(model_path), score_file_sha256=sha256(result_path))
    (ROOT/'reports'/NAME/'recovered_probability.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)


if __name__=='__main__':
    run()
