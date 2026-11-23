"""Build the fixed previously evaluated quarter-weight contender for an optional submission."""
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

from broad_partition_evidence_n23 import ART, ROOT, write
from exaone_validation_data import CATEGORICAL, compact_features
from kaggle_workflow import sha256, validate_submission
from spend_simplex_likelihood_n23 import design

VERSION = 'n23_budget_quarter_incumbent'
NAME = 'proposal_' + VERSION


def source(name, test):
    directory = ART/name
    path = directory/'submission.csv'
    reviews = list((ROOT/'reports'/name/'online_attempts').glob('*.json'))
    assert len(reviews)==1
    receipt = json.loads(reviews[0].read_text())
    assert sha256(path)==receipt['file_sha256'] and receipt['evaluation']['Status']=='COMPLETE'
    hard = pd.read_csv(path)
    soft = pd.read_csv(directory/'test_probabilities.csv',float_precision='round_trip')
    for frame in [hard,soft]:
        np.testing.assert_array_equal(frame.PassengerId,test.PassengerId)
    np.testing.assert_array_equal(soft.probability,soft.reload_probability)
    np.testing.assert_array_equal(soft.probability>=.5,hard.Transported)
    return soft.probability.to_numpy(), hard.Transported.to_numpy(bool), receipt['file_sha256']


def export():
    train = pd.read_csv(ROOT/'data/raw/train.csv')
    test = pd.read_csv(ROOT/'data/raw/test.csv')
    x, query = [design(compact_features(frame))[0] for frame in [train,test]]
    assert not set(train.PassengerId.str.split('_').str[0]) & set(test.PassengerId.str.split('_').str[0])
    base, _, base_hash = source('proposal_v108_party_bills',test)
    _, incumbent, release_hash = source('proposal_v131_party_consensus',test)
    out = ART/NAME
    out.mkdir(exist_ok=True)
    model = CatBoostClassifier(iterations=480,depth=6,learning_rate=.045,l2_leaf_reg=6,
                               thread_count=2,random_seed=345,allow_writing_files=False,verbose=False)
    model.fit(x,train.Transported,cat_features=CATEGORICAL)
    file = out/'budget.cbm'
    model.save_model(str(file))
    def probabilities(fitted):
        alt = fitted.predict_proba(query)[:,1]
        return np.where((base>=.5)==incumbent, .75*base+.25*alt, incumbent.astype(float))
    p = probabilities(model)
    q = probabilities(CatBoostClassifier().load_model(str(file)))
    np.testing.assert_array_equal(p,q)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    pd.DataFrame(dict(PassengerId=test.PassengerId,probability=p,
                      reload_probability=q)).to_csv(out/'test_probabilities.csv',index=False)
    pd.DataFrame(dict(PassengerId=test.PassengerId,Transported=p>=.5)).to_csv(out/'submission.csv',index=False)
    check = validate_submission(out/'submission.csv',ROOT/'data/raw/test.csv')
    comparisons = {}
    for receipt in (ROOT/'reports').glob('*/online_attempts/*.json'):
        record = json.loads(receipt.read_text())
        assert record.get('file_sha256')!=check['sha256'], receipt
        csv = ART/receipt.parent.parent.name/'submission.csv'
        if csv.exists() and sha256(csv)==record.get('file_sha256'):
            previous = pd.read_csv(csv)
            np.testing.assert_array_equal(previous.PassengerId,test.PassengerId)
            comparisons[receipt.parent.parent.name] = int(np.sum(previous.Transported.to_numpy()!=(p>=.5)))
    assert min(comparisons.values())>0
    selection = dict(version=VERSION,threshold=.5)
    manifest = dict(**check,version=VERSION,model_sha256=sha256(file),reload_difference=0,
                    fit_people=len(train),anchors={'v108':base_hash,'v131':release_hash},
                    preserved_incumbent_overrides=int(((base>=.5)!=incumbent).sum()),
                    parent_changed=int(np.sum((p>=.5)!=incumbent)),
                    matched_previous_csv=len(comparisons),minimum_difference=min(comparisons.values()),
                    nearest_exports=[n for n,v in comparisons.items() if v==min(comparisons.values())])
    for target in [out,ROOT/'reports'/NAME]:
        write(target/'manifest.json',manifest)
        write(target/'selection.json',selection)
    print(manifest,flush=True)


if __name__=='__main__':
    with threadpool_limits(2):export()
