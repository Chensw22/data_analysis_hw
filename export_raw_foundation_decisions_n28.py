"""Freeze verified test decisions based on agreement between independent data views."""
import argparse
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from archive_source_harm_veto_n28 import PARENTS
from export_budget_quarter_n23 import source
from foundation_support_for_mechanisms_n28 import decisions
from kaggle_workflow import validate_submission,sha256
from repeated_archive_misses_n26 import ROOT,ART
from relational_error_corrector import write

NAMES={'majority_plus_foundation_raw':'proposal_n28_joint_foundation_raw',
       'any_tied_and_raw':'proposal_n28_tied_raw_crosscheck',
       'unanimous_foundations_and_raw':'proposal_n28_independent_foundation_raw'}


def apply(rule,z):
    parent=z['probability']
    hard=parent>=.5
    flag=decisions(hard,z['versions'],z['raw'],z['seven'])[rule]
    result=np.where(flag,1.-parent,parent)
    result[flag&(parent==.5)]=np.nextafter(.5,0)
    return result


def export(rule):
    name=NAMES[rule]
    test=pd.read_csv(ROOT/'data/raw/test.csv')
    ids=test.PassengerId.to_numpy(str)
    n=len(pd.read_csv(ROOT/'data/raw/train.csv',usecols=['PassengerId']))
    hashes={};versions=[]
    for key,parent in PARENTS.items():
        soft,hard,hashes[parent]=source(parent,test)
        if key=='direction':prob=soft
        versions.append(hard)
    versions=np.column_stack(versions)
    with np.load(ART/'nov28_foundation_support_for_mechanisms/aligned.npz') as z:
        np.testing.assert_array_equal(z['ids'][n:],ids)
        np.testing.assert_array_equal(z['ordered'][n:],z['random'][n:])
        args=dict(probability=prob,versions=versions,seven=z['seven'][n:],raw=z['random'][n:],ids=ids)
    report=json.loads((ROOT/'reports/nov23_matched_learning_curve/foundation_support_for_mechanisms.json').read_text())
    out=ART/name;out.mkdir(exist_ok=True)
    model=out/'inputs.npz'
    np.savez_compressed(model,**args)
    pred=apply(rule,args)
    with np.load(model) as z:again=apply(rule,z)
    np.testing.assert_array_equal(pred,again)
    csv=out/'submission.csv'
    pd.DataFrame(dict(PassengerId=ids,Transported=pred>=.5)).to_csv(csv,index=False)
    pd.DataFrame(dict(PassengerId=ids,probability=pred,reload_probability=again)).to_csv(out/'test_probabilities.csv',index=False)
    checked=validate_submission(csv,ROOT/'data/raw/test.csv')
    changed=int(np.sum((pred>=.5)!=versions[:,0]))
    assert all(r[rule]['test']==changed for r in report.values())
    prior=[]
    for path in (ROOT/'reports').glob('*/online_attempts/*.json'):
        row=json.loads(path.read_text())
        if row.get('evaluation',{}).get('Status')!='COMPLETE':continue
        assert row['file_sha256']!=checked['sha256']
        f=ART/path.parents[1].name/'submission.csv'
        if f.exists() and sha256(f)==row['file_sha256']:
            df=pd.read_csv(f)
            np.testing.assert_array_equal(df.PassengerId,ids)
            prior.append((int(np.sum(df.Transported!=(pred>=.5))),path.parents[1].name))
    nearest=min(prior);assert nearest[0]>0
    version=name.removeprefix('proposal_')
    manifest=dict(version=version,**checked,model_sha256=sha256(model),training_rows=n,reload_difference=0,
                  parents=hashes,changed=changed)
    selection=dict(version=version,rule=rule,nearest=nearest,
                   evidence={k:s[rule] for k,s in report.items()},
                   limitation='Exploratory selection on reused bookings. The three foundation opinions are shared across validations.')
    for d in [out,ROOT/'reports'/name]:
        write(d/'manifest.json',manifest);write(d/'selection.json',selection)
    print(name,changed,nearest,checked['sha256'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('rule',choices=NAMES)
    with threadpool_limits(1):export(parser.parse_args().rule)
