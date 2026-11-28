"""Use receipt deletion directions to adjudicate the paid branch of the scored three-state model."""
import argparse
import json

import numpy as np
import pandas as pd

from export_budget_quarter_n23 import source
from export_consensus_review_n26 import dig,write
from kaggle_workflow import validate_submission
from price_implied_deck_n26 import FEES
from repeated_archive_misses_n26 import ART,cohort,count
from unanimous_error_reach_n27 import ROOT,inputs

BASE='proposal_n27_directional_foundation'
ROUTES='proposal_n27_three_payment_states'
RECEIPTS='proposal_n27_complete_receipt_average'
NAME='proposal_n28_deletion_confirmed_routes'


def apply(parent,route,receipt,paid,banks):
    original=parent>=.5
    moves=(route>=.5)!=original
    confirms=np.ones(len(original),bool)
    for curve in banks:
        delta=(curve[:,1:].mean(axis=1)-curve[:,0])*np.where(original,-1.,1.)
        confirms &= delta>0
    chosen=moves & (~paid | confirms)
    return np.where((receipt>=.5)!=original,receipt,np.where(chosen,route,parent))


def run(export):
    raw,frame,y,_,fold,first,edits,_,opinions,_=inputs()
    n=len(y)
    base=first^edits
    ids=frame.PassengerId.to_numpy(str)
    paid=(frame[FEES].notna().all(axis=1)&frame[FEES].gt(0).any(axis=1)).to_numpy()
    _,_,known,fresh=cohort(raw,y,fold)
    common=((opinions[:n]>=.5)==base[:n,None]).all(1)
    test=frame.iloc[n:]
    parent,hard,digest=source(BASE,test)
    route_p,route_h,route_digest=source(ROUTES,test)
    receipt_p,receipt_h,receipt_digest=source(RECEIPTS,test)
    report={}
    with np.load(ART/'nov27_payment_state_routes/aligned.npz') as z:
        routes={k:z['parent']^z[k].any(1) for k in ['random','ordered']}
    for k in routes:
        with np.load(ART/'nov27_complete_receipt_counterfactual'/f'aligned_{k}_2.npz') as z:
            np.testing.assert_array_equal(z['ids'],ids)
            np.testing.assert_array_equal(z['y'],y)
            np.testing.assert_array_equal(z['anchor'],base)
            banks=np.stack([z['curve_bank0'],z['curve_bank1']])
            receipt=base^(z['average_random_calibrated']&z['average_ordered_calibrated'])
        np.testing.assert_array_equal(receipt[n:],receipt_h)
        np.testing.assert_array_equal(routes[k][n:],route_h)
        candidate=apply(base,routes[k],receipt,paid,banks)>=.5
        for name,previous in [('base',base),('routes',routes[k]),('receipt',receipt)]:
            flag=candidate!=previous
            wrong=previous[:n]!=y
            report.setdefault(k,{})[name]=dict(all=count(flag[:n],known,wrong),fresh=count(flag[:n],known&fresh,wrong),
                common=count(flag[:n],known&common,wrong),test=int(flag[n:].sum()),
                accuracy=float((candidate[:n][known]==y[known]).mean()))
        if k=='random': expected=candidate[n:]
        else: np.testing.assert_array_equal(candidate[n:],expected)
    print(report,flush=True)
    write(ROOT/'reports/nov23_matched_learning_curve/deletion_confirmed_routes.json',report)
    if not export:return
    dest=ART/NAME
    dest.mkdir(exist_ok=True)
    args=dict(parent=parent,route=route_p,receipt=receipt_p,paid=paid[n:],banks=banks[:,n:])
    path=dest/'inputs.npz'
    np.savez_compressed(path,ids=ids[n:],**args)
    p=apply(**args)
    with np.load(path) as z: repeat=apply(**{k:z[k] for k in args})
    np.testing.assert_array_equal(p,repeat)
    np.testing.assert_array_equal(p>=.5,expected)
    csv=dest/'submission.csv'
    pd.DataFrame(dict(PassengerId=ids[n:],Transported=expected)).to_csv(csv,index=False)
    pd.DataFrame(dict(PassengerId=ids[n:],probability=p,reload_probability=repeat)).to_csv(dest/'test_probabilities.csv',index=False)
    checked=validate_submission(csv,ROOT/'data/raw/test.csv')
    assert all(json.loads(p.read_text()).get('file_sha256')!=checked['sha256'] for p in (ROOT/'reports').glob('*/online_attempts/*.json'))
    manifest=dict(**checked,version=NAME.removeprefix('proposal_'),training_rows=n,model_sha256=dig(path),
             parents={BASE:digest,ROUTES:route_digest,RECEIPTS:receipt_digest},reload_difference=0,source_file='route_with_receipt_deletions_n28.py')
    for folder in [dest,ROOT/'reports'/NAME]:
        folder.mkdir(exist_ok=True)
        write(folder/'manifest.json',manifest)
        write(folder/'selection.json',dict(version=manifest['version'],results=report,rule='Require both receipt reference banks to support paid routing when a facility is hidden on average; retain independently scored receipt changes.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--export',action='store_true')
    run(parser.parse_args().export)
