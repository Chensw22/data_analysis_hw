"""Dispatch two scored model families by whether all seven archived opinions support the parent."""
import json
import numpy as np
import pandas as pd

from export_budget_quarter_n23 import source
from export_consensus_review_n26 import dig,write
from kaggle_workflow import validate_submission
from repeated_archive_misses_n26 import ART,cohort,count
from route_with_receipt_deletions_n28 import BASE,ROUTES,RECEIPTS
from unanimous_error_reach_n27 import ROOT,inputs

NAME='proposal_n28_unanimous_payment_dispatch'


def run():
    raw,frame,y,_,fold,first,changes,_,opinions,_=inputs()
    n=len(y)
    base=first^changes
    common=((opinions>=.5)==base[:,None]).all(1)
    _,_,known,fresh=cohort(raw,y,fold)
    test=frame.iloc[n:]
    ids=frame.PassengerId.to_numpy(str)
    sources={name:source(name,test) for name in [BASE,ROUTES,RECEIPTS]}
    with np.load(ART/'nov27_payment_state_routes/aligned.npz') as z:
        np.testing.assert_array_equal(z['parent'],base)
        routed={tag:z['parent']^z[tag].any(1) for tag in ('random','ordered')}
    report={}
    for tag in routed:
        with np.load(ART/'nov27_complete_receipt_counterfactual'/f'aligned_{tag}_2.npz') as z:
            np.testing.assert_array_equal(z['ids'],ids)
            np.testing.assert_array_equal(z['y'],y)
            receipt=base^(z['average_random_calibrated']&z['average_ordered_calibrated'])
        decisions={BASE:base,ROUTES:routed[tag],RECEIPTS:receipt}
        candidate=np.where(common,routed[tag],receipt)
        for name,par in decisions.items():
            np.testing.assert_array_equal(par[n:],sources[name][1])
            flag=candidate!=par
            wrong=par[:n]!=y
            report.setdefault(tag,{})[name]=dict(all=count(flag[:n],known,wrong),fresh=count(flag[:n],known&fresh,wrong),
                common=count(flag[:n],known&common[:n],wrong),changes_test=int(flag[n:].sum()),
                accuracy=float((candidate[:n][known]==y[known]).mean()))
    write(ROOT/'reports/nov23_matched_learning_curve/unanimous_payment_dispatch.json',report)
    print(report,flush=True)
    if not all(report[tag][BASE]['all']['fixed']>report[tag][BASE]['all']['harmed'] and
               report[tag][BASE]['accuracy']>.83 for tag in routed):return
    dest=ART/NAME
    dest.mkdir(exist_ok=True)
    model=dest/'inputs.npz'
    np.savez_compressed(model,ids=ids[n:],common=common[n:],routed=sources[ROUTES][0],receipt=sources[RECEIPTS][0])
    p=np.where(common[n:],sources[ROUTES][0],sources[RECEIPTS][0])
    with np.load(model) as z:repeat=np.where(z['common'],z['routed'],z['receipt'])
    np.testing.assert_array_equal(p,repeat)
    np.testing.assert_array_equal(p>=.5,candidate[n:])
    csv=dest/'submission.csv'
    pd.DataFrame(dict(PassengerId=ids[n:],Transported=p>=.5)).to_csv(csv,index=False)
    pd.DataFrame(dict(PassengerId=ids[n:],probability=p,reload_probability=repeat)).to_csv(dest/'test_probabilities.csv',index=False)
    check=validate_submission(csv,ROOT/'data/raw/test.csv')
    assert not any(json.loads(path.read_text()).get('file_sha256')==check['sha256'] for path in (ROOT/'reports').glob('*/online_attempts/*.json'))
    m=dict(**check,version=NAME.removeprefix('proposal_'),source_file='dispatch_on_unanimous_reference_n28.py',
           model_sha256=dig(model),reload_difference=0,training_rows=n,parents={name:src[2] for name,src in sources.items()})
    for folder in [dest,ROOT/'reports'/NAME]:
        folder.mkdir(exist_ok=True)
        write(folder/'manifest.json',m)
        write(folder/'selection.json',dict(version=m['version'],rule='Use the three-state model on unanimous-parent decisions and the receipt-deletion model elsewhere.',results=report))


if __name__=='__main__':run()
