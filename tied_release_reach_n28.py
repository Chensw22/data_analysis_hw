"""Count distinct disagreements among score-tied files without using test outcomes."""
import json
import math
from decimal import Decimal

import numpy as np
import pandas as pd

from kaggle_workflow import sha256
from unanimous_error_reach_n27 import ROOT


def run():
    best=json.loads((ROOT/'reports/current_best_submission.json').read_text())
    champion=Decimal(str(best['public_score']))
    ids=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId']).PassengerId
    decisions={}
    for path in sorted((ROOT/'reports').glob('*/online_attempts/*.json')):
        row=json.loads(path.read_text())
        scored=row.get('evaluation',{})
        if scored.get('Status')!='COMPLETE' or Decimal(str(scored['Public Score']))!=champion:continue
        name=path.parents[1].name
        csv=ROOT/'artifacts'/name/'submission.csv'
        assert sha256(csv)==row['file_sha256']
        frame=pd.read_csv(csv)
        np.testing.assert_array_equal(frame.PassengerId,ids)
        decisions[name]=frame.Transported.to_numpy(bool)
    base=decisions[best['proposal']]
    different=np.stack([dec!=base for dec in decisions.values()]).any(0)
    compatible=[k for k in range(len(ids)+1) if k*100000//len(ids)==int(champion*100000)]
    assert len(compatible)==1
    needed=math.floor(Decimal('0.85')*len(ids))+1-compatible[0]
    result=dict(baseline=best['proposal'],score=float(champion),rows=len(ids),compatible_correct=compatible[0],
                versions=list(decisions),differing_rows=int(different.sum()),agreeing_rows=int((~different).sum()),
                net_more_correct_required=needed,minimum_net_from_agreements=max(0,needed-int(different.sum())),
                by_file={name:int(np.sum(v!=base)) for name,v in decisions.items()},
                note='Assumes the compatible 4277-passenger scoring denominator. The bound treats every differing row as correctable; it does not estimate or identify test labels.')
    (ROOT/'reports/nov23_matched_learning_curve/tied_release_reach_n28.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)


if __name__=='__main__':run()
