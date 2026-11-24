"""Count broad and repeated rescues of the frozen bagged edit candidate."""
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,banks,data,latest_reference
from full_roster_edit_resampling_n24 import run as replay
from full_roster_six_failure_archive_n24 import prediction_sets
from kaggle_workflow import sha256


def run():
    name = 'proposal_n24_bagged_net_edits'
    frozen = json.loads((ROOT/'reports'/name/'selection.json').read_text())
    current,historical,test = replay()
    assert sha256(ROOT/frozen['evidence'])==frozen['evidence_sha256']
    csv = ROOT/'artifacts'/name/'submission.csv'
    manifest = json.loads((ROOT/'reports'/name/'manifest.json').read_text())
    assert sha256(csv)==manifest['sha256']
    np.testing.assert_array_equal(pd.read_csv(csv).Transported,test)
    raw,y,g,fold = data()
    _,oy,_,rows,of,_,champ,_,_,_,_,_,_ = aligned()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    _,panels = prediction_sets(raw,y,g,fold,rows,of)
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as z:
        for k,v in [('passenger',raw.PassengerId),('fold',fold),('y',y)]:
            np.testing.assert_array_equal(z[k],v)
        hard = z['remaining']
    solo = raw.groupby(g).PassengerId.transform('size').to_numpy()==1
    report = {}
    corrections,harms = [],[]
    for (tag,ix,base,pred),(six,archive) in zip([('full',np.arange(len(y)),ref,current),
                          ('historical',rows,champ,historical)],panels,strict=True):
        agree = ((six>=.5)==base[:,None]).all(1)
        all_agree = agree & ((archive>=.5)==base[:,None]).all(1)
        fixed = (base!=y[ix]) & (pred==y[ix])
        harmed = (base==y[ix]) & (pred!=y[ix])
        corrections.append(fixed)
        harms.append(harmed)
        scopes = dict(all=np.ones(len(ix),bool),six_agree=agree,six_and_archive_agree=all_agree,
                      hard_cohort=hard[ix],solo=solo[ix],grouped=~solo[ix])
        report[tag] = {k:dict(rows=int(s.sum()),baseline_errors=int((s & (base!=y[ix])).sum()),
                              fixed=int((s&fixed).sum()),harmed=int((s&harmed).sum())) for k,s in scopes.items()}
    report['paired'] = dict(fixed_both=int((corrections[0][rows]&corrections[1]).sum()),
                            harmed_both=int((harms[0][rows]&harms[1]).sum()))
    path = ROOT/'reports/nov23_matched_learning_curve/bagged_net_edit_cases.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(report,flush=True)


if __name__=='__main__':
    with threadpool_limits(1):
        run()
