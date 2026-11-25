"""Rebuild the nine-vote release on historical folds and verify its feature pipeline on Kaggle rows."""
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import BILLS
from exaone_validation_data import compact_features
from export_budget_quarter_n23 import source
from export_selected_dissent_n24 import NAME
from full_roster_adaptive_bill_prior_n23 import EVIDENCE
from full_roster_bill_prior_balance_n23 import predict_priors
from full_roster_booking_validation_n23 import FAMILIES,OUT as ORIGINAL,predictor
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,data
from full_roster_disagreement_calibration_n24 import stats
from full_roster_permutation_selection_n24 import ensemble_probability,feature_bank
from full_roster_selected_tree_replay_n24 import original_prediction,prepared_vote
from kaggle_workflow import sha256

OUT=ROOT/'artifacts/nov25_historical_nine'


def run():
    raw,y,g,_=data()
    _,oy,og,rows,f,_,v131,_,_,_,_,old,_=aligned()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=lambda c:c!='Name')
    roster=pd.concat([raw,test],ignore_index=True)
    groups=roster.PassengerId.str.split('_').str[0].to_numpy()
    x=compact_features(roster)
    cue=['CryoSleep',*BILLS]
    rest=x.drop(columns=cue)
    ordinary={k:predictor(k,roster,y) for k in ('cat','budget')}
    def calculate(t,q,fold,three):
        prior_train=t[~np.isin(groups[t],groups[q])]
        extra=predict_priors(x,cue,rest,y,groups,prior_train,q)
        scores={k:fit(t,q,fold) for k,fit in ordinary.items()}
        scores.update({k:three[:,i] for i,k in enumerate(('mitra96','v44_without_name','tabicl'))})
        return np.column_stack([scores[k] for k in (*FAMILIES,'mitra96')]+[extra[k] for k in EVIDENCE])
    OUT.mkdir(exist_ok=True)
    components=np.full((len(rows),9),np.nan)
    for fold in np.unique(f):
        t,q=rows[f!=fold],rows[f==fold]
        path=OUT/f'fold_{fold}.npz'
        if not path.exists():
            scores=calculate(t,q,int(fold),old[f==fold])
            np.savez_compressed(path,t=t,q=q,y=y[t],scores=scores)
        with np.load(path) as z:
            for k,v in [('t',t),('q',q),('y',y[t])]:
                np.testing.assert_array_equal(z[k],v)
            components[f==fold]=z['scores']
    assert np.isfinite(components).all()
    historical_tree=original_prediction(rows,f,g,y,'historical')
    hist=prepared_vote(historical_tree,v131,components)
    short=prepared_vote(historical_tree,v131,old)
    # Full-size replay has an independently scored CSV and frozen raw probabilities for the tree.
    anchor,_,_=source('proposal_online_first_counts15',test)
    prior,base,_=source('proposal_v131_party_consensus',test)
    scored,truth,digest=source(NAME,test)
    other=[anchor]
    for key in ('v44_without_name','tabicl'):
        with np.load(ORIGINAL/f'original_{key}.npz') as z:
            np.testing.assert_array_equal(z['test_ids'],test.PassengerId)
            other.append(z['test'])
    t=np.arange(len(y))
    q=np.arange(len(y),len(roster))
    assert not np.isin(groups[t],groups[q]).any()
    votes=calculate(t,q,0,np.column_stack(other))
    model_path=ROOT/'artifacts'/NAME/'ensemble.joblib'
    declared=json.loads((ROOT/'reports'/NAME/'manifest.json').read_text())
    assert sha256(model_path)==declared['model_sha256']
    model=joblib.load(model_path)
    query,fields,_=feature_bank(test)
    query=pd.DataFrame(query,columns=fields).reindex(columns=model['fields'],fill_value=0).to_numpy()
    tree=ensemble_probability(model['models'],query)
    eligible=((votes>=.5)==base[:,None]).all(1)
    np.testing.assert_allclose(np.where(eligible,tree,prior),scored,atol=1e-14,rtol=0)
    np.testing.assert_array_equal(prepared_vote(tree,base,votes),truth)
    np.savez_compressed(OUT/'historical.npz',rows=rows,passenger=raw.PassengerId.iloc[rows].to_numpy(str),
                        y=oy,fold=f,v131=v131,nine=hist,scores=components,tree=historical_tree)
    report=dict(votes=[*FAMILIES,'mitra96',*EVIDENCE],scored_csv_sha256=digest,
                verified_test_rows=len(test),test_eligible=int(eligible.sum()),changes_to_v131=int((truth!=base).sum()),
                historical_rows=len(oy),historical_eligible=int(((components>=.5)==v131[:,None]).all(1).sum()),
                nine_to_v131=stats(hist,v131,oy,og,f),three_to_v131=stats(short,v131,oy,og,f),
                nine_to_three=stats(hist,short,oy,og,f),historical_prior_training_sizes=[int((~np.isin(g[rows[f!=v]],g[rows[f==v]])).sum()) for v in np.unique(f)],
                historical_passengers_with_training_peers=int(sum(
                    np.isin(g[rows[f==fold]],g[rows[f!=fold]]).sum() for fold in np.unique(f))))
    (ROOT/'reports/nov23_matched_learning_curve/historical_nine_release.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report,flush=True)


if __name__=='__main__':
    with threadpool_limits(2):
        run()
