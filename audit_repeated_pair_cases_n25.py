"""Describe repeated pair-model reversals with their correct-prediction denominators."""
import json
import numpy as np
from threadpoolctl import threadpool_limits

from full_roster_adaptive_bill_prior_n23 import EVIDENCE
from full_roster_booking_validation_n23 import FAMILIES
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import BILLS,ROOT,banks,data,latest_reference
from full_roster_selected_tree_replay_n24 import original_prediction,prepared_vote
from full_roster_six_failure_archive_n24 import prediction_sets
from historical_nine_release_n25 import OUT as HISTORY
from passenger_pair_outcomes_n25 import TAG


def near_scores(tag,ids,fold,groups,y):
    scores=np.empty(len(ids))
    for k in np.unique(fold):
        mask=fold==k
        q=ids[mask]
        t=ids[~mask&~np.isin(groups[ids],groups[q])]
        with np.load(ROOT/'artifacts'/f'nov24_{tag[0]}'/f'{tag[1]}_{k}.npz') as z:
            for key,value in [('t',t),('q',q),('truth',y[t])]:
                np.testing.assert_array_equal(z[key],value)
            scores[mask]=z['p'][:,1]
    return scores


def run():
    raw,y,g,f=data()
    _,oy,_,rows,of,*_=aligned()
    names,h,b=banks(raw,y,f)
    ref=latest_reference(y,g,f,names,h,b)
    ix=np.arange(len(y))
    inc=prepared_vote(original_prediction(ix,f,g,y,'full'),ref,b[:,[names.index(k) for k in (*FAMILIES,'mitra96',*EVIDENCE)]])
    with np.load(HISTORY/'historical.npz') as z:
        np.testing.assert_array_equal(z['rows'],rows)
        hist=z['nine']
    _,panels=prediction_sets(raw,y,g,f,rows,of)
    size=raw.groupby(g).PassengerId.transform('size').to_numpy()
    positive=raw[BILLS].gt(0).sum(axis=1).to_numpy()
    strata={'solo':size==1,'multi':size>1,'earth':raw.HomePlanet.eq('Earth').to_numpy(),
            'cryo':raw.CryoSleep.eq(True).to_numpy(),'no_positive_bill':positive==0,
            'one_or_two_services':(positive==1)|(positive==2),'three_plus_services':positive>=3,
            'missing_bill':raw[BILLS].isna().any(axis=1).to_numpy(),'all_fields_present':~raw.isna().any(axis=1).to_numpy()}
    report={}
    for (tag,ids,fold,base),panel in zip([('full',ix,f,inc),('historical',rows,of,hist)],panels,strict=True):
        same=((np.column_stack(panel)>=.5)==base[:,None]).all(1)
        p=[near_scores((t,tag),ids,fold,g,y) for t in (TAG,TAG+'_repeat')]
        both=((p[0]>=.5)!=base)&((p[1]>=.5)!=base)
        wrong=base!=y[ids]
        records={}
        for name,sub in {'all':np.ones(len(ids),bool),'originally_false':~base,'originally_true':base,
                         **{k:m[ids] for k,m in strata.items()}}.items():
            mask=same&sub
            records[name]=dict(eligible=int(mask.sum()),original_errors=int((mask&wrong).sum()),
                               repeated_fixed=int((mask&both&wrong).sum()),repeated_harmed=int((mask&both&~wrong).sum()))
        report[tag]=records
        print(tag,{k:v for k,v in records.items() if k in ('all','originally_false','originally_true','three_plus_services','no_positive_bill')},flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/repeated_pair_case_profiles.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
