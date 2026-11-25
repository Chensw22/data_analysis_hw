"""Locate which stages of the scored tree decision rule change coverage in deployment."""
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from export_budget_quarter_n23 import source
from export_selected_tree_pair_n25 import NAME,NINE,SEEDS,query_score,decisions
from full_roster_adaptive_bill_prior_n23 import EVIDENCE
from full_roster_booking_validation_n23 import FAMILIES
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,banks,data,latest_reference
from full_roster_permutation_selection_n24 import feature_bank
from full_roster_selected_tree_replay_n24 import original_prediction,prepared_vote,refit


def stages(tree,anchor,champ,orig):
    free=(orig==champ)&((anchor>=.5)==orig)
    opposed=free&((tree>=.5)!=orig)
    taken=opposed&(abs(tree.clip(.003,.997)-.5)>abs(anchor.clip(.003,.997)-.5))
    return np.column_stack([np.ones(len(tree),bool),free,opposed,taken])


def strata(raw,group,base,anchor):
    size=raw.groupby(group).PassengerId.transform('size').to_numpy()
    return np.column_stack([base,size==1,np.abs(anchor-.5)>.15,raw.CryoSleep.eq(True),
                            raw[['FoodCourt','Spa','VRDeck','RoomService','ShoppingMall']].gt(0).any(axis=1)]).astype(int)


def run():
    raw,y,g,f=data()
    names,h,full=banks(raw,y,f)
    base=latest_reference(y,g,f,names,h,full)
    nine=full[:,[names.index(k) for k in (*FAMILIES,'mitra96',*EVIDENCE)]]
    original=original_prediction(np.arange(len(y)),f,g,y,'full')
    champ=prepared_vote(original,base,nine)
    p=full[:,names.index('mitra96')]
    alt=np.mean([refit(raw,y,g,seed) for seed in SEEDS],axis=0)
    a=stages(alt,p,champ,base)
    datafile=ROOT/'data/raw/test.csv'
    test=pd.read_csv(datafile,usecols=lambda c:c!='Name')
    tg=test.PassengerId.str.split('_').str[0].to_numpy()
    pt,_,_=source('proposal_online_first_counts15',test)
    _,bt,_=source('proposal_v131_party_consensus',test)
    cp,ct,_=source(NINE,test)
    _,official,_=source(NAME,test)
    x,fields,_=feature_bank(test)
    qt=query_score(joblib.load(ROOT/'artifacts'/NAME/'ensemble.joblib'),pd.DataFrame(x,columns=fields),tg)
    np.testing.assert_array_equal(decisions(qt,pt,ct,bt,cp)>=.5,official)
    b=stages(qt,pt,ct,bt)
    np.testing.assert_array_equal(b[:,-1],official!=ct)
    assert tuple([a[:,-1].sum(),b[:,-1].sum()])==(201,157)
    truth=champ!=y
    sx,tx=strata(raw,g,champ,p),strata(test,tg,ct,pt)
    groups=['originally_true','solo','anchor_over_65pct','cryo','has_positive_bill']
    out={}
    for name,local,targ in [('all',np.ones(len(y),bool),np.ones(len(test),bool)),
                            *((s,sx[:,j]==1,tx[:,j]==1) for j,s in enumerate(groups)),
                            *((f'complement_{s}',sx[:,j]==0,tx[:,j]==0) for j,s in enumerate(groups))]:
        changed=local & a[:,-1]
        out[name]=dict(local_stages=a[local].sum(0).tolist(),test_stages=b[targ].sum(0).tolist(),
                       corrected=int((changed&truth).sum()),harmed=int((changed&~truth).sum()))
    # Direct standardization of the local edit rate to the size of each observed test stratum.
    # Use only pre-edit fields and three coarse binary splits to retain data in every cell.
    std=[]
    for cell in range(8):
        local=sx[:,:3]@np.array([1,2,4])==cell
        targ=tx[:,:3]@np.array([1,2,4])==cell
        assert local.sum()>0
        std.append(dict(cell=cell,local=int(local.sum()),test=int(targ.sum()),
                        expected_test_edits=float(targ.sum()*a[local,-1].mean()),actual_test_edits=int(b[targ,-1].sum())))
    report=dict(stages=['all','eligible','challenger_opposes','accepted'],cohorts=out,standardization=std,
                mix_expected_edits=sum(item['expected_test_edits'] for item in std))
    print(json.dumps(report['cohorts'],ensure_ascii=False),report['mix_expected_edits'],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/selected_tree_pair_transfer.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
