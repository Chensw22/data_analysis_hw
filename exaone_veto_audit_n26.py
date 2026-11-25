"""Verify a single exact EXAONE vote can rule out production sleeper or party edits."""
import json

import joblib
import numpy as np
import pandas as pd

from disjoint_tree_fields_n25 import ART,DEST
from export_budget_quarter_n23 import source
from exaone_validation_data import compact_features
from full_roster_booking_validation_n23 import ROOT,data
from party_negative_rescue import family_contrasts,risk
from sleeper_consensus_scope import cohort
from sleeper_manifest_risk import design


def branches(p108,can_lower,can_raise,exaone):
    base=p108>=.5
    lower=base&can_lower
    raise_it=~base&can_raise
    cannot_lower=lower&(exaone<.5)
    cannot_raise=raise_it&(exaone>=.5)
    return lower,raise_it,cannot_lower,cannot_raise


def audit(p108,can_lower,can_raise,exaone,v131):
    base=p108>=.5
    lower,raise_it,cannot_lower,cannot_raise=branches(p108,can_lower,can_raise,exaone)
    known=~(lower|raise_it)|cannot_lower|cannot_raise
    np.testing.assert_array_equal(v131[known],base[known])
    return dict(rows=len(base),possible_lower=int(lower.sum()),possible_raise=int(raise_it.sum()),
        vetoed_lower=int(cannot_lower.sum()),vetoed_raise=int(cannot_raise.sum()),total_vetoed=int((cannot_lower|cannot_raise).sum()),
        known_with_veto=int(known.sum()))


def run():
    raw,y,g,_=data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())
    roster=pd.concat([raw,test],ignore_index=True)
    features,controls,size,x=family_contrasts(roster)
    awake=x.CryoSleep.ne('True').to_numpy()
    sleeping=cohort(x)
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        rows,hf,old=z['rows'],z['fold'],z['v131']
    with np.load(ART/'sleeper_manifest_risk.npz') as z:
        np.testing.assert_array_equal(z['dev'],rows)
        np.testing.assert_array_equal(z['fold'],hf)
        witness=z['manifest_berth_party_a']
    with np.load(ART/'v108_exact_oof/oof.npz') as z:
        np.testing.assert_array_equal(z['dev_ids'],rows)
        p=z['v108']
    with np.load(ART/'run_054/EXAONE_v91_CAST_default8_seed20261112.npz') as z:
        for name,value in [('dev_ids',rows),('fold_id',hf)]:
            np.testing.assert_array_equal(z[name],value)
        e=z['prob']
    family=np.empty(len(rows))
    for k in np.unique(hf):
        b=hf==k; q=rows[b];t=rows[~b&~np.isin(g[rows],g[q])]
        family[b]=risk(features,controls,y,t,q,size,awake)[:,0]
    result={'historical':audit(p,sleeping[rows]&~witness,awake[rows]&(size[rows]>1)&(family>=.60),e,old)}
    q=np.arange(len(raw),len(roster))
    predicted=source('proposal_online_first_exaone_fixed',test)[0]
    p=source('proposal_v108_party_bills',test)[0]
    prior=source('proposal_v131_party_consensus',test)[1]
    witness=joblib.load(ART/'proposal_v128_manifest_confirmation/model.joblib').predict_proba(design(compact_features(test),True))[:,1]
    family=risk(features,controls,y,np.arange(len(y)),q,size,awake)[:,0]
    result['competition']=audit(p,sleeping[q]&(witness<.5),awake[q]&(size[q]>1)&(family>=.60),predicted,prior)
    DEST.with_name('exaone_veto_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)


if __name__=='__main__':
    run()
