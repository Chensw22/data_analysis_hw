"""Export a pair of 80%-training selected tree predictions with held-out-order replay checks."""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from export_budget_quarter_n23 import source
from export_selected_dissent_n24 import NAME as NINE
from full_roster_conditional_pool_n23 import ROOT,data
from full_roster_permutation_selection_n24 import ensemble_probability,feature_bank,fit_ensemble
from kaggle_workflow import sha256,validate_submission
from relational_error_corrector import write

NAME='proposal_n25_selected_tree_pair'
SEEDS=(11251,11252)


def assign(groups,seed):
    # All passengers on one future order use the same held-out fold model.
    return np.array([int.from_bytes(hashlib.sha256(f'{seed}/{g}'.encode()).digest()[:8],'big')%5 for g in groups])


def query_score(bundle,test_x,test_groups):
    fields=bundle['fields']
    x=test_x.reindex(columns=fields,fill_value=0).to_numpy()
    panels=[]
    for seed in SEEDS:
        which=assign(test_groups,seed)
        panel=np.full(len(x),np.nan)
        for fold,(columns,fits) in enumerate(bundle['models'][seed]):
            target=which==fold
            panel[target]=ensemble_probability(fits,x[target][:,columns])
        assert np.isfinite(panel).all()
        panels.append(panel)
    return np.mean(panels,axis=0)


def decisions(tree,anchor,champ,orig,champ_score):
    live=(champ==orig) & ((anchor>=.5)==orig) & ((tree>=.5)!=orig)
    # Absolute log odds are monotonic in absolute distance from 0.5.
    tree_strength=abs(tree.clip(.003,.997)-.5)
    anchor_strength=abs(anchor.clip(.003,.997)-.5)
    return np.where(live & (tree_strength>anchor_strength),tree,champ_score)


def export():
    raw,y,g,_=data()
    test_path=ROOT/'data/raw/test.csv'
    test=pd.read_csv(test_path,usecols=lambda col:col!='Name')
    tg=test.PassengerId.str.split('_').str[0].to_numpy()
    assert not np.isin(tg,g).any()
    x,fields,_=feature_bank(raw)
    tx,test_fields,_=feature_bank(test)
    tx=pd.DataFrame(tx,columns=test_fields)
    anchor,_,anchor_digest=source('proposal_online_first_counts15',test)
    _,orig,orig_digest=source('proposal_v131_party_consensus',test)
    champ_score,champ,champ_digest=source(NINE,test)
    bundle=dict(fields=fields,models={})
    out=ROOT/'artifacts'/NAME
    out.mkdir(exist_ok=True)
    for seed in SEEDS:
        bundle['models'][seed]=[]
        for f,(t,q) in enumerate(GroupKFold(5,shuffle=True,random_state=seed).split(x,groups=g)):
            path=ROOT/'artifacts/nov24_selected_tree_replay'/f'{seed}_{f}.npz'
            with np.load(path) as z:
                for key,value in [('t',t),('q',q),('truth',y[t]),('fields',fields)]:
                    np.testing.assert_array_equal(z[key],value)
                cols,held=z['columns'],z['p']
            models=fit_ensemble(x[t][:,cols],y[t],497)
            np.testing.assert_allclose(ensemble_probability(models,x[q][:,cols]),held,atol=1e-12,rtol=0)
            bundle['models'][seed].append((cols,models))
            print(seed,f,flush=True)
    model=out/'ensemble.joblib'
    joblib.dump(bundle,model)
    p=decisions(query_score(bundle,tx,tg),anchor,champ,orig,champ_score)
    replay=decisions(query_score(joblib.load(model),tx,tg),anchor,champ,orig,champ_score)
    np.testing.assert_array_equal(p,replay)
    assert ((p>=0)&(p<=1)).all() and np.all((p[champ!=orig]>=.5)==champ[champ!=orig])
    hard=p>=.5
    pd.DataFrame(dict(PassengerId=test.PassengerId,probability=p,reload_probability=replay)).to_csv(out/'test_probabilities.csv',index=False)
    pd.DataFrame(dict(PassengerId=test.PassengerId,Transported=hard)).to_csv(out/'submission.csv',index=False)
    check=validate_submission(out/'submission.csv',test_path)
    assert check['rows']==4277 and any(hard!=champ)
    old_digests={json.loads(f.read_text()).get('file_sha256') for f in (ROOT/'reports').glob('*/online_attempts/*.json')}
    assert check['sha256'] not in old_digests
    evidence='reports/nov23_matched_learning_curve/selected_tree_order_transfer.json'
    local=json.loads((ROOT/evidence).read_text())['11251+11252/mean']
    manifest=dict(**check,version=NAME.removeprefix('proposal_'),model_sha256=sha256(model),reload_difference=0,
                  training_rows=len(y),parents=dict(v96=anchor_digest,v131=orig_digest,nine=champ_digest),
                  edits_vs_best=int((hard!=champ).sum()),edits_vs_v131=int((hard!=orig).sum()),evidence=evidence)
    selection=dict(version=manifest['version'],seeds=list(SEEDS),folds=5,threshold=.5,local_net=local['net'],
                   decision='Keep nine-vote and v131 overrides. Adopt the opposite mean tree prediction when its absolute margin exceeds the anchor margin.',
                   assignment='First 8 SHA-256 bytes of seed/booking as unsigned big-endian modulo 5, per seed.')
    for folder in (out,ROOT/'reports'/NAME):
        write(folder/'selection.json',selection)
        write(folder/'manifest.json',manifest)
    print(manifest,flush=True)


if __name__=='__main__':
    with threadpool_limits(3):
        export()
