"""Freeze the two-panel booking-resampled correction to the v96 score, preserving v131 overrides."""
import json

import numpy as np
import pandas as pd
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits

from full_roster_edit_resampling_n24 import ROOT,calibrate,inputs,shift,subsample,test_inputs
from kaggle_workflow import sha256,validate_submission
from relational_error_corrector import write

VERSION = 'n24_bagged_net_edits'
NAME = 'proposal_'+VERSION
SEEDS = (0,1443,1444,1445)


def train():
    raw,y,g,fold,_,_,_,sets = inputs()
    result = []
    for seed in SEEDS:
        models = []
        for j,(panel,p,x) in enumerate(sets):
            for f in np.unique(fold):
                q = np.flatnonzero(fold==f)
                t = np.flatnonzero((panel==panel[q[0]]) & (fold!=f))
                t = subsample(t,g,seed+f+9*j if seed else 0)
                assert not np.isin(g[t],g[q]).any()
                models.append(calibrate(x,p,y,g,t))
        result.append(models)
    return dict(version=VERSION,by_seed=result,seeds=SEEDS,training_rows=len(raw)),g


def predict(model,query,p,incumbent,champ):
    assert len(model['by_seed'])==4 and all(len(models)==16 for models in model['by_seed'])
    estimates = [np.mean([shift(m,query) for m in block],axis=0) for block in model['by_seed']]
    correction = np.mean(estimates,axis=0)
    free = (p>=.5)==champ
    return np.where(free,expit(logit(p.clip(.003,.997))+correction),incumbent)


def export():
    model,g = train()
    test,query,p,incumbent,champ,parents = test_inputs(g)
    report = ROOT/'reports'/NAME
    out = ROOT/'artifacts'/NAME
    model_file = out/'calibrators.json'
    write(model_file,model)
    scores = predict(model,query,p,incumbent,champ)
    reloaded = predict(json.loads(model_file.read_text()),query,p,incumbent,champ)
    np.testing.assert_array_equal(scores,reloaded)
    assert np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
    hard = scores>=.5
    changed = hard!=champ
    evidence = ROOT/'reports/nov23_matched_learning_curve/edit_resampling.json'
    benchmark = json.loads(evidence.read_text())['bagged']
    assert changed.sum()==benchmark['query_changes']==41 and (hard & changed).sum()==benchmark['query_to_true']==31
    assert not changed[(p>=.5)!=champ].any()
    csv = out/'submission.csv'
    pd.DataFrame(dict(PassengerId=test.PassengerId,Transported=hard)).to_csv(csv,index=False)
    pd.DataFrame(dict(PassengerId=test.PassengerId,probability=scores,reload_probability=reloaded)
                 ).to_csv(out/'test_probabilities.csv',index=False)
    checked = validate_submission(csv,ROOT/'data/raw/test.csv')
    distances = {}
    def distance(name,digest):
        previous = ROOT/'artifacts'/name/'submission.csv'
        assert sha256(previous)==digest
        df = pd.read_csv(previous)
        np.testing.assert_array_equal(df.PassengerId,test.PassengerId)
        return int(np.count_nonzero(df.Transported!=hard))
    for receipt in (ROOT/'reports').glob('*/online_attempts/*.json'):
        record = json.loads(receipt.read_text())
        if record.get('evaluation',{}).get('Status')=='COMPLETE' and receipt.parents[1].name!=NAME:
            distances[receipt.parents[1].name] = distance(receipt.parents[1].name,record['file_sha256'])
    pending = {}
    for name in ('proposal_n24_selected_dissent','proposal_n24_solo_random_nine'):
        frozen = json.loads((ROOT/'reports'/name/'manifest.json').read_text())
        pending[name] = distance(name,frozen['sha256'])
    nearest = min(distances.values())
    assert nearest>0 and min(pending.values())>0
    manifest = dict(**checked,version=VERSION,model_sha256=sha256(model_file),training_rows=len(g),parents=parents,
             reload_difference=0,changes_vs_v131=int(changed.sum()),changed_to_true=int((changed & hard).sum()),
             changed_to_false=int((changed & ~hard).sum()),models=64,seeds=SEEDS,completed_exports=len(distances),
             closest_completed_changes=nearest,closest_completed=[k for k,v in distances.items() if v==nearest],
             distance_to_queued=pending)
    selection = dict(version=VERSION,threshold=.5,method='Mean log-odds correction from four samples of two panels, eight fold calibrators each.',
             preserve='Keep v131 whenever it differs from v96.',evidence=str(evidence.relative_to(ROOT)),evidence_sha256=sha256(evidence))
    for directory in (out,report):
        write(directory/'manifest.json',manifest)
        write(directory/'selection.json',selection)
    print(manifest)


if __name__=='__main__':
    with threadpool_limits(1):
        export()
