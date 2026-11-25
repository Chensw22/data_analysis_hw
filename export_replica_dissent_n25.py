"""Freeze the independently replayed half-roster filter for the scored tree on structurally certified orders."""
import hashlib
import json

import numpy as np
import pandas as pd

from export_budget_quarter_n23 import source
from kaggle_workflow import sha256,validate_submission
from full_roster_booking_validation_n23 import ROOT,data
from relational_error_corrector import write

NAME='proposal_n25_replica_supported_dissent'
PARENT='proposal_n25_certified_tree'
BEST='proposal_n24_selected_dissent'
PANELS=('original','rotated')
# Separate, immutable assignments mimic the two out-of-fold half-roster predictions.
SALTS=(251101,251102)


def membership(groups,salt):
    return np.array([hashlib.sha256(f'{salt}/{g}'.encode()).digest()[0]%2 for g in groups])


def predict(spec,test,groups,baseline,parent):
    sources=[]
    for j,panel in enumerate(PANELS):
        scores=[]
        for part in (0,1):
            key=f'{panel}_{part}'
            path=ROOT/spec['components'][key]['file']
            assert sha256(path)==spec['components'][key]['sha256']
            with np.load(path) as z:
                np.testing.assert_array_equal(z['ids'],test.PassengerId)
                scores.append(z['prediction'])
        sources.append(np.stack(scores))
    assigned=np.column_stack([panel[membership(groups,salt),np.arange(len(test))]
                              for panel,salt in zip(sources,spec['salts'],strict=True)])
    inc=baseline>=.5
    def apply(a,b):
        support=np.where(inc,np.minimum(a,b)>=.5,np.maximum(a,b)<=.5)
        return np.where(support,parent,baseline)
    p=apply(assigned[:,0],assigned[:,1])
    sensitivity=[(apply(a,b)>=.5)!=inc for a in sources[0] for b in sources[1]]
    return p,np.stack(sensitivity)


def run():
    raw,y,g,f=data()
    test_file=ROOT/'data/raw/test.csv'
    test=pd.read_csv(test_file,usecols=lambda c:c!='Name')
    groups=test.PassengerId.str.split('_').str[0].to_numpy()
    assert not np.isin(g,groups).any()
    best,inc,ih=source(BEST,test)
    parent,expanded,eh=source(PARENT,test)
    assert (expanded!=inc).sum()==67
    components={}
    for j,panel in enumerate(PANELS):
        fold=(f%2,(f//2)%2)[j]
        for part in (0,1):
            path=ROOT/'artifacts/nov25_mitra_replica_queries'/f'{panel}_{part}.npz'
            with np.load(path) as z:
                t=np.flatnonzero(fold!=part)
                np.testing.assert_array_equal(z['t'],t)
                np.testing.assert_array_equal(z['truth'],y[t])
            components[f'{panel}_{part}']=dict(file=str(path.relative_to(ROOT)),sha256=sha256(path))
    spec=dict(components=components,salts=list(SALTS),parents={BEST:ih,PARENT:eh},filter='Both half-trained replicas support the incumbent or abstain at p=0.5.')
    out=ROOT/'artifacts'/NAME
    out.mkdir(exist_ok=True)
    model=out/'composition.json'
    write(model,spec)
    p,sensitivity=predict(spec,test,groups,best,parent)
    reloaded,again=predict(json.loads(model.read_text()),test,groups,best,parent)
    np.testing.assert_array_equal(p,reloaded)
    np.testing.assert_array_equal(sensitivity,again)
    hard=p>=.5
    assert np.isfinite(p).all() and np.all(~(hard!=inc)|(expanded!=inc))
    assert 0<(hard!=inc).sum()<67
    pd.DataFrame(dict(PassengerId=test.PassengerId,Transported=hard)).to_csv(out/'submission.csv',index=False)
    pd.DataFrame(dict(PassengerId=test.PassengerId,probability=p,reload_probability=reloaded)).to_csv(out/'test_probabilities.csv',index=False)
    check=validate_submission(out/'submission.csv',test_file)
    old={json.loads(r.read_text()).get('file_sha256') for r in (ROOT/'reports').glob('*/online_attempts/*.json')}
    assert check['sha256'] not in old
    selected=(hard!=inc)
    summary=dict(**check,version=NAME.removeprefix('proposal_'),parents=spec['parents'],model_sha256=sha256(model),
                 training_rows=len(y),component_training_rows=[int((f%2!=part).sum()) for part in (0,1)]+
                    [int(((f//2)%2!=part).sum()) for part in (0,1)],reload_difference=0,
                 edits_vs_best=int(selected.sum()),original_edits=67,edits_to_true=int((selected&hard).sum()),
                 alternative_fixed_assignments=sensitivity.sum(1).tolist(),
                 accepted_on_every_assignment=int(sensitivity.all(0).sum()),
                 selected_supported_by_all=int((sensitivity.all(0)&selected).sum()))
    for folder in (out,ROOT/'reports'/NAME):
        write(folder/'manifest.json',summary)
        write(folder/'selection.json',dict(version=summary['version'],threshold=.5,rule=spec['filter'],
            validation='reports/nov23_matched_learning_curve/certified_tree_release.json',policy='replica_filter'))
    print(summary,flush=True)


if __name__=='__main__':
    run()
