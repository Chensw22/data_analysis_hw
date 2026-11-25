"""Reconstruct the same 99 expert opinions on the unlabelled competition passengers."""
import json
import numpy as np
import pandas as pd
from export_consensus_review_n26 import ART, BAND, FEATURES, dig
from export_six_way_dispute_n26 import NAME as SIX_NAME, PARENT as FARE_NAME, combine as six_model
from full_roster_booking_validation_n23 import ROOT, data
from held_expert_portfolios_n26 import NEW, SIX
from price_implied_deck_n26 import build_features, OUT as DECK
from solo_selection_likelihood_n26 import fit_model
from tabicl_consensus_review_n26 import OUT as VALIDATION

OUT=ART/'nov26_competition_expert_panel'
REPORT=ROOT/'reports/nov23_matched_learning_curve/competition_expert_panel.json'


def frozen(directory,filename):
    path=ART/directory/filename
    manifest=json.loads((ROOT/'reports'/directory/'manifest.json').read_text())
    assert dig(path)==manifest['model_sha256']
    return np.load(path)


def run():
    raw,y,g,_=data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())
    people=pd.concat([raw,test],ignore_index=True)
    ids=test.PassengerId.to_numpy(str)
    groups=people.PassengerId.str[:4].to_numpy()
    assert not np.isin(g,groups[len(y):]).any()
    with np.load(VALIDATION/'full.npz') as z:
        names=z['names'].tolist()
    with frozen(BAND,'frozen_votes.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        fine=z['p']
        columns=z['names'].tolist()
    tags=np.array([s.split('/')[0] for s in columns])
    parts=[np.column_stack([fine[:,tags==n].mean(1) for n in names[:18]]),fine[:,[columns.index(n) for n in names[18:80]]]]
    input_hashes=json.loads((ART/'proposal_n26_consensus_review/input_hashes.json').read_text())
    extras=[]
    for name in ('bills','context'):
        file=FEATURES/f'{name}.npz'
        assert dig(file)==input_hashes[name]
        with np.load(file) as z:
            np.testing.assert_array_equal(z['passenger'],ids)
            extras.extend(z['names'].tolist())
            parts.append(z['p'])
    np.testing.assert_array_equal(extras,names[80:])
    p={}
    with frozen(FARE_NAME,'scores.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        p.update(neural_deck_raw=z['raw'],neural_deck_fine=z['fine'])
    with frozen(SIX_NAME,'predictions.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        for j,k in enumerate(['register_all','register_paid','sequence_paid','tree_route_exact','swap','neural_route_exact']):
            p[k]=z['challengers'][:,j]
        np.testing.assert_array_equal(parts[0]>=.5,z['family_votes'])
        predicted=six_model(z['parent'],np.column_stack([p[n] for n in SIX]),z['family_votes'],z['eligible'])>=.5
    official=pd.read_csv(ART/SIX_NAME/'submission.csv')
    np.testing.assert_array_equal(official.PassengerId,ids)
    np.testing.assert_array_equal(official.Transported,predicted)
    frame=build_features(people,groups)
    errors={}
    with np.load(DECK/'tree_full_0.npz') as z:
        for k in ('raw','fine'):
            x=frame[k]
            t,q=z['t'],z['q']
            replay=fit_model(x.iloc[t],y[t]).predict_proba(x.iloc[q])[:,1]
            errors[k]=float(np.max(abs(replay-z[k])))
            np.testing.assert_allclose(replay,z[k],atol=1.e-12,rtol=0)
            p['tree_deck_'+k]=fit_model(x.iloc[:len(y)],y).predict_proba(x.iloc[len(y):])[:,1]
    scores=np.column_stack([*parts,*[p[n] for n in NEW]])
    assert scores.shape==(len(ids),99) and np.isfinite(scores).all()
    dissent=(scores>=.5)!=predicted[:,None]
    OUT.mkdir(exist_ok=True)
    target=OUT/'scores.npz'
    np.savez_compressed(target,passenger=ids,scores=scores,names=[*names,*NEW],prediction=predicted)
    r=dict(rows=len(ids),columns=scores.shape[1],trained_extra_trees=2,fold_zero_replay_max=errors,
           sha256=dig(target),no_dissent=int((~dissent.any(1)).sum()),one_or_more_dissent=int(dissent.any(1).sum()),
           newly_contested_by_ten=int((~dissent[:,:89].any(1)&dissent[:,89:].any(1)).sum()))
    REPORT.write_text(json.dumps(r,indent=2)+'\n')
    print(r,flush=True)


if __name__=='__main__':
    run()
