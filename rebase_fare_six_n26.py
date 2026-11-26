"""Apply the frozen fare and six-expert rules directly to the publicly scored nine-vote incumbent."""
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from arbitration_against_scored_nine_n26 import champions
from exhaustive_family_quorums_n26 import summary
from export_consensus_review_n26 import ART, dig, write
from forecast_frozen_queue_n26 import NAMES
from held_expert_portfolios_n26 import DECK, ROOT, SIX, fare_model, panel_predictions, six_model, sources
from kaggle_workflow import validate_submission
from six_way_against_scored_nine_n26 import test_decisions

NAME = 'proposal_n26_champion_fare_six'
REPORT = ROOT/'reports/nov23_matched_learning_curve/rebase_fare_six.json'


def predict(anchor, raw, fine, six, families, eligible):
    fare = fare_model(anchor, raw, fine, eligible)
    joint = six_model(fare, six, families, eligible)
    direct = six_model(anchor, six, families, eligible)
    return {'fare':fare, 'six_alone':direct, 'joint':joint}


def evaluate():
    raw, y, g, views = sources()
    parent, known, hist, hi = champions()
    np.testing.assert_array_equal(views['historical']['ids'], hi)
    new = ~np.isin(g, g[hi])
    result = {}
    with np.load(DECK/'deck_estimates.npz') as z:
        eligible = z['eligible']
    for tag, v in views.items():
        ids = v['ids']
        p = panel_predictions(v, tag)
        a = (parent if tag=='full' else hist)[ids]
        mask = known[ids] if tag=='full' else np.ones(len(ids),bool)
        scores = predict(a,p['neural_deck_raw'],p['neural_deck_fine'],
                         np.column_stack([p[n] for n in SIX]),v['votes'][:,:18],eligible[ids])
        scopes = {'all':mask}
        if tag=='full':
            scopes['outside_historical_orders'] = mask & new[ids]
            scopes['earlier_releases_agree'] = mask & v['common'] & (v['anchor']==a)
        result[tag] = {}
        for key, probability in scores.items():
            b = probability>=.5
            result[tag][key] = {n:summary(b!=a, a!=y[ids], m, v['fold']) for n,m in scopes.items()}
    return result


def export():
    ids = pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId']).PassengerId
    keys = ['nine','review','fare','six']
    originals = {k:test_decisions(NAMES[k],ids) for k in keys}
    inputs = {}
    hashes = {}
    for label, key, file, cols in [('fare','fare','scores.npz',['eligible','raw','fine']),
                                    ('six','six','predictions.npz',['eligible','challengers','family_votes'])]:
        path = ART/NAMES[key]/file
        hashes[label] = dig(path)
        manifest = json.loads((ROOT/'reports'/NAMES[key]/'manifest.json').read_text())
        assert hashes[label] == manifest['model_sha256']
        with np.load(path) as z:
            np.testing.assert_array_equal(z['passenger'],ids)
            for col in cols:
                if col in inputs:
                    np.testing.assert_array_equal(inputs[col],z[col])
                inputs[col] = z[col]
    assert inputs['challengers'].shape==(len(ids),6) and inputs['family_votes'].shape==(len(ids),18)
    def run(anchor):
        return predict(anchor,inputs['raw'],inputs['fine'],inputs['challengers'],inputs['family_votes'],inputs['eligible'])
    replay = run(originals['review'][0])
    np.testing.assert_array_equal(replay['fare']>=.5,originals['fare'][0])
    np.testing.assert_array_equal(replay['joint']>=.5,originals['six'][0])
    probabilities = run(originals['nine'][0])
    out = ART/NAME
    out.mkdir(exist_ok=True)
    model = out/'scores.npz'
    np.savez_compressed(model,passenger=ids.to_numpy(str),anchor=originals['nine'][0],**inputs)
    with np.load(model) as z:
        rebuilt = predict(z['anchor'],z['raw'],z['fine'],z['challengers'],z['family_votes'],z['eligible'])['joint']
    p = probabilities['joint']
    np.testing.assert_array_equal(p,rebuilt)
    assert np.isfinite(p).all() and 0<=p.min()<=p.max()<=1
    csv = out/'submission.csv'
    pd.DataFrame(dict(PassengerId=ids,Transported=p>=.5)).to_csv(csv,index=False)
    pd.DataFrame(dict(PassengerId=ids,probability=p,reload_probability=rebuilt)).to_csv(out/'test_probabilities.csv',index=False)
    check = validate_submission(csv,ROOT/'data/raw/test.csv')
    previous = [json.loads(f.read_text())['file_sha256'] for f in (ROOT/'reports').glob('*/online_attempts/*.json')]
    assert check['sha256'] not in previous
    changes = {k:int(((v>=.5)!=originals['nine'][0]).sum()) for k,v in probabilities.items()}
    manifest = dict(**check,version='n26_champion_fare_six',training_rows=8693,parents={NAMES[k]:originals[k][1] for k in keys},
                    input_artifact_hashes=hashes,model_sha256=dig(model),reload_difference=0,changes_from_champion=changes)
    selection = dict(version=manifest['version'],rule='Apply the two original fare and six-expert decisions to the scored nine-vote incumbent without inheriting the earlier dispute policies.',
                     evidence=str(REPORT.relative_to(ROOT)),origin='Exploratory after seeing the two stages improve versus their actual predecessors online.')
    for folder in (out, ROOT/'reports'/NAME):
        folder.mkdir(exist_ok=True)
        write(folder/'manifest.json',manifest)
        write(folder/'selection.json',selection)
    return manifest


if __name__=='__main__':
    with threadpool_limits(1):
        local=evaluate()
        manifest=export()
    write(REPORT,dict(local=local,test=manifest))
    print({tag:{n:(v['all']['fixed'],v['all']['harmed']) for n,v in d.items()} for tag,d in local.items()})
    print(manifest['changes_from_champion'])
