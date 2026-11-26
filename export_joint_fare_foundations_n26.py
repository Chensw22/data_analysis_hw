"""Keep the 111-change base and add only paid passengers opposed by both fare foundation models."""
import json
import numpy as np
import pandas as pd

from export_consensus_review_n26 import ART, dig, write
from forecast_frozen_queue_n26 import NAMES
from full_roster_booking_validation_n23 import ROOT, data
from kaggle_workflow import validate_submission
from nori_fare_party_n26 import ART as NORI, SHA
from paired_fare_foundations_n26 import policies
from rebase_fare_six_n26 import NAME as BASE
from six_way_against_scored_nine_n26 import test_decisions

NAME='proposal_n26_joint_fare_foundations'


def combine(parent,champion,eligible,raw,fine):
    hard=parent>=.5
    changes=policies(hard,eligible&(hard==champion),raw,fine)['fare_pair']
    probability=np.where(changes,fine.clip(0,1).mean(1),parent)
    np.testing.assert_array_equal(probability>=.5,hard^changes)
    return probability,changes


def run():
    raw_train,y,g,_=data()
    ids=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId']).PassengerId
    test_orders=ids.str.split('_').str[0]
    assert not np.isin(g,test_orders).any()
    q=np.arange(len(y),len(y)+len(ids))
    champion,champion_sha=test_decisions(NAMES['nine'],ids)
    base,base_sha=test_decisions(BASE,ids)
    base_prob=pd.read_csv(ART/BASE/'test_probabilities.csv',float_precision='round_trip')
    np.testing.assert_array_equal(base_prob.PassengerId,ids)
    parent=base_prob.probability.to_numpy()
    np.testing.assert_array_equal(parent,base_prob.reload_probability)
    np.testing.assert_array_equal(base,parent>=.5)
    fare_path=ART/NAMES['fare']/'scores.npz'
    fare_hash=dig(fare_path)
    assert fare_hash==json.loads((ROOT/'reports'/NAMES['fare']/'manifest.json').read_text())['model_sha256']
    with np.load(fare_path) as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        eligible,tab_raw,tab_fine=z['eligible'],z['raw'],z['fine']
    nori={}
    sources={'TabICL':fare_hash}
    for mode in ['raw','own']:
        path=NORI/f'{mode}_test.npz'
        sources['Nori_'+mode]=dig(path)
        with np.load(path) as z:
            for key,val in [('passenger',ids),('t',np.arange(len(y))),('labels',y),('q',q)]:
                np.testing.assert_array_equal(z[key],val)
            assert str(z['sha'])==SHA and np.isfinite(z['p']).all()
            nori[mode]=z['p']
    raw=np.column_stack([tab_raw,nori['raw']])
    fine=np.column_stack([tab_fine,nori['own']])
    args=dict(parent=parent,champion=champion,eligible=eligible,raw=raw,fine=fine)
    probability,changes=combine(**args)
    out=ART/NAME
    out.mkdir(exist_ok=True)
    model=out/'inputs.npz'
    np.savez_compressed(model,passenger=ids.to_numpy(str),**args)
    with np.load(model) as z:
        replay,_=combine(**{k:z[k] for k in args})
    np.testing.assert_array_equal(probability,replay)
    csv=out/'submission.csv'
    pd.DataFrame(dict(PassengerId=ids,Transported=probability>=.5)).to_csv(csv,index=False)
    pd.DataFrame(dict(PassengerId=ids,probability=probability,reload_probability=replay)).to_csv(out/'test_probabilities.csv',index=False)
    checked=validate_submission(csv,ROOT/'data/raw/test.csv')
    assert all(json.loads(f.read_text())['file_sha256']!=checked['sha256'] for f in (ROOT/'reports').glob('*/online_attempts/*.json'))
    added=int(changes.sum())
    total=int(((probability>=.5)!=champion).sum())
    assert total==added+int((base!=champion).sum())
    manifest=dict(**checked,version='n26_joint_fare_foundations',training_rows=len(raw_train),model_sha256=dig(model),
                  parents={BASE:base_sha,NAMES['nine']:champion_sha},inputs=sources,reload_difference=0,
                  extra_test_changes=added,total_changes_from_champion=total)
    selection=dict(version=manifest['version'],rule='On complete positive bills unchanged by the clean incumbent recipe, use a new decision only when both fare-conditioned TabICL and Nori predict it.',
                   evidence='reports/nov23_matched_learning_curve/paired_fare_foundations.json',
                   scope='Exploratory. The incremental gain is concentrated inside the earlier development population.')
    for folder in (ROOT/'reports'/NAME,out):
        folder.mkdir(exist_ok=True)
        write(folder/'manifest.json',manifest)
        write(folder/'selection.json',selection)
    print(manifest['sha256'],added,total)


if __name__=='__main__':
    run()
