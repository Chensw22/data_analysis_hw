"""Freeze separately calibrated positive and negative foundation dissent against the scored 111 file."""
import json
import numpy as np
import pandas as pd
from directional_foundation_transfer_n27 import OUT,calibrate
from export_consensus_review_n26 import ART,ROOT,dig,write
from full_roster_booking_validation_n23 import data
from kaggle_workflow import validate_submission
from rebase_fare_six_n26 import NAME as BASE
from six_way_against_scored_nine_n26 import test_decisions

NAME='proposal_n27_directional_foundation'


def apply(parent,scores,cuts):
    side=parent>=.5
    median=np.median(scores,axis=1)
    risk=np.where(side,1-median,median)
    action=(risk[:,None]>=cuts[:,side.astype(int)].T).sum(1)>len(cuts)/2
    p=np.where(action,median,parent)
    np.testing.assert_array_equal(p>=.5,side^action)
    return p


def run():
    raw,y,g,f=data()
    ids=pd.read_csv(ROOT/'data/raw/test.csv').PassengerId
    side,parent_sha=test_decisions(BASE,ids)
    prob=pd.read_csv(ART/BASE/'test_probabilities.csv',float_precision='round_trip')
    np.testing.assert_array_equal(prob.PassengerId,ids)
    np.testing.assert_array_equal(prob.probability>=.5,side)
    np.testing.assert_array_equal(prob.probability,prob.reload_probability)
    parent=prob.probability.to_numpy()
    panel_file=ART/'nov26_latent_passenger_difficulty/panel.npz'
    with np.load(panel_file) as z:
        np.testing.assert_array_equal(z['passenger'][:len(y)],raw.PassengerId)
        np.testing.assert_array_equal(z['passenger'][len(y):],ids)
        np.testing.assert_array_equal(z['fold'],f)
        np.testing.assert_array_equal(z['y'],y)
        np.testing.assert_array_equal(z['names'][-3:],['Mitra','TabICL fare','Nori fare'])
        known=z['known']
        full=z['scores'][:,-3:]
    with np.load(ART/'nov26_nori_fare_party/aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId)
        np.testing.assert_array_equal(z['known'],known)
        base=z['champion111'].astype(bool)
    scores=full[len(y):]
    med=np.median(full[:len(y)],axis=1)
    risk=np.where(base,1-med,med)
    cuts=np.stack([calibrate(risk,base,y,np.flatnonzero(known&(f==k)),1.) for k in range(8)])
    with np.load(OUT/'thresholds.npz') as z:
        np.testing.assert_array_equal(z['cuts'],cuts)
        np.testing.assert_array_equal(z['scores'],scores)
    dest=ART/NAME
    dest.mkdir(exist_ok=True)
    model=dest/'inputs.npz'
    np.savez_compressed(model,passenger=ids.to_numpy(str),parent=parent,scores=scores,cuts=cuts)
    p=apply(parent,scores,cuts)
    with np.load(model) as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        repeat=apply(z['parent'],z['scores'],z['cuts'])
    np.testing.assert_array_equal(p,repeat)
    with np.load(OUT/'test.npz') as z:
        np.testing.assert_array_equal(p>=.5,z['median_held_1'])
    csv=dest/'submission.csv'
    pd.DataFrame({'PassengerId':ids,'Transported':p>=.5}).to_csv(csv,index=False)
    pd.DataFrame({'PassengerId':ids,'probability':p,'reload_probability':repeat}).to_csv(dest/'test_probabilities.csv',index=False)
    check=validate_submission(csv,ROOT/'data/raw/test.csv')
    assert not any(json.loads(path.read_text())['file_sha256']==check['sha256'] and path.parents[1].name!=NAME
                   for path in (ROOT/'reports').glob('*/online_attempts/*.json'))
    edit=(p>=.5)!=side
    assert edit.sum()==16 and not side[edit].any()
    version='n27_directional_foundation'
    manifest=dict(**check,version=version,training_rows=len(y),parents={BASE:parent_sha},input_artifact_hashes={'panel':dig(panel_file)},
                  model_sha256=dig(model),reload_difference=0,changed=int(edit.sum()),changed_orders=int(ids[edit].str[:4].nunique()),
                  source_file='export_directional_foundation_n27.py')
    selection=dict(version=version,rule='When a majority of eight booking-fold threshold pairs approve the opposite median foundation decision, replace the scored 111 decision.',
                   evidence='reports/nov23_matched_learning_curve/directional_foundation_n27.json', fresh='Random 6 fixed/6 harmed, contiguous and repeated random 5/4.')
    for folder in [dest,ROOT/'reports'/NAME]:
        folder.mkdir(exist_ok=True)
        write(folder/'manifest.json',manifest)
        write(folder/'selection.json',selection)
    print(manifest)


if __name__=='__main__':run()
