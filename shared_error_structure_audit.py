"""Descriptive profiles of existing OOF shared errors; no fitted prediction rule."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import audit_top5_references as audit
import exaone_validation_data as data

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'reports/error_rescue_20261117'
LOCAL = ROOT/'artifacts/error_rescue_20261117'
ALIGNED = ROOT/'artifacts/data_error_diagnosis_20261117/aligned_rows.csv'


def main():
    previous = json.loads((ROOT/'reports/data_error_diagnosis_20261117/gating_feasibility.json').read_text())
    expected = previous['alignment_review']['sources'][str(ALIGNED.relative_to(ROOT))]
    if hashlib.sha256(ALIGNED.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Previously audited OOF table changed.')
    inputs = data.load_data()
    rows = pd.read_csv(ALIGNED, dtype={'PassengerId':str})
    assert np.array_equal(rows.row_id, inputs['dev']) and np.array_equal(rows.y, inputs['y'])
    assert np.array_equal(rows.PassengerId, inputs['passengers'])
    train, test, _ = audit.read_development_only(ROOT/'data/raw/train.csv', ROOT/'data/raw/test.csv', inputs['dev'])
    raw = pd.concat([train, test], ignore_index=True)
    group = raw.PassengerId.str.split('_').str[0]
    counts = group.map(group.value_counts())
    cabin_count = raw.Cabin.map(raw.Cabin.value_counts())
    sleeping = raw.CryoSleep.astype(float)
    known = sleeping.notna().astype(int)
    denom = known.groupby(group).transform('sum')-known
    num = sleeping.fillna(0).groupby(group).transform('sum')-sleeping.fillna(0)
    peer_rate = num/denom.replace(0,np.nan)
    frame = raw.iloc[inputs['dev']].reset_index(drop=True)
    family = ['v44','v81','TabDPT','Kumo','EXAONE']
    wrong = np.column_stack([(rows['p_'+m].to_numpy()>=.5)!=rows.y.to_numpy() for m in family])
    hard = wrong.all(axis=1); easy = ~wrong.any(axis=1)
    assert hard.sum()==914 and easy.sum()==5263
    group_size = counts.iloc[inputs['dev']].to_numpy()
    cabin_size = cabin_count.iloc[inputs['dev']].to_numpy()
    peers = peer_rate.iloc[inputs['dev']].to_numpy()
    sleeping_slice = frame.CryoSleep.fillna('missing').astype(str)
    planet = frame.HomePlanet.fillna('missing').astype(str)
    deck = frame.Cabin.str.split('/').str[0].fillna('missing')
    bills = frame[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']]
    spending = np.where(bills.isna().any(axis=1),'missing',np.where(bills.sum(axis=1)==0,'zero','positive'))
    group_bin = np.where(group_size==1,'solo',np.where(group_size==2,'2',np.where(group_size<=4,'3-4','5+')))
    cabin_bin = np.where(np.isnan(cabin_size),'missing',np.where(cabin_size==1,'solo','shared'))
    peer_bin = np.where(np.isnan(peers),'no_known_peers',np.where(peers==0,'all_awake',np.where(peers==1,'all_asleep','mixed')))
    slices = {'HomePlanet':planet,'CryoSleep':sleeping_slice,'Deck':deck,'spending':spending,
              'GroupSize':group_bin,'CabinOccupancy':cabin_bin,'PeerCryo':peer_bin,
              'planet_sleep':planet+' / '+sleeping_slice,
              'planet_spending':planet+' / '+pd.Series(spending),
              'sleep_peer':sleeping_slice+' / '+pd.Series(peer_bin),
              'planet_group':planet+' / '+pd.Series(group_bin)}
    profiles=[]
    for name, values in slices.items():
        values=np.asarray(values)
        for value in sorted(set(values)):
            mask=values==value;n=int(mask.sum())
            profiles.append(dict(slice=name,value=value,rows=n,shared_errors=int(hard[mask].sum()),
                shared_error_rate=float(hard[mask].mean()),share_of_all_shared_errors=float(hard[mask].sum()/hard.sum()),
                v81_errors=int(wrong[mask,1].sum()),v81_accuracy=float(1-wrong[mask,1].mean()),
                all_correct=int(easy[mask].sum()),target_rate=float(rows.y.to_numpy()[mask].mean())))
    tr_groups=set(train.PassengerId.str.split('_').str[0]);te_groups=test.PassengerId.str.split('_').str[0]
    coverage=dict(test_rows=len(test),test_with_training_group=int(te_groups.isin(tr_groups).sum()),
        test_with_training_cabin=int(test.Cabin.isin(set(train.Cabin.dropna())).sum()),
        interpretation='No training-label neighbor is available for any test row through exact group or cabin; random-fold label propagation would have a different coverage regime.')
    OUT.mkdir(parents=True,exist_ok=True);LOCAL.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(profiles).to_csv(OUT/'structure_slices.csv',index=False)
    samples=pd.concat([rows[['row_id','PassengerId','fold','y','family_wrong_count']],
        frame.drop(columns=['PassengerId','Name','Transported'],errors='ignore')],axis=1)
    samples['GroupSize']=group_size;samples['CabinOccupants']=cabin_size;samples['PeerCryoSleepRate']=peers
    for name in family:samples['p_'+name]=rows['p_'+name]
    samples.loc[hard].to_csv(LOCAL/'shared_error_cases.csv',index=False)
    report=dict(scope='Descriptive reused-development error structure; no accuracy selection or model fit',
        rows=len(rows),shared_errors=int(hard.sum()),all_five_correct=int(easy.sum()),
        test_label_neighbor_coverage=coverage,
        shared_errors_with_group_peers=int((hard & (group_size>1)).sum()),
        shared_errors_with_known_peer_cryo=int((hard & ~np.isnan(peers)).sum()),
        shared_errors_without_group_peers=int((hard & (group_size==1)).sum()),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),aligned_sha256=expected,
        limitations=['Slice differences do not prove a causal error mechanism or label corruption.',
          'v96/v99 lack OOF and test labels; their full-training fitted predictions must not be presented as error repair evaluation.',
          'Raw row-level cases remain in ignored artifacts; public reports contain aggregates only.'])
    (OUT/'structure_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['limitations','source_sha256','aligned_sha256']},indent=2))


if __name__=='__main__':main()
