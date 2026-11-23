"""Locate unrecovered shared errors and compare them with correct unanimous votes in the same observable states."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from broad_geographic_transfer_n23 import BILLS, load
from broad_matched_model_changes_n23 import read_bank
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write


def main():
    row,y,frame,groups,_ = load()
    with np.load(ART/'nov23_release_vote_information/panel.npz') as panel:
        np.testing.assert_array_equal(row,panel['row'])
        np.testing.assert_array_equal(y,panel['y'])
        votes=panel['votes']
        unanimous=(votes==votes[0,:,0][None,:,None]).all((0,2))
        anchor=votes[0,:,0]
        wrong=unanimous&(anchor!=y)
        right=unanimous&(anchor==y)
    bank, blended=[], []
    for seed in map(str,SEEDS):
        with np.load(ART/f'nov23_joint_observed_state/{seed}.npz') as z:
            np.testing.assert_array_equal(row,z['row'])
            np.testing.assert_array_equal(y,z['y'])
            hist = read_bank(row,y,z['fold'],seed).reshape(len(row),-1)
            new = np.column_stack([z[key] for key in ('sleep_conditional','sleep_direct','basket_conditional','basket_direct')])
            bank.append((np.c_[hist,new]>=.5)==y[:,None])
            joint=.75*z['base']+.25*z['basket_conditional']
            blended.append(np.where((z['base']>=.5)==z['ref'],joint>=.5,z['ref'])==y)
    hits=np.stack(bank)
    stable=hits.all(0).any(1)
    any_hit=hits.any((0,2))
    risk=frame[['VRDeck','Spa','RoomService']].gt(0).any(axis=1)
    paid=frame[BILLS].gt(0).any(axis=1)
    state=pd.Series(np.where(risk,'room_spa_or_vr',np.where(paid,'food_or_shop_only','no_positive_bill')))
    cross=pd.DataFrame(dict(state=state,sleep=frame.CryoSleep,planet=frame.HomePlanet,decision=anchor))
    def counts(mask):
        return dict(common_wrong=int((mask&wrong).sum()), stable_rescue=int((mask&wrong&stable).sum()),
                    no_stable_rescue=int((mask&wrong&~stable).sum()),never_correct=int((mask&wrong&~any_hit).sum()),
                    quarter_basket_correct=int((mask&wrong&np.all(blended,axis=0)).sum()),common_right=int((mask&right).sum()))
    cells={}
    for (s,d),keys in cross.groupby(['state','decision']).indices.items():
        cells[f'{s}__pred_{int(d)}']=counts(np.isin(np.arange(len(y)),keys))
    # Evaluate a predeclared majority correction in the matched unanimous population only.
    # All booking groups scored here are held out from estimation of that cell's error frequency.
    grouped={}
    ids=np.flatnonzero(unanimous)
    for fields in [['state','decision'],['state','decision','sleep'],['state','decision','sleep','planet']]:
        codes=cross.groupby(fields,dropna=False).ngroup().to_numpy()
        n=codes.max()+1
        pred=anchor.copy()
        for t,q in GroupKFold(5,shuffle=True,random_state=717).split(ids,groups=groups[ids]):
            t,q=ids[t],ids[q]
            assert not np.intersect1d(groups[t],groups[q]).size
            wrong_count=np.bincount(codes[t],weights=wrong[t],minlength=n)
            population=np.bincount(codes[t],minlength=n)
            rate=(wrong_count+12*wrong[t].mean())/(population+12)
            switch=q[rate[codes[q]]>.5]
            pred[switch]=~pred[switch]
        grouped['+'.join(fields)]=paired(pred[ids]==y[ids],anchor[ids]==y[ids])
    report=dict(total=counts(np.ones(len(y),bool)),cells=cells,calibrated=grouped,
                models=hits.shape[2],stable_shared_right_loss=int((right&(~hits).all(0).any(1)).sum()))
    write(ROOT/'reports/nov23_adaptive_manifest_splines/common_census.json',report)
    print(report)


if __name__=='__main__':main()
