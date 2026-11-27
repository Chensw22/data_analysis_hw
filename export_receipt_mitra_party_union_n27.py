"""Freeze a single union of three independently motivated challengers to the official 16-change release."""
import json

import numpy as np
import pandas as pd
from scipy.special import expit,logit

from export_broad_unanimous_party_n27 import ballot
from export_consensus_review_n26 import ROOT,ART,write,dig
from full_roster_booking_validation_n23 import data
from kaggle_workflow import validate_submission
from price_implied_deck_n26 import FEES
from repeated_archive_misses_n26 import cohort,count
from six_way_against_scored_nine_n26 import test_decisions

NAME='proposal_n27_receipt_mitra_party_union'
BEST='proposal_n27_directional_foundation'
FIRST='proposal_n26_champion_fare_six'


def apply(z):
    base=z['parent']>=.5
    first=z['first']
    paid=z['eligible']
    mitra=z['mitra']
    specific=paid&(base==first)&((mitra[:,0]>=.5)==first)&((mitra[:,1]>=.5)!=first)&((mitra[:,2]>=.5)==first)
    specific &= ((z['other_fares']>=.5)!=first[:,None]).any(1)
    curve=z['curve']
    odds=lambda a:logit(a.clip(.005,.995))
    step=np.nan_to_num(odds(curve[:,1:].mean(1))-odds(curve[:,0]))
    sign=np.where(base,-1.,1.)
    moved=z['native'][:,None]+step[:,None]*z['weights'].mean(axis=1)
    counter=paid&((moved*sign[:,None])>0).all(1)
    peer,peer_score=ballot(z['parent'],z['global_scores'],z['experts'],z['opinions'])
    votes=np.column_stack([specific,counter,peer])
    signals=np.column_stack([mitra[:,1],expit(moved.mean(1)),peer_score])
    edited=votes.any(1)
    score=np.where(edited,np.where(votes,signals,0.).sum(1)/votes.sum(1).clip(1),z['parent'])
    np.testing.assert_array_equal(score>=.5,base^edited)
    return votes,score


def run():
    raw,y,g,f=data();n=len(y)
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns)
    frame=pd.concat([raw,test],ignore_index=True);ids=frame.PassengerId.to_numpy(str)
    _,blind,known,fresh=cohort(raw,y,f)
    hard,sha=test_decisions(BEST,test.PassengerId)
    prev,first_sha=test_decisions(FIRST,test.PassengerId)
    parent_prob=pd.read_csv(ART/BEST/'test_probabilities.csv',float_precision='round_trip')
    np.testing.assert_array_equal(ids[n:],parent_prob.PassengerId)
    np.testing.assert_array_equal(hard,parent_prob.probability>=.5)
    sources=dict(anchor=ART/'nov27_dissent_likelihood_transfer/aligned.npz',mitra=ART/'nov27_mitra_fare/aligned.npz',
        panel=ART/'nov26_latent_passenger_difficulty/panel.npz',expert=ART/'nov27_broad_population_expert_crosscheck/aligned.npz',
        expert_actions=ART/'nov27_broad_population_expert_crosscheck/unanimous_party_actions.npz')
    with np.load(sources['anchor']) as z:
        np.testing.assert_array_equal(ids,z['passenger']);first=z['first'];fixed=first^z['random_median'];alternate=first^z['contiguous_median']
    np.testing.assert_array_equal(first[n:],prev);np.testing.assert_array_equal(fixed[n:],hard)
    with np.load(sources['panel']) as z:
        np.testing.assert_array_equal(ids,z['passenger']);other=z['scores'][:,-2:]
    with np.load(sources['mitra']) as z:
        np.testing.assert_array_equal(ids,z['ids']);mitra=np.column_stack([z[k] for k in ['raw','fare','shuffled']]);mm=z['111_specific_with_support']
    with np.load(sources['expert_actions']) as z:
        np.testing.assert_array_equal(ids,z['ids']);np.testing.assert_array_equal(fixed,z['parent']);opin=z['opinions']
        outcomes={mode:z[mode] for mode in ('random','ordered')}
    fee=frame[FEES]
    paid=fee.notna().all(axis=1).to_numpy()&fee.gt(0).any(axis=1).to_numpy()
    dest=ART/NAME;dest.mkdir(exist_ok=True)
    local={};pack=None;votes=None
    for mode,em in [('random','random'),('ordered','contiguous')]:
        path=ART/'nov27_complete_receipt_counterfactual'/f'aligned_{mode}_2.npz'
        sources['counter_'+mode]=path
        with np.load(path) as z:
            np.testing.assert_array_equal(ids,z['ids']);np.testing.assert_array_equal(y,z['y']);np.testing.assert_array_equal(fixed,z['anchor'])
            cf=z['average_random_calibrated']&z['average_ordered_calibrated']
            fold=z['fold'];curve=z['curve_banks_mean'][n:];native=z['native'][n:]
            weights=np.stack([z['weights_average_random'],z['weights_average_ordered']])
        with np.load(sources['expert']) as z:
            np.testing.assert_array_equal(ids,z['passenger']);global_scores=z[em+'_global'][n:];experts=z[em+'_party'][n:]
        assert len(np.unique(fold))==8
        inspect={}
        for bn,b in [('registered',fixed),('same_calibration',fixed if mode=='random' else alternate)]:
            atoms=np.column_stack([mm&(first==b),cf&(fixed==b),outcomes[mode]&(fixed==b)])
            flags=dict(Mitra=atoms[:,0],counterfactual=atoms[:,1],party=atoms[:,2],
                       two_branch=atoms[:,:2].any(1),union=atoms.any(1))
            def metrics(pred,base):
                act=(pred!=base)[:n];w=base[:n]!=y;delta=act*np.where(w,1,-1)
                return dict(all=count(act,known,w),fresh=count(act,known&fresh,w),formerly_unanimous=count(act,blind,w),
                    fold_net=[int(delta[known&(fold==k)].sum()) for k in range(8)],test=int((pred[n:]!=base[n:]).sum()))
            inspect[bn]={k:{'vs_16':metrics(b^v,b),'vs_111':metrics(b^v,first)} for k,v in flags.items()}
            inspect[bn]['test_pair_overlap']=(atoms[n:].astype(int).T@atoms[n:].astype(int)).tolist()
        local[mode]=inspect
        p=dict(passenger=ids[n:],parent=parent_prob.probability.to_numpy(),first=first[n:],eligible=paid[n:],
               mitra=mitra[n:],other_fares=other[n:],curve=curve,native=native,weights=weights,
               global_scores=global_scores,experts=experts,opinions=opin[n:])
        actual,score=apply(p)
        np.testing.assert_array_equal(actual,atoms[n:])
        if pack is None:pack=p;votes=actual;prob=score
        else:np.testing.assert_array_equal(actual,votes)
    packet=dest/'inputs.npz';np.savez_compressed(packet,**pack)
    with np.load(packet) as z:
        np.testing.assert_array_equal(z['passenger'],ids[n:])
        rv,rp=apply(z)
    np.testing.assert_array_equal(rv,votes);np.testing.assert_array_equal(rp,prob)
    assert votes.sum(0).tolist()==[5,10,3] and (votes.sum(1)<=1).all()
    checkfile=dest/'submission.csv'
    pd.DataFrame(dict(PassengerId=ids[n:],Transported=rp>=.5)).to_csv(checkfile,index=False)
    pd.DataFrame(dict(PassengerId=ids[n:],probability=prob,reload_probability=rp)).to_csv(dest/'test_probabilities.csv',index=False)
    check=validate_submission(checkfile,ROOT/'data/raw/test.csv')
    assert all(json.loads(a.read_text())['file_sha256']!=check['sha256'] for a in (ROOT/'reports').glob('*/online_attempts/*.json') if a.parents[1].name!=NAME)
    manifest=dict(**check,version=NAME.removeprefix('proposal_'),training_rows=n,changed=int(votes.any(1).sum()),
        changed_orders=int(test.PassengerId[votes.any(1)].str[:4].nunique()),reload_difference=0,
        parents={BEST:sha,FIRST:first_sha},model_sha256=dig(packet),source_file='export_receipt_mitra_party_union_n27.py',
        input_artifact_hashes={str(path.relative_to(ART)):dig(path) for path in sources.values()})
    selection=dict(version=manifest['version'],rule='On complete paid bills, retain a new Mitra fare decision with matching raw/shuffle controls plus an independent foundation vote, or apply the calibrated two-context average leave-one-service-out increment. On seven-way agreement, permit the two party specialists when both show sufficiently large increments over their respective global controls. Keep all other decisions.',metrics=local,
        limitations='Two independently retrained complete-receipt and party challengers, with the seven-opinion gate, original Mitra predictions, and registered parent replay shared between protocols. The historical-unseen subset overlaps the main validation sample.')
    for directory in [dest,ROOT/'reports'/NAME]:
        directory.mkdir(exist_ok=True)
        write(directory/'manifest.json',manifest);write(directory/'selection.json',selection)
    print(manifest,{k:local[k]['registered']['union'] for k in local},flush=True)


if __name__=='__main__':run()
