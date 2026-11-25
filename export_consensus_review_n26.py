"""Export the double-confirmation TabICL reviewer after replaying all fifteen validation fits."""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from booking_mix_confidence_transfer_n26 import assignments,partition
from exaone_validation_data import compact_features
from full_roster_booking_validation_n23 import ROOT
from release_confidence_review_n26 import summarize
from tabicl_consensus_review_n26 import OUT as VALIDATION,matrices,sources

NAME='proposal_n26_consensus_review'
ART=ROOT/'artifacts'
FEATURES=ART/'nov26_review_release'
OUT=ART/NAME
PARENT='proposal_n26_joint_dispute_awake'
BAND='proposal_n26_dispute_band'


def dig(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,record):
    path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(record,indent=2)+'\n')


def prepare():
    from export_budget_quarter_n23 import source
    raw,y,g,v=sources()
    old,cur=v['historical'],v['full']
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=lambda c:c!='Name')
    native=source('proposal_online_first_counts15',test)[0]
    frozen=ART/BAND/'frozen_votes.npz'
    assert dig(frozen)==json.loads((ROOT/'reports'/BAND/'manifest.json').read_text())['model_sha256']
    with np.load(frozen) as z:
        np.testing.assert_array_equal(z['passenger'],test.PassengerId)
        p=z['p']
        cols=z['names'].tolist()
    masks=[np.array([n.split('/')[0]==family for n in cols]) for family in old['names'][:18]]
    assert np.sum(np.stack(masks),axis=0).tolist()==[1]*len(cols)
    parts=[np.column_stack([p[:,mask].mean(1) for mask in masks]),p[:,[cols.index(n) for n in old['names'][18:80]]]]
    extra=[]
    hashes={BAND:dig(frozen)}
    for name in ('bills','context'):
        path=FEATURES/f'{name}.npz'
        with np.load(path) as z:
            np.testing.assert_array_equal(z['passenger'],test.PassengerId)
            extra.extend(z['names'].tolist())
            parts.append(z['p'])
        hashes[name]=dig(path)
    np.testing.assert_array_equal(extra,old['names'][80:])
    bank=np.column_stack(parts)
    assert bank.shape==(len(test),89) and np.isfinite(bank).all() and np.all((bank>=0)&(bank<=1))
    par_file=ART/PARENT/'submission.csv'
    hashes[PARENT]=dig(par_file)
    assert hashes[PARENT]==json.loads((ROOT/'reports'/PARENT/'manifest.json').read_text())['sha256']
    par=pd.read_csv(par_file)
    soft=pd.read_csv(ART/PARENT/'test_probabilities.csv',float_precision='round_trip')
    np.testing.assert_array_equal(par.PassengerId,test.PassengerId)
    np.testing.assert_array_equal(soft.PassengerId,test.PassengerId)
    np.testing.assert_array_equal(soft.probability>=.5,par.Transported)
    block=summarize(bank,native,par.Transported.to_numpy(bool))
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'test_inputs.npz',passenger=test.PassengerId.to_numpy(str),names=old['names'],
                        parent_probability=soft.probability.to_numpy(),**block)
    write(OUT/'input_hashes.json',hashes)


def train():
    import torch
    from joint_party_tabicl_n26 import model
    from tabicl_routed_context_n23 import DIGEST,WEIGHT
    assert dig(WEIGHT)==DIGEST
    torch.set_num_threads(2)
    raw,y,g,v=sources()
    old,cur=v['historical'],v['full']
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=lambda c:c!='Name')
    tg=test.PassengerId.str.split('_').str[0]
    assert not np.isin(g,tg).any()
    frame=compact_features(test).assign(booking_size=tg.map(tg.value_counts()).to_numpy())
    roster=pd.concat([raw,frame],ignore_index=True)
    with np.load(OUT/'test_inputs.npz') as z:
        np.testing.assert_array_equal(z['passenger'],test.PassengerId)
        target={n:z[n] for n in ('basic','relative','anchor','off')}
    target['ids']=len(y)+np.arange(len(test))
    src_fold=partition(tg,'source:',3)
    bucket=partition(tg,'query:',5)
    for f,k,t,q in assignments(old,cur,g):
        ix=np.flatnonzero((src_fold==f)&(bucket==k))
        path=OUT/f'review_{f}_{k}.npz'
        digest=dig(OUT/'test_inputs.npz')
        if not path.exists():
            xt,held=matrices(raw,old,cur,t,q)
            xt2,testx=matrices(roster,old,target,t,ix)
            pd.testing.assert_frame_equal(xt[1],xt2[1])
            result={}
            errors=[]
            with np.load(VALIDATION/f'pred_bookingmix_{f}_{k}.npz') as z:
                np.testing.assert_array_equal(z['q'],q)
                np.testing.assert_array_equal(z['t'],old['ids'][t])
                for n,labels in [('error',old['wrong'][t]),('outcome',y[old['ids'][t]])]:
                    est=model(7601).fit(xt[1],labels)
                    p=est.predict_proba(held[1])[:,1]
                    if n=='outcome':
                        p=np.where(cur['anchor'][q],1-p,p)
                    saved=z['identities_'+n]
                    errors.append(float(np.max(np.abs(p-saved))))
                    np.testing.assert_allclose(p,saved,atol=3e-5,rtol=0)
                    np.testing.assert_array_equal(p>.5,saved>.5)
                    pred=est.predict_proba(testx[1])[:,1]
                    result[n]=np.where(target['anchor'][ix],1-pred,pred) if n=='outcome' else pred
            with path.with_suffix('.tmp').open('wb') as h:
                np.savez_compressed(h,test=ix,t=old['ids'][t],q=q,replay_error=errors,source=digest,**result)
            path.with_suffix('.tmp').replace(path)
        with np.load(path) as z:
            for n,val in [('test',ix),('t',old['ids'][t]),('q',q),('source',digest)]:
                np.testing.assert_array_equal(z[n],val)
        print(f,k,len(ix),flush=True)


def combine(scores,parent):
    risk=scores.min(1)
    changed=risk>.5
    corrected=np.where(parent>=.5,1-risk,risk)
    return np.where(changed,corrected,parent)


def export():
    from kaggle_workflow import validate_submission
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId'])
    with np.load(OUT/'test_inputs.npz') as z:
        np.testing.assert_array_equal(z['passenger'],test.PassengerId)
        parent=z['parent_probability']
        agree=z['agreement']
    p=np.full((len(test),2),np.nan)
    e=[]
    for f in range(3):
        for k in range(5):
            with np.load(OUT/f'review_{f}_{k}.npz') as z:
                ix=z['test']
                assert np.isnan(p[ix]).all()
                p[ix]=np.column_stack([z['error'],z['outcome']])
                e.extend(z['replay_error'].tolist())
    assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    file=OUT/'review_scores.npz'
    np.savez_compressed(file,passenger=test.PassengerId.to_numpy(str),scores=p,parent=parent)
    probability=combine(p,parent)
    with np.load(file) as z:
        np.testing.assert_array_equal(z['passenger'],test.PassengerId)
        repeat=combine(z['scores'],z['parent'])
    np.testing.assert_array_equal(probability,repeat)
    edits=(probability>=.5)!=(parent>=.5)
    np.testing.assert_array_equal(edits,(p>.5).all(1))
    pd.DataFrame(dict(PassengerId=test.PassengerId,Transported=probability>=.5)).to_csv(OUT/'submission.csv',index=False)
    pd.DataFrame(dict(PassengerId=test.PassengerId,probability=probability,reload_probability=repeat)).to_csv(OUT/'test_probabilities.csv',index=False)
    checked=validate_submission(OUT/'submission.csv',ROOT/'data/raw/test.csv')
    nine=pd.read_csv(ART/'proposal_n24_selected_dissent/submission.csv')
    np.testing.assert_array_equal(nine.PassengerId,test.PassengerId)
    original=nine.Transported.to_numpy(bool)
    parent_changed=(parent>=.5)!=original
    eligible_common=agree&~parent_changed
    assert edits.any() and all(json.loads(f.read_text())['file_sha256']!=checked['sha256']
                               for f in (ROOT/'reports').glob('*/online_attempts/*.json'))
    manifest=dict(**checked,version=NAME.removeprefix('proposal_'),training_rows=8693,
                  historical_calibration_population=6954,parents={PARENT:dig(ART/PARENT/'submission.csv')},
                  model_sha256=dig(file),reload_difference=0,upstream= json.loads((OUT/'input_hashes.json').read_text()),
                  review_fits=30,local_replay_max_probability_error=max(e),local_replay_decision_differences=0,
                  parent_changes=int(edits.sum()),nine_changes=int(np.sum((probability>=.5)!=nine.Transported)),
                  undo_parent_edits=int((edits&parent_changed).sum()),additional_edits=int((edits&~parent_changed).sum()),
                  unanimous_rows=int(eligible_common.sum()),unanimous_edits=int((eligible_common&edits).sum()),
                  to_true=int((edits&(probability>=.5)).sum()),to_false=int((edits&(probability<.5)).sum()))
    selection=dict(version=manifest['version'],rule='Flip the 214-edit parent when both TabICL targets predict an error with probability above 0.5.',
                   allocation='Each test booking is assigned one source fold and one calibration bucket by the validation hash.',
                   evidence='reports/nov23_matched_learning_curve/tabicl_consensus_review.json')
    for directory in (OUT,ROOT/'reports'/NAME):
        write(directory/'manifest.json',manifest)
        write(directory/'selection.json',selection)
    print(manifest,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','train','export'])
    {'prepare':prepare,'train':train,'export':export}[parser.parse_args().action]()
