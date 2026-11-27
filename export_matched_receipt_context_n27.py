"""Freeze replicated receipt-mask evidence on three missing-bill passengers."""
import json
import numpy as np
import pandas as pd
from scipy.special import logit

from export_consensus_review_n26 import ART,ROOT,write,dig
from export_directional_foundation_n27 import NAME as BASE
from kaggle_workflow import validate_submission
from masked_reference_receipts_n27 import OUT,FEES
from six_way_against_scored_nine_n26 import test_decisions

NAME='proposal_n27_matched_receipt_context'


def combine(parent,incomplete,base_confidence,tree,augmented,tab):
    side=parent>=.5
    change=incomplete & ((augmented>=.5)!=side) & ((tree>=.5)==side)
    signed=np.where(side,1.,-1.)*base_confidence
    for estimates in tab:
        matched,original,incorrect=[logit(p.clip(.005,.995)) for p in estimates]
        change &= ((signed+matched-original>=0)!=side)&((signed+matched-incorrect>=0)!=side)
    return np.where(change,augmented,parent)


def run():
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId',*FEES])
    ids=test.PassengerId
    n=len(pd.read_csv(ROOT/'data/raw/train.csv',usecols=['PassengerId']))
    side,parent_sha=test_decisions(BASE,ids)
    probabilities=pd.read_csv(ART/BASE/'test_probabilities.csv',float_precision='round_trip')
    np.testing.assert_array_equal(probabilities.PassengerId,ids)
    parent=probabilities.probability.to_numpy()
    np.testing.assert_array_equal(parent>=.5,side)
    np.testing.assert_array_equal(parent,probabilities.reload_probability)
    panel=ART/'nov26_latent_passenger_difficulty/panel.npz'
    with np.load(panel) as z:
        np.testing.assert_array_equal(z['names'][-3:],['Mitra','TabICL fare','Nori fare'])
        np.testing.assert_array_equal(z['passenger'][n:],ids)
        confidence=abs(logit(np.median(z['scores'][n:,-3:],axis=1).clip(.005,.995)))
    partial=ART/'nov27_partial_receipt_deck/aligned.npz'
    with np.load(partial) as z:
        np.testing.assert_array_equal(z['passenger'][n:],ids)
        tree,aug=z['base'][n:],z['augmented'][n:]
    files=[OUT/'aligned.npz',OUT/'replica/aligned.npz']
    tabs=[]
    for path in files:
        with np.load(path) as z:
            np.testing.assert_array_equal(z['passenger'][n:],ids)
            np.testing.assert_array_equal(z['anchor'][n:],side)
            tabs.append(np.stack([z['tab_'+k][n:] for k in ('matched','original','wrong')]))
    with np.load(files[-1]) as z:
        expected=z['action_replicate_independent_shift'][n:]
    dest=ART/NAME
    dest.mkdir(exist_ok=True)
    model=dest/'inputs.npz'
    args=dict(parent=parent,incomplete=test[FEES].isna().any(axis=1).to_numpy(),
              base_confidence=confidence,tree=tree,augmented=aug,tab=np.stack(tabs))
    p=combine(**args)
    np.savez_compressed(model,passenger=ids.to_numpy(str),**args)
    with np.load(model) as z:
        again=combine(**{k:z[k] for k in args})
    np.testing.assert_array_equal(p,again)
    np.testing.assert_array_equal((p>=.5)!=side,expected)
    assert expected.sum()==3 and np.isfinite(p).all()
    path=dest/'submission.csv'
    pd.DataFrame(dict(PassengerId=ids,Transported=p>=.5)).to_csv(path,index=False)
    pd.DataFrame(dict(PassengerId=ids,probability=p,reload_probability=again)).to_csv(dest/'test_probabilities.csv',index=False)
    check=validate_submission(path,ROOT/'data/raw/test.csv')
    assert not any(json.loads(r.read_text())['file_sha256']==check['sha256'] and r.parents[1].name!=NAME
                   for r in (ROOT/'reports').glob('*/online_attempts/*.json'))
    manifest=dict(**check,version='n27_matched_receipt_context',training_rows=n,parents={BASE:parent_sha},
                  input_artifact_hashes={str(f.relative_to(ART)):dig(f) for f in [panel,partial,*files]},
                  model_sha256=dig(model),reload_difference=0,changed=int(expected.sum()),changed_orders=int(ids[expected].str[:4].nunique()),
                  source_file='export_matched_receipt_context_n27.py')
    selection=dict(version=manifest['version'],reference_rule='Require positive matched-mask shifts relative to original and cyclic incorrect masks sufficient to cross the incumbent in both independent TabICL banks; require an augmentation-only tree change.',
                   reference_evidence='reports/nov23_matched_learning_curve/masked_reference_receipts_replica.json',
                   model_file='inputs.npz',support=dict(fixed=6,harmed=0,fresh_fixed=1,fresh_harmed=0))
    for folder in (dest,ROOT/'reports'/NAME):
        folder.mkdir(exist_ok=True)
        write(folder/'manifest.json',manifest)
        write(folder/'selection.json',selection)
    print(check['sha256'],ids[expected].to_list())


if __name__=='__main__':
    run()
