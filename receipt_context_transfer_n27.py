"""Test whether learning from priced training passengers transfers to passengers with no usable fare."""
import json
import numpy as np
import pandas as pd

from full_roster_booking_validation_n23 import ROOT, data
from nori_booking_regression_n26 import packet_dir
from price_implied_deck_n26 import OUT as DECK, FEES
from nori_fare_party_n26 import ART as NORI
from repeated_archive_misses_n26 import ART, cohort, count
from kaggle_workflow import sha256

OUT=ART/'nov27_receipt_context_transfer'
REPORT=ROOT/'reports/nov23_matched_learning_curve/receipt_context_transfer_n27.json'


def policies(anchor,ineligible,raw,fine):
    sign=np.where(anchor,-1.,1.)[:,None]
    old=(raw>=.5)!=anchor[:,None]
    new=(fine>=.5)!=anchor[:,None]
    raised=(fine-raw)*sign>0
    return dict(raw_pair=ineligible&old.all(1),fine_pair=ineligible&new.all(1),
                strengthened=ineligible&new.all(1)&raised.all(1),
                new_opposition=ineligible&new.all(1)&~old.all(1),
                both_crossed=ineligible&new.all(1)&~old.any(1),
                newly_confirmed=ineligible&new.all(1)&~old.all(1)&raised.all(1))


def run():
    raw,y,g,f=data()
    base,blind,known,fresh=cohort(raw,y,f)
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns)
    frame=pd.concat([raw,test],ignore_index=True)
    n=len(y)
    old=np.empty((len(frame),2))
    fine=old.copy()
    for k in range(8):
        t,q=np.flatnonzero(f!=k),np.flatnonzero(f==k)
        with np.load(DECK/f'neural_full_{k}.npz') as z:
            for key,v in [('t',t),('q',q),('y',y[t])]:np.testing.assert_array_equal(z[key],v)
            old[q,0],fine[q,0]=z['raw'],z['fine']
        for dest,path in [(old,packet_dir(0)/f'fold{k}.npz'),(fine,NORI/f'own{k}.npz')]:
            with np.load(path) as z:
                for key,v in [('t',t),('q',q),('labels',y[t]),('ids' if 'ids' in z else 'passenger',raw.PassengerId.iloc[q])]:
                    np.testing.assert_array_equal(z[key],v)
                dest[q,1]=z['p']
    name='proposal_n26_joint_fare_foundations'
    path=ART/name/'inputs.npz'
    assert sha256(path)==json.loads((ROOT/'reports'/name/'manifest.json').read_text())['model_sha256']
    with np.load(path) as z:
        np.testing.assert_array_equal(z['passenger'],test.PassengerId)
        old[n:],fine[n:]=z['raw'],z['fine']
        expected=z['eligible']
        best111=z['parent']>=.5
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        anchor=z['anchor']
        np.testing.assert_array_equal(z['passenger'],frame.PassengerId)
        np.testing.assert_array_equal(anchor[:n],base)
    paid=frame[FEES].gt(0).any(axis=1).to_numpy()
    eligible=paid & frame[FEES].notna().all(axis=1).to_numpy()
    np.testing.assert_array_equal(eligible[n:],expected)
    np.testing.assert_array_equal(anchor[n:][~expected],best111[~expected])
    size=frame.PassengerId.str.split('_').str[0].map(frame.PassengerId.str.split('_').str[0].value_counts()).to_numpy()
    result={}
    arrays={}
    for source,area in [('ineligible',~eligible),('missing_paid',paid&~eligible),('unpaid',~paid)]:
        actions=policies(anchor,area,old,fine)
        for k,act in actions.items():
            tag=source+'_'+k
            delta=act[:n]*np.where(base!=y,1,-1)
            scopes=dict(all=known,fresh=known&fresh,common=blind,solo=known&(size[:n]==1),party=known&(size[:n]>1))
            result[tag]=dict(test_edits=int(act[n:].sum()),by_fold=[int(delta[known&(f==j)].sum()) for j in range(8)],
                             scopes={a:count(act[:n],m,base!=y) for a,m in scopes.items()})
            arrays[tag]=act
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'actions.npz',passenger=frame.PassengerId.to_numpy(str),y=y,fold=f,anchor=anchor,old=old,fine=fine,**arrays)
    REPORT.write_text(json.dumps(result,indent=2)+'\n')
    print([(k,v['scopes']['all'],v['scopes']['fresh'],v['test_edits']) for k,v in result.items() if k.startswith('ineligible')])


if __name__=='__main__':
    run()
