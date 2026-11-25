"""Compare the fixed candidate to widening the same rule across the full vote distribution."""
import json
from decimal import Decimal

import numpy as np
import pandas as pd

from disjoint_tree_fields_n25 import ART,DEST
from export_dispute_band_n26 import NAME, NINE, ROOT
from family_identity_transfer_n26 import panels_and_reference
from kaggle_workflow import sha256


def actions(p,names):
    n=(p>=.5).sum(1)
    minority=np.minimum(n,p.shape[1]-n)
    tree=p[:,[k for k,n in enumerate(names) if n.startswith('permutation_selection/')]].mean(1)>=.5
    return tree,{'moderate':(minority>=4)&(minority<=16),'above_sixteen':minority>16,
                 'four_or_more':minority>=4,'one_or_more':minority>=1,'unrestricted':np.ones(len(p),bool)}


def run():
    y,g,f,rows,names,p,h,_,_,reference=panels_and_reference()
    prior,lo,hi=reference
    known=prior>=0
    ref=np.where(prior.astype(bool),hi,lo)
    unseen=~np.isin(g,g[rows])
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        old=z['nine']
    best=json.loads((ROOT/'reports/current_best_submission.json').read_text())
    assert best['proposal']==NINE
    oldfile=ART/NINE/'submission.csv'
    assert sha256(oldfile)==best['csv_sha256']
    scored=pd.read_csv(oldfile)
    with np.load(ART/NAME/'frozen_votes.npz') as z:
        np.testing.assert_array_equal(z['passenger'],scored.PassengerId)
        np.testing.assert_array_equal(z['anchor']>=.5,scored.Transported)
        np.testing.assert_array_equal(z['names'],names)
        test=z['p']
    n=len(scored)
    # Project's aggregate-score audit identified a common denominator of 4,277.
    observed=json.loads((ROOT/'reports/nov23_flip_capacity/primary_scores.json').read_text())
    observed_scores=observed['public_scores']
    values=observed_scores.values() if isinstance(observed_scores,dict) else observed_scores
    digits={int(Decimal(str(s))*100000) for s in values}
    digits.add(int(Decimal(str(best['public_score']))*100000))
    denominators=[]
    for possible in range(1,n+1):
        # Allow either five-digit rounding or truncation on each published aggregate score.
        if all(any(-possible<=2*(100000*k-score*possible)<2*possible
                   for k in (score*possible//100000,score*possible//100000+1)) for score in digits):
            denominators.append(possible)
    assert denominators==[n]
    correct=round(float(best['public_score'])*n)
    goal=85*n//100+1
    report={}
    for tag,probs,anchor,label,mask in [('full_known',p,ref,y,known),('new_known',p,ref,y,known&unseen),
                ('historical',h,old,y[rows],np.ones(len(rows),bool)),
                ('competition',test,scored.Transported.to_numpy(),None,np.ones(n,bool))]:
        tree,routes=actions(probs,names)
        for name,use in routes.items():
            flip=(tree!=anchor)&use&mask
            d=dict(eligible=int((mask&use).sum()),edits=int(flip.sum()))
            if label is None:
                d.update(upper_accuracy_if_every_edit_helps=min(1,(correct+int(flip.sum()))/n))
                if name=='moderate':
                    test_csv=pd.read_csv(ART/NAME/'submission.csv')
                    np.testing.assert_array_equal(anchor^flip,test_csv.Transported)
            else:
                d.update(fixed=int((flip&(tree==label)).sum()),harmed=int((flip&(tree!=label)).sum()))
            report.setdefault(name,{})[tag]=d
    out=dict(denominator=n,baseline_correct=correct,required_for_strict_85=goal,minimum_net_change=goal-correct,
             evidence='Equal-weight aggregate-score denominator from published scores; ceilings assume all changed rows help.',
             routes=report)
    DEST.with_name('release_band_scope.json').write_text(json.dumps(out,indent=2)+'\n')
    for name,entry in report.items():print(name,entry)
    print('required net corrections',goal-correct)


if __name__=='__main__':
    run()
