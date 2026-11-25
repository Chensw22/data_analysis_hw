"""Audit genuine versus permuted service identity on every held unique-positive basket."""
import json
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from disjoint_tree_fields_n25 import ART,DEST,stats
from evaluate_inverse_tabicl_n26 import OUT,odds,ratio
from family_identity_transfer_n26 import panels_and_reference
from full_roster_booking_validation_n23 import data
from grouped_peer_release_replay_n25 import edit_effect_bounds


def run():
    raw,y,g,fold=data()
    _,_,_,rows,_,archive,_,native,_,ref=panels_and_reference()
    count=(archive>=.5).sum(1)
    hard=(count==0)&raw.CryoSleep.eq(False).to_numpy()&~raw.PassengerId.str[:4].duplicated(keep=False).to_numpy()
    new=~np.isin(g,g[rows])
    arrays={}
    absent=np.ones(len(y),bool)
    for k in range(8):
        with np.load(OUT/f'{k}_paid_identity.npz') as z:
            q=z['q']
            np.testing.assert_array_equal(z['passenger'],raw.PassengerId.iloc[q])
            assert (fold[q]==k).all() and absent[q].all()
            absent[q]=False
            masked=z['prior']
            actual=ratio(z['observed'])
            shuffled=ratio(z['shuffled'])
        with np.load(ART/f'nov25_passenger_state_contexts/{k}_paid.npz') as z:
            all_paid=z['q']
            idx=np.searchsorted(all_paid,q)
            np.testing.assert_array_equal(all_paid[idx],q)
            regular=z['p'][idx]
        scores={'forward_paid':regular,'masked':masked,'restored':expit(odds(masked)+actual),
                'shuffled':expit(odds(masked)+shuffled),'likelihood_only':expit(actual),'shuffled_likelihood_only':expit(shuffled)}
        for name,p in scores.items():
            arrays.setdefault(name,native.copy())[q]=p
    assert (hard&y).sum()==143
    result={}
    for name,p in arrays.items():
        a,b=edit_effect_bounds(p>=.5,native>=.5,*ref,y)
        known=ref[0]>=0
        np.testing.assert_array_equal(a[known],b[known])
        table={}
        for tag,cohort in [('known',known),('new',new&known),('common',known&((count==0)|(count==68))),
                           ('awake_solo_negative',known&hard),('new_awake_solo_negative',known&new&hard)]:
            table[tag]=stats(a,cohort,fold,g)
        for tag,m in [('all',hard&~absent),('new',hard&new&~absent)]:
            table['ranking_'+tag]=dict(rows=int(m.sum()),errors=int(y[m].sum()),auc=float(roc_auc_score(y[m],p[m])))
        table['whole_bounds']=[int(a.sum()),int(b.sum())]
        print(name,[(key,table[key]['fixed'],table[key]['harmed']) for key in ('known','new','awake_solo_negative')],table['ranking_new'])
        result[name]=table
    DEST.with_name('paid_identity_audit.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    run()
