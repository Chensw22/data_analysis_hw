"""Check passenger-level overlap of the two candidates that qualified for online tests."""
import json
import numpy as np
import pandas as pd

from awake_context_ranks_n25 import OUT as NEURAL
from disjoint_tree_fields_n25 import ART,DEST,OUT as VIEWS,stats
from full_roster_booking_validation_n23 import ROOT,data
from full_roster_permutation_selection_n24 import OUT as RANDOM
from grouped_peer_release_replay_n25 import edit_effect_bounds
from grouped_peer_teacher_n25 import OUT
from kaggle_workflow import sha256


def run():
    raw,y,g,fold=data()
    awake=raw.CryoSleep.eq(False).to_numpy()
    effects={name:[] for name in ('tree','neural','union','intersection')}
    masks=[]
    for k in np.unique(fold):
        with np.load(OUT/f'release{k}.npz') as z:
            q=z['q']
            np.testing.assert_array_equal(q,np.flatnonzero(fold==k))
            p,prior,lo,hi=(z[key] for key in ('native','v131','nine_on_low_branch','nine_on_high_branch'))
        with np.load(RANDOM/f'full_{k}.npz') as z:
            np.testing.assert_array_equal(q,z['query'])
            tree=(.75*p+.25*z['prob'][:,2]>=.5)!=(p>=.5)
        for suffix in ('','_repeat'):
            with np.load(VIEWS/f'full_{k}{suffix}.npz') as z:
                np.testing.assert_array_equal(q,z['q'])
                tree &= ((z['p']>=.5)!=(p[:,None]>=.5)).sum(1)>=2
        neur=np.zeros(len(q),bool)
        with np.load(NEURAL/f'{k}_personal.npz') as z:
            np.testing.assert_array_equal(q[awake[q]],z['q'])
            neur[awake[q]]=(z['p']>=.5)!=(p[awake[q]]>=.5)
        masks.append((q,prior>=0))
        for name,change in [('tree',tree),('neural',neur),('union',tree|neur),('intersection',tree&neur)]:
            a,b=edit_effect_bounds((p>=.5)^change,p>=.5,prior,lo,hi,y[q])
            effects[name].append((a,b))
    row=np.concatenate([q for q,_ in masks])
    known=np.concatenate([m for _,m in masks])
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        new_order=~np.isin(g[row],g[z['rows']])
    report={}
    for name,blocks in effects.items():
        a,b=(np.concatenate([v[i] for v in blocks]) for i in range(2))
        r={}
        for key,m in [('all',np.ones(len(row),bool)),('other_five',fold[row]>=3),('unseen',new_order)]:
            r[key]=dict(bounds=[int(a[m].sum()),int(b[m].sum())],known=stats(a,known&m,fold[row],g[row]))
        report[name]=r
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId'])
    vals=[]
    for name in ('proposal_n24_selected_dissent','proposal_n25_confirmed_random_fields','proposal_n25_awake_personal_tabicl'):
        path=ART/name/'submission.csv'
        assert sha256(path)==json.loads((ROOT/'reports'/name/'manifest.json').read_text())['sha256']
        v=pd.read_csv(path)
        np.testing.assert_array_equal(v.PassengerId,test.PassengerId)
        vals.append(v.Transported.to_numpy(bool))
    nine,tree,neural=vals
    report['test']=dict(tree=int((nine!=tree).sum()),neural=int((nine!=neural).sum()),
                        overlap=int(((nine!=tree)&(nine!=neural)).sum()),union=int(((nine!=tree)|(nine!=neural)).sum()))
    DEST.with_name('confirmed_awake_union.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['test'],[(k,report[k]['all']['bounds'],report[k]['all']['known']['fixed'],report[k]['all']['known']['harmed']) for k in effects])


if __name__=='__main__':
    run()
