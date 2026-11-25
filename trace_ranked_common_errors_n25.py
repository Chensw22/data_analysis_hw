"""Count distinct rescues, collateral errors and corroboration across model families by original vote split."""
import json

import numpy as np
from scipy.special import logit

from adjudicate_amount_dissent_n25 import counts, load_fold
from disjoint_tree_fields_n25 import ART,DEST
from evaluate_bill_precision_n25 import TAB, GROUPS
from full_roster_booking_validation_n23 import data
from full_roster_six_failure_archive_n24 import prediction_sets
from grouped_peer_release_replay_n25 import edit_effect_bounds
from matched_error_ranker_n25 import OUT as RANK


def run():
    raw,y,g,f=data()
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        rows,hf=z['rows'],z['fold']
    _,panels=prediction_sets(raw,y,g,f,rows,hf)
    votes,old=[(np.column_stack(panel)>=.5).sum(1) for panel in panels]
    same=np.minimum(votes,68-votes)
    repeated=np.zeros(len(y),bool)
    repeated[rows]=(same[rows]==0)&((old==0)|(old==68))&((old>34)==(votes[rows]>34))
    new=~np.isin(g,g[rows])
    result={}
    agreements={}
    for k in range(8):
        q,t=np.flatnonzero(f==k),np.flatnonzero(f!=k)
        native,prior,lo,hi,_,flags,_=load_fold(k,q,t,raw,y)
        with np.load(RANK/f'{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            margin=np.abs(logit(z['native'].clip(.002,.998)))
            pair=z['matched']>margin
            ordinary=z['likelihood']>margin
        independent=[]
        for name in GROUPS:
            with np.load(TAB/f'{k}_{name}.npz') as z:
                np.testing.assert_array_equal(z['q'],q)
                independent.append((z['p']>=.5)!=native)
        transformer=np.column_stack(independent).any(1)
        neural=flags.any(1)
        routes={'paired':pair, 'ordinary':ordinary,'generative':neural, 'transformer':transformer,
                'union':pair|ordinary|neural|transformer, 'pair_alone':pair&~(ordinary|neural|transformer),
                'pair_new_to_precision':pair&~neural,'pair_and_precision':pair&neural,
                'pair_and_ordinary':pair&ordinary,
                'pair_and_transformer':pair&transformer,
                'pair_and_full_manifest':pair&flags[:,0],
                'pair_and_three_sources':pair&neural&transformer}
        for name,change in routes.items():
            a,b=edit_effect_bounds(native^change,native,prior,lo,hi,y[q])
            known=prior>=0
            np.testing.assert_array_equal(a[known],b[known])
            result.setdefault(name,[]).append((q,known,a,b))
        with np.load(ART/f'nov25_grouped_peer_teacher/release{k}.npz') as z:
            ref=z['nine']
            # Prior known is the conservative scope shared with all transfer comparisons.
            assert (ref[prior>=0]>=0).all()
            agreements[k]=(q,prior>=0,ref)
    scopes={}
    ix,known,ref=(np.concatenate([part[i] for part in agreements.values()]) for i in range(3))
    for name,mask in [('all',np.ones(len(y),bool)),('unanimous',same==0),('one_to_three',(same>=1)&(same<=3)),
                      ('four_to_sixteen',(same>=4)&(same<=16)),('seventeen_plus',same>=17),
                      ('repeated',repeated),('new',new),('new_unanimous',new&(same==0))]:
        use=known&mask[ix]
        scopes[name]=dict(rows=int(use.sum()),initial_errors=int((use&(ref!=y[ix])).sum()))
    report={}
    for action,parts in result.items():
        seq,m,a,b=(np.concatenate([v[i] for v in parts]) for i in range(4))
        np.testing.assert_array_equal(seq,ix)
        np.testing.assert_array_equal(m,known)
        groups={'all':known,'unanimous':known&(same[ix]==0),'one_to_three':known&(same[ix]>=1)&(same[ix]<=3),
                'four_to_sixteen':known&(same[ix]>=4)&(same[ix]<=16),'seventeen_plus':known&(same[ix]>=17),
                'repeated':known&repeated[ix],'new':known&new[ix],'new_unanimous':known&new[ix]&(same[ix]==0)}
        r={key:counts(a,mask,f[ix]) for key,mask in groups.items()}
        r['bounds']=[int(a.sum()),int(b.sum())]
        report[action]=r
        print(action,[(s,r[s]['fixed'],r[s]['harmed']) for s in ('all','unanimous','new')])
    DEST.with_name('trace_ranked_common_errors.json').write_text(json.dumps(dict(base=scopes,actions=report),indent=2)+'\n')
    print(scopes)


if __name__=='__main__':
    run()
