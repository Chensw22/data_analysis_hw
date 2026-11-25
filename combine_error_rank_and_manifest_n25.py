"""Cross-fit a correction decision using independently trained pair rankings and manifest evidence."""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from adjudicate_amount_dissent_n25 import counts, load_fold, odds
from conditional_manifest_evidence_n25 import OUT as FIELDS
from disjoint_tree_fields_n25 import ART,DEST
from full_roster_booking_validation_n23 import data
from full_roster_six_failure_archive_n24 import prediction_sets
from grouped_peer_release_replay_n25 import edit_effect_bounds
from matched_error_ranker_n25 import OUT as RANK


def run():
    raw,y,g,f=data()
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        rows,hf=z['rows'],z['fold']
    _,panels=prediction_sets(raw,y,g,f,rows,hf)
    archive=np.column_stack(panels[0])
    votes,old_votes=[(np.column_stack(panel)>=.5).sum(1) for panel in panels]
    majority=votes>34
    common=(votes==0)|(votes==68)
    repeated=np.zeros(len(y),bool)
    repeated[rows]=common[rows]&((old_votes==0)|(old_votes==68))&((old_votes>34)==majority[rows])
    new=~np.isin(g,g[rows])
    results={}
    for k in range(8):
        q,t=np.flatnonzero(f==k),np.flatnonzero(f!=k)
        ref,prior,lo,hi,features,_,_=load_fold(k,q,t,raw,y)
        sign=np.where(ref,-1.,1.)[:,None]
        crowd=sign*odds(archive[q])
        anchor=np.column_stack([features['anchor'], np.quantile(crowd,[.05,.5,.95],axis=1).T])
        with np.load(FIELDS/f'fold{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            field=sign*z['observed']
        with np.load(RANK/f'{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            rank=z['matched']
            baseline=z['likelihood']
            random=z['random_order']
        matrices={'crowd':anchor,'fields':np.column_stack([anchor,field]),
            'ranking':np.column_stack([anchor,rank]),
            'joint':np.column_stack([anchor,field,rank]),
            'ordinary_training':np.column_stack([anchor,field,baseline]),
            'shuffled_pairs':np.column_stack([anchor,field,random])}
        outcomes=y[q]!=ref
        splits={'full':list(GroupKFold(4,shuffle=True,random_state=3207).split(q,groups=g[q])),
                'unseen_orders':[(np.flatnonzero(~new[q]),np.flatnonzero(new[q]))]}
        for protocol,partitions in splits.items():
            for train,test in partitions:
                assert not np.isin(g[q[train]],g[q[test]]).any()
                for name,x in matrices.items():
                    lr=LogisticRegression(C=.2,solver='liblinear',max_iter=500).fit(x[train],outcomes[train])
                    choice=lr.predict_proba(x[test])[:,1]>.5
                    a,b=edit_effect_bounds(ref[test]^choice,ref[test],prior[test],lo[test],hi[test],y[q[test]])
                    known=prior[test]>=0
                    np.testing.assert_array_equal(a[known],b[known])
                    results.setdefault((protocol,name),[]).append((q[test],known,a,b))
    report={}
    for (protocol,name),parts in results.items():
        ix,m,a,b=(np.concatenate([v[j] for v in parts]) for j in range(4))
        np.testing.assert_array_equal(np.sort(ix),np.arange(len(y)) if protocol=='full' else np.flatnonzero(new))
        r=dict(bounds=[int(a.sum()),int(b.sum())])
        for key,mask in [('known',m),('consensus',m&common[ix]),('repeated',m&repeated[ix]),
                         ('common_mistakes',m&common[ix]&(majority[ix]!=y[ix]))]:
            r[key]=counts(a,mask,f[ix])
        report.setdefault(protocol,{})[name]=r
        print(protocol,name,r['bounds'],r['known']['fixed'],r['known']['harmed'],r['common_mistakes']['fixed'])
    DEST.with_name('combine_error_rank_and_manifest.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    run()
