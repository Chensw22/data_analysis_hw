"""Compare the actual candidate decisions to both their queued parent and the scored nine-vote model."""
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from booking_mix_confidence_transfer_n26 import audit
from family_identity_transfer_n26 import panels_and_reference
from tabicl_consensus_review_n26 import ROOT,OUT,sources,splits
from solo_archive_recovery_n26 import ticker


def run():
    raw,y,g,views=sources()
    *_,ref=panels_and_reference()
    with np.load(ROOT/'artifacts/nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['original_prior'],ref[0])
        prior=z['prior']
    nine_full=np.where(prior>0,ref[2],ref[1]).astype(bool)
    with np.load(ROOT/'artifacts/nov25_historical_nine/historical.npz') as z:
        nine_old=z['nine'].astype(bool)
    size=raw.booking_size.to_numpy()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId'])
    tg=test.PassengerId.str.split('_').str[0]
    weights=np.bincount(tg.map(tg.value_counts()).clip(upper=4),minlength=5)[1:]/len(tg)
    batches={}
    for key,src,dst,t,q in splits(views,g):
        with np.load(OUT/f'pred_{key}.npz') as z:
            np.testing.assert_array_equal(z['q'],dst['ids'][q])
            revise=(z['identities_error']>.5)&(z['identities_outcome']>.5)
        tag=key.split('_')[0] if key!='historical_to_new' else key
        batches.setdefault(tag,[]).append((q,revise))
    report={}
    for tag,values in batches.items():
        q=np.concatenate([z[0] for z in values])
        flip=np.concatenate([z[1] for z in values])
        v=views['historical' if tag=='historical' else 'full']
        nine=(nine_old if tag=='historical' else nine_full)[q]
        ids=v['ids'][q]
        label=y[ids]
        parent=v['anchor'][q]
        new=parent^flip
        assert v['known'][q].all()
        prev=nine!=parent
        all_rows=np.ones(len(q),bool)
        r={'total':len(q),'accuracy':{k:float((v==label).mean()) for k,v in [('nine',nine),('parent',parent),('review',new)]},
           'parent_vs_nine':ticker(all_rows,nine!=label,prev),'review_vs_nine':ticker(all_rows,nine!=label,nine!=new),
           'review_reversing_parent_edits':ticker(prev,parent!=label,flip),
           'review_on_parent_unchanged':ticker(~prev,parent!=label,flip)}
        if tag=='bookingmix':
            rng=np.random.default_rng(7430)
            draws=[]
            for b in range(1,5):
                n=len(np.unique(g[ids[size[ids].clip(max=4)==b]]))
                draws.append(rng.multinomial(n,np.full(n,1/n),size=2000))
            for name,base in [('parent',parent),('nine',nine)]:
                gain=(new==label).astype(int)-(base==label)
                r['weighted_vs_'+name]=audit(gain,ids,size,g,weights,draws)
            fresh=~np.isin(g[ids],g[views['historical']['ids']])
            r['single_source_on_fresh']=ticker(fresh,parent!=label,flip)
        report[tag]=r
        print(tag,r['accuracy'],r['review_reversing_parent_edits'],r['review_on_parent_unchanged'],flush=True)
    path=ROOT/'reports/nov23_matched_learning_curve/consensus_review_vs_scored_nine.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
