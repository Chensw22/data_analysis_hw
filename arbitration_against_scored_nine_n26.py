"""Learn who wins when the actually scored champion and the pending release disagree."""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from family_identity_transfer_n26 import panels_and_reference
from target_mix_expert_routing_n26 import ROOT,batches,ticker


def champions():
    y,g,fold,rows,_,_,_,_,_,ref=panels_and_reference()
    with np.load(ROOT/'artifacts/nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['fold'],fold)
        np.testing.assert_array_equal(z['original_prior'],ref[0])
        known=z['prior']>=0
        full=np.where(z['prior']==1,ref[2],ref[1]).astype(bool)
    with np.load(ROOT/'artifacts/nov25_historical_nine/historical.npz') as z:
        np.testing.assert_array_equal(z['rows'],rows)
        np.testing.assert_array_equal(z['y'],y[rows])
        hist=np.zeros(len(y),bool)
        hist[rows]=z['nine'].astype(bool)
    return full,known,hist,rows


def fit(x,target,z,weights):
    if np.unique(target[weights>0]).size<2:
        return np.full(len(z),(weights@target+1)/(weights.sum()+2))
    scaler=StandardScaler().fit(x,sample_weight=weights)
    model=LogisticRegression(C=.025,max_iter=500).fit(scaler.transform(x).clip(-5,5),target,sample_weight=weights)
    return model.predict_proba(scaler.transform(z).clip(-5,5))[:,1]


def run():
    full,known,hist,historical_ids=champions()
    items={}
    for b in batches(True):
        assert known[b.ids].all()
        if b.tag=='full':
            assert known[b.train_ids].all()
            orig=full
        else:
            assert np.isin(b.train_ids,historical_ids).all()
            orig=hist
        parent=b.query_cell%2!=0
        old=b.train_cell%2!=0
        mt=old!=orig[b.train_ids]
        mq=parent!=full[b.ids]
        # If the champion disagrees it fixes the current release precisely when the latter is wrong.
        target=b.train_error
        # The score quantiles measure evidence against the pending release, in either class direction.
        xt=np.column_stack([b.train_fields,mt])
        xq=np.column_stack([b.query_fields,mq])
        levels=[]
        for side in (False,True):
            active=mt&(old==side)
            levels.append((target[active].sum()+5)/(active.sum()+10))
        directional=np.asarray(levels)[parent.astype(int)]
        exclusive=fit(xt,target,xq,mt.astype(float))
        shared=fit(xt,target,xq,np.where(mt,1.,.15))
        actions={'class_rate':mq&(directional>.5),'paired_fit':mq&(exclusive>.5),
                 'borrow_other_cases':mq&(shared>.5),'both_fits':mq&(exclusive>.5)&(shared>.5),
                 'both_plus_direction_rate':mq&(exclusive>.5)&(shared>.5)&(directional>.5)}
        items.setdefault(b.tag,[]).append((b.ids,b.query_error,parent,mq,b.strict,b.query_cell,actions))
    report={}
    for tag,rows in items.items():
        ids,wrong,parent,diff,strict,cells=[np.concatenate([r[k] for r in rows]) for k in range(6)]
        assert len(ids)==len(np.unique(ids))
        masks=dict(disagreements=diff,now_positive=diff&parent,now_negative=diff&~parent,
                   original_common=diff&strict,added_common=diff&(cells>=4)&~strict)
        result={}
        packet={}
        for name in rows[0][-1]:
            action=np.concatenate([r[-1][name] for r in rows])
            assert not (action&~diff).any()
            packet[name]=action
            result[name]={n:ticker(mask,wrong,action) for n,mask in masks.items()}
        report[tag]=result
        print(tag,{n:(r['disagreements']['fixed'],r['disagreements']['harmed']) for n,r in result.items()},flush=True)
        folder=ROOT/'artifacts/nov26_champion_arbitration'
        folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/f'{tag}.npz',ids=ids,parent=parent,wrong=wrong,disagreement=diff,**packet)
    (ROOT/'reports/nov23_matched_learning_curve/arbitration_against_scored_nine.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
