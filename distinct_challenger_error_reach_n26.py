"""Locate errors shared by frozen releases and count which additional information sources reach them."""
import json
import numpy as np
from threadpoolctl import threadpool_limits

from expanded_joint_error_witnesses_n26 import load_scores
from frozen_queue_shared_errors_n26 import paired
from outside_panel_coverage_n26 import archives, projection
from price_implied_deck_n26 import FEES
from tabnet_attention_reliability_n26 import signals
from target_mix_expert_routing_n26 import ROOT,sources

RESULT=ROOT/'reports/nov23_matched_learning_curve/distinct_challenger_error_reach.json'


def run():
    raw,y,g,f,known,versions=paired()
    parent=versions[:,-1]
    same=known&(versions==parent[:,None]).all(1)
    _,truth,groups,fold,views,pred=projection(include_full=True)
    for a,b in [(truth,y),(groups,g),(fold,f),(views['full'][0],known),(views['full'][1][known],parent[known])]:
        np.testing.assert_array_equal(a,b)
    paid=raw[FEES].gt(0).any(axis=1).to_numpy()
    panel=pred!=parent[:,None]
    panel[~paid,-2:]=False
    old,_=archives(raw,y,g,f,True)
    outside=(np.column_stack(list(old.values()))>=.5)!=parent[:,None]
    additional=load_scores(raw,y,g,{'full':sources()[-1]['full']})['full']
    extra=(additional>=.5)!=parent[:,None]
    networks=signals(raw,y,g,f)[0]
    tabstar=np.empty(len(y)); density=np.empty(len(y))
    for k in range(8):
        t,q=np.flatnonzero(f!=k),np.flatnonzero(f==k)
        for name,key,array in [('original_tabstar/tuned','finetuned',tabstar),('ordinal_factor_density/fold','conditional_bills',density)]:
            with np.load(ROOT/f'artifacts/nov26_{name}{k}.npz') as z:
                np.testing.assert_array_equal(z['q'],q)
                np.testing.assert_array_equal(z['train_y'],y[t])
                if 't' in z:
                    np.testing.assert_array_equal(z['t'],t)
                array[q]=z[key]
    blocks=dict(original_99=panel,other_archives=outside,supplementary=extra,
                tabstar=(tabstar[:,None]>=.5)!=parent[:,None],bill_density=(density[:,None]>=.5)!=parent[:,None],
                tabnet_recipes=(networks>=.5)!=parent[:,None])
    signature=np.column_stack([b.any(1) for b in blocks.values()])
    count=signature.sum(1)
    wrong=y!=parent
    solo=raw.groupby(g).PassengerId.transform('size').eq(1).to_numpy()
    gaps={}
    for name,mask in dict(all=same,solo=same&solo,multiple=same&~solo,
                          awake_paid_negative=same&~parent&paid&raw.CryoSleep.eq(False).to_numpy(),
                          sleeping_no_positive_true=same&parent&~paid&raw.CryoSleep.eq(True).to_numpy()).items():
        gaps[name]={}
        for n,m in {'all':mask,'none':mask&(count==0),'single_block':mask&(count==1),'two_or_more':mask&(count>=2)}.items():
            gaps[name][n]=dict(rows=int(m.sum()),wrong=int((m&wrong).sum()),correct=int((m&~wrong).sum()))
    def counts(m):
        return dict(fixed=int((same&wrong&m).sum()),harmed=int((same&~wrong&m).sum()))
    order={}
    union=np.zeros(len(y),bool)
    for j,name in enumerate(blocks):
        act=signature[:,j]
        order[name]=dict(models=blocks[name].shape[1],any=counts(act),added_in_list_order=counts(act&~union),exclusive=counts(act&(count==1)))
        union|=act
    assert same.sum()==7891 and (same&wrong).sum()==1181
    assert sum(v['added_in_list_order']['fixed'] for v in order.values())+gaps['all']['none']['wrong']==1181
    report=dict(baseline='Public champion and nine November 25-26 candidate decisions before TabNet routing',blocks=order,scopes=gaps,
                interpretation='A model opposing a binary baseline fixes each baseline error it opposes and harms every originally correct case it opposes. Block counts pool potentially related predictions and are descriptive, not a deployable selector.')
    RESULT.write_text(json.dumps(report,indent=2)+'\n')
    print(order,gaps,flush=True)


if __name__=='__main__':
    with threadpool_limits(1):
        run()
