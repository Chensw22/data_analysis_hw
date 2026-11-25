"""Calibrate independently trained model families using labels their first-stage fits never saw."""
import json
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from nested_release_review_n26 import OUT as NESTED
from outside_panel_coverage_n26 import archives, projection
from held_expert_portfolios_n26 import ROOT, sources, panel_predictions, ticker


def fit_probability(x,train,query,y):
    mean=x[train].mean(0)
    deviation=x[train].std(0).clip(.35)
    tr=((x[train]-mean)/deviation).clip(-6,6)
    ask=((x[query]-mean)/deviation).clip(-6,6)
    return LogisticRegression(C=.15,max_iter=500).fit(tr,y[train]).predict_proba(ask)[:,1]


def run():
    raw,y,g,f,views,original=projection()
    _,_,_,panels=sources()
    extras,_=archives(raw,y,g,f,True)
    # EXAONE uses another partition, so its support could contain other people in this fold.
    families=json.loads((ROOT/'reports/nov23_matched_learning_curve/outside_panel_coverage.json').read_text())['family_votes']
    focus=panel_predictions(panels['full'],'full')
    base=np.column_stack([focus[n] for n in ['neural_deck_raw','tree_deck_raw','neural_deck_fine','tree_deck_fine',
                        'neural_route_exact','tree_route_exact','register_all','register_paid','sequence_paid','swap']])
    convert=lambda x:logit(np.clip(x,.005,.995))
    original_x=convert(base)
    center=np.median(original_x,axis=1)
    designs={'reference':original_x}
    ext={}
    for name in ['Mitra','TabDPT','LimiX2','mask_net','bill_flow','profiles']:
        col=convert(np.mean([extras[n] for n in families[name]],axis=0))
        ext[name]=np.column_stack([col,(col-center),np.abs(col-center)])
    registry=['Mitra','TabDPT','LimiX2']
    gen=['mask_net','bill_flow','profiles']
    for name,fs in [('foundation',registry),('generative',gen),('all_outside',registry+gen)]:
        designs[name]=np.column_stack([original_x,*[ext[k] for k in fs]])
    report={}
    # The external accuracy signals in any one meta-fit all come from the same group-held base fold.
    # Thus neither the held queries nor their booking peers were used to fit the expert inputs of that meta-fit.
    for tag in ['bookingmix','historical_to_new']:
        mask,reference=views[tag]
        fresh=views['historical_to_new'][0]
        records=[]
        for k in np.unique(f):
            members=np.flatnonzero(f==k)
            if tag=='bookingmix':
                splits=GroupKFold(3,shuffle=True,random_state=7421).split(members,groups=g[members])
                pairs=[(members[t],members[q][mask[members[q]]]) for t,q in splits]
            else:
                train=members[~np.isin(g[members],g[fresh])]
                pairs=[(train,members[mask[members]])]
            for t,q in pairs:
                if not len(q):
                    continue
                assert not np.isin(g[t],g[q]).any()
                preds={n:fit_probability(x,t,q,y)>=.5 for n,x in designs.items()}
                actions={n:vote!=reference[q] for n,vote in preds.items()}
                for n in ['foundation','generative','all_outside']:
                    actions[n+'_only']=actions[n]&~actions['reference']
                actions['two_sources']=actions['generative_only']&actions['foundation_only']
                records.append((q,actions))
        ids=np.concatenate([r[0] for r in records])
        np.testing.assert_array_equal(np.sort(ids),np.flatnonzero(mask))
        base=reference[ids]
        unanimous=(original[ids]==base[:,None]).all(1)
        scope={'all':np.ones(len(ids),bool),'all_99_agree':unanimous,'some_dissent':~unanimous}
        r={n:{key:ticker(m,base!=y[ids],np.concatenate([i[1][n] for i in records])) for key,m in scope.items()}
           for n in records[0][1]}
        NESTED.mkdir(exist_ok=True)
        np.savez_compressed(NESTED/f'outsiders_{tag}.npz',ids=ids,wrong=base!=y[ids],
                            **{n:np.concatenate([i[1][n] for i in records]) for n in records[0][1]})
        report[tag]=r
        print(tag,[(n,v['all']['fixed'],v['all']['harmed']) for n,v in r.items()],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/within_fold_outside_evidence.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
