"""Learn whether support specialization identifies wrong incumbent predictions among disputed cases."""
import json

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from exhaustive_family_quorums_n26 import summary
from full_state_context_n26 import OUT as FULL
from latent_passenger_difficulty_n26 import OUT as PANEL,ROOT
from matched_state_context_n26 import inputs,OUT as SMALL

REPORT=ROOT/'reports/nov23_matched_learning_curve/adjudicate_state_support.json'


def run():
    frame,y,g,f,masks=inputs()
    n=len(y)
    with np.load(PANEL/'panel.npz') as panel, np.load(SMALL/'aligned.npz') as small, np.load(FULL/'aligned.npz') as full:
        for z in (panel,small,full):
            np.testing.assert_array_equal(z['passenger'],frame.PassengerId)
            np.testing.assert_array_equal(z['fold'],f)
        base,known,common,fresh=[panel[k] for k in ('anchor','known','common','fresh')]
        # The first 18 columns are family averages, the final three are the actual foundation scores.
        assert list(panel['names'][-3:])==['Mitra','TabICL fare','Nori fare']
        family=panel['scores'][:,:-3]
        forecasters=panel['scores'][:,-3:]
        state=full['p']
        local=small['specific']
        matched=small['matched_prior_restored']
    sign=np.where(base,-1.,1.)
    def score(p):
        return sign*logit(np.clip(p,.001,.999))
    s,l,m=map(score,[state,local,matched])
    a,b,c=[score(col) for col in forecasters.T]
    active=np.isfinite(s)&((s>0)|(l>0)|(a>0)|(b>0)|(c>0))
    # Both paths get the same rows selected before labels; adding local-minus-global
    # and local-minus-matched isolates information in the changed support.
    controls=np.column_stack([base,masks['awake_paid'],a,b,c,score(family.mean(axis=1)),
                              family.std(axis=1),m,np.maximum.reduce([a,b,c])])
    treatment=np.column_stack([controls,s-b,l-m,s-l,np.minimum(s,l),np.maximum(s,l)])
    scopes=dict(all=known,common=common,fresh=known&fresh,hard_awake=common&masks['awake_paid'][:n]&~base[:n],
                hard_sleep=common&masks['sleep_unpaid'][:n]&base[:n])
    test=np.flatnonzero(active[n:])+n
    wrong=(y!=base[:n])
    output={}
    cached={}
    for ordered in [False,True]:
        pred=np.zeros((len(g),2))
        testp=[]
        for k in range(8):
            ids=np.flatnonzero(known&active[:n]&(f==k))
            inner=GroupKFold(3,shuffle=not ordered,random_state=66+k if not ordered else None)
            if ordered:
                blocks=np.array_split(np.unique(g[ids]),3)
                folds=[(ids[~np.isin(g[ids],q)],ids[np.isin(g[ids],q)]) for q in blocks]
            else:
                folds=[(ids[i],ids[j]) for i,j in inner.split(ids,groups=g[ids])]
            for fit,query in [*folds,(ids,test)]:
                assert not np.isin(g[fit],g[query]).any()
                pair=[]
                for x in (controls,treatment):
                    model=make_pipeline(StandardScaler(),LogisticRegression(C=.12,max_iter=400))
                    model.fit(x[fit],wrong[fit])
                    pair.append(model.predict_proba(x[query])[:,1])
                pair=np.column_stack(pair)
                if query[0]>=n:
                    testp.append(pair)
                else:
                    pred[query]=pair
        pred[test]=np.mean(testp,axis=0)
        sham,new=(pred>=.5).T
        actions=dict(control=sham,specialized=new,new_only=new&~sham,removed=sham&~new)
        key='ordered' if ordered else 'random'
        cached[key]=pred
        output[key]={k:dict(test_changes=int(v[n:].sum()),scopes={s:summary(v[:n],wrong,m,f) for s,m in scopes.items()}) for k,v in actions.items()}
    only=[(v[:,1]>=.5)&(v[:,0]<.5) for v in cached.values()]
    for name,v in [('both_new_only',np.logical_and.reduce(only)),('both_specialized',np.logical_and.reduce([v[:,1]>=.5 for v in cached.values()]))]:
        output[name]=dict(test_changes=int(v[n:].sum()),scopes={s:summary(v[:n],wrong,m,f) for s,m in scopes.items()})
    np.savez_compressed(FULL/'adjudication.npz',passenger=frame.PassengerId.to_numpy(str),fold=f,y=y,**cached)
    REPORT.write_text(json.dumps(output,indent=2)+'\n')
    print([(k,[(i,j['scopes']['all']['net']) for i,j in v.items()]) for k,v in output.items() if k in ('random','ordered')])
    print(output['both_new_only'])


if __name__=='__main__':
    run()
