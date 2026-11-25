"""Retest archived raw-data challengers against all nine historical votes and exact solo decisions."""
import json

import numpy as np
from threadpoolctl import threadpool_limits

from certified_tree_release_n25 import AUDIT,HISTORY,structural_inputs
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,data
from full_roster_six_failure_archive_n24 import prediction_sets
from production_anchor_cpu_replay_n25 import certified
import pandas as pd


def run():
    raw,y,g,f=data()
    _,oy,og,rows,of,_,oldbase,_,_,_,_,old,_=aligned()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=lambda c:c!='Name')
    fields=structural_inputs(pd.concat([raw,test],ignore_index=True))
    with np.load(AUDIT/'oof.npz') as z:
        for k,v in [('row',np.arange(len(y))),('y',y),('fold',f)]:
            np.testing.assert_array_equal(z[k],v)
        cert=z['certified_v131_equals_v96']
        fullbase=z['v96']
        fullnine=z['certified_nine'][cert].astype(bool)
    with np.load(HISTORY/'historical.npz') as z:
        np.testing.assert_array_equal(z['rows'],rows)
        np.testing.assert_array_equal(z['y'],oy)
        oldnine=z['nine']
    hc=certified(old[:,0],*[col[rows] for col in fields])
    np.testing.assert_array_equal(oldbase[hc],old[:,0][hc]>=.5)
    cols,panels=prediction_sets(raw,y,g,f,rows,of)
    families=np.array([key.split('/')[0] for key in cols])
    group=np.unique(families)
    keys=['response_channels','random_projections',*group.tolist()]
    changed=[]
    results={}
    for (tag,mask,p,inc,truth,fold,ids),pair in zip([
        ('entire_certified',cert,fullbase[cert],fullnine,y[cert],f[cert],np.flatnonzero(cert)),
        ('historical_certified',hc,old[hc,0],oldnine[hc],oy[hc],of[hc],rows[hc])],panels,strict=True):
        six,archive=[table[mask] for table in pair]
        rival=np.column_stack([np.median(six[:,:3],axis=1),np.median(six[:,3:],axis=1),
                               *[np.median(archive[:,families==k],axis=1) for k in group]])
        free=(p>=.5)==inc
        new=~np.isin(g[ids],og)
        pred={'direct':rival>=.5,'quarter':(.75*p[:,None]+.25*rival)>=.5}
        dissent={mode:free[:,None]&(z!=inc[:,None]) for mode,z in pred.items()}
        changed.append((ids,inc,dissent))
        for mode,choices in dissent.items():
            tab={}
            for k,edit in zip(keys,choices.T,strict=True):
                agree=np.where(inc==truth,-1,1)*edit
                tab[k]=dict(corrected=int((agree==1).sum()),harmed=int((agree==-1).sum()),
                            fold_net=[int(agree[fold==v].sum()) for v in np.unique(fold)])
                if tag=='entire_certified':
                    tab[k]['new_orders_net']=int(agree[new].sum())
            results.setdefault(tag,{})[mode]=tab
    a,ia,ca=changed[0]
    b,ib,cb=changed[1]
    shared,ai,bi=np.intersect1d(a,b,return_indices=True)
    for mode in ca:
        agree=ca[mode][ai]&cb[mode][bi]
        right=agree&((ia[ai]!=y[shared])&(ib[bi]!=y[shared]))[:,None]
        wrong=agree&((ia[ai]==y[shared])&(ib[bi]==y[shared]))[:,None]
        results.setdefault('same_passengers',{})[mode]={k:dict(corrected=int(right[:,i].sum()),harmed=int(wrong[:,i].sum())) for i,k in enumerate(keys)}
        for k in keys:
            here=results['entire_certified'][mode][k]
            hist=results['historical_certified'][mode][k]
            if min(here['corrected']-here['harmed'],hist['corrected']-hist['harmed'],here['new_orders_net'])>0:
                print(mode,k,here,hist,results['same_passengers'][mode][k],flush=True)
    results['populations']=dict(full=len(a),historical=len(b),intersection=len(shared))
    (ROOT/'reports/nov23_matched_learning_curve/archived_challengers_true_nine.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
