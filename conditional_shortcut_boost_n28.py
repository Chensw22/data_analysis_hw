"""Learn errors left by a strong coarse manifest rule with an order-excluded logistic offset."""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from full_roster_two_view_reliability_n24 import design
from price_implied_deck_n26 import FEES
from repeated_archive_misses_n26 import ART, cohort, count
from unanimous_error_reach_n27 import ROOT, inputs


def baseline(table, train, labels, groups):
    keys = pd.DataFrame({'cell':table[train], 'party':groups[train], 'y':labels[train].astype(int)})
    by_cell = keys.groupby('cell').y.agg(['sum','count'])
    combined = keys.groupby(['cell','party']).y.agg(['sum','count'])
    total = by_cell.reindex(table,fill_value=0).to_numpy()
    same = combined.reindex(pd.MultiIndex.from_arrays([table,groups]),fill_value=0).to_numpy()
    events, members = (total-same).T
    return logit((events+5)/(members+10))


def experiment_inputs():
    raw,frame,y,g,f,first,direction,_,opinions,_=inputs()
    n=len(y)
    parent=first^direction
    groups=frame.PassengerId.str[:4].to_numpy()
    x=np.column_stack(design(frame,groups))
    amounts=frame[FEES]
    paid=amounts.gt(0).any(axis=1)
    dominant=amounts.fillna(0).idxmax(axis=1).where(paid,'none')
    cryo=frame.CryoSleep.astype('string').fillna('?')
    cell=(frame.HomePlanet.fillna('?')+'|'+cryo+'|'+dominant).to_numpy(str)
    _,_,known,fresh=cohort(raw,y,f)
    common=((opinions[:n]>=.5)==parent[:n,None]).all(1)
    wrong=parent[:n]!=y
    masks=dict(all=known,new_orders=known&fresh,common=known&common,
               awake_paid=known&common&raw.CryoSleep.eq(False).to_numpy()&paid.to_numpy()[:n],
               sleep_unpaid=known&common&raw.CryoSleep.eq(True).to_numpy()&~paid.to_numpy()[:n])
    assert (wrong&masks['common']).sum()==862
    ordered=np.zeros(n,int)
    for j,part in enumerate(np.array_split(np.unique(g),8)):
        ordered[np.isin(g,part)]=j
    odds=lambda p:logit(np.clip(p,.002,.998))
    anchor=np.where(parent,1.,-1.)*abs(odds(np.median(opinions[:,1:4],axis=1)))
    return frame,y,f,parent,groups,x,cell,known,fresh,wrong,masks,ordered,anchor


def run():
    frame,y,f,parent,groups,x,cell,known,fresh,wrong,masks,ordered,anchor=experiment_inputs()
    n=len(y)
    report,saved={},{}
    for tag,partition in [('random',f),('ordered',ordered)]:
        scores=np.empty((len(frame),5))
        for k in range(9):
            t=np.flatnonzero(partition!=k)
            q=np.flatnonzero(partition==k) if k<8 else np.arange(n,len(frame))
            assert not np.isin(groups[t],groups[q]).any()
            coarse=baseline(cell,t,y,groups)
            base=lgb.Dataset(x[t],label=y[t],free_raw_data=False)
            raw_scores=[]
            for offset in [np.zeros(len(t)),coarse[t]]:
                def objective(z,data):
                    p=expit(z+offset)
                    return p-y[t],np.maximum(p*(1-p),1e-5)
                params=dict(objective=objective,num_threads=2,verbosity=-1,force_col_wise=True,
                            learning_rate=.045,num_leaves=21,min_data_in_leaf=36,lambda_l2=5)
                model=lgb.train(params,base,num_boost_round=225)
                raw_scores.append(model.predict(x[q],raw_score=True,num_threads=2))
            direct,resid=raw_scores
            scores[q]=np.column_stack([coarse[q],direct,resid+coarse[q],resid+.5*coarse[q],resid])
        saved[tag]=scores
        votes=scores>=0
        dissent=votes!=parent[:,None]
        incremental=dissent[:,2]&((anchor+scores[:,2]-scores[:,1]>=0)!=parent)
        rules=dict(ordinary=dissent[:,1], offset=dissent[:,2], attenuated=dissent[:,3],
                   offset_increment=incremental,
                   challenged=incremental & ~dissent[:,0] & dissent[:,3] & dissent[:,4])
        report[tag]=dict(accuracy=np.mean(votes[:n]==y[:,None],axis=0).tolist(),rules={})
        for name,flag in rules.items():
            saved[tag+'_'+name]=flag
            report[tag]['rules'][name]=dict(scopes={s:count(flag[:n],m,wrong) for s,m in masks.items()},
                accuracy=float(np.mean((parent[:n]^flag[:n])[known]==y[known])),test=int(flag[n:].sum()))
        print(tag,report[tag]['rules']['challenged'],flush=True)
    np.testing.assert_allclose(saved['random'][n:],saved['ordered'][n:])
    dest=ART/'nov28_conditional_shortcut_boost'
    dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'aligned.npz',passenger=frame.PassengerId.to_numpy(str),y=y,fold=f,**saved)
    (ROOT/'reports/nov23_matched_learning_curve/conditional_shortcut_boost.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
