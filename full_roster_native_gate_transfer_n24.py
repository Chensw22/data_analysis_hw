"""Apply a fixed, fold-safe spending gate to two independently fitted sets of query scores."""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from full_roster_adaptive_bill_prior_n23 import EVIDENCE,proposal_design
from full_roster_champion_overlap_n24 import aligned
from full_roster_native_prior_gate_n24 import native_scores
from full_roster_repartition_gate_n24 import ROOT,panel_data


def run():
    _,y,g,rows,oldfold,newfold,champion,_,cold,_,_,old,_=aligned()
    raw,labels,groups,fold,names,half,full,baseline,base,cols,rotated=panel_data()
    native=native_scores(raw,labels,groups,rows,oldfold,old)
    old_conf,ox=proposal_design(native[:,:5],native[:,5:])
    new_conf,nx=proposal_design(full[:,:5],full[:,[names.index(k) for k in EVIDENCE]])
    locked=champion!=(native[:,4]>=.5)
    add=~np.isin(np.arange(len(labels)),rows)
    output={}
    for key, panel, source in [('canonical',fold%2,half[:,cols]),('rotated',(fold//2)%2,rotated)]:
        eligible,inputs=proposal_design(source[:,:5],source[:,5:])
        old_flip=np.zeros(len(y),bool)
        new_flip=np.zeros(len(labels),bool)
        for f in np.unique(fold):
            t=(panel==panel[np.flatnonzero(fold==f)[0]])&(fold!=f)&eligible
            assert not np.isin(groups[t],groups[fold==f]).any()
            model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=500))
            model.fit(inputs[t],(source[t,4]>=.5)!=labels[t])
            q=np.flatnonzero((newfold==f)&old_conf&~locked)
            old_flip[q]=model.predict_proba(ox[q])[:,1]>=.5
            q=np.flatnonzero((fold==f)&new_conf)
            new_flip[q]=model.predict_proba(nx[q])[:,1]>=.5
        a,b=champion^old_flip,base^new_flip
        if key=='canonical':
            np.testing.assert_array_equal(b,baseline)
        def describe(pred,reference,target,group,split,mask):
            d=((pred==target).astype(int)-(reference==target))[mask]
            return dict(rows=int(mask.sum()),correct=int((pred[mask]==target[mask]).sum()),fixed=int((d==1).sum()),
                        harmed=int((d==-1).sum()),fold_net=[int(d[split[mask]==f].sum()) for f in np.unique(split[mask])],
                        ci_pp=interval(d,group[mask]))
        output[key]=dict(historical=describe(a,champion,y,g,oldfold,np.ones(len(y),bool)),
                         historical_new_orders=describe(a,champion,y,g,oldfold,cold),
                         full=describe(b,base,labels,groups,fold,np.ones(len(labels),bool)),
                         additional_new_orders=describe(b,base,labels,groups,fold,add&~np.isin(groups,g)),
                         additional_shared_orders=describe(b,base,labels,groups,fold,add&np.isin(groups,g)),
                         matched_eligible=int((old_conf&~locked&new_conf[rows]).sum()),
                         same_flips=int((old_flip&new_flip[rows]).sum()),
                         same_corrected=int((old_flip&new_flip[rows]&(a==y)&(b[rows]==y)).sum()),
                         same_harmed=int((old_flip&new_flip[rows]&(a!=y)&(b[rows]!=y)).sum()))
        print(key,output[key],flush=True)
    path=ROOT/'reports/nov23_matched_learning_curve/native_gate_transfer.json'
    path.write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
