"""Compare challengers across the whole roster when smaller copies of the anchor support its label."""
import json

import numpy as np
from threadpoolctl import threadpool_limits

from export_selected_tree_pair_n25 import SEEDS
from full_roster_adaptive_bill_prior_n23 import EVIDENCE
from full_roster_booking_validation_n23 import FAMILIES
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT
from full_roster_disagreement_calibration_n24 import stats
from full_roster_repartition_gate_n24 import panel_data
from full_roster_selected_tree_replay_n24 import original_prediction,prepared_vote,refit
from full_roster_six_failure_archive_n24 import prediction_sets


def choose(inc,base,p,rivals,replicas):
    free=(base==inc)&((p>=.5)==inc)
    support=((replicas>=.5)==inc[:,None]).all(1)
    # Every challenger faces the same eligibility rules. The two-tree vote
    # and probability margin are computed without observing passenger outcomes.
    flips=(rivals>=.5)!=inc[:,None]
    offered=free&support
    confident=abs(rivals.clip(.003,.997)-.5)>abs(p.clip(.003,.997)-.5)[:,None]
    return {'all_dissent':flips&offered[:,None],
            'stronger':flips&confident&offered[:,None]}


def run():
    raw,y,g,f,names,h,full,ref,_,cols,rot=panel_data()
    _,oy,og,rows,of,_,historical,_,_,_,_,old,_=aligned()
    ix=np.arange(len(y))
    j=names.index('mitra96')
    replicas=np.column_stack([h[:,j],rot[:,cols.index(j)]])
    a=original_prediction(ix,f,g,y,'full')
    b=original_prediction(rows,of,g,y,'historical')
    nine=full[:,[names.index(k) for k in (*FAMILIES,'mitra96',*EVIDENCE)]]
    current=prepared_vote(a,ref,nine)
    past=prepared_vote(b,historical,old)
    fits=[np.mean([refit(raw,y,g,s) for s in pair],axis=0) for pair in [(1021,1022),SEEDS]]
    columns,panels=prediction_sets(raw,y,g,f,rows,of)
    families=np.array([k.split('/')[0] for k in columns])
    family_names=np.unique(families)
    labels=['original_selected_tree','discovery_pair','fresh_pair','both_pairs_mean',
            'response_channels','projection',*family_names.tolist()]
    report={}
    overlap=[]
    for (tag,ids,fold,inc,base,p,tree),(small,archive) in zip([
        ('full',ix,f,current,ref,full[:,j],a),('historical_three_vote',rows,of,past,historical,old[:,0],b)],panels,strict=True):
        values=np.column_stack([tree,fits[0][ids],fits[1][ids],np.mean(fits,axis=0)[ids],
                                np.median(small[:,:3],axis=1),np.median(small[:,3:],axis=1),
                                *[np.median(archive[:,families==k],axis=1) for k in family_names]])
        methods=choose(inc,base,p,values,replicas[ids])
        strict=((np.column_stack([small,archive])>=.5)==inc[:,None]).all(1)&(inc!=y[ids])
        part={}
        new=~np.isin(g[ids],og)
        for mode,flip in methods.items():
            policies={}
            for i,k in enumerate(labels):
                change=flip[:,i]
                corrected=(inc!=y[ids])&change
                harmed=(inc==y[ids])&change
                policies[k]=dict(corrected=int(corrected.sum()),harmed=int(harmed.sum()),
                                 fold_net=[int(corrected[fold==v].sum()-harmed[fold==v].sum()) for v in np.unique(fold)],
                                 common_fixed=int((change&strict).sum()))
                if tag=='full':
                    policies[k]['new_orders_net']=int(corrected[new].sum()-harmed[new].sum())
            part[mode]=policies
        overlap.append((inc,methods))
        report[tag]=part
    (inc,fp),(oldbase,hp)=overlap
    report['matched']={}
    for mode in fp:
        same=fp[mode][rows]&hp[mode]
        correct=((inc[rows]!=oy)&(oldbase!=oy))[:,None]&same
        harm=((inc[rows]==oy)&(oldbase==oy))[:,None]&same
        report['matched'][mode]={k:[int(correct[:,n].sum()),int(harm[:,n].sum())] for n,k in enumerate(labels)}
    path=ROOT/'reports/nov23_matched_learning_curve/replica_supported_reference.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    for mode in fp:
        for k in labels:
            v=report['full'][mode][k]
            oldv=report['historical_three_vote'][mode][k]
            if k in labels[:4] or min(v['corrected']-v['harmed'],oldv['corrected']-oldv['harmed'],v['new_orders_net'])>0:
                print(mode,k,v['corrected'],v['harmed'],oldv['corrected'],oldv['harmed'],v['new_orders_net'],flush=True)


if __name__=='__main__':
    with threadpool_limits(2):
        run()
