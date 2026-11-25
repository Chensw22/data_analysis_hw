"""Evaluate and freeze the tree correction on the subset with an exact incumbent replay."""
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from export_selected_tree_pair_n25 import NINE as BEST, NAME as PARENT
from export_budget_quarter_n23 import source
from historical_nine_release_n25 import OUT as HISTORY
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT
from full_roster_disagreement_calibration_n24 import stats
from full_roster_repartition_gate_n24 import panel_data
from full_roster_selected_tree_replay_n24 import prepared_vote,original_prediction
from full_roster_booking_validation_n23 import FAMILIES
from full_roster_adaptive_bill_prior_n23 import EVIDENCE
from kaggle_workflow import sha256,validate_submission
from production_anchor_cpu_replay_n25 import certified,cohort,family_contrasts,peer,get_challenger
from paired_mitra_replicas_n25 import routes
from relational_error_corrector import write

NAME='proposal_n25_certified_tree'
AUDIT=ROOT/'artifacts/nov25_production_anchor_cpu'


def structural_inputs(roster):
    _,_,size,x=family_contrasts(roster)
    cabin_path=ROOT/'artifacts/v108_exact_oof/cabin_updates_all_roster.json'
    patched=json.loads(cabin_path.read_text())
    ids=np.arange(len(roster))
    cabin=np.isin(ids,[int(row) for row in patched])
    _,known,_,_=peer.context()
    return (cabin,known,cohort(x),x.CryoSleep.ne('True').to_numpy()&(size>1))


def run():
    raw,y,g,f,names,h,full,_,_,cols,rot=panel_data()
    _,oy,og,rows,of,_,oldbase,_,_,_,_,old,_=aligned()
    test_path=ROOT/'data/raw/test.csv'
    test=pd.read_csv(test_path,usecols=lambda c:c!='Name')
    fields=structural_inputs(pd.concat([raw,test],ignore_index=True))
    ix=np.arange(len(y))
    q=np.arange(len(y),len(y)+len(test))
    j=names.index('mitra96')
    copies=np.column_stack([h[:,j],rot[:,cols.index(j)]])
    new=~np.isin(g,og)
    nine=full[:,[names.index(k) for k in (*FAMILIES,'mitra96',*EVIDENCE)]]
    tree=original_prediction(ix,f,g,y,'full')
    fresh=get_challenger(y,g)
    with np.load(HISTORY/'historical.npz') as z:
        for k,v in [('rows',rows),('fold',of),('y',oy),('v131',oldbase)]:
            np.testing.assert_array_equal(z[k],v)
        oldinc=z['nine']
    report={}
    evals=[]
    for tag,ids,fold,p,inc in [('full',ix,f,full[:,j],prepared_vote(tree,full[:,j]>=.5,nine)),
                               ('historical',rows,of,old[:,0],oldinc)]:
        cert=certified(p,*[v[ids] for v in fields])
        np.testing.assert_equal(cert.sum(),4111 if tag=='full' else 3300)
        if tag=='full':
            with np.load(AUDIT/'oof.npz') as z:
                for k,v in [('row',ids),('fold',fold),('y',y),('certified_v131_equals_v96',cert)]:
                    np.testing.assert_array_equal(z[k],v)
                np.testing.assert_array_equal(inc[cert],z['certified_nine'][cert])
        else:
            np.testing.assert_array_equal(oldbase[cert],(p[cert]>=.5))
        # Only certified rows can be edited; the exact block can be scored without
        # assuming anything about the unobserved incumbent decisions outside it.
        offered=((inc==(p>=.5)) & ((fresh[ids]>=.5)!=inc) &
                 (abs(fresh[ids].clip(.003,.997)-.5)>abs(p.clip(.003,.997)-.5)))
        supported=routes(inc,p,copies[ids],fresh[ids],p>=.5)['tree_without_any_replica']
        opts={'all_offers':cert&offered,'replica_filter':cert&supported}
        entry={}
        for k,edit in opts.items():
            v=stats((inc^edit)[cert],inc[cert],y[ids][cert],g[ids][cert],fold[cert])
            v['certified_rows']=int(cert.sum())
            v['net_pp_entire_roster']=100*v['net']/len(ids)
            if tag=='full':
                mask=cert&new
                v['unseen_orders']=stats((inc^edit)[mask],inc[mask],y[mask],g[mask],fold[mask])
            entry[k]=v
        report[tag]=entry
        evals.append((inc,opts))
    (a,first),(b,second)=evals
    for k in first:
        repeat=first[k][rows]&second[k]
        report.setdefault('matched',{})[k]=dict(fixed=int((repeat&(a[rows]!=oy)&(b!=oy)).sum()),
                                                harmed=int((repeat&(a[rows]==oy)&(b==oy)).sum()))
    p,_,ph=source('proposal_online_first_counts15',test)
    soft,inc,bh=source(BEST,test)
    parent,decision,eh=source(PARENT,test)
    cert=certified(p,*[v[q] for v in fields])
    with np.load(AUDIT/'certified_test.npz') as z:
        np.testing.assert_array_equal(z['passenger'],test.PassengerId)
        np.testing.assert_array_equal(z['certified_v131_equals_v96'],cert)
    spec=dict(parents={BEST:bh,PARENT:eh,'v96':ph},mask_passengers=test.PassengerId[cert].tolist(),
              rule='Use the scored challenger only where all four v131 adjustments are structurally inapplicable.')
    out=ROOT/'artifacts'/NAME
    out.mkdir(exist_ok=True)
    model=out/'eligible_orders.json'
    write(model,spec)
    rebuilt=test.PassengerId.isin(json.loads(model.read_text())['mask_passengers']).to_numpy()
    pred=np.where(cert,parent,soft)
    replay=np.where(rebuilt,parent,soft)
    np.testing.assert_array_equal(pred,replay)
    hard=pred>=.5
    assert cert.sum()==1989 and (hard!=inc).sum()==67 and np.all(hard[cert]==decision[cert])
    pd.DataFrame(dict(PassengerId=test.PassengerId,probability=pred,reload_probability=replay)).to_csv(out/'test_probabilities.csv',index=False)
    pd.DataFrame(dict(PassengerId=test.PassengerId,Transported=hard)).to_csv(out/'submission.csv',index=False)
    check=validate_submission(out/'submission.csv',test_path)
    digests={json.loads(r.read_text()).get('file_sha256') for r in (ROOT/'reports').glob('*/online_attempts/*.json')}
    manifest=dict(**check,version=NAME.removeprefix('proposal_'),parents=spec['parents'],model_sha256=sha256(model),
                  training_rows=len(y),reload_difference=0,eligible=int(cert.sum()),edits_vs_best=int((hard!=inc).sum()))
    for folder in (out,ROOT/'reports'/NAME):
        write(folder/'manifest.json',manifest)
        write(folder/'selection.json',dict(version=manifest['version'],rule=spec['rule'],threshold=.5,
             local_evidence='reports/nov23_matched_learning_curve/certified_tree_release.json'))
    report['test']=dict(eligible=int(cert.sum()),edits=int((hard!=inc).sum()),previously_scored=check['sha256'] in digests)
    write(ROOT/'reports/nov23_matched_learning_curve/certified_tree_release.json',report)
    print({tag:{k:(v['corrected'],v['harmed']) for k,v in report[tag].items()} for tag in ('full','historical')},
          report['matched'],report['test'],flush=True)


if __name__=='__main__':
    with threadpool_limits(2):
        run()
