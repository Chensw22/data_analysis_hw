"""Incrementally reveal any spending, the service pattern and the positive amounts."""
import json

import numpy as np
from scipy.special import logit
import torch
from torch.nn import functional as F
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,banks,data,latest_reference
from full_roster_gate_probability_n23 import gate_probabilities
from full_roster_same_model_calibration_n23 import learn
from full_roster_zero_truncated_bills_n24 import inputs,fit,pattern_log_likelihood


def reveal(model,args,mode):
    x,observed,paid,amt,asleep=args
    latent=model.body(x)
    prior=F.log_softmax(model.prior(latent),dim=1)
    params=model.params(latent).reshape(len(x),2,16)
    def likelihood(mask):
        return pattern_log_likelihood(params[:,:,:5],params[:,:,5],observed[:,None,:],mask[:,None,:],mode,asleep)
    empty=likelihood(torch.zeros_like(paid)).clamp_max(0)
    any_service=torch.where(paid.any(1)[:,None],torch.log((-torch.expm1(empty)).clamp_min(1e-12)),empty)
    states=[prior,prior+any_service,prior+likelihood(paid),model(*args,mode)]
    return torch.stack([s.softmax(1)[:,1] for s in states],dim=1)


def calibrate(prob,anchor,protect,y,g,fold):
    z=logit(prob.clip(1e-6,1-1e-6))
    effects=np.diff(z,axis=1)
    offset=logit(anchor.clip(.001,.999))
    features={'single_score':(z[:,3]-offset)[:,None],
              'activity':effects[:,:1],'services':effects[:,1:2],'amounts':effects[:,2:3],
              'separate':np.column_stack([z[:,0]-offset,effects])}
    outputs={}
    for key,x in features.items():
        out=protect.copy()
        for f in np.unique(fold):
            outer=np.flatnonzero(fold==f)
            for a,b in GroupKFold(4,shuffle=True,random_state=462).split(outer,groups=g[outer]):
                t,q=outer[a],outer[b]
                assert not np.isin(g[t],g[q]).any()
                # Fit without reading historical override flags from calibration records.
                scale=np.maximum(x[t].std(0),.1)
                w=learn(x[t]/scale,y[t],offset[t],np.zeros(x.shape[1]))
                q=q[protect[q]==(anchor[q]>=.5)]
                out[q]=offset[q]+(x[q]/scale)@w>=0
        outputs[key]=out
    return outputs


def run():
    _,y,g,rows,oldfold,_,champ,_,_,_,_,old,_=aligned()
    raw,labels,groups,fold=data()
    names,half,full=banks(raw,labels,fold)
    ref=latest_reference(labels,groups,fold,names,half,full)
    gate,_=gate_probabilities(names,half,full,labels,groups,fold)
    x,bills,sleep=inputs(raw)
    predictions=[]
    for tag,ix,splits in [('historical',rows,oldfold),('full',np.arange(len(raw)),fold)]:
        prob=np.empty((len(ix),4))
        for f in np.unique(splits):
            q=ix[splits==f]
            t=ix[(splits!=f)&~np.isin(groups[ix],groups[q])]
            assert not np.isin(groups[t],groups[q]).any()
            prob[splits==f]=fit(x,bills,sleep,labels,t,q,'truncated',1981+int(f),reveal)
        with np.load(ROOT/'artifacts/nov24_zero_truncated/truncated.npz') as archive:
            np.testing.assert_array_equal(archive[tag],prob[:,3])
        predictions.append(prob)
    old_p,new_p=predictions
    a=calibrate(old_p,old[:,0],champ,y,g,oldfold)
    b=calibrate(new_p,gate,ref,labels,groups,fold)
    shared=((full>=.5)!=labels[:,None]).all(1)
    hard=shared[rows]&(champ!=y)
    unseen=~np.isin(groups,g)
    def describe(z,base,target,group,splits):
        d=(z==target).astype(int)-(base==target)
        return dict(fixed=int((d==1).sum()),harmed=int((d==-1).sum()),accuracy=float((z==target).mean()),
                    ci_pp=interval(d,group),fold_net=[int(d[splits==f].sum()) for f in np.unique(splits)])
    report={}
    for k,score in b.items():
        report[k]=dict(historical=describe(a[k],champ,y,g,oldfold),full=describe(score,ref,labels,groups,fold),
                       unseen=describe(score[unseen],ref[unseen],labels[unseen],groups[unseen],fold[unseen]),
                       common_fixed=int((shared&(score==labels)).sum()),
                       stable_hard_fixed=int((hard&(a[k]==y)&(score[rows]==y)).sum()))
        print(k,[(r,report[k][r]['fixed'],report[k][r]['harmed']) for r in ['historical','full','unseen']],flush=True)
    path=ROOT/'reports/nov23_matched_learning_curve/spending_evidence.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(2):
        run()
