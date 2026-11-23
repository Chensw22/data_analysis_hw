"""Separate the incremental contribution of joint outcome modeling from the reduced-feature view."""
import numpy as np
from scipy.special import logit

from broad_geographic_transfer_n23 import load
from broad_booking_policy_replay_n23 import interval
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write


def decide(base, ref, joint, direct):
    prior = logit(base.clip(.002, .998))
    delta = logit(joint.clip(.002,.998)) - logit(direct.clip(.002,.998))
    locked = (base >= .5) != ref
    return {k:np.where(locked,ref,v) for k,v in {
        'full_change':prior+delta>=0, 'half_change':prior+.5*delta>=0,
        'quarter_joint':.75*base+.25*joint>=.5, 'quarter_direct':.75*base+.25*direct>=.5}.items()}


def main():
    row,y,_,group,_ = load()
    with np.load(ART/'nov23_release_vote_information/panel.npz') as panel:
        np.testing.assert_array_equal(row,panel['row'])
        wrong=(panel['votes']!=y[None,:,None]).all((0,2))
        right=(panel['votes']==y[None,:,None]).all((0,2))
    plans,hist={},{}
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_joint_observed_state/{seed}.npz') as z:
            np.testing.assert_array_equal(z['row'],row)
            np.testing.assert_array_equal(z['y'],y)
            base,ref,fold=z['base'],z['ref'],z['fold']
            votes={'reference':ref}
            for view in ['basket','sleep','mean']:
                v=decide(base,ref,z[view+'_conditional'],z[view+'_direct'])
                votes.update({view+'_'+key:p for key,p in v.items()})
        h={k:v==y for k,v in votes.items()}
        plans[seed]={k:{**paired(a,ref==y),'fold_net':[int((a[fold==f].astype(int)-(ref==y)[fold==f]).sum()) for f in np.unique(fold)]}
                     for k,a in h.items()}
        if seed!='v131':
            for k,v in h.items():hist.setdefault(k,[]).append(v)
    reference=np.stack(hist['reference'])
    stable={k:dict(accuracy_pct=100*float(np.mean(v)),gain95_pp=interval((np.stack(v).astype(float)-reference).mean(0),group),
                   common_correct=int((np.all(v,axis=0)&wrong).sum()), common_right_lost=int((~np.any(v,axis=0)&right).sum()))
            for k,v in hist.items()}
    write(ROOT/'reports/nov23_adaptive_manifest_splines/state_transfer.json',dict(plans=plans,stable=stable))
    for k in h:
        print(k,[d[k]['fixed']-d[k]['harmed'] for d in plans.values()])


if __name__=='__main__':main()
