"""Volume-matched comparison of new methods after accounting for 50 previous classifier votes."""
import numpy as np
from broad_error_placement_control_n23 import read,SOURCES
from broad_booking_policy_n23 import EXTRA
from archive_error_recovery_n23 import ART,ROOT,SEEDS,panel,predictions
from relational_error_corrector import write


def run():
    row,y,base,archive,_,_,_=panel()
    extra=[predictions(row,y,s,EXTRA) for s in SEEDS]
    historical=np.concatenate([archive,np.stack([v[1] for v in extra])],axis=2)>=.5
    def get(folder,key):
        return np.stack([read(row,y,extra,i,folder,key) for i in range(len(SEEDS))])
    previous=np.stack([get(*SOURCES[k]) for k in ("shared","bills","rotation","mondrian")],axis=2)
    ref=base>=.5
    correct=ref==y
    repeat=~correct.any(0)
    old=repeat&~(historical==y[None,:,None]).all(0).any(1)
    remaining=old&~(previous==y[None,:,None]).all(0).any(1)
    never=repeat&~(historical==y[None,:,None]).any((0,2))
    assert old.sum()-remaining.sum()==58
    sources={n:("broad_passenger_diffusion",n) for n in ("direct","loss_vote","loss_fields","base_direct","base_direct_loss")}
    sources.update({n:("broad_diffusion_moments",n) for n in ("mean","mean_cov")})
    sources.update({"raw_after_panel":("broad_conditional_raw_signal","full_panel_real_fields"),
                    "weighted_after_panel":("broad_honest_field_weights","full_panel_honest"),
                    "conditional_ratio":("broad_label_ratio","transfer")})
    bank=np.stack([get(*item) for item in sources.values()])
    def count(v):
        good=v==y
        stable=good.all(1)
        counts=np.column_stack([(stable&remaining).sum(1),(stable&never).sum(1),(stable&repeat).sum(1),
                               (good.mean(1)-correct.mean(0)).sum(1)])
        union=stable[:7].any(0)
        return np.vstack([counts,[(union&remaining).sum(),(union&never).sum(),(union&repeat).sum(),0]])
    archive50=np.concatenate([historical,previous],axis=2)
    dissent=(archive50!=ref[:,:,None]).mean((0,2))
    confidence=np.maximum(base,1-base).mean(0)
    codes=(ref*np.array([1,2,4])[:,None]).sum(0)
    strata=codes*20+4*np.digitize(confidence,np.quantile(confidence,[.2,.4,.6,.8]))+np.digitize(dissent,[1e-9,.05,.2])
    blocks=[np.flatnonzero(strata==s) for s in np.unique(strata)]
    rng=np.random.default_rng(841)
    draws=[]
    for _ in range(800):
        shuffle=np.arange(len(y))
        for k in blocks:
            shuffle[k]=rng.permutation(k)
        draws.append(count(bank[:,:,shuffle]))
    simulated=np.stack(draws)
    actual=count(bank)
    result={}
    for i,n in enumerate([*sources,"diffusion_union"]):
        result[n]={}
        for j,k in enumerate(("new_after58","historical212","persistent","mean_net")):
            if k=="mean_net" and n=="diffusion_union":
                continue
            values=simulated[:,i,j]
            result[n][k]=dict(observed=float(actual[i,j]),null_mean=float(values.mean()),
                              central95=np.quantile(values,[.025,.975]).tolist())
    write(ROOT/"reports/nov23_broad_conditional_raw_signal/placement.json",dict(remaining=int(remaining.sum()),results=result))
    print(result["diffusion_union"])


if __name__=="__main__":
    run()
