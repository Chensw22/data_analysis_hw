"""Use confirmed sleepers to anchor a latent cabin-versus-activity outcome mixture."""
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART, ROOT, SEEDS, panel
from conditional_field_value_n23 import field_blocks
from conditional_view_density_n23 import fit_score
from grouped_five_way_coverage_n23 import verify
from relational_error_corrector import write
from spend_simplex_likelihood_n23 import BILLS


def soft_classifier(x, target, query):
    # Fractional binomial responses use the exact weighted positive and
    # negative contributions, retaining a normal classifier at prediction.
    n=len(target)
    return expit(fit_score(np.tile(x,(2,1)),np.r_[np.ones(n),np.zeros(n)],query,None,
                           weight=np.r_[target,1-target]))


def mixture(views,y,t,q,sleep,rule):
    location,activity,gating=views
    allrows=np.r_[t,q]
    sleepers=sleep[t]
    membership=np.where(sleepers,.94,.25)
    for step in range(5):
        loc=expit(fit_score(location[t],y[t],location[allrows],None,weight=membership))
        fac=expit(fit_score(activity[t],y[t],activity[allrows],None,weight=1-membership))
        if rule=="fixed":
            gate=np.where(sleep[allrows],1.,.25)
        else:
            unknown=~sleepers if rule=="anchored" else np.ones(len(t),bool)
            gate=soft_classifier(gating[t[unknown]],membership[unknown],gating[allrows]).clip(.025,.975)
            if rule=="anchored":
                gate[sleep[allrows]]=1.
        if step==4:
            return gate[len(t):]*loc[len(t):]+(1-gate[len(t):])*fac[len(t):],gate[len(t):]
        first=np.where(y[t],loc[:len(t)],1-loc[:len(t)])*gate[:len(t)]
        second=np.where(y[t],fac[:len(t)],1-fac[:len(t)])*(1-gate[:len(t)])
        membership=first/(first+second).clip(1e-12)
        # Mild shrinkage prevents latent branch starvation inside small leaves.
        membership=.96*membership+.02
        if rule in ("fixed","anchored"):
            membership[sleepers]=1.


def run():
    row,y,bref,bank,_,_,_=panel()
    base_hits=(bref>=.5)==y
    bank_hits=(bank>=.5)==y[None,:,None]
    same=(bank_hits.mean(1)>=base_hits.mean(1)[:,None]-.01).all(0)
    weak=~base_hits.any(0)&bank_hits.all(0).any(1)&~bank_hits.all(0)[:,same].any(1)
    deep=~base_hits.any(0)&~bank_hits.any(axis=(0,2))
    assert (weak.sum(),deep.sum())==(335,252)
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{part}.csv",usecols=lambda c:c not in {"Transported","Name"})
                   for part in ("train","test")],ignore_index=True)
    blocks,group=field_blocks(raw,row)
    common=[blocks["origin_destination"],blocks["age_sleep_vip"]]
    location=np.column_stack([*common,blocks["cabin_booking"]])
    activity=np.column_stack([*common,blocks["bill_states"],blocks["bill_amounts"]])
    npos=raw[BILLS].iloc[row].gt(0).sum(axis=1).to_numpy()
    gate=np.column_stack([*common,blocks["bill_states"],np.log1p(raw[BILLS].iloc[row].sum(axis=1)),npos])
    all_inputs=np.column_stack(list(blocks.values()))
    sleepy=raw.CryoSleep.eq(True).to_numpy()[row]
    earth=raw.HomePlanet.eq("Earth").to_numpy()[row]
    awake=raw.CryoSleep.eq(False).to_numpy()[row]
    strata={"all":np.ones(len(row),bool),"earth_sleep_zero":earth&sleepy&(npos==0),
            "earth_awake_multiple_bills":earth&awake&(npos>=2)}
    assert [m.sum() for m in strata.values()]==[6954,1106,2030]
    out=ART/"nov23_berth_activity"
    out.mkdir(exist_ok=True)
    report={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            fold,base=z["fold"],z["base"]
            verify(z,row,y,fold)
        probs={name:np.full(len(y),np.nan) for name in ["direct","fixed","free","anchored"]}
        gates={k:np.full(len(y),np.nan) for k in ("fixed","free","anchored")}
        for f in np.unique(fold):
            t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
            assert not np.intersect1d(group[t],group[q]).size
            probs["direct"][q]=expit(fit_score(all_inputs[t],y[t],all_inputs[q],None))
            for name in gates:
                probs[name][q],gates[name][q]=mixture((location,activity,gate),y,t,q,sleepy,name)
            print(seed,"fold",f,flush=True)
        before=(base>=.5)==y
        metrics={}
        for name,p in probs.items():
            assert np.isfinite(p).all()
            for method,s in [(name,p),(name+"_quarter",.75*base+.25*p)]:
                ok=(s>=.5)==y
                metrics[method]={}
                for scope,m in strata.items():
                    metrics[method][scope]=dict(rows=int(m.sum()),accuracy=float(ok[m].mean()),
                        fixed=int((ok&~before&m).sum()),harmed=int((~ok&before&m).sum()),
                        weak335=int((ok&weak&m).sum()),deep252=int((ok&deep&m).sum()))
        report[seed]=metrics
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**probs,**{"gate_"+k:v for k,v in gates.items()})
        write(ROOT/"reports/nov23_berth_activity/result.json",dict(results=report))
        print(seed,metrics["anchored"]["all"],metrics["anchored_quarter"]["all"],flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
