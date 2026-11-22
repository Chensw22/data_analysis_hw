"""Learn whether a third-order prediction adds actionable evidence beyond lower-order models."""
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from conditional_calibration_n23 import offsets
from grouped_five_way_coverage_n23 import verify
from population_rescue_map_n23 import ART, ROOT, SEEDS, read_panel
from relational_error_corrector import write


def run():
    row=pd.read_csv(ART/"nov23_global_error_inventory.csv").row.to_numpy()
    group=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["PassengerId"]).PassengerId.iloc[row].str.split("_").str[0].to_numpy()
    y, hits, others=read_panel()
    persistent=~hits.any(0)
    hard=persistent & ~others.any(axis=(0,1))
    out=ART/"nov23_selective_fields"
    out.mkdir(exist_ok=True)
    results={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_field_factors/{seed}.npz") as data:
            fold,base=data["fold"],data["base"]
            verify(data,row,y,fold)
            models=np.stack([data[f"order{i}"] for i in [1,2,3]],axis=1)
        scores=logit(np.column_stack([base,models]).clip(.005,.995))
        z=scores[:,0]
        first,second,third=np.diff(scores,axis=1).T
        strength=np.minimum(abs(z),4)
        # Compare successive corrections relative to the preceding model's prediction.
        # Interactions let the strength/direction of the incumbent's opinion modulate
        # the new evidence. All row labels used hereafter come from calibration bookings.
        common=np.column_stack([np.ones(len(y)),z,first,second,first*strength,second*strength,second*np.sign(z)])
        new=np.column_stack([third,third*strength,third*np.sign(z),third*np.sign(second)])
        probabilities={key:np.full(len(y),np.nan) for key in ["lower_order","all_orders","permuted_higher_order"]}
        rng=np.random.default_rng(199)
        for k in np.unique(fold):
            pool=np.flatnonzero(fold==k)
            assert not np.intersect1d(group[pool],group[fold!=k]).size
            # Within these cells the reference and the order-two classifier have similar opinions.
            # Permuting order-three innovations in each cell tests their passenger-specific content.
            cells=np.digitize(base,[.2,.4,.6,.8])*3+np.digitize(models[:,1],[.33,.67])
            shuffled=new.copy()
            for c in np.unique(cells[pool]):
                ids=pool[cells[pool]==c]
                shuffled[ids]=new[rng.permutation(ids)]
            for a,b in GroupKFold(3,shuffle=True,random_state=117).split(pool,groups=group[pool]):
                t,q=pool[a],pool[b]
                assert not np.intersect1d(group[t],group[q]).size
                for key,extra in [("lower_order",None),("all_orders",new),("permuted_higher_order",shuffled)]:
                    x=common if extra is None else np.column_stack([common,extra])
                    penalty=np.full(x.shape[1],20.)
                    penalty[:2]=2.
                    probabilities[key][q]=offsets(x,z,y,t,q,penalty)
        old=(base>=.5)==y
        metrics={}
        for key,p in probabilities.items():
            assert np.isfinite(p).all()
            ok=(p>=.5)==y
            metrics[key]=dict(accuracy=float(ok.mean()),fixed=int((ok&~old).sum()),harmed=int((~ok&old).sum()),
                              recovered1070=int(ok[persistent].sum()),recovered466=int(ok[hard].sum()))
        results[seed]=metrics
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**probabilities)
    write(ROOT/"reports/nov23_selective_fields/result.json",dict(results=results,
          calibration="Whole bookings cross-fitted within each held-out base fold. Diagnostic error labels and masks were excluded from inputs."))
    print(results,flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
