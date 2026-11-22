"""Separate marginal path instability from excess errors at comparable final scores."""
import numpy as np
from broad_neural_path_n23 import OUT,REPORT,SEEDS,write


def run():
    report={}
    for name in [*map(str,SEEDS),"v131"]:
        with np.load(OUT/f"{name}.npz") as z:
            path=z["trajectories"].mean(1)
            base,ref,y,fold=z["base"],z["ref"],z["y"],z["fold"]
        changed=((path>=.5)!=(path[:,[-1]]>=.5)).any(1)
        errors=ref!=y
        code=10000*fold.astype(int)+100*(base*20).astype(int)+(path[:,-1]*20).astype(int)
        rng=np.random.default_rng(245)
        draws=np.zeros(4000,int)
        for key in np.unique(code):
            c=code==key
            err=int(errors[c].sum())
            n=int(changed[c].sum())
            if n:
                draws+=rng.hypergeometric(err,int(c.sum())-err,n,size=len(draws))
        report[name]=dict(changed=int(changed.sum()),changed_errors=int(errors[changed].sum()),
                          changed_error_rate=float(errors[changed].mean()),unchanged_error_rate=float(errors[~changed].mean()),
                          matched_mean=float(draws.mean()),random_interval=np.quantile(draws,[.025,.975]).tolist(),cells=int(np.unique(code).size))
    write(REPORT/"matched.json",report)


if __name__=="__main__":
    run()
