"""Learn piecewise cabin-location offsets on training-only residuals of Earth sleepers."""
import numpy as np
import pandas as pd
from scipy.special import expit
from pathlib import Path
import relational_error_corrector as ref

ROOT=Path(__file__).resolve().parent


def fit_segments(position, raw, y):
    # Up to 12 candidate intervals per side. BIC charges an offset and location for each extra segment.
    boundaries=np.unique(np.quantile(position,np.linspace(0,1,13)))[1:-1]
    cells=np.searchsorted(boundaries,position)
    bins=len(boundaries)+1
    cost=np.full(bins+1,np.inf)
    cost[0]=0
    paths={0:[]}
    for end in range(1,bins+1):
        for start in range(end):
            mask=(cells>=start)&(cells<end)
            if mask.sum()<24 or not np.isfinite(cost[start]):
                continue
            offset=ref.logit(raw[mask]).astype(float)
            shift=0.
            for _ in range(12):
                p=expit(offset+shift)
                step=((p-y[mask]).sum()+shift)/(np.sum(p*(1-p))+1.)
                shift-=step
                if abs(step)<1e-8:
                    break
            log_odds=offset+shift
            total=cost[start]+np.sum(np.logaddexp(0,log_odds)-y[mask]*log_odds)+.5*shift**2
            if start:
                total+=np.log(len(y))
            if total<cost[end]:
                cost[end]=total
                paths[end]=paths[start]+[(end,shift)]
    segments=paths[bins]
    return [float(boundaries[end-1]) for end,_ in segments[:-1]],[float(delta) for _,delta in segments]


def run():
    inputs,splits,_,bases,evidence=ref.overlap.load_authenticated()
    dev,y=inputs["dev"],inputs["y"]
    x=inputs["features"].iloc[dev]
    eligible=(x.HomePlanet.eq("Earth")&x.CryoSleep.eq("True")&x.Deck.eq("G")&x.CabinNumber.notna()).to_numpy()
    side=x.Side.to_numpy()
    pos=x.CabinNumber.to_numpy()
    raw=bases["rawMitra"]
    updated=raw.astype(float).copy()
    fits=[]
    for fold,(a,b) in enumerate(splits):
        inner=ref.load_inner(fold,a,b,dev,y,evidence)
        for s in ["P","S"]:
            tr=eligible[a]&(side[a]==s)
            va=b[eligible[b]&(side[b]==s)]
            cuts,shifts=fit_segments(pos[a][tr],inner[tr],y[a][tr])
            updated[va]=expit(ref.logit(raw[va])+np.array(shifts)[np.searchsorted(cuts,pos[va])])
            fits.append(dict(fold=fold,side=s,cuts=cuts,offsets=shifts))
    cat=pd.read_csv(ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(cat.row_id,dev)
    hard=cat.all_recipes_still_wrong.to_numpy(bool)
    result=dict(metrics=ref.score(updated,y),vs_raw=ref.pair(updated,raw,y),
                eligible=ref.pair(updated,raw,y,eligible),hard=ref.pair(updated,raw,y,hard),
                folds=[ref.pair(updated[b],raw[b],y[b])["net_correct"] for _,b in splits],fits=fits)
    ref.write(ROOT/"reports/top3_stage/cabin_change_points.json",result)
    print(result)


if __name__=="__main__":
    run()
