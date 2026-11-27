"""Test whether positions on different decks share residual risk after alignment on the booking manifest."""
import json
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from booking_mix_confidence_transfer_n26 import partition
from exaone_validation_data import compact_features, CATEGORICAL
from full_roster_booking_validation_n23 import ROOT, data
from repeated_archive_misses_n26 import ART, cohort, count

OUT = ART/'nov27_cross_deck_clock'
REPORT = ROOT/'reports/nov23_matched_learning_curve/cross_deck_booking_clock.json'


def geometry(frame):
    cabin=frame.Cabin.str.split('/',expand=True)
    tags=(cabin[0]+cabin[2]).fillna('?').to_numpy(str)
    berth=pd.to_numeric(cabin[1]).to_numpy(float)
    booking=frame.PassengerId.str.split('_').str[0].astype(int).to_numpy()
    clock=np.full((len(frame),3),np.nan)
    for lane in set(tags)-{'?'}:
        ix=np.flatnonzero(tags==lane)
        m=pd.DataFrame({'berth':berth[ix],'booking':booking[ix]}).drop_duplicates()
        m=m.groupby('berth').booking.median().sort_index()
        if len(m)<12:
            continue
        # Estimate a common boarding coordinate from local booking numbers on this lane, without labels.
        shared=m.rolling(17,center=True,min_periods=1).median()
        rank=np.searchsorted(m.index,berth[ix])/max(1,len(m)-1)
        aligned=np.interp(berth[ix],m.index,shared)/max(booking)
        clock[ix]=np.column_stack([aligned,rank,(aligned+.37)%1])
    return tags,clock


def adjustment(t,q,base,y,lanes,clock):
    valid=t[np.isfinite(clock[t,0])]
    residual=y[t]-base
    index=pd.Series(residual,index=t)
    out=np.zeros((len(q),3))
    for lane in np.unique(lanes[q]):
        ask=np.flatnonzero((lanes[q]==lane)&np.isfinite(clock[q,0]))
        ref=valid[lanes[valid]!=lane]
        if not len(ask) or not len(ref):
            continue
        # Shift only reference coordinates for the negative control.
        for j in range(3):
            order=np.argsort(clock[ref,j])
            coord=clock[ref[order],j]
            r=index.loc[ref[order]].to_numpy()
            cumulative=np.r_[0.,np.cumsum(r)]
            loc=clock[q[ask],min(j,1) if j<2 else 0]
            lo,hi=np.searchsorted(coord,loc-.012),np.searchsorted(coord,loc+.012)
            out[ask,j]=(cumulative[hi]-cumulative[lo])/(hi-lo+24)
    return out


def run():
    raw,y,g,f=data()
    _,blind,known,fresh=cohort(raw,y,f)
    n=len(y)
    frame=pd.concat([raw,pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns)],ignore_index=True)
    lanes,clock=geometry(frame)
    x=pd.get_dummies(compact_features(frame),columns=CATEGORICAL,dtype=float).to_numpy()
    ids=frame.PassengerId.to_numpy(str)
    all_g=frame.PassengerId.str.split('_').str[0].to_numpy()
    splits=partition(g,'cross_deck_residual',3)
    def fit(t,q):
        model=HistGradientBoostingClassifier(max_iter=165,max_leaf_nodes=18,l2_regularization=4,early_stopping=False,random_state=513)
        return model.fit(x[t],y[t]).predict_proba(x[q])[:,1]
    base=np.empty(len(frame))
    shifts=np.empty((len(frame),3))
    for k in range(9):
        tr=np.flatnonzero(f!=k)
        q=np.flatnonzero(f==k) if k<8 else np.arange(n,len(frame))
        assert not np.isin(all_g[tr],all_g[q]).any()
        pred=np.empty(len(tr))
        for j in range(3):
            held=splits[tr]==j
            pred[held]=fit(tr[~held],tr[held])
        base[q]=fit(tr,q)
        shifts[q]=adjustment(tr,q,pred,y,lanes,clock)
    p=np.clip(base[:,None]+shifts,.001,.999)
    with np.load(ART/'nov27_dissent_likelihood_transfer/aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        first=z['first']
        parent=first^z['random_median']
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        native=np.median(z['scores'][:,-3:],axis=1)
    link=lambda v:logit(np.clip(v,.001,.999))
    # Carry only the extra spatial odds to the scored anchor's original direction.
    prior=np.where(parent,1.,-1.)*abs(link(native))
    flags=(prior[:,None]+link(p)-link(base)[:,None]>=0)!=parent[:,None]
    plans={name:a for name,a in zip(('aligned','rank','shifted'),flags.T)}
    plans['aligned_only']=flags[:,0]&~flags[:,1]&~flags[:,2]
    plans['aligned_and_rank']=flags[:,0]&flags[:,1]&~flags[:,2]
    report={'standalone_accuracy':dict(zip(('plain','aligned','rank','shifted'),((np.column_stack([base,p])[:n]>=.5)==y[:,None]).mean(0).tolist())),'rules':{}}
    for k,a in plans.items():
        wrong=parent[:n]!=y
        delta=a[:n]*np.where(wrong,1,-1)
        report['rules'][k]=dict(all=count(a[:n],known,wrong),fresh=count(a[:n],fresh&known,wrong),common=count(a[:n],blind,wrong),
                                 by_fold=[int(delta[(f==j)&known].sum()) for j in range(8)],test=int(a[n:].sum()))
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'aligned.npz',passenger=ids,y=y,fold=f,clock=clock,base=base,p=p,**plans)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(report)


if __name__=='__main__':
    with threadpool_limits(1):
        run()
