"""Compare unanimous errors to their actual denominators at matched confidence and direction."""
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,BILLS,banks,data,latest_reference
from full_roster_six_failure_archive_n24 import prediction_sets


def groups(raw,g):
    fees = raw[BILLS]
    count = fees.gt(0).sum(axis=1)
    highest = fees.fillna(0).idxmax(axis=1).where(count>0,'none')
    head = highest.str.split().str[0]
    cat = dict(planet=raw.HomePlanet,destination=raw.Destination,sleep=raw.CryoSleep,
               deck=raw.Cabin.str.split('/').str[0],side=raw.Cabin.str.split('/').str[2],
               positive_services=count.clip(upper=3),largest_service=head,
               order_size=raw.groupby(g).PassengerId.transform('size').clip(upper=3),
               age=pd.cut(raw.Age,[-1,12,17,29,49,100]),missing_bill=fees.isna().any(axis=1))
    names,columns = [],[]
    for field,values in cat.items():
        values = values.astype('string').fillna('?').to_numpy(str)
        for val in np.unique(values):
            names.append(field+'='+val)
            columns.append(values==val)
    return names,np.column_stack(columns)


def compare(mask,err,p,g,fold):
    # Center both covariates inside each score band; retain the booking as the
    # resampling unit for the simultaneous, descriptive uncertainty check.
    band = np.digitize(p,np.r_[np.arange(.5,.98,.025),.99,.997])
    cells = np.unique(np.column_stack([fold,band]),axis=0,return_inverse=True)[1].ravel()
    expected = np.zeros(mask.shape[1])
    contribution = np.zeros(mask.shape)
    for k in np.unique(cells):
        s = cells==k
        expected += mask[s].sum(0)*err[s].mean()
        contribution[s] = (mask[s]-mask[s].mean(0))*(err[s,None]-err[s].mean())
    counts = mask.sum(0)
    failures = err.astype(int)@mask.astype(int)
    d = failures-expected
    np.testing.assert_allclose(contribution.sum(0),d,atol=1e-9)
    orders,inv = np.unique(g,return_inverse=True)
    scores = np.zeros((len(orders),mask.shape[1]))
    np.add.at(scores,inv,contribution)
    scores -= scores.mean(0)
    se = np.sqrt((scores*scores).sum(0))
    use = (counts>=40)&(expected>=5)&(se>1e-9)
    assert use.any()
    score = np.divide(d,se,out=np.zeros_like(d),where=se>0)
    rng = np.random.default_rng(975)
    signs = rng.choice([-1.,1.],(999,len(orders)))
    maxima = np.abs(signs@scores[:,use]/se[use]).max(1)
    return [dict(rows=int(n),wrong=int(e),expected=float(v),excess=float(diff),
           eligible=bool(ok),within_direction_adjusted_p=(float((1+(maxima>=abs(z)).sum())/1000) if ok else None))
            for n,e,v,diff,ok,z in zip(counts,failures,expected,d,use,score)]


def run():
    raw,y,g,fold = data()
    _,_,_,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    _,panels = prediction_sets(raw,y,g,fold,rows,of)
    labels,mask = groups(raw,g)
    report = {}
    for (tag,ix,f,b,p),(six,arch) in zip([('full',np.arange(len(y)),fold,ref,full[:,names.index('mitra96')]),
                 ('historical',rows,of,champ,old[:,0])],panels,strict=True):
        strict = ((np.c_[six,arch]>=.5)==b[:,None]).all(1)
        observed = b!=y[ix]
        report[tag] = {}
        for sign in (False,True):
            s = strict & (b==sign)
            r = dict(zip(labels,compare(mask[ix][s],observed[s],np.where(b,p,1-p)[s],g[ix][s],f[s]),strict=True))
            report[tag][str(sign)] = dict(rows=int(s.sum()),wrong=int((s&observed).sum()),profiles=r)
            print(tag,sign,sorted([(k,v['wrong'],v['rows'],round(v['excess'],1),v['within_direction_adjusted_p'])
                      for k,v in r.items() if v['eligible']],key=lambda a:a[-1])[:4],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/unanimous_error_profiles.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
