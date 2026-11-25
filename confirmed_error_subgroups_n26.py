"""Discover recurring understandable error groups and independently estimate their error rates."""
import json
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from audit_fare_routes_n26 import DECK, held_fare
from export_fare_deck_n26 import combine
from full_roster_booking_validation_n23 import ROOT, data
from price_implied_deck_n26 import FEES
from solo_archive_recovery_n26 import ticker
from tabicl_consensus_review_n26 import OUT as REVIEW, sources, splits


def source_terms(frame, support):
    names, masks = [], []
    for key in frame:
        col = frame[key]
        if col.dtype.kind in 'biuf':
            cuts = np.unique(col.iloc[support].dropna().quantile([.25,.50,.75]))
            terms = [(f'{key} <= {val:.4g}', col <= val) for val in cuts]
            terms += [(f'{key} > {val:.4g}', col > val) for val in cuts]
        else:
            terms = [(f'{key} = {val}', col == val) for val in sorted(col.iloc[support].dropna().unique())]
        for name, values in terms:
            m = values.to_numpy(bool)
            if 5 <= m[support].sum() <= len(support)-5:
                names.append(name)
                masks.append(m)
    return names, np.array(masks)


def find_groups(frame, labels, train, confirm, query):
    names, masks = source_terms(frame, train)
    base = (labels[train].sum()+1)/(len(train)+2)
    if not len(masks):
        return np.full(len(query),base),[]
    train_mask=masks[:,train].astype(float)
    total=train_mask@train_mask.T
    positives=train_mask[:,labels[train]]@train_mask[:,labels[train]].T
    n=np.maximum(1,total)
    smooth=(positives+8*base)/(n+8)
    utility=(smooth-base)*np.sqrt(n)
    allowed=np.triu((total>=6)&(total<=len(train)*.75)&(smooth>base))
    choices=np.argwhere(allowed)
    order=np.argsort(-utility[allowed],kind='stable')
    panel=[]
    for i,j in choices[order]:
        keep=masks[i]&masks[j]
        region=keep[train]
        if any(np.sum(region&prev)/np.sum(region|prev)>.65 for prev,_,_ in panel):
            continue
        expression=names[i] if i==j else names[i]+' & '+names[j]
        panel.append((region,keep,expression))
        if len(panel)==10:
            break
    baseline=(labels[confirm].sum()+1)/(len(confirm)+2)
    score=np.full(len(query),baseline)
    rules=[]
    for _,keep,expression in panel:
        hit=confirm[keep[confirm]]
        if not len(hit):
            continue
        risk=(labels[hit].sum()+8*baseline)/(len(hit)+8)
        score[keep[query]]=np.maximum(score[keep[query]],risk)
        rules.append(dict(rule=expression,confirmation_rows=len(hit),confirmation_errors=int(labels[hit].sum()),
                          estimated_risk=float(risk)))
    return score,rules


def run():
    raw,y,g,_=data()
    panels=sources()[-1]
    fare={tag:held_fare(v,tag,with_routes=False) for tag,v in panels.items()}
    with np.load(DECK/'deck_estimates.npz') as z:
        eligible=z['eligible'][:len(y)]
        registered=z['label'][:len(y)]
        agree=(registered>=0)&(z['fine'][:len(y)].argmax(1)==registered)
        deck_conf=z['fine'][:len(y)].max(1)
    frame=raw[['HomePlanet','Destination','Age']].copy()
    fields=raw.Cabin.str.split('/',expand=True)
    frame['deck'],frame['side'],frame['berth']=fields[0],fields[2],pd.to_numeric(fields[1])
    frame['booking']=g.astype(int)
    frame['party_size']=pd.Series(g).map(pd.Series(g).value_counts()).to_numpy()
    frame['bill']=raw[FEES].sum(axis=1,min_count=5)
    frame['active_services']=raw[FEES].gt(0).sum(axis=1)
    frame['deck_probability']=deck_conf
    for col in FEES:
        frame[col+'_share']=raw[col]/frame.bill.where(frame.bill>0)
    frames={}
    for tag,v in panels.items():
        ids=v['ids']
        local=frame.iloc[ids].reset_index(drop=True).copy()
        local['original_error_odds']=v['off']
        local['fare_model_error_probability']=np.where(v['anchor'],1-fare[tag]['neural_deck_fine'],fare[tag]['neural_deck_fine'])
        frames[tag]=local
    bundles={}
    for key,src,dst,t,q in splits(panels,g):
        if not (key.startswith('bookingmix') or key=='historical_to_new'):
            continue
        a,b=src['ids'][t],dst['ids'][q]
        with np.load(REVIEW/f'pred_{key}.npz') as z:
            np.testing.assert_array_equal(z['q'],b)
            parent=dst['anchor'][q]^((z['identities_error']>.5)&(z['identities_outcome']>.5))
        current=combine(parent,fare['full']['neural_deck_raw'][q],fare['full']['neural_deck_fine'][q],eligible[b])>=.5
        m=src['common'][t]&agree[a]&eligible[a]
        choose=dst['common'][q]&agree[b]&eligible[b]
        np.testing.assert_array_equal(src['anchor'][t][m],combine(src['anchor'][t][m],fare['historical']['neural_deck_raw'][t][m],
                                    fare['historical']['neural_deck_fine'][t][m],eligible[a][m])>=.5)
        np.testing.assert_array_equal(current[choose],dst['anchor'][q][choose])
        xf=pd.concat([frames['historical'].iloc[t[m]],frames['full'].iloc[q[choose]]],ignore_index=True)
        target=np.r_[src['wrong'][t][m],np.zeros(choose.sum(),bool)]
        grp=g[a[m]]
        assert not np.isin(grp,g[b]).any()
        unique=np.unique(grp)
        rng=np.random.default_rng(6180)
        selected=rng.permutation(unique)[::2]
        halves=np.isin(grp,selected)
        shuffled=target.copy()
        # Preserve dependence on the original risk score under the negative control.
        risk=src['off'][t][m]
        base_bin=np.searchsorted(np.quantile(risk,[.25,.5,.75]),risk)
        for k in range(4):
            which=np.flatnonzero(base_bin==k)
            shuffled[which]=rng.permutation(target[which])
        ids=np.arange(len(grp),len(xf))
        scores={}
        details=[]
        for n,d in [('real',target),('permuted',shuffled)]:
            forecasts=[]
            for mask in [halves,~halves]:
                prediction,rules=find_groups(xf,d,np.flatnonzero(mask),np.flatnonzero(~mask),ids)
                forecasts.append(prediction)
                if n=='real' and key=='historical_to_new':
                    details.extend(rules)
            scores[n]=np.mean(forecasts,axis=0)
        scores['original']=expit(dst['off'][q][choose])
        tag='bookingmix' if key.startswith('bookingmix') else key
        bundles.setdefault(tag,[]).append((b[choose],current[choose]!=y[b[choose]],scores,details))
    report={}
    for tag,items in bundles.items():
        ids,wrong=[np.concatenate([v[j] for v in items]) for j in range(2)]
        assert len(np.unique(ids))==len(ids)
        r={}
        for key in items[0][2]:
            p=np.concatenate([v[2][key] for v in items])
            # Fixed passenger order breaks risk-score ties without consulting outcomes.
            order=np.lexsort((ids,-p))
            top=order[:int(np.ceil(len(ids)/10))]
            r[key]=dict(**ticker(np.ones(len(ids),bool),wrong,p>.5),auc=float(roc_auc_score(wrong,p)),
                        top_tenth_size=len(top),top_tenth_errors=int(wrong[top].sum()),max_risk=float(p.max()))
        r['confirmation_rules']=[rule for item in items for rule in item[3]]
        report[tag]=r
        print(tag, {k:v for k,v in r.items() if k!='confirmation_rules'},flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/confirmed_error_subgroups.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    run()
