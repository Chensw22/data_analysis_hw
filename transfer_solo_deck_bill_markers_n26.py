"""Check whether matched deck and food-service differences select actual false positives."""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from full_roster_booking_validation_n23 import data
from tabnet_attention_transfer_n26 import RESULT
from target_mix_expert_routing_n26 import ticker
from worst_booking_group_ensemble_n26 import order_splits


def model_design(rows, opinions, deck_g, food, spa):
    count = opinions[:, 1:].mean(axis=1)
    levels = np.column_stack([count, np.sqrt(count), opinions[:, 1:19]])
    markers = np.column_stack([deck_g[rows], food[rows], spa[rows], deck_g[rows] & food[rows]])
    return levels, np.column_stack([levels, markers, markers * count[:, None]])


def run():
    raw, _, group, _ = data()
    solo = raw.groupby(group).PassengerId.transform('size').eq(1).to_numpy()
    deck_g = raw.Cabin.str.startswith('G/', na=False).to_numpy()
    # Both known zeros and missing fees appeared as "no recorded positive" in the matched analysis.
    no_food = ~raw.FoodCourt.gt(0).to_numpy()
    no_spa = ~raw.Spa.gt(0).to_numpy()
    by_tag = {}
    for b,_,_,_,ts,qs,f,xt,xq in order_splits():
        t,q = b.train_ids,b.ids
        train = ts & solo[t] & (b.train_cell%2==1) & xt[:,1:].any(1)
        use = qs & solo[q] & (b.query_cell%2==1) & xq[:,1:].any(1)
        assert not np.isin(group[t[train]],group[q]).any()
        tr = model_design(t,xt,deck_g,no_food,no_spa)
        te = model_design(q,xq,deck_g,no_food,no_spa)
        pred = []
        for at,aq in zip(tr,te,strict=True):
            model = LogisticRegression(C=.3,max_iter=350).fit(at[train], b.train_error[train])
            pred.append(model.predict_proba(aq)[:,1])
        evidence = deck_g[q] & no_food[q]
        actions = dict(deck_g=deck_g[q],no_recorded_foodcourt=no_food[q],both_markers=evidence,
                       opinions_only=pred[0]>.5,opinions_and_markers=pred[1]>.5,
                       markers_required=(pred[1]>.5) & (pred[0]<=.5))
        by_tag.setdefault(b.tag,[]).append((q,use,b.query_error,f,actions,int(train.sum())))
    report = {}
    for tag, batches in by_tag.items():
        q,use,wrong,fold = (np.concatenate([b[i] for b in batches]) for i in range(4))
        assert len(q)==len(np.unique(q))
        report[tag]={'training_support':[b[-1] for b in batches]}
        for name in batches[0][4]:
            act = np.concatenate([b[4][name] for b in batches])
            r = ticker(use,wrong,act)
            r['net_by_fold']=[int((use*act*np.where(wrong,1,-1)*(fold==k)).sum()) for k in range(8)]
            report[tag][name]=r
        print(tag,{n:(v['fixed'],v['harmed']) for n,v in report[tag].items() if isinstance(v,dict)},flush=True)
    report['scope']='Solo members with an agreed positive release decision and a countervote from the 99 comparable experts. Historical discovery fits use only eligible whole bookings. Fields were noticed in a separate retrospective matched analysis which included some evaluation rows.'
    (RESULT.parent/'transfer_solo_deck_bill_markers.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
