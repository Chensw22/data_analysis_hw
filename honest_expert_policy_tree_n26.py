"""Learn shallow expert assignment trees from observed paired gains, then confirm their leaves on other orders."""
import json
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from target_mix_expert_routing_n26 import ROOT, batches, ticker


def leaf(values, edits):
    # Models taking no action have zero benefit. Require gain above a one-standard-error penalty.
    conservative = values - np.sqrt(edits.clip(min=0))
    index = conservative.argmax(axis=-1)
    return index, np.take_along_axis(conservative,index[...,None],axis=-1).squeeze(-1)


def tree(fields, candidates, wrong, build, confirm, test, query, groups):
    assert not np.isin(groups[build],groups[confirm]).any()
    binary = []
    held_binary = []
    for j,col in enumerate(fields.T):
        observed = np.unique(col[build])
        if len(observed)<2:
            continue
        thresholds = [(observed[0]+observed[1])/2] if len(observed)==2 else np.unique(np.quantile(col[build],[.2,.4,.6,.8]))
        for cut in thresholds:
            binary.append(col>cut)
            held_binary.append(test[:,j]>cut)
    tr = np.asarray(binary,float)
    te = np.asarray(held_binary,bool)
    gain = candidates * (2*wrong.astype(float)[:,None]-1)
    direct = np.zeros(len(test),bool)
    certified = direct.copy()

    def descend(ids, valid, ask, depth):
        edits = candidates[ids].sum(0)
        value = gain[ids].sum(0)
        best,score = leaf(value,edits)
        if depth<2 and len(ids)>=70:
            splits = tr[:,ids]
            size = splits.sum(1)
            left_edits = splits@candidates[ids].astype(float)
            left_gain = splits@gain[ids]
            _,left = leaf(left_gain,left_edits)
            _,right = leaf(value-left_gain,edits-left_edits)
            utility = left+right
            utility[(size<30)|(len(ids)-size<30)] = -np.inf
            j = int(utility.argmax())
            if utility[j]>score+3:
                for sign in (False,True):
                    descend(ids[tr[j,ids]==sign],valid[tr[j,valid]==sign],ask[te[j,ask]==sign],depth+1)
                return
        direct[ask] = query[ask,best]
        # The validation half can veto the chosen expert; it does not select a replacement.
        proposals = candidates[valid,best]
        if proposals.any() and gain[valid,best].sum()>0:
            certified[ask] = query[ask,best]
    descend(build,confirm,np.arange(len(test)),0)
    return direct,certified


def run():
    records = {}
    for b in batches(True):
        raw,checked = [],[]
        for a,c in GroupKFold(2,shuffle=True,random_state=5238).split(b.groups,groups=b.groups):
            first,confirmed = tree(b.train_fields,b.train_actions,b.train_error,a,c,b.query_fields,b.query_actions,b.groups)
            raw.append(first)
            checked.append(confirmed)
        actions = {'raw_union':raw[0]|raw[1], 'raw_intersection':raw[0]&raw[1],
                   'confirmed_union':checked[0]|checked[1],'confirmed_intersection':checked[0]&checked[1]}
        records.setdefault(b.tag,[]).append((b.ids,b.query_error,b.strict,b.query_cell,actions))
    report = {}
    for tag,items in records.items():
        ids,wrong,strict,cell = [np.concatenate([r[i] for r in items]) for i in range(4)]
        assert len(np.unique(ids))==len(ids)
        scopes = {'all':np.ones(len(ids),bool),'original_unanimous':strict,
                  'added_unanimous':(cell>=4)&~strict,'contested':cell<4}
        r = {}
        for name in items[0][-1]:
            action = np.concatenate([item[-1][name] for item in items])
            r[name] = {n:ticker(m,wrong,action) for n,m in scopes.items()}
        report[tag] = r
        print(tag,{n:(v['all']['fixed'],v['all']['harmed']) for n,v in r.items()},flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/honest_expert_policy_tree.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
