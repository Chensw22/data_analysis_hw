"""Discover expert-dissent rules by net corrections and replicate on separate bookings."""
import json
from itertools import combinations

import numpy as np
from sklearn.model_selection import GroupKFold

from broad_booking_policy_replay_n23 import interval
from full_roster_adaptive_bill_prior_n23 import EVIDENCE, proposal_design
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_direct_reference_risk_n23 import choose_reference
from full_roster_unexpected_peers_n23 import reference_score


def rules(scores, reference, probability):
    middle = (probability > .3) & (probability < .7)
    dissent = (scores >= .5) != reference[:,None]
    masks, labels = [], []
    subsets = [(i,) for i in range(scores.shape[1])] + list(combinations(range(scores.shape[1]), 2))
    for pair in subsets:
        active = middle & dissent[:,pair].all(axis=1)
        for direction, scope in [('both',middle),('positive',reference),('negative',~reference)]:
            masks.append(active & scope)
            labels.append((pair,direction))
    return np.column_stack(masks), labels


def discover(mask, truth, ref):
    uncovered = np.ones(len(truth), bool)
    ordered = []
    sign = np.where(truth != ref, 1, -1)
    for _ in range(5):
        evidence = mask[uncovered]
        n = evidence.sum(axis=0)
        net = sign[uncovered] @ evidence
        merits = np.where(n>=12, net-np.sqrt(n), -np.inf)
        best = int(np.argmax(merits))
        if merits[best] <= 0:
            break
        ordered.append(best)
        uncovered &= ~mask[:, best]
    return ordered


def run():
    raw,y,group,fold = data()
    names, half, full = banks(raw,y,fold)
    ref = latest_reference(y,group,fold,names,half,full)
    alt = names.index('balanced_full')
    p = reference_score(full,ref,alt)
    middle = (p>.3) & (p<.7)
    proposal, old = proposal_design(half[:,:5], half[:,[names.index(k) for k in EVIDENCE]])
    available, labels = rules(full,ref,p)
    modes = ['discovery','replicated','replicated_strict']
    pred = {k:ref.copy() for k in modes}
    visits = []
    for f in np.unique(fold):
        t = (fold!=f) & (fold%2==f%2)
        outer = np.flatnonzero(fold==f)
        rt = np.zeros(len(y),bool)
        for h in np.unique(fold[t]):
            train,held = t & (fold!=h),t & (fold==h)
            assert not np.isin(group[train],group[held | (fold==f)]).any()
            rt[held] = choose_reference(half,old,proposal,y,train,held)
        hp = reference_score(half[t],rt[t],alt)
        donor_masks, donor_labels = rules(half[t],rt[t],hp)
        assert donor_labels == labels
        ordered = discover(donor_masks,y[t],rt[t])
        visits.append(dict(fold=int(f), rules=[([names[j] for j in labels[i][0]],labels[i][1]) for i in ordered]))
        if ordered:
            pred['discovery'][outer] ^= available[np.ix_(outer,ordered)].any(axis=1)
        for a,b in GroupKFold(4,shuffle=True,random_state=462).split(outer,groups=group[outer]):
            donors,q = outer[a],outer[b]
            assert not np.isin(group[donors],group[q]).any()
            for mode in modes[1:]:
                accepted = np.zeros(len(donors),bool)
                active = np.zeros(len(q),bool)
                for index in ordered:
                    new = available[donors,index] & ~accepted
                    votes = np.where(ref[donors[new]]==y[donors[new]],-1,1)
                    threshold = np.sqrt(len(votes)) if mode=='replicated_strict' else 0
                    if len(votes)>=5 and votes.sum()>threshold:
                        accepted |= new
                        active |= available[q,index]
                pred[mode][q] ^= active
    shared = ((full>=.5)!=y[:,None]).all(1)
    pickable = ((full>=.5)==y[:,None]).any(1)
    report = dict(middle=int(middle.sum()),middle_wrong=int((middle & (y!=ref)).sum()),
                  middle_errors_with_a_correct_expert=int((middle & (y!=ref) & pickable).sum()),
                  middle_right_challenged=int((middle & (y==ref) & ((full>=.5)!=y[:,None]).any(1)).sum()),
                  discovery_rules=visits, results={})
    report['score_ranges'] = {}
    for key, mask in [('low',p<=.3),('middle',middle),('high',p>=.7)]:
        report['score_ranges'][key] = dict(passengers=int(mask.sum()), errors=int((ref[mask]!=y[mask]).sum()),
                                           all_experts_wrong=int(shared[mask].sum()))
    with np.load(ROOT/'artifacts/run_002/split_indices.npz') as z:
        dev = np.sort(z['development'])
    for k, prediction in pred.items():
        delta = (prediction == y).astype(int) - (ref == y)
        report['results'][k] = dict(accuracy=float((prediction == y).mean()),fixed=int((delta==1).sum()),
            harmed=int((delta==-1).sum()), additional_net=int(delta.sum()-delta[dev].sum()),
            common_fixed=int((delta[shared]==1).sum()), fold_net=[int(delta[fold==f].sum()) for f in np.unique(fold)],
            ci_pp=interval(delta,group))
    (ROOT/'reports/nov23_matched_learning_curve/middle_disagreement_rules.json').write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items() if k!='discovery_rules'})


if __name__ == '__main__':
    run()
