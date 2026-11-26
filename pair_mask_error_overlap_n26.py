"""Trace distinct repairs to labelled references and observed facility subsets, then validate their intersections."""
import json
import numpy as np

from booking_mix_confidence_transfer_n26 import partition
from full_roster_bill_subset_route_n24 import BILLS, MASK
from full_roster_booking_validation_n23 import ROOT, data
from repeated_archive_misses_n26 import OUT, cohort, count

REPORT = ROOT / 'reports/nov23_matched_learning_curve/pair_mask_error_overlap.json'
PAIR = 'nov24_passenger_pair_outcomes'
BILL = 'nov24_bill_subset_route'
MODES = ('raw_tree','near','random','near_random_mean')


def columns(names, keys):
    return [names.index(k) for k in keys]


def analyze(votes, names):
    pair_idx = columns(names, [f'{folder}/{j}' for folder in (PAIR, PAIR+'_repeat') for j in range(4)])
    mask_idx = columns(names, [f'{BILL}/{j}' for j in range(32)])
    pair = votes[:,pair_idx].reshape(len(votes),2,4)
    masks = votes[:,mask_idx]
    other = np.delete(votes, pair_idx + mask_idx, axis=1).any(1)
    pa, ma = pair.any(axis=(1,2)), masks.any(1)
    results = {'both':pa & ma, 'pair_only':pa & ~ma, 'mask_only':ma & ~pa, 'union':pa | ma,
               'pair_unique_among_48':pa & ~ma & ~other, 'mask_unique_among_48':ma & ~pa & ~other,
               'both_unique_among_48':ma & pa & ~other}
    near, random = pair[:,:,1].any(1), pair[:,:,2].any(1)
    results.update(near_only=near&~random, random_only=random&~near, near_and_random=near&random,
                   raw_only=pair[:,:,0].any(1)&~pair[:,:,1:].any(axis=(1,2)))
    for j, mode in enumerate(MODES):
        results[mode+'_both_seeds'] = pair[:,:,j].all(1)
    for size in range(6):
        include = MASK.sum(1)==size
        results[f'any_visible_{size}'] = masks[:,include].any(1)
        results[f'exclusively_visible_{size}'] = masks[:,include].any(1) & ~masks[:,~include].any(1)
    for j, fee in enumerate(BILLS):
        exposed, hidden = masks[:,MASK[:,j]].any(1), masks[:,~MASK[:,j]].any(1)
        results[fee+'_only_when_hidden'] = hidden & ~exposed
        results[fee+'_only_when_visible'] = exposed & ~hidden
    for i, inc in enumerate(MASK):
        results['show_' + ('+'.join(np.array(BILLS)[inc]) if inc.any() else 'none')] = masks[:,i]
    return pair,masks,results


def run():
    raw,y,g,f=data()
    base,blind,known,fresh=cohort(raw,y,f)
    with np.load(OUT/'aligned.npz') as z:
        for k,v in [('passenger',raw.PassengerId),('y',y),('fold',f)]:
            np.testing.assert_array_equal(z[k],v)
        rows=z['rows']
        names=z['names'].tolist()
        def dissent(scores,b):
            return (scores-.5)*np.where(b,-1.,1.)[:,None]>0
        full=dissent(z['full'],base)
        old=dissent(z['historical'],base[rows])
    amounts = raw[BILLS].to_numpy(float)
    protocols = [('full',np.arange(len(y)),full), ('full_matched',rows,full[rows]),
                 ('same_column_both',rows,full[rows]&old)]
    answer = {}
    for name,ids,votes in protocols:
        pair,masks,actions = analyze(votes,names)
        # Distinguish losing a known positive bill from hiding only a zero or an already missing amount.
        hides_charge = (amounts[ids] > 0).astype(int) @ (~MASK).T > 0
        incomplete = ~masks[:,31]
        charged = (masks & hides_charge).any(1) & incomplete
        no_charge = (masks & ~hides_charge).any(1) & incomplete
        actions.update(only_if_positive_removed=charged & ~no_charge,
                       masking_without_losing_positive_only=no_charge & ~charged,
                       either_mask_type=charged & no_charge,
                       complete_mask_already_opposes=masks[:,31])
        answer[name] = {s:{k:count(a,scope[ids],base[ids]!=y[ids]) for k,a in actions.items()}
                        for s,scope in [('known',known),('blind',blind),('fresh',fresh&known)]}
    # Match each exact bill mask with the same pair model's disagreement on both reference seeds.
    # Choose on different bookings within the same outer fold. The abstention rule needs no labels.
    pair,masks,_ = analyze(full,names)
    both = pair[:,:,1:].all(1)
    rules=(both[:,:,None] & masks[:,None,:]).reshape(len(y),-1)
    signed=rules*np.where(base!=y,1,-1)[:,None]
    checks={}
    for split in ['random','ordered']:
        parts=partition(g,'pair_mask_check',4) if split=='random' else np.zeros(len(y),int)
        selected=np.zeros(len(y),bool)
        choices=[]
        for k in range(8):
            if split=='ordered':
                for j,group in enumerate(np.array_split(np.unique(g[f==k]),4)):
                    parts[(f==k)&np.isin(g,group)]=j
            for j in range(4):
                test=(f==k)&(parts==j)
                fit=(f==k)&(parts!=j)&known
                assert not np.isin(g[fit],g[test]).any()
                scores=signed[fit].sum(0)
                best=int(scores.argmax())
                if scores[best]>0:
                    selected[test]=rules[test,best]
                    choices.append([k,j,MODES[1+best//32],int(best%32),int(scores[best])])
        checks[split]=dict(choices=choices,scopes={s:count(selected,v,base!=y) for s,v in
                                                   [('all',known),('blind',blind),('fresh',known&fresh)]},
                           fold_net=[int(np.where(base!=y,1,-1)[(f==k)&selected&known].sum()) for k in range(8)])
    answer['heldout_choice']=checks
    REPORT.write_text(json.dumps(answer,indent=2)+'\n')
    for name,ids,votes in protocols:
        print(name,{k:answer[name]['blind'][k] for k in ('both','pair_only','mask_only','union','near_only','random_only','near_and_random')})
    print({k:v['scopes'] for k,v in checks.items()})


if __name__=='__main__':
    run()
