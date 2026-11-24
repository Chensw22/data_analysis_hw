"""Count repeated failures and successful alternatives across the entire solo population."""
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from full_roster_bill_neural_corroboration_n24 import load_scores
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,BILLS,banks,data,latest_reference
from full_roster_gate_probability_n23 import gate_probabilities
from full_roster_location_controls_n24 import spatial

EXPERIMENTS = {
    'predicate_tables': ('main','fixed','learned'),
    'two_view_reliability': ('registration','spending','mean','pooled','specialist','margins'),
    'honest_failure_partitions': ('natural','natural_honest','enriched','enriched_honest','placebo','placebo_honest'),
    'failure_sampling_offset': ('natural','enriched','remove_offset','in_fit_offset','placebo'),
    'spending_shape': ('free','pooled_direction','basket_direction','reversed'),
    'fixed_bill_margins': ('personal','margins','measured','reconstructed','same_fit')
}


def cached(folder,tag,rows,fold,g,y):
    blocks,positions = [],[]
    for f in np.unique(fold):
        mask = fold==f
        q = rows[mask]
        t = rows[~mask & ~np.isin(g[rows],g[q])]
        with np.load(ROOT/f'artifacts/nov24_{folder}/{tag}_{f}.npz') as z:
            for k,v in [('q',q),('t',t),('truth' if 'truth' in z else 'y',y[t])]:
                np.testing.assert_array_equal(z[k],v)
            blocks.append(z['p'])
        positions.append(np.flatnonzero(mask))
    return np.vstack(blocks)[np.argsort(np.concatenate(positions))]


def run():
    raw,y,g,fold = data()
    _,oy,_,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    gate,_ = gate_probabilities(names,h,full,y,g,fold)
    solo = raw.groupby(g).PassengerId.transform('size').to_numpy()==1
    errors = solo & (ref!=y)
    ar = ROOT/'artifacts/nov24_remaining_cases'
    scores = {}
    for filename,cols in [('after_corroboration',['remaining']),('audit',['scores']),
                          ('archive_scores',['full','historical','names'])]:
        with np.load(ar/f'{filename}.npz') as z:
            for k,v in [('passenger',raw.PassengerId),('fold',fold),('y',y)]:
                np.testing.assert_array_equal(z[k],v)
            if filename=='archive_scores':
                np.testing.assert_array_equal(z['row'],rows)
                np.testing.assert_array_equal(z['old_fold'],of)
            scores.update({k:z[k] for k in cols})
    hard = solo & scores['remaining']
    confidence = np.where(ref,gate,1-gate)
    paid = raw[BILLS].gt(0).sum(axis=1).to_numpy()
    masks = dict(all=solo,negative=~ref,positive=ref,uncertain=confidence<.7,
                 high=confidence>=.8,complete=~raw.isna().any(axis=1),negative_multiple=(~ref)&(paid>=2))
    report = {'profiles':{k:dict(rows=int((v&solo).sum()),errors=int((v&errors).sum()),hard=int((v&hard).sum()))
                          for k,v in masks.items()},'coverage':{},'new_archives':{}}
    def hit(p):
        return ((p>=.5)==y[:,None]).any(1)
    covered = hit(np.column_stack([scores['scores'],scores['full']]))
    report['coverage']['initial_59_still_wrong'] = int((errors&~covered).sum())
    ix = np.arange(len(y))
    art = ROOT/'artifacts'
    fields = art/'nov24_field_interactions'
    extra = [load_scores(fields/'oof.npz',raw,y,fold,('additive','pair','triple')),
             load_scores(fields/'third_order_seeds.npz',raw,y,fold,['scores'])[:,1:],
             load_scores(art/'nov24_booking_transport/oof.npz',raw,y,fold,('individual','independent','transport')),
             cached('bill_amount_frequency','full',ix,fold,g,y)]
    covered |= hit(np.column_stack(extra))
    np.testing.assert_array_equal(errors&~covered,hard)
    candidates = {name:(p,a) for name,p,a in zip(scores['names'],scores['full'].T,scores['historical'].T)}
    for folder,keys in EXPERIMENTS.items():
        b = cached(folder,'full',ix,fold,g,y)
        a = cached(folder,'historical',rows,of,g,y)
        assert b.shape[1]==a.shape[1]==len(keys)
        report['new_archives'][folder] = int((hard&hit(b)).sum())
        covered |= hit(b)
        candidates.update({folder+'/'+key:(p,q) for key,p,q in zip(keys,b.T,a.T)})
    report['coverage']['after_additional_29_still_wrong'] = int((errors&~covered).sum())
    cabin = raw.Cabin.str.extract(r'^([^/]+)/(\d+)/([PS])$')
    pos = pd.to_numeric(cabin[1]).to_numpy()
    deck = cabin[0].fillna('?').to_numpy(str)
    lane = (cabin[0].fillna('?')+cabin[2].fillna('?')).to_numpy(str)
    xy = np.column_stack([g.astype(float)/5,pos])
    state = np.char.add(raw.CryoSleep.astype(str).to_numpy(str),np.where(paid>0,'P','Z'))
    report['geometry'] = {}
    for k,pools in [('lane',lane),('lane_activity',np.char.add(lane,state)),('mirrored_activity',np.char.add(deck,state))]:
        p,_,_ = spatial(y,g,fold,pools,xy,np.isfinite(pos),h[:,names.index('mitra96')])
        right = (p>=.5)==y
        covered |= right
        report['geometry'][k] = dict(corrected=int((errors&right).sum()),
                      harmed=int((solo&~errors&~right).sum()), hard_fixed=int((hard&right).sum()))
    report['coverage']['after_spatial_still_wrong'] = int((errors&~covered).sum())
    report['screen'] = []
    anchor = full[:,names.index('mitra96')]
    for name,(b,a) in candidates.items():
        for w in (1.,.25):
            p = (w*b+(1-w)*gate)>=.5
            past = (w*a+(1-w)*old[:,0])>=.5
            if w<1:
                past = np.where((old[:,0]>=.5)==champ,past,champ)
            fix = int((errors&(p==y)).sum())
            harm = int((solo&~errors&(p!=y)).sum())
            old_fix = int((solo[rows]&(champ!=oy)&(past==oy)).sum())
            old_harm = int((solo[rows]&(champ==oy)&(past!=oy)).sum())
            physical = np.where((anchor>=.5)==ref,(w*b+(1-w)*anchor)>=.5,ref) if w<1 else p
            actual_fix = int((errors&(physical==y)).sum())
            actual_harm = int((solo&~errors&(physical!=y)).sum())
            report['screen'].append(dict(physical=[actual_fix,actual_harm],
                                         physical_min_net=min(actual_fix-actual_harm,old_fix-old_harm),method=name,weight=w,full=[fix,harm],history=[old_fix,old_harm],
                                         min_net=min(fix-harm,old_fix-old_harm)))
    report['screen'].sort(key=lambda item:item['min_net'],reverse=True)
    path = ROOT/'reports/nov23_matched_learning_curve/solo_coverage.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(report['profiles'],report['coverage'],report['screen'][0])


if __name__=='__main__':
    with threadpool_limits(2):
        run()
