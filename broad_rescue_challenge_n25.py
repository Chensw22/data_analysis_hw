"""Estimate mistake rates in other bookings that witnessed the same proposed reversal."""
import json
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from common_exception_audit_n25 import inputs
from disjoint_tree_fields_n25 import ART,DEST,stats
from full_roster_booking_validation_n23 import data
from full_roster_six_failure_archive_n24 import prediction_sets
from grouped_peer_teacher_n25 import OUT as REF
from screen_joint_disagreements_n25 import collect


def ranks(train, query):
    columns=[]
    for j in range(train.shape[1]):
        ordered=np.sort(train[:,j])
        columns.append((np.searchsorted(ordered,query[:,j],side='left')+np.searchsorted(ordered,query[:,j],side='right'))/(2*len(train)))
    return np.column_stack(columns)


def witness(distance,groups,outcomes,prior):
    # Only one passenger represents each independent training booking in each neighborhood.
    probs=[]
    for d in distance:
        ordered=np.argsort(d,kind='stable')
        _,idx=np.unique(groups[ordered],return_index=True)
        selected=ordered[np.sort(idx)[:48]]
        rates=[]
        for k in (12,48):
            a=selected[:k]
            rates.append((outcomes[a].sum()+4*prior)/(len(a)+4))
        probs.append(rates)
    return np.array(probs)


def run():
    raw,y,group,f=data()
    n=len(y)
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        r,hf,old_ref=z['rows'],z['fold'],z['nine']
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId.iloc[r])
        np.testing.assert_array_equal(z['y'],y[r])
    nine=np.full(n,-1)
    for k in np.unique(f):
        with np.load(REF/f'release{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],np.flatnonzero(f==k))
            nine[z['q']]=z['nine']
    ref=np.r_[nine,old_ref]
    index=np.r_[np.arange(n),r]
    truth=y[index]
    groups=group[index]
    _,panels=prediction_sets(raw,y,group,f,r,hf)
    archive=np.vstack([np.column_stack(v) for v in panels])
    alt=np.vstack([collect(np.arange(n),f,y,group,'fold'),collect(r,hf,y,group,'historical')])
    sign=ref==1
    against=np.where(sign[:,None],1-alt,alt)
    arch=np.where(sign[:,None],1-archive,archive)
    discrepancy=(alt>=.5)!=sign[:,None]
    votes=((archive>=.5)!=sign[:,None]).sum(1)
    known=ref>=0
    eligible=known & discrepancy.any(1)
    errors=truth!=ref
    registered,paid=inputs(raw)
    raw_features=np.column_stack([registered,paid])[index]
    a=np.column_stack([votes/68,np.quantile(arch,[.1,.25,.5,.75,.9],axis=1).T])
    # Match patterns of source disagreement separately from the overall level.
    b=np.column_stack([against,discrepancy.mean(1),against.max(1)])
    fresh=~np.isin(group,group[r])
    splits={'historical_to_new_orders':[(n+np.arange(len(r)),np.flatnonzero(fresh))],'same_fitted_base':[]}
    for k in np.unique(f):
        within=np.flatnonzero(f==k)
        for t,q in GroupKFold(4,shuffle=True,random_state=213).split(within,groups=group[within]):
            splits['same_fitted_base'].append((within[t],within[q]))
    # Keep the same untouched orders while changing the number of historical witness bookings.
    # Use disjoint groups at each size and report every repeat, not the most favorable partition.
    historical_orders=np.random.default_rng(173).permutation(np.unique(group[r]))
    for pieces in (2,8):
        for j,part in enumerate(np.array_split(historical_orders,pieces)):
            ii=np.flatnonzero(np.isin(group[r],part))
            splits[f'historical_{pieces}_part_{j}']=[(n+ii,np.flatnonzero(fresh))]
    seeds=(79,173,281,379,479,577,673,773)
    for seed in seeds:
        ordered=np.random.default_rng(seed).permutation(np.unique(group[r]))
        for percentage in (10,30):
            subset=ordered[:round(percentage*len(ordered)/100)]
            ii=np.flatnonzero(np.isin(group[r],subset))
            splits[f'size_{percentage}_seed_{seed}']=[(n+ii,np.flatnonzero(fresh))]
    sampled_votes={}
    report={}
    for protocol,parts in splits.items():
        metrics={}
        predictions={name:[] for name in ['archive','source_pattern','source_and_manifest','archive_and_manifest','permuted_source']}
        rows=[]
        for fold,(t,q) in enumerate(parts):
            assert not np.isin(groups[t],groups[q]).any()
            q=q[known[q]]
            rows.extend(q)
            p={name:np.zeros((len(q),2)) for name in predictions}
            for side in [0,1]:
                tr=t[eligible[t]&(ref[t]==side)]
                mask=eligible[q]&(ref[q]==side)
                qq=q[mask]
                if not len(qq):continue
                assert len(tr)
                base=cdist(ranks(a[tr],a[qq]),ranks(a[tr],a[tr]),metric='cityblock')/a.shape[1]
                alt_dist=cdist(ranks(b[tr],b[qq]),ranks(b[tr],b[tr]),metric='cityblock')/b.shape[1]
                raw_dist=cdist(raw_features[qq],raw_features[tr],metric='hamming')
                # Permute source patterns only among similar archive-confidence deciles.
                order=np.arange(len(tr))
                rng=np.random.default_rng(602+fold+side)
                bin_idx=np.minimum((10*a[tr,3]).astype(int),9)
                for level in np.unique(bin_idx):
                    cell=np.flatnonzero(bin_idx==level)
                    order[cell]=rng.permutation(cell)
                for name,d in [('archive',base),('source_pattern',base+alt_dist),
                               ('source_and_manifest',base+alt_dist+2*raw_dist),('archive_and_manifest',base+2*raw_dist),
                               ('permuted_source',base+alt_dist[:,order])]:
                    p[name][mask]=witness(d,groups[tr],errors[tr],errors[tr].mean())
            for name in predictions:predictions[name].append(p[name])
        ids=np.asarray(rows)
        assert len(np.unique(ids))==len(ids)
        for name,blocks in predictions.items():
            pair=np.vstack(blocks)
            if (protocol.startswith('size_') or protocol=='historical_to_new_orders') and name in ('source_and_manifest','archive_and_manifest'):
                sampled_votes[protocol,name]=(ids.copy(),pair[:,1]>.5)
            for rule,action in [('12',pair[:,0]>.5),('48',pair[:,1]>.5),('both',(pair>.5).all(1))]:
                details={}
                for label,mask in [('complete',np.ones(len(ids),bool)),('unanimous',votes[ids]==0),('within_three',votes[ids]<=3),
                                   ('new',fresh[ids])]:
                    fixed=mask&action&errors[ids]
                    harmed=mask&action&~errors[ids]
                    details[label]=dict(rows=int(mask.sum()),errors=int((mask&errors[ids]).sum()),eligible=int((mask&eligible[ids]).sum()),
                                        fixed=int(fixed.sum()),harmed=int(harmed.sum()),
                                        by_fold=[int(fixed[f[ids]==k].sum()-harmed[f[ids]==k].sum()) for k in np.unique(f[ids])])
                metrics[name+'/'+rule]=details
        report[protocol]=metrics
    size_audit={}
    for percentage in (10,30,100):
        protocols=[f'size_{percentage}_seed_{s}' for s in seeds] if percentage<100 else ['historical_to_new_orders']
        results={}
        for name in ('source_and_manifest','archive_and_manifest'):
            ids,first=sampled_votes[protocols[0],name]
            matrix=[]
            for protocol in protocols:
                i,decisions=sampled_votes[protocol,name]
                np.testing.assert_array_equal(i,ids)
                matrix.append(decisions)
            m=np.vstack(matrix)
            def pair(mask):
                return [int((mask&errors[ids]).sum()),int((mask&~errors[ids]).sum())]
            hits=[pair(v) for v in m]
            results[name]=dict(fixed_harmed_by_seed=hits,net_by_seed=[a-b for a,b in hits],
                               unanimous_across_seeds=pair(m.all(0)),changed_only_in_some_seeds=int((m.any(0)&~m.all(0)).sum()))
        results['source_minus_control_by_seed']=np.subtract(results['source_and_manifest']['net_by_seed'],
            results['archive_and_manifest']['net_by_seed']).tolist()
        size_audit[percentage]=results
    ids,with_sources=sampled_votes['historical_to_new_orders','source_and_manifest']
    other_ids,without_sources=sampled_votes['historical_to_new_orders','archive_and_manifest']
    np.testing.assert_array_equal(ids,other_ids)
    paired=(with_sources.astype(int)-without_sources.astype(int))*np.where(errors[ids],1,-1)
    size_audit['paired_100_percent']=stats(paired,np.ones(len(ids),bool),f[ids],group[ids])
    out={key:report[key] for key in ('same_fitted_base','historical_to_new_orders')}
    for pieces in (2,8):
        out[f'disjoint_historical_parts_{pieces}']=[
            {method:[report[f'historical_{pieces}_part_{j}'][method]['complete'][k] for k in ('fixed','harmed')]
             for method in ('source_and_manifest/48','archive_and_manifest/48')} for j in range(pieces)]
    out['same_new_orders_size_audit']=dict(seeds=seeds,rows=len(ids),sizes=size_audit)
    DEST.with_name('broad_rescue_challenge_n25.json').write_text(json.dumps(out,indent=2)+'\n')
    print('Paired source contribution on all available historical witness orders:',size_audit['paired_100_percent'])


if __name__=='__main__':
    with threadpool_limits(2):
        run()
