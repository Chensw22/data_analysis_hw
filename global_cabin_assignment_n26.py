"""Narrow missing cabin tracks using a covering matching for every booking with no known cabin."""
import json
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from full_roster_booking_validation_n23 import ROOT

DEST = ROOT/'reports/nov23_matched_learning_curve/global_cabin_assignment.json'
OUT = ROOT/'artifacts/nov26_global_cabin_assignment'


def candidates(frame):
    group = frame.PassengerId.str[:4].astype(int).to_numpy()
    pieces = frame.Cabin.str.split('/',expand=True)
    track = (pieces[0]+pieces[2]).fillna('').to_numpy()
    number = pd.to_numeric(pieces[1]).to_numpy()
    unknown = track == ''
    targets = np.unique(group[unknown])
    required = np.setdiff1d(targets,group[~unknown])
    gaps = {}
    choices = {int(g):{} for g in targets}
    for name in sorted(set(track)-{''}):
        occupied = np.unique(np.stack((group[track==name],number[track==name].astype(int)),axis=1),axis=0)
        bookings, nums = occupied.T
        assert (np.diff(bookings)>0).all() and (np.diff(nums)>0).all()
        for target,at in zip(targets, np.searchsorted(bookings,targets)):
            if at < len(bookings) and bookings[at]==target:
                choices[target][name] = (int(nums[at]),int(nums[at]),None)
            else:
                low = int(nums[at-1] + 1) if at else 0
                high = int(nums[at] - 1) if at < len(nums) else None
                if high is None or low<=high:
                    gap = (name,int(at))
                    gaps[gap] = high-low+1 if high is not None else len(targets)
                    choices[target][name] = low,high,gap
    needed = Counter(info[2] for g in required for info in choices[g].values())
    starts = {}
    ncols = 0
    for gap,capacity in gaps.items():
        # Each test reserves at most one extra slot for a partially observed booking.
        slots = min(capacity, needed[gap]+1)
        starts[gap]=np.arange(ncols,ncols+slots)
        ncols+=slots
    ir,ic=[],[]
    for i,g in enumerate(required):
        for _,_,gap in choices[g].values():
            columns=starts[gap]
            ir.extend([i]*len(columns))
            ic.extend(columns)
    graph = csr_matrix((np.ones(len(ir),np.int8),(ir,ic)),shape=(len(required),ncols))
    match = maximum_bipartite_matching(graph,perm_type='column')
    if (match<0).any():
        raise ValueError('Some wholly missing bookings cannot be covered')
    req = {g:i for i,g in enumerate(required)}
    allowed={}
    for g,options in choices.items():
        keep={}
        include = np.ones(len(required),bool)
        if g in req:
            include[req[g]]=False
        for name,(low,high,gap) in options.items():
            if gap is not None:
                slot = starts[gap][0]
                if (match[include]==slot).any():
                    other = np.ones(ncols,bool)
                    other[slot]=False
                    if (maximum_bipartite_matching(graph[include][:,other],perm_type='column')<0).any():
                        continue
            keep[name] = (low,high,gap)
        assert keep
        allowed[g]=keep
    # Within a track, a booking forced into this gap must leave room for earlier
    # and later forced bookings. Open terminal gaps keep their unbounded upper end.
    forced = {}
    for g in required:
        if len(allowed[g])==1:
            gap=next(iter(allowed[g].values()))[2]
            forced.setdefault(gap,[]).append(g)
    records=[]
    for i in np.flatnonzero(unknown):
        g=group[i]
        before = choices[g]
        after=allowed[g]
        output={}
        for track_name,(low,high,gap) in after.items():
            if gap is not None:
                same=forced.get(gap,[])
                low+=int(np.sum(np.asarray(same)<g))
                if high is not None:
                    high-=int(np.sum(np.asarray(same)>g))
            assert high is None or low<=high
            output[track_name]=[low,high]
        old = {key:[lo,hi] for key,(lo,hi,_) in before.items()}
        records.append(dict(row=int(i),local=old,global_options=output))
    return records


def count(records):
    def exact(opts):
        return len(opts)==1 and next(iter(opts.values()))[0]==next(iter(opts.values()))[1]
    result={'missing':len(records),'reduced_tracks':0,'reduced_intervals':0}
    for which in ['local','global_options']:
        result[which+'_unique_track']=sum(len(r[which])==1 for r in records)
        result[which+'_unique_cabin']=sum(exact(r[which]) for r in records)
    result['reduced_tracks']=sum(len(r['local'])>len(r['global_options']) for r in records)
    result['reduced_intervals']=sum(r['local']!=r['global_options'] for r in records)
    return result


def run():
    roster=pd.concat([pd.read_csv(ROOT/f'data/raw/{split}.csv',usecols=['PassengerId','Cabin'])
                      for split in ('train','test')],ignore_index=True)
    n=len(pd.read_csv(ROOT/'data/raw/train.csv',usecols=['PassengerId']))
    exact=candidates(roster)
    result={'all':count(exact),'train':count([r for r in exact if r['row']<n]),
            'competition':count([r for r in exact if r['row']>=n])}
    OUT.mkdir(exist_ok=True)
    (OUT/'missing.json').write_text(json.dumps(exact))
    groups=roster.PassengerId.str[:4].to_numpy()
    # Every known passenger is hidden once along with all others from their booking.
    batches=np.array_split(np.random.default_rng(9983).permutation(np.unique(groups)),40)
    checked=[]
    for k, batch in enumerate(batches):
        mask=np.isin(groups,batch)&roster.Cabin.notna().to_numpy()
        masked=roster.copy()
        masked.loc[mask,'Cabin']=np.nan
        records=candidates(masked)
        for row in records:
            i=row['row']
            if not mask[i]:
                continue
            deck,number,side=roster.Cabin.iloc[i].split('/')
            lo,hi=row['global_options'][deck+side]
            assert lo<=int(number) and (hi is None or int(number)<=hi)
            checked.append(row)
        if k%10==9:
            print('completed masking batches',k+1,flush=True)
    assert len(checked)==roster.Cabin.notna().sum() and len({r['row'] for r in checked})==len(checked)
    result['masked_known']=count(checked)
    result['masked_known']['true_cabin_excluded']=0
    (OUT/'masked_known.json').write_text(json.dumps(checked))
    DEST.write_text(json.dumps(result,indent=2)+'\n')
    print(result,flush=True)


if __name__=='__main__':
    run()
