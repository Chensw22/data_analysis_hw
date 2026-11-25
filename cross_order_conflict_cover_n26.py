"""Find disjoint, opposite-outcome neighbours and freeze a training-only ambiguity filter."""
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from full_roster_booking_validation_n23 import ROOT,data
from exhaustive_family_quorums_n26 import inputs,summary
from price_implied_deck_n26 import FEES


def relations(raw,g,full):
    cabin=raw.Cabin.str.split('/',expand=True)
    bill=raw[FEES]
    # Within each exact categorical cell, distances are maxima of the numeric distances.
    key=raw[['HomePlanet','CryoSleep']].copy()
    key['present_bills']=bill.notna().to_numpy().astype(int) @ (1<<np.arange(5))
    key['positive_bills']=bill.gt(0).to_numpy().astype(int) @ (1<<np.arange(5))
    key['known_age']=raw.Age.notna()
    numerical=[raw.Age.fillna(0)/12,np.log1p(bill.fillna(0))/np.log(4)]
    if full:
        key[['Destination','VIP']]=raw[['Destination','VIP']]
        key[['deck','side']]=cabin[[0,2]]
        key['party']=raw.groupby(g).PassengerId.transform('size').clip(upper=4)
        numerical.append(pd.to_numeric(cabin[1]).fillna(0)/180)
    x=np.column_stack(numerical)
    edges=[]
    for rows in key.groupby(list(key),dropna=False,sort=False).indices.values():
        e=cKDTree(x[rows]).query_pairs(1.,p=np.inf,output_type='ndarray')
        if len(e):
            a,b=rows[e].T
            distinct=g[a]!=g[b]
            edges.append(np.column_stack([a[distinct],b[distinct]]))
    a,b=np.concatenate(edges).T
    d=np.max(abs(x[a]-x[b]),axis=1)
    return a,b,d


def cover(edges,left,right):
    # König alternating paths from unmatched left vertices give a minimum vertex cover.
    if not len(edges):
        return np.empty(0,int)
    graph=csr_matrix((np.ones(len(edges),dtype=bool),(edges[:,0],edges[:,1])),shape=(len(left),len(right)))
    match=maximum_bipartite_matching(graph,perm_type='column')
    L=match<0
    R=np.zeros(len(right),bool)
    while True:
        new_r=np.asarray(graph.T@L) & ~R
        if not new_r.any():
            break
        R |= new_r
        L |= (match>=0)&R[np.maximum(match,0)]
    chosen=np.r_[left[~L],right[R]]
    assert len(chosen)==np.sum(match>=0)
    assert ((~L)[edges[:,0]]|R[edges[:,1]]).all()
    return chosen


def min_cover(a,b,y,eligible):
    cross=eligible[a]&eligible[b]&(y[a]!=y[b])
    a,b=a[cross],b[cross]
    left,right=np.flatnonzero(eligible&~y),np.flatnonzero(eligible&y)
    i=np.searchsorted(left,np.where(y[a],b,a))
    j=np.searchsorted(right,np.where(y[a],a,b))
    edges=np.column_stack([i,j])
    low=cover(edges,left,right)
    high=cover(edges[:,::-1],right,left)
    assert len(low)==len(high)
    # Both orientations are optimal but can differ when several equally small removals exist.
    return len(low),np.union1d(low,high)


def predict(a,b,d,eligible,y,g,removed):
    # Each eligible query draws at most one vote from each other training booking.
    a,b=np.r_[a,b],np.r_[b,a]
    d=np.tile(d,2)
    pick=eligible[a]&~eligible[b]&~np.isin(a,removed)
    df=pd.DataFrame(dict(q=b[pick],t=a[pick],tg=g[a[pick]],d=d[pick]))
    df=df.sort_values(['q','d','t']).drop_duplicates(['q','tg'])
    df=df.groupby('q',sort=False).head(9)
    df['label']=y[df.t]
    means=df.groupby('q').label.agg(['size','mean'])
    p=np.full(len(y),.5)
    enough=means['size']>=3
    p[means.index[enough]]=means['mean'][enough]
    return p


def common_tradeoff(a,b,cur,y):
    # One endpoint is a common error and the other was correct under the same original prediction.
    select=cur['common'][a]&cur['common'][b]&(cur['parent'][a]==cur['parent'][b])&(y[a]!=y[b])
    left,right=a[select],b[select]
    errors=np.where(cur['wrong'][left],left,right)
    count,_=min_cover(left,right,y,cur['common'])
    return dict(distinct_errors_with_correct_neighbor=len(np.unique(errors)),disjoint_error_correct_pairs=count)


def run():
    raw,y,g,f=data()
    _,cur,_=inputs()
    parent,wrong=cur['parent'],cur['wrong']
    result={}
    for full in (False,True):
        aa,bb,dist=relations(raw,g,full)
        for radius in (.5,1.):
            ok=dist<=radius
            a,b,d=aa[ok],bb[ok],dist[ok]
            alternate=y[a]!=y[b]
            conflict=np.zeros(len(y),bool)
            conflict[np.r_[a[alternate],b[alternate]]]=True
            full_cover,_=min_cover(a,b,y,cur['common'])
            n=len(y)
            plain=np.full(n,.5);clean=plain.copy()
            for k in np.unique(f):
                t=f!=k
                assert not np.isin(g[t],g[~t]).any()
                _,dropped=min_cover(a,b,y,t)
                for dst,ban in [(plain,np.array([],int)),(clean,dropped)]:
                    dst[~t]=predict(a,b,d,t,y,g,ban)[~t]
            def decide(p):
                return np.where(parent,p<=1/3,p>=2/3)
            first,second=decide(plain),decide(clean)
            record={'common_rows':int(cur['common'].sum()),'common_errors_with_any_conflicting_neighbor':int((cur['common']&wrong&conflict).sum()),
                    'opposite_label_different_booking_pairs':int(alternate.sum()),'common_disjoint_conflicts':full_cover,'scopes':{}}
            for name,scope in [('all',cur['known']),('common',cur['common']),('new',cur['known']&cur['new'])]:
                record['scopes'][name]={m:summary(s,wrong,scope,f) for m,s in [
                    ('original',first),('edited',second),('only_edited',second&~first),('both',first&second)]}
            record.update(common_tradeoff(a,b,cur,y))
            name=('registration' if full else 'spending')+'_'+str(radius)
            result[name]=record
            print(name,full_cover,record['common_errors_with_any_conflicting_neighbor'],
                  [(m,s['fixed'],s['harmed']) for m,s in record['scopes']['common'].items()],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/cross_order_conflict_cover.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
