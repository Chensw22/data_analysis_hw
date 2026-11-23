"""Quantify outcome discrimination between distinct payer bookings at similar incumbent probabilities."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import GroupKFold

from archive_error_recovery_n23 import panel
from broad_bill_exception_n23 import strata
from broad_boosting_path_n23 import calibrated
from broad_geographic_transfer_n23 import BILLS, load
from broad_partition_evidence_n23 import ART, SEEDS, historical, write
from paid_conditional_tilt_n23 import REPORT


def ranker_probabilities(z, group, paid):
    ref, base, fold, y = [z[k] for k in ('ref','base','fold','y')]
    eligible = paid & ~ref & (base < .5)
    scores = np.c_[logit(base.clip(.002,.998)), z['teacher'], z['ordinary'], z['matched']]
    p = base.copy()
    for f in np.unique(fold):
        ix = np.flatnonzero((fold == f) & eligible)
        for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ix,groups=group[ix]):
            t,q=ix[a],ix[b]
            p[q]=calibrated(scores[t],scores[q],y,t)
    return p


def panel_scores(row, y, group, paid):
    rows, truth, baseline, historical_scores, _, _, _ = panel()
    np.testing.assert_array_equal(rows,row)
    np.testing.assert_array_equal(truth,y)
    out = dict(base=baseline, archive_mean=historical_scores.mean(axis=2), conditional=[], pairwise=[])
    for i,seed in enumerate(SEEDS):
        for folder,method in [('paid_pairwise_tree','pairwise'),('paid_conditional_tilt','conditional')]:
            with np.load(ART/f'nov23_{folder}/{seed}.npz') as z:
                for k,v in [('row',row),('y',y),('base',baseline[i])]:
                    np.testing.assert_array_equal(z[k],v)
                score=ranker_probabilities(z,group,paid) if method=='pairwise' else z['conditional']
                out[method].append(score)
    return {k:np.asarray(v) for k,v in out.items()}


def paired_intervals(group,cases,donors,differences,traits):
    # Two pairs can share a booking even though no individual is matched twice.
    bookings,inv=np.unique(np.r_[group[cases],group[donors]],return_inverse=True)
    n=len(cases)
    edges=coo_matrix((np.ones(n),(inv[:n],inv[n:])),shape=(len(bookings),len(bookings)))
    count,component=connected_components(edges,directed=False)
    pair_component=component[inv[:n]]
    np.testing.assert_array_equal(pair_component,component[inv[n:]])
    size=np.bincount(pair_component,minlength=count)
    values=np.column_stack([*differences.values(),*traits.values()])
    total=np.zeros((count,values.shape[1]))
    np.add.at(total,pair_component,values)
    draws=[]
    rng=np.random.default_rng(709)
    for _ in range(50):
        batch=rng.integers(count,size=(100,count))
        draws.append(total[batch].sum(1)/size[batch].sum(1)[:,None])
    samples=np.vstack(draws)
    lower,upper=np.quantile(samples,[.025,.975],axis=0)
    avg=values.mean(0)
    names=[*differences,*traits]
    lo=len(differences)
    sd=samples[:,lo:].std(0,ddof=1)
    joint=np.quantile(np.max(abs(samples[:,lo:]-avg[lo:])/np.maximum(sd,1.e-12),axis=1),.95)
    return dict(components=int(count),largest_component_pairs=int(size.max()),draws=len(samples),
        pointwise={k:[float(lower[i]),float(avg[i]),float(upper[i])] for i,k in enumerate(names)},
        simultaneous_traits={k:[float(avg[lo+i]-joint*sd[i]),float(avg[lo+i]+joint*sd[i])]
                             for i,k in enumerate(traits)})


def main():
    rows,y,frame,g,_=load()
    ids,labels,masks=historical()
    np.testing.assert_array_equal(rows,ids)
    np.testing.assert_array_equal(y,labels)
    paid=strata(frame)==2
    scores=panel_scores(rows,y,g,paid)
    base=scores['base']
    negative=(base<.5).all(0)&paid
    cases=np.flatnonzero(negative&y)
    controls=np.flatnonzero(negative&~y)
    assert len(cases)==428 and len(controls)==2550
    cost=np.square(base[:,cases,None]-base[:,controls][:,None,:]).sum(0)
    cost[g[cases,None]==g[controls]]=1e5
    idx,j=linear_sum_assignment(cost)
    np.testing.assert_array_equal(idx,np.arange(len(cases)))
    donors=controls[j]
    assert len(np.unique(donors))==len(cases) and (g[cases]!=g[donors]).all()
    raw=frame[BILLS].fillna(0)
    known=raw.sum(axis=1)
    fields=dict(earth=frame.HomePlanet=='Earth',deck_g=frame.Deck=='G',solo=frame.BookingSize==1,
                high_known_total=known>=1000,
                spa_vr_room_ge75=raw[['Spa','VRDeck','RoomService']].sum(axis=1)>=.75*known,
                missing_bill=frame[BILLS].isna().any(axis=1))
    phenotype={k:{name:float(x.iloc[ix].mean()) for name,ix in
                  [('cases',cases),('all_controls',controls),('matched_controls',donors)]} for k,x in fields.items()}
    goals=[]
    for p in base:
        needed=int(np.floor(.85*len(y)))+1-int(((p>=.5)==y).sum())
        goals.append(dict(net_fixes=needed,pair_wins=int(np.ceil((len(cases)+needed)/2))))
    comparisons={}
    for method,p in scores.items():
        delta=p[:,cases]-p[:,donors]
        each=(delta>0)+.5*(delta==0)
        comparisons[method]=dict(wins=each.sum(1).tolist(),pairs=len(cases),
            expected_net=(2*each.sum(1)-len(cases)).tolist(),
            mean_score_pair_accuracy=float(((delta.mean(0)>0)+.5*(delta.mean(0)==0)).mean()))
    # Include a range of common demographic and bill summaries in the simultaneous comparison.
    fields.update({f'planet_{k}':frame.HomePlanet==k for k in ['Europa','Mars']})
    fields.update({f'deck_{k.lower()}':frame.Deck==k for k in 'ABCDEF'})
    fields.update(under18=frame.Age<18,at_least40=frame.Age>=40,two_services=(raw>0).sum(axis=1)>=2)
    for service in BILLS:
        fields['dominant_'+service]=raw.idxmax(axis=1)==service
    assert len(fields)==22
    traits={}
    for k,x in fields.items():
        values=x.to_numpy(dtype=float)
        traits[k]=values[cases]-values[donors]
    def choices(p):
        gap=p.mean(0)[cases]-p.mean(0)[donors]
        return (gap>0).astype(float)+.5*(gap==0)
    anchor=choices(base)
    differences={k:choices(p)-anchor for k,p in scores.items() if k!='base'}
    uncertainty=paired_intervals(g,cases,donors,differences,traits)
    report=dict(rows=len(y),cases=len(cases),controls=len(controls),
                risk={name:base[:,ix].mean(1).tolist() for name,ix in
                      [('cases',cases),('all_controls',controls),('matched_controls',donors)]},
                maximum_risk_gap_quantiles=np.quantile(abs(base[:,cases]-base[:,donors]).max(0),[.5,.9,1.]).tolist(),
                phenotype=phenotype,targets=goals,comparisons=comparisons,uncertainty=uncertainty)
    write(REPORT/'matched_passengers.json',report)
    print(report,flush=True)


if __name__=='__main__':
    main()
