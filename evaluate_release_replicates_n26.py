"""Measure when recalibrating the complete release on disjoint bookings improves its hard decisions."""
import json
import numpy as np
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from full_roster_booking_validation_n23 import ROOT
from evaluate_nested_release_n26 import OUT as NESTED, DECK, REVIEW, protocols, sources, outcome_scores, panel_predictions, replay
from release_field_contributions_n26 import evidence
from replicate_release_calibration_n26 import OUT
from solo_archive_recovery_n26 import ticker


def describe(big,small,decision,other):
    center=logit(big.clip(.005,.995))
    deviations=logit(small.clip(.005,.995))-center[:,None,:]
    return np.column_stack([deviations.mean(1),deviations.std(1),np.min(small,axis=2).mean(1),
                            (other!=decision[:,None]).mean(1)])


def fitted(x,labels,z,weights=None):
    scale=StandardScaler().fit(x,sample_weight=weights)
    model=LogisticRegression(C=.12,max_iter=500).fit(scale.transform(x).clip(-7,7),labels,sample_weight=weights)
    return model.predict_proba(scale.transform(z).clip(-7,7))[:,1]


def record_scores(key,src,dst,t,q,p,s,enabled,y,paid):
    a,b=src['ids'][t],dst['ids'][q]
    with np.load(OUT/f'{key}.npz') as z:
        np.testing.assert_array_equal(z['ids'],a)
        np.testing.assert_array_equal(z['outer'],b)
        np.testing.assert_array_equal(z['train_y'],y[a])
        which=z['fold'][:,None]!=np.arange(3)
        tr=z['train'][which].reshape(len(a),2,2)
        te=z['query']
    with np.load(NESTED/f'{key}.npz') as z:
        np.testing.assert_array_equal(z['ids'],a)
        big_t=z['risk']
    with np.load(REVIEW/f'pred_{key}.npz') as z:
        np.testing.assert_array_equal(z['q'],b)
        big_q=np.column_stack([z['identities_error'],z['identities_outcome']])
    assert np.isfinite(tr).all() and np.isfinite(te).all()
    origin='full' if key.startswith('full_') else 'historical'
    def train_view(risk):
        parent,_,fields=evidence(src,s[origin],p[origin],t,risk,enabled[a])
        return parent,fields
    def query_view(risk):
        parent,_,fields=evidence(dst,s['full'],p['full'],q,risk,enabled[b])
        return parent,fields
    old,xt=train_view(big_t)
    parent,xq=query_view(big_q)
    np.testing.assert_array_equal(parent,replay(key,q,dst,p['full'],enabled))
    trains=[train_view(tr[:,k]) for k in range(2)]
    queries=[query_view(te[:,k]) for k in range(3)]
    et=np.column_stack([v[0] for v in trains])
    eq=np.column_stack([v[0] for v in queries])
    alt=eq!=parent[:,None]
    actions={'one_opposes':alt.any(1),'majority_opposes':alt.sum(1)>=2,'all_oppose':alt.all(1)}
    mean=te.mean(1)
    # Shrink the full predictions toward the three separate fits, or extend the observed change in odds.
    empirical=query_view(mean)[0]
    actions['pooled_small_scores']=empirical!=parent
    actions['shrink_toward_small']=query_view((big_q+mean)/2)[0]!=parent
    odds=2*logit(big_q.clip(.005,.995))-logit(te.clip(.005,.995)).mean(1)
    actions['extend_to_larger_support']=query_view(expit(odds))[0]!=parent
    primary=fitted(xt,old!=y[a],xq)
    train_p=np.column_stack([xt,describe(big_t,tr,old,et)])
    # Training has two clean small fits per person. Average over all three corresponding query pairs.
    tested=[]
    for i,j in [(0,1),(0,2),(1,2)]:
        aug=np.column_stack([xq,describe(big_q,te[:,[i,j]],parent,eq[:,[i,j]])])
        tested.append(fitted(train_p,old!=y[a],aug))
    informed=np.mean(tested,axis=0)
    actions['primary_risk']=primary>.5
    actions['replicate_informed_risk']=informed>.5
    actions['replicate_informed_extra']=(informed>.5)&(primary<=.5)
    # Replicate decisions share one outcome per passenger. Preserve the total weight of that passenger.
    xaug=np.concatenate([xt,*[v[1] for v in trains]])
    yaug=np.concatenate([old!=y[a],*[v[0]!=y[a] for v in trains]])
    risk=fitted(xaug,yaug,xq,np.full(len(xaug),1/3))
    actions['augmented_examples']=risk>.5
    actions['risk_and_unanimous_replication']=(risk>.5)&alt.all(1)
    actions['new_risk_and_replication']=(risk>.5)&(primary<=.5)&alt.any(1)
    strict=((s['full'][q]>=.5)==parent[:,None]).all(1)
    dry=~paid[b]
    domain=np.where(dry,((s['full'][q,:-2]>=.5)==parent[:,None]).all(1),strict)
    return b,parent,parent!=y[b],strict,domain,actions


def run():
    from price_implied_deck_n26 import FEES
    raw,y,g,panels=sources()
    paid=raw[FEES].gt(0).any(axis=1).to_numpy()
    p={tag:panel_predictions(v,tag) for tag,v in panels.items()}
    s={tag:outcome_scores(v,p[tag]) for tag,v in panels.items()}
    with np.load(DECK/'deck_estimates.npz') as z:
        enabled=z['eligible']
    items={}
    for key,src,dst,t,q in protocols(panels,g,include_full=True):
        assert not np.isin(g[src['ids'][t]],g[dst['ids'][q]]).any()
        tag='full' if key.startswith('full_') else ('bookingmix' if key.startswith('bookingmix') else key)
        items.setdefault(tag,[]).append(record_scores(key,src,dst,t,q,p,s,enabled,y,paid))
    report={}
    for tag,rows in items.items():
        ids,parent,wrong,strict,domain=[np.concatenate([r[j] for r in rows]) for j in range(5)]
        assert len(ids)==len(np.unique(ids))
        groups={'all':np.ones(len(ids),bool),'predicted_yes':parent,'predicted_no':~parent,
                'original_common':strict,'added_common':domain&~strict,'contested':~domain}
        r={}
        out={}
        for name in rows[0][-1]:
            act=np.concatenate([v[-1][name] for v in rows])
            out[name]=act
            r[name]={k:ticker(scope,wrong,act) for k,scope in groups.items()}
        np.savez_compressed(OUT/f'decisions_{tag}.npz',ids=ids,parent=parent,wrong=wrong,**out)
        report[tag]=r
        print(tag,[(n,v['all']['fixed'],v['all']['harmed']) for n,v in r.items()],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/replicate_release_calibration.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
