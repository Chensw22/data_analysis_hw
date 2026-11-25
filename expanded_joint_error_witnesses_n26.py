"""Find additional coverage from frozen, previously evaluated full-roster models on both releases' errors."""
import json
import numpy as np
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
from arbitration_against_scored_nine_n26 import champions,fit
from outside_panel_coverage_n26 import archives,projection
from target_mix_expert_routing_n26 import ROOT,batches,sources,ticker
from price_implied_deck_n26 import FEES

FAMILIES=['booking_transport_classifier','conditional_age_density','conditional_bill_deviations',
          'confidence_ordered_booking','constrained_track_odds','joint_manifest_forest','joint_party_tabicl',
          'latent_receipt_channel','manifest_role_joint','neighbor_booking_composition','solo_selection_likelihood']


def decode(name,z,solo):
    if name=='booking_transport_classifier':
        return z['p'][:,4]
    if name=='conditional_age_density':
        density=z['true_year'].clip(1e-9,1)
        return np.where(z['age']>=0,expit(logit(z['prior'].clip(.001,.999))+np.log(density[:,1]/density[:,0])),z['direct'])
    if name=='conditional_bill_deviations':
        return expit(z['deviation'])
    if name=='confidence_ordered_booking':
        return z['conditional']
    if name=='constrained_track_odds':
        return expit(z['prior']+z['real'])
    if name=='joint_manifest_forest':
        return np.mean([z[k][:,1] for k in ['service_pair','route','sleep_side','origin_berth']],axis=0)
    if name=='joint_party_tabicl':
        return z['joint']
    if name=='latent_receipt_channel':
        return expit(z['prior']+z['channel'])
    if name=='manifest_role_joint':
        mass=z['real'][np.arange(len(solo)),z['role']]
        return mass[:,1]/mass.sum(1).clip(1e-12)
    if name=='neighbor_booking_composition':
        return z['near']
    if name=='solo_selection_likelihood':
        a,b=[z[n].clip(.002,.998) for n in ('true_0','true_1')]
        ratio=np.where(solo,np.log(b/a),np.log((1-b)/(1-a)))
        return expit(logit(z['blind'].clip(.002,.998))+ratio)
    raise ValueError(name)


def load_scores(raw,y,g,panels):
    solo=raw.PassengerId.groupby(g).transform('size').eq(1).to_numpy()
    output={}
    for tag,v in panels.items():
        p=np.full((len(y),len(FAMILIES)),np.nan)
        for k in np.unique(v['fold']):
            q=v['ids'][v['fold']==k]
            for j,name in enumerate(FAMILIES):
                path=ROOT/f'artifacts/nov26_{name}/{tag}_{k}.npz'
                with np.load(path) as z:
                    t=z['t']
                    assert np.isin(t,v['ids'][v['fold']!=k]).all() and not np.isin(g[t],g[q]).any()
                    np.testing.assert_array_equal(z['q'],q)
                    label=next(n for n in ('train_y','truth','labels') if n in z)
                    np.testing.assert_array_equal(z[label],y[t])
                    if 'passenger' in z:
                        np.testing.assert_array_equal(z['passenger'],raw.PassengerId.iloc[q])
                    pred=decode(name,z,solo[q])
                    assert pred.shape==q.shape and np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
                    p[q,j]=pred
        assert np.isfinite(p[v['ids']]).all()
        output[tag]=p
    return output


def run():
    raw,y,g,f,views,original=projection(include_full=True)
    ext,_=archives(raw,y,g,f,True)
    outside=np.column_stack([p>=.5 for p in ext.values()])
    nine,known,hist,old_ids=champions()
    probabilities=load_scores(raw,y,g,sources()[-1])
    dry=~raw[FEES].gt(0).any(axis=1).to_numpy()
    bags={}
    for b in batches(True):
        origin='full' if b.tag=='full' else 'historical'
        old=b.train_cell%2!=0
        parent=b.query_cell%2!=0
        a,q=b.train_ids,b.ids
        same_t=old==(nine[a] if origin=='full' else hist[a])
        same_q=parent==nine[q]
        assert known[q].all()
        o=(original[q]!=parent[:,None])
        o[dry[q],-2:]=False
        blind=same_q&~o.any(1)&~(outside[q]!=parent[:,None]).any(1)
        qt=probabilities[origin][a]
        qq=probabilities['full'][q]
        sign=lambda parent:np.where(parent,-1,1)[:,None]
        signed_t=logit(qt.clip(.002,.998))*sign(old)
        signed_q=logit(qq.clip(.002,.998))*sign(parent)
        labels=b.train_error
        base=fit(b.train_fields,labels,b.query_fields,same_t.astype(float))
        combined=fit(np.column_stack([b.train_fields,signed_t]),labels,
                     np.column_stack([b.query_fields,signed_q]),same_t.astype(float))
        votes=(qq>=.5)!=parent[:,None]
        actions={n:same_q&votes[:,j] for j,n in enumerate(FAMILIES)}
        actions.update(any=same_q&votes.any(1),majority=same_q&(votes.sum(1)>=6),
                       pooled_prior=same_q&(base>.5),enhanced=same_q&(combined>.5),
                       added_by_models=same_q&(combined>.5)&(base<=.5),
                       addition_confirmed=same_q&(combined>.5)&(base<=.5)&(votes.sum(1)>=6))
        bags.setdefault(b.tag,[]).append((q,b.query_error,same_q,blind,actions))
    report={}
    for tag,items in bags.items():
        ids,wrong,joint,blind=[np.concatenate([v[j] for v in items]) for j in range(4)]
        assert len(ids)==len(np.unique(ids))
        r={}
        for name in items[0][-1]:
            action=np.concatenate([v[-1][name] for v in items])
            r[name]={n:ticker(m,wrong,action) for n,m in [('common_decision',joint),('uncovered_by_previous_models',blind)]}
        assert r['any']['uncovered_by_previous_models']['errors']=={'full':179,'bookingmix':147,'historical_to_new':25}[tag]
        print(tag,{n:r[n] for n in ['any','enhanced','added_by_models']},flush=True)
        report[tag]=r
    (ROOT/'reports/nov23_matched_learning_curve/expanded_joint_error_witnesses.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
