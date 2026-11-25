"""Use paid bookings as a masked auxiliary domain for travelers with no visible service use."""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from full_roster_booking_validation_n23 import ROOT, data
from exaone_validation_data import compact_features
from price_implied_deck_n26 import FEES,OUT as FARE
from booking_mix_confidence_transfer_n26 import assignments
from tabicl_consensus_review_n26 import sources,OUT as REVIEW

OUT=ROOT/'artifacts/nov26_masked_activity_transfer'


def design(raw,g):
    x=compact_features(raw).drop(columns=FEES)
    observed=raw[FEES].to_numpy(float)
    pos=observed>0
    zero=~pos.any(1)
    missing=np.isnan(observed)@(1<<np.arange(5))
    group=pd.Series(g)
    size=group.groupby(g).transform('size').to_numpy()
    x['booking_size']=size
    def other(values):
        return pd.Series(values).groupby(g).transform('sum').to_numpy()-values
    x['paid_peers']=other((~zero).astype(int))
    x['sleeping_peers']=other(raw.CryoSleep.eq(True).astype(int).to_numpy())
    x['awake_peers']=other(raw.CryoSleep.eq(False).astype(int).to_numpy())
    x['child_peers']=other(raw.Age.lt(13).astype(int).to_numpy())
    for j,fee in enumerate(FEES):
        x['peers_using_'+fee]=other(pos[:,j].astype(int))
        x['other_bill_'+fee]=np.log1p(other(np.nan_to_num(observed[:,j])).clip(0))
    return x,zero,missing


def train(folds=None):
    import torch
    from joint_party_tabicl_n26 import model
    from tabicl_routed_context_n23 import WEIGHT,DIGEST
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    torch.set_num_threads(2)
    raw,y,g,fold=data()
    frame,zero,missing=design(raw,g)
    OUT.mkdir(exist_ok=True)
    for k in np.unique(fold):
        if folds is not None and k not in folds:
            continue
        path=OUT/f'fold{k}.npz'
        if path.exists(): continue
        t=np.flatnonzero(fold!=k)
        q=np.flatnonzero((fold==k)&zero)
        assert not np.isin(g[t],g[q]).any()
        empty=t[zero[t]]
        paid=t[~zero[t]]
        # Hide all positive bills in the auxiliary domain. Assign observed missingness patterns
        # like real zero-visible rows, within sleep/solo strata. This is simulated transfer.
        sleep=raw.CryoSleep.astype('string').fillna('?').to_numpy()
        ctx=np.char.add(sleep.astype(str),(frame.booking_size.to_numpy()==1).astype(str))
        pattern=missing.copy()
        rng=np.random.default_rng(9843+k)
        for level in np.unique(ctx[paid]):
            donors=empty[ctx[empty]==level]
            buyers=paid[ctx[paid]==level]
            pattern[buyers]=rng.choice(missing[donors] if len(donors) else missing[empty],len(buyers))
        x=frame.copy()
        for j,fee in enumerate(FEES):
            x['missing_'+fee]=(pattern&(1<<j))!=0
        source=zero[t].astype(int)
        strata=ctx[t]+y[t].astype(str)
        sham=source.copy()
        for level in np.unique(strata):
            loc=np.flatnonzero(strata==level)
            sham[loc]=rng.permutation(source[loc])
        result={}
        for name,ids,labels in [('real_only',empty,y[empty]),('mixed',t,y[t]),
                                ('domain',t,2*source+y[t]),('sham',t,2*sham+y[t])]:
            est=model(4233+int(k)).fit(x.iloc[ids],labels)
            p=est.predict_proba(x.iloc[q]);cls=est.classes_.astype(int)
            if name in ('domain','sham'):
                result[name]=p[:,cls==3].ravel()/p[:,cls>=2].sum(1)
                result[name+'_marginal']=p[:,cls%2==1].sum(1)
            else:
                result[name]=p[:,cls==1].ravel()
        # Another training population admits a paid passenger only if a naturally missing service
        # pattern can physically hide all of the passenger's positive fees.
        active=(raw.iloc[paid][FEES].to_numpy()>0)@(1<<np.arange(5))
        feasible=paid[(pattern[paid]&active)==active]
        ids=np.r_[empty,feasible]
        model_fit=model(4233+int(k)).fit(x.iloc[ids],y[ids])
        result['compatible']=model_fit.predict_proba(x.iloc[q])[:,1]
        with path.with_suffix('.tmp').open('wb') as h:
            np.savez_compressed(h,t=t,q=q,truth=y[t],paid=len(paid),compatible_count=len(feasible),**result)
        path.with_suffix('.tmp').replace(path)
        print(k,len(empty),len(paid),len(feasible),flush=True)


def route():
    from catboost import CatBoostClassifier
    from scipy.special import logit,expit
    raw,y,g,fold=data()
    OUT.mkdir(exist_ok=True)
    x,zero,missing=design(raw,g)
    dest=x.Destination.to_numpy(str)
    x['missing_count']=np.isnan(raw[FEES].to_numpy(float)).sum(1)
    ctx=x.drop(columns='Destination')
    def fit(df,target):
        cats=[name for name in df if df[name].dtype==object or pd.api.types.is_string_dtype(df[name])]
        return CatBoostClassifier(iterations=310,depth=4,learning_rate=.05,l2_leaf_reg=8,
            cat_features=cats,random_seed=4781,verbose=False,allow_writing_files=False,thread_count=2).fit(df,target)
    for k in np.unique(fold):
        t=np.flatnonzero((fold!=k)&zero);q=np.flatnonzero((fold==k)&zero)
        assert not np.isin(g[t],g[q]).any()
        full=fit(x.iloc[t],y[t]).predict_proba(x.iloc[q])[:,1]
        prior=fit(ctx.iloc[t],y[t]).predict_proba(ctx.iloc[q])[:,1]
        strata=x.HomePlanet.astype(str)+'/'+x.CryoSleep.astype(str)+'/'+(x.booking_size==1).astype(str)
        strata=strata.to_numpy()[t];sham=y[t].copy()
        rng=np.random.default_rng(4800+int(k))
        for c in np.unique(strata):
            loc=np.flatnonzero(strata==c)
            sham[loc]=rng.permutation(sham[loc])
        result={'route_direct':full,'route_blind':prior}
        for n,labels in [('route_real',y[t]),('route_sham',sham)]:
            fit_route=fit(ctx.iloc[t].assign(hypothesis=labels.astype(int)),dest[t])
            idx=pd.Index(fit_route.classes_).get_indexer(dest[q]);assert (idx>=0).all()
            scores=np.stack([fit_route.predict_proba(ctx.iloc[q].assign(hypothesis=h))[np.arange(len(q)),idx]
                             for h in [0,1]])
            result[n]=expit(logit(prior)+np.log(scores[1]/scores[0]))
        np.savez_compressed(OUT/f'route{k}.npz',q=q,**result)


def score():
    raw,y,g,fold=data()
    _,_,_,v=sources()
    _,zero,missing=design(raw,g)
    old,cur=v['historical'],v['full']
    valid=np.zeros(len(y),bool);parent=cur['anchor'].copy()
    for f,k,t,q in assignments(old,cur,g):
        with np.load(REVIEW/f'pred_bookingmix_{f}_{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            parent[q]^=(z['identities_error']>.5)&(z['identities_outcome']>.5)
        valid[q]=True
    with np.load(FARE/'deck_estimates.npz') as z:
        eligible=z['eligible'][:len(y)]
    for k in np.unique(fold):
        ids=np.flatnonzero(fold==k)
        with np.load(FARE/f'neural_full_{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],ids)
            parent[ids]^=eligible[ids]&((z['raw']>=.5)==parent[ids])&((z['fine']>=.5)!=parent[ids])
    assert not (eligible&zero).any()
    forecasts={};support=[]
    for k in np.unique(fold):
        q=np.flatnonzero((fold==k)&zero)
        with np.load(OUT/f'fold{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            np.testing.assert_array_equal(z['truth'],y[z['t']])
            support.append([int(z['paid']),int(z['compatible_count'])])
            for name in ('real_only','mixed','domain','sham','domain_marginal','compatible'):
                forecasts.setdefault(name,np.zeros(len(y)))[q]=z[name]
        with np.load(OUT/f'route{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            for n in ['route_real','route_sham','route_direct','route_blind']:
                forecasts.setdefault(n,np.zeros(len(y)))[q]=z[n]
    peer_count=pd.Series(~zero).groupby(g).transform('sum').to_numpy()
    common=cur['common']
    fresh=~np.isin(g,g[old['ids']])
    scopes={'all':valid,'no_visible_bill':valid&zero,'complete_zero':valid&zero&(missing==0),
            'incomplete_zero':valid&zero&(missing!=0),'known_sleep':valid&zero&raw.CryoSleep.eq(True).to_numpy(),
            'known_awake':valid&zero&raw.CryoSleep.eq(False).to_numpy(),'unknown_sleep':valid&zero&raw.CryoSleep.isna().to_numpy(),
            'has_paid_peer':valid&zero&(peer_count>0),'no_paid_peer':valid&zero&(peer_count==0),
            'unanimous_zero':valid&zero&common,'fresh':valid&fresh,'fresh_zero':valid&fresh&zero,
            'fresh_unanimous_zero':valid&fresh&zero&common}
    for nm in ('complete_zero','incomplete_zero','known_sleep','known_awake','unknown_sleep','has_paid_peer','no_paid_peer'):
        scopes[nm+'_unanimous']=scopes[nm]&common
        scopes['fresh_'+nm]=scopes[nm]&fresh
    wrong=parent!=y
    def counts(mask,flip):
        return dict(rows=int(mask.sum()),errors=int((wrong&mask).sum()),
                    fixed=int((flip&mask&wrong).sum()),harmed=int((flip&mask&~wrong).sum()))
    disagreement={n:zero&((p>=.5)!=parent) for n,p in forecasts.items()}
    for n in ('mixed','domain','compatible','sham'):
        disagreement[n+'_induced']=disagreement[n]&~disagreement['real_only']
    for n in ('route_real','route_sham'):
        disagreement[n+'_induced']=disagreement[n]&~disagreement['route_direct']
    disagreement['domain_and_route']=disagreement['domain']&disagreement['route_real']
    disagreement['domain_and_compatible']=disagreement['domain']&disagreement['compatible']
    result={n:{s:counts(m,change) for s,m in scopes.items()} for n,change in disagreement.items()}
    known_common=valid&zero&common
    population={}
    for kind,field in [('origin',raw.HomePlanet.fillna('?')),('deck',raw.Cabin.str.split('/').str[0].fillna('?')),
                       ('destination',raw.Destination.fillna('?')),('solo',raw.PassengerId.groupby(g).transform('size')==1),
                       ('original_transport_prediction',parent)]:
        field=np.asarray(field)
        population[kind]={str(level):{'rows':int((known_common&(field==level)).sum()),
                'errors':int((known_common&(field==level)&wrong).sum())} for level in np.unique(field[known_common])}
    # Measure ranking power even when no probability justifies reversing a vote.
    # A fixed tenth is descriptive across held-out orders, not a calibrated flip policy.
    from sklearn.metrics import average_precision_score
    ranking={}
    risks={name:np.where(parent,1-p,p) for name,p in forecasts.items()}
    direction=np.where(parent==cur['anchor'],1,-1)
    risks['existing_anchor_confidence']=direction*cur['off']
    risks['existing_panel_median_confidence']=direction*(cur['off']+cur['basic'][:,3])
    for name,surprise in risks.items():
        ranking[name]={}
        for label,mask in [('unanimous_zero',known_common),('fresh_unanimous_zero',known_common&fresh)]:
            ids=np.flatnonzero(mask)
            order=ids[np.argsort(-surprise[ids],kind='stable')]
            band=order[:int(np.ceil(.1*len(order)))]
            ranking[name][label]=dict(
                rows=len(ids),errors=int(wrong[ids].sum()),average_precision=float(average_precision_score(wrong[ids],surprise[ids])),
                top_tenth_rows=len(band),top_tenth_errors=int(wrong[band].sum()),
            )
    all_sleep=raw.CryoSleep.astype('string').fillna('?').to_numpy()
    by_sleep={str(c):dict(real_zero=int((zero&(all_sleep==c)).sum()),paid_source=int((~zero&(all_sleep==c)).sum()))
              for c in np.unique(all_sleep)}
    report=dict(
                reference='fare-informed 98-passenger test candidate (3bb61e7); grouped OOF decisions',
                comparison='Fixed 0.5 cutoff on no-visible-positive rows. Other rows retain the reference prediction.',
                common_definition='Every original model family and archived prediction agrees with the original anchor.',
                support_columns=['paid training passengers', 'positive charges all hidden by sampled missingness pattern'],
                eligible_paid_per_fold=support,all_passengers_by_sleep=by_sleep,
                bookings_by_scope={key:len(np.unique(g[mask])) for key,mask in scopes.items()},
                common_zero_profile=population,unanimous_error_ranking=ranking,methods=result)
    (ROOT/'reports/nov23_matched_learning_curve/masked_activity_transfer.json').write_text(json.dumps(report,indent=2)+'\n')
    print([(n,d['all']['fixed'],d['all']['harmed'],d['unanimous_zero']['fixed'],d['unanimous_zero']['harmed']) for n,d in result.items()])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('train','route','score'))
    parser.add_argument('--folds',nargs='+',type=int,choices=range(8))
    args=parser.parse_args()
    if args.action=='train':
        train(args.folds)
    else:
        {'route':route,'score':score}[args.action]()
