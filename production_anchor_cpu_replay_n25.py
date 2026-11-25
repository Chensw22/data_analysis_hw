"""Audit the historical release chain and bound two unavailable committee gates on the eight-fold roster."""
import json

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import export_consensus_routes as routes
import peer_bill_correction as peer
import relational_error_corrector as ref
from confirm_sleeper_scope import confirm
from exact_fold_error_selector import cached
from export_selected_tree_pair_n25 import NAME,NINE,SEEDS,decisions,query_score
from export_budget_quarter_n23 import source
from exaone_validation_data import compact_features
from full_roster_adaptive_bill_prior_n23 import EVIDENCE
from full_roster_booking_validation_n23 import FAMILIES
from full_roster_conditional_pool_n23 import ROOT,banks,data,latest_reference
from full_roster_permutation_selection_n24 import feature_bank
from full_roster_selected_tree_replay_n24 import original_prediction,prepared_vote
from kaggle_workflow import sha256
from party_negative_rescue import family_contrasts,risk,decide
from route_consensus_errors import FAMILIES as PRODUCTION_FAMILIES
from sleeper_consensus_scope import apply as scope,cohort
from sleeper_manifest_risk import design

ART=ROOT/'artifacts'
OUT=ART/'nov25_production_anchor_cpu'
TALLY=ROOT/'reports/nov23_matched_learning_curve/production_anchor_cpu.json'


def offsets(base,mask,shift):
    p=base.astype(float).copy()
    p[mask]=expit(ref.logit(p[mask])+shift[mask])
    return p


def differences(a,b):
    return int(np.count_nonzero(a!=b))


def get_challenger(y,g):
    panels=[]
    for seed in SEEDS:
        p=np.full(len(y),np.nan)
        for fold in range(5):
            with np.load(ART/f'nov24_selected_tree_replay/{seed}_{fold}.npz') as z:
                t,q=z['t'],z['q']
                assert not np.isin(g[t],g[q]).any() and np.isnan(p[q]).all()
                np.testing.assert_array_equal(y[t],z['truth'])
                p[q]=z['p']
        assert np.isfinite(p).all()
        panels.append(p)
    return np.mean(panels,axis=0)


def score(base,anchor,challenger,orig_tree,votes,y):
    champ=prepared_vote(orig_tree,base,votes)
    alternative=decisions(challenger,anchor,champ,base,champ.astype(float))>=.5
    flip=alternative!=champ
    delta=flip*np.where(champ!=y,1,-1)
    return np.column_stack([flip,delta])


def certified(anchor,cabin_patch,has_peer_bills,earth_sleep,active_family):
    # If the first two stages cannot touch the probability, the sign is known
    # exactly when applying the one-direction-only v128 and v131 prerequisites.
    pos=anchor>=.5
    return ~cabin_patch & ~has_peer_bills & ~(pos & earth_sleep) & ~(~pos & active_family)


def exact_historical(raw,y,g,roster,features,controls,size,awake,cabin):
    inp,_,f,bases,_=ref.overlap.load_authenticated()
    rows=inp['dev']
    np.testing.assert_array_equal(raw.PassengerId.iloc[rows],inp['passengers'])
    np.testing.assert_array_equal(y[rows],inp['y'])
    def get(path,key):
        return cached(path+'.npz',key,rows,f)
    v96=get('v96_oof_completion/oof','prob')
    exact='v108_exact_oof/oof'
    path=ART/(exact+'.npz')
    assert sha256(path)==json.loads((ROOT/'reports/v108_exact_oof/result.json').read_text())['oof_sha256']
    np.testing.assert_array_equal(v96,get(exact,'v96'))
    v106=np.empty_like(get(exact,'v106'))
    for k in np.unique(f):
        mask=f==k
        with np.load(ART/f'v108_exact_oof/fold{k}.npz') as z:
            np.testing.assert_array_equal(z['query_ids'],rows[mask])
            np.testing.assert_array_equal(z['v96'],v96[mask])
            v106[mask]=z['v106']
    np.testing.assert_array_equal(v106,get(exact,'v106'))
    members,known,_,fields=peer.context()
    np.testing.assert_array_equal(members,roster.PassengerId)
    shift=get(exact,'peer_shift')
    np.testing.assert_array_equal(get(exact,'peer_eligible'),known[rows])
    v108=offsets(v106,known[rows],shift)
    np.testing.assert_array_equal(v108,get(exact,'v108'))
    five=np.column_stack([bases[k] for k in PRODUCTION_FAMILIES])
    v127,_=scope(v108>=.5,five>=.5,get('replicate_error_committee','original_either'),inp['features'].iloc[rows])
    np.testing.assert_array_equal(v127,get('sleeper_consensus_scope','original'))
    v128=confirm(v108>=.5,v127,get('sleeper_manifest_risk','manifest_berth_party_a'))
    archived=pd.read_csv(ART/'nov21_party_negative/oof.csv')
    np.testing.assert_array_equal(archived.row,rows)
    np.testing.assert_array_equal(v128,archived.v128)
    experts=np.empty((len(rows),2))
    for k in np.unique(f):
        q=rows[f==k]; t=rows[(f!=k)&~np.isin(g[rows],g[q])]
        experts[f==k]=risk(features,controls,y,t,q,size,awake)
    v131=decide(v128,v108,five,experts,awake[rows]&(size[rows]>1))['family_consensus']
    np.testing.assert_array_equal(v131,archived.family_consensus)
    with np.load(ART/'nov23_incumbent_budget_transfer/oof.npz') as z:
        np.testing.assert_array_equal(rows,z['row'])
        np.testing.assert_array_equal(v131,z['v131'])
    seq=[v96>=.5,v106>=.5,v108>=.5,v128,v131]
    cert=certified(v96,cabin[rows],known[rows],cohort(inp['features'].iloc[rows]),awake[rows]&(size[rows]>1))
    for step in seq[1:]:
        np.testing.assert_array_equal(step[cert],seq[0][cert])
    return dict(rows=len(rows),certified_v96_rows=int(cert.sum()),changes=[differences(a,b) for a,b in zip(seq,seq[1:])],
                correct=[int((a==y[rows]).sum()) for a in seq],peer_probability_max_difference=0,
                equality_checks=['v96 native OOF','v106 exact outer query batches','v108 frozen peer offsets',
                                 'v127 from v118 and five votes','v128 cohort witness','v131 refitted family decision'])


def deployed(raw,y,test,features,controls,size,awake,cabin):
    from export_nov21_error_batch import test_probabilities
    n=len(y); q=np.arange(n,n+len(test)); testids=test.PassengerId.to_numpy(str)
    x=compact_features(test)
    p96,_,_=source('proposal_online_first_counts15',test)
    p106,_,_=source('proposal_nov18_peer_cabin',test)
    _,known,_,pf=peer.context()
    pack=joblib.load(ART/'proposal_v108_party_bills/peer_coefficients.joblib')
    shifts=np.zeros(len(q)); use=known[q]
    shifts[use]=peer.apply(pack,pf[q[use]])
    p108=offsets(p106,use,shifts)
    saved108,_,_=source(routes.BASE,test)
    np.testing.assert_allclose(saved108,p108,atol=1e-14,rtol=0)
    five=np.column_stack([test_probabilities(k,testids) for k in routes.HISTORICAL])
    child=routes.official_decisions('proposal_v118_error_union',testids)[0]
    parent,_=scope(p108>=.5,five>=.5,child,x)
    np.testing.assert_array_equal(parent,routes.official_decisions('proposal_v127_sleeper_scope',testids)[0])
    witness=joblib.load(ART/'proposal_v128_manifest_confirmation/model.joblib')
    manifest=witness.predict_proba(design(x,True))[:,1]
    v128=confirm(p108>=.5,parent,manifest>=.5)
    np.testing.assert_array_equal(v128,routes.official_decisions('proposal_v128_manifest_confirmation',testids)[0])
    family=risk(features,controls,y,np.arange(n),q,size,awake)
    v131=decide(v128,p108,five,family,awake[q]&(size[q]>1))['family_consensus']
    np.testing.assert_array_equal(v131,routes.official_decisions('proposal_v131_party_consensus',testids)[0])
    seq=[p96>=.5,p106>=.5,p108>=.5,v128,v131]
    X,labels,_=feature_bank(test)
    challenger=query_score(joblib.load(ART/NAME/'ensemble.joblib'),pd.DataFrame(X,columns=labels),test.PassengerId.str.split('_').str[0].to_numpy())
    soft,b,_=source(NINE,test)
    output=decisions(challenger,p96,b,v131,soft)>=.5
    np.testing.assert_array_equal(output,source(NAME,test)[1])
    cert=certified(p96,cabin[q],use,cohort(x),awake[q]&(size[q]>1))
    for step in seq[1:]:
        np.testing.assert_array_equal(step[cert],seq[0][cert])
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/'certified_test.npz',row=np.arange(len(test)),passenger=testids,
                        certified_v131_equals_v96=cert,v96=p96)
    return dict(rows=len(test),certified_v96_rows=int(cert.sum()),certified_tree_changes=differences(output[cert],b[cert]),
                certified_nine_changes=differences(b[cert],v131[cert]),changes=[differences(a,b) for a,b in zip(seq,seq[1:])],
                peer_probability_max_difference=float(abs(p108-saved108).max()),tree_changes=differences(output,b))


def main():
    raw,y,g,f=data(); n=len(y)
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=lambda c:c!='Name')
    roster=pd.concat([raw,test],ignore_index=True)
    features,controls,size,x=family_contrasts(roster)
    awake=x.CryoSleep.ne('True').to_numpy()
    cabin_keys=json.loads((ART/'v108_exact_oof/cabin_updates_all_roster.json').read_text())
    cabin_all=np.isin(np.arange(len(roster)),[int(i) for i in cabin_keys])
    historical=exact_historical(raw,y,g,roster,features,controls,size,awake,cabin_all)
    official=deployed(raw,y,test,features,controls,size,awake,cabin_all)
    names,half,full=banks(raw,y,f)
    anchor=full[:,names.index('mitra96')]
    nine_votes=full[:,[names.index(k) for k in (*FAMILIES,'mitra96',*EVIDENCE)]]
    orig_tree=original_prediction(np.arange(n),f,g,y,'full')
    tree=get_challenger(y,g)
    _,known,_,pf=peer.context()
    known=known[:n]
    v108=np.full(n,np.nan); witness=np.full(n,np.nan); family=np.full(n,np.nan)
    manifest_x=design(x,True)
    sleeping=cohort(x)
    for k in np.unique(f):
        q=np.flatnonzero(f==k); t=np.flatnonzero(f!=k)
        # Half-panel teacher predictions for these three folds used only opposite-parity
        # training labels. Unlike full OOF, these never encode labels from the query fold.
        cal=t[(f[t]%2==k%2)&known[t]]
        pack=peer.fit_offset(pf[cal],half[cal,names.index('mitra96')],y[cal])
        use=known[q]
        shift=np.zeros(len(q)); shift[use]=peer.apply(pack,pf[q[use]])
        v108[q]=offsets(anchor[q],use,shift)
        model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1000))
        m=t[sleeping[t]]
        model.fit(manifest_x[m],y[m]); witness[q]=model.predict_proba(manifest_x[q])[:,1]
        family[q]=risk(features,controls,y,t,q,size,awake)[:,0]
    base=v108>=.5
    may_lower=base&sleeping[:n]&(witness<.5)
    may_raise=~base&awake[:n]&(size[:n]>1)&(family>=.60)
    assert not (may_lower&may_raise).any()
    possible=may_lower|may_raise
    approx_votes=full[:,[names.index(k) for k in FAMILIES]]>=.5
    proxy=base ^ ((may_lower&approx_votes.all(1)) | (may_raise&~approx_votes.any(1)))
    ref_old=latest_reference(y,g,f,names,half,full)
    solo=size[:n]==1
    stages={}
    for name,b in [('earlier_half_gate',ref_old),('peer_offset_only',base),('necessary_conditions_suffice',base^possible),
                   ('available_family_votes',proxy)]:
        metrics=score(b,anchor,tree,orig_tree,nine_votes,y)
        stages[name]=dict(protected_against_v96=differences(b,anchor>=.5),correct=int((b==y).sum()),
            cuts=int(metrics[:,0].sum()),fixed=int((metrics[:,1]==1).sum()),harmed=int((metrics[:,1]==-1).sum()),
            net=int(metrics[:,1].sum()),solo_cuts=int(metrics[solo,0].sum()))
    # Conservative partial-identification interval. Unknown committee votes may take
    # either value on each permitted row, conditional on this approximate input v108.
    a=score(base,anchor,tree,orig_tree,nine_votes,y)
    b=score(base^possible,anchor,tree,orig_tree,nine_votes,y)
    cabin=cabin_all[:n]
    cert=certified(anchor,cabin,known,sleeping[:n],awake[:n]&(size[:n]>1))
    np.testing.assert_array_equal(proxy[cert],anchor[cert]>=.5)
    by_certificate={}
    for name,starting in [('earlier_half_gate',ref_old),('available_family_votes',proxy),('v96',anchor>=.5)]:
        effects=score(starting,anchor,tree,orig_tree,nine_votes,y)
        by_certificate[name]={key:dict(rows=int(mask.sum()),cuts=int(effects[mask,0].sum()),
            fixed=int((effects[mask,1]==1).sum()),harmed=int((effects[mask,1]==-1).sum())) for
            key,mask in [('certified',cert),('outside',~cert)]}
    cert_out=score(anchor>=.5,anchor,tree,orig_tree,nine_votes,y)[cert]
    nine_cert=prepared_vote(orig_tree,anchor>=.5,nine_votes)
    exact_subset=dict(rows=int(cert.sum()),nine_edits=int(((nine_cert!=(anchor>=.5))&cert).sum()),
                      corrected=int((cert_out[:,1]==1).sum()),harmed=int((cert_out[:,1]==-1).sum()),
                      net=int(cert_out[:,1].sum()),area='v131==v96 by structural exclusion of all four stages')

    counts={}
    for title,mask in [('all',np.ones(n,bool)),('outside_cabin_patch',~cabin)]:
        counts[title]=dict(n=int(mask.sum()),tree_cuts=[int(np.minimum(a[:,0],b[:,0])[mask].sum()),int(np.maximum(a[:,0],b[:,0])[mask].sum())],
                           tree_net=[int(np.minimum(a[:,1],b[:,1])[mask].sum()),int(np.maximum(a[:,1],b[:,1])[mask].sum())])
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/'certified.npz',row=np.arange(n),passenger=raw.PassengerId.to_numpy(str),
                        fold=f,y=y,v96=anchor,certified_v131_equals_v96=cert)
    np.savez_compressed(OUT/'oof.npz',row=np.arange(n),passenger=raw.PassengerId.to_numpy(str),fold=f,y=y,certified_v131_equals_v96=cert,certified_nine=np.where(cert,nine_cert,-1),v96=anchor,peer_offset_probability=v108,
                        witness=witness,family=family,may_lower=may_lower,may_raise=may_raise,proxy=proxy,cabin_patch=cabin)
    report=dict(exact_historical=historical,exact_test=official,certified_eightfold_subset=exact_subset,eight_fold=stages,by_certificate=by_certificate,
                pending=dict(cabin_patch_rows=int(cabin.sum()),sleeper_eligible_without_committees=int(may_lower.sum()),
                             family_eligible_without_committees=int(may_raise.sum())),
                conditional_on_approximate_v108=counts,
                limits=['Official and historical v106 use saved native Mitra query results; no eight-fold patched query probabilities available.',
                        'Peer offset in eightfold uses same-parity held-out Mitra96 instead of exact original rawMitra inner predictions.',
                        'The four available family votes substitute for original five; v118 union is not available for these eight folds.',
                        'No probabilistic confidence statement: bounds vary the missing committee decisions only, conditional on approximate v108.'])
    TALLY.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    with threadpool_limits(2):
        main()
