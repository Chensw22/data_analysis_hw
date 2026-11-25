"""Test if continuous outer-only opinions separate transported awake spenders under unanimous negative votes."""
import json

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from attentive_tabnet_n26 import MODELS
from family_identity_transfer_n26 import panels_and_reference
from held_expert_portfolios_n26 import NEW, panel_predictions
from solo_archive_recovery_n26 import archive
from distinct_challenger_error_reach_n26 import population
from expanded_joint_error_witnesses_n26 import FAMILIES,load_scores
from outside_panel_coverage_n26 import archives
from tabnet_attention_reliability_n26 import signals
from target_mix_expert_routing_n26 import ROOT

ART=ROOT/'artifacts'
REPORT=ROOT/'reports/nov23_matched_learning_curve/awake_paid_soft_unanimity.json'


def panels(raw,y,g,f):
    old,_=archives(raw,y,g,f,True)
    # EXAONE was built on a different split; another passenger's OOF feature can include
    # query labels in its training. It can stratify the final audit but cannot fit a router here.
    del old['EXAONE_grouped']
    full={'ids':np.arange(len(y)), 'fold':f}
    extra=load_scores(raw,y,g,{'full':full})['full']
    net=signals(raw,y,g,f)[0]
    ends=np.empty((len(y),2))
    for k in np.unique(f):
        t,q=np.flatnonzero(f!=k),np.flatnonzero(f==k)
        assert not np.isin(g[t],g[q]).any()
        for i,(name,col) in enumerate([('original_tabstar/tuned','finetuned'),('ordinal_factor_density/fold','conditional_bills')]):
            with np.load(ART/f'nov26_{name}{k}.npz') as z:
                np.testing.assert_array_equal(z['q'],q)
                np.testing.assert_array_equal(z['train_y'],y[t])
                if 't' in z:
                    np.testing.assert_array_equal(z['t'],t)
                ends[q,i]=z[col]
    return np.column_stack([*old.values(),extra,net,ends]), [*old,*FAMILIES,*MODELS,'tuned_tabstar','conditional_bills']


def original_scores(y,g,f):
    # Use raw outcomes before the queue signed-log-odds transformation; none of these
    # features are computed from a reviewer's error or any other within-query outcome.
    ids=np.arange(len(y))
    _,_,ff,_,variant_names,original,*_=panels_and_reference()
    np.testing.assert_array_equal(ff,f)
    with np.load(ART/'nov26_tabicl_consensus_review/full.npz') as z:
        names=z['names'].tolist()
        saved_votes=z['votes']
    families=names[:18]
    tags=np.array([n.split('/')[0] for n in variant_names])
    scores=[original[:,tags==n].mean(1) for n in families]
    archive_names,old=archive('full',ids,f,g,y,ART/'nov24_remaining_cases/archive_scores.npz',probabilities=True)
    assert archive_names==names[18:]
    legacy=np.column_stack([*scores,old])
    np.testing.assert_array_equal(legacy>=.5,saved_votes)
    full=dict(ids=ids,fold=f)
    last=panel_predictions(full,'full')
    # The stream fits take t from the corresponding deck file, and save q only.
    # Cabin geometry and zero-truncated fits save one complete cross-fit array.
    # Both generate each held fold using labels outside it (see the two source scripts).
    for k in np.unique(f):
        t,q=ids[f!=k],ids[f==k]
        assert not np.isin(g[t],g[q]).any()
        for name in ['nov26_price_implied_deck/tree','nov26_price_implied_deck/neural',
                     'nov26_fare_routes/tree','nov26_fare_routes/neural','nov26_disjoint_charge_swaps/tree',
                     'nov26_global_cabin_assignment/tabicl','nov26_latent_bill_rounding/outcome',
                     'nov26_latent_bill_rounding/charge_classifier','nov26_latent_bill_rounding/charge_tabicl']:
            with np.load(ART/f'{name}_full_{k}.npz') as z:
                np.testing.assert_array_equal(z['t'],t)
                np.testing.assert_array_equal(z['q'],q)
                for key in ('y','train_y'):
                    if key in z:
                        np.testing.assert_array_equal(z[key],y[t])
    return np.column_stack([legacy,*[last[n] for n in NEW]]),names+list(NEW)


def fit(x, y, query):
    scale=StandardScaler().fit(x)
    m=LogisticRegression(C=.1,max_iter=450).fit(scale.transform(x),y)
    return m.predict_proba(scale.transform(query))[:,1]


def scorecard(p,scope,y,g,f,top):
    ix=np.flatnonzero(scope)
    s=p[ix]
    truth=y[ix]
    action=s>=.5
    fixed=action&truth
    harmed=action&~truth
    return dict(rows=len(ix),errors=int(truth.sum()),base_rate=float(truth.mean()),auc=float(roc_auc_score(truth,s)),
                average_precision=float(average_precision_score(truth,s)),maximum=float(s.max()),
                at_half=dict(corrected=int(fixed.sum()),harmed=int(harmed.sum()),
                             by_fold=[int(((fixed.astype(int)-harmed)*(f[ix]==k)).sum()) for k in range(8)]),
                per_fold_top_tenth=dict(rows=int(top[ix].sum()),corrected=int((top[ix]&truth).sum()),
                                       harmed=int((top[ix]&~truth).sum())),
                individual_fold_auc=[float(roc_auc_score(truth[f[ix]==k],s[f[ix]==k])) for k in range(8)])


def run():
    raw,y,g,f,parent,same,count,_,_,_,_,paid=population()
    active=raw.CryoSleep.eq(False).to_numpy() & paid
    unanimous=same & ~parent & active & (count==0)
    assert unanimous.sum()==1435 and y[unanimous].sum()==127
    scores,names=panels(raw,y,g,f)
    silent=(scores<.5).all(1)
    assert silent[unanimous].all() and np.isfinite(scores).all()
    logits=logit(scores.clip(.002,.998))
    mean=logits.mean(1,keepdims=True)
    spread=logits-mean
    original,original_names=original_scores(y,g,f)
    assert original.shape==(len(y),99) and (original[unanimous]<.5).all()
    expanded=np.column_stack([original,scores])
    assert expanded.shape==(len(y),126)
    expanded_logits=logit(expanded.clip(.002,.998))
    scalar_scores=np.mean(scores,axis=1)
    values={'equal_soft_vote':scalar_scores, 'original_99_equal_soft_vote':original.mean(1),
            'all_126_equal_soft_vote':expanded.mean(1)}
    values['all_126_individual_profiles']=np.full(len(y),np.nan)
    for features in ('mean_logit','individual_profiles','sorted_profiles'):
        values[features]=np.full(len(y),np.nan)
    # The meta-learner sees only raw awake/paid membership and unaltered outcome labels.
    # In each outer fold all opinions came from models fitted on the *other* seven folds.
    # A second split within that outer fold separates calibration and query order labels.
    for k in np.unique(f):
        rows=np.flatnonzero(active&(f==k))
        for tr,te in GroupKFold(3,shuffle=True,random_state=17300).split(rows,groups=g[rows]):
            t,q=rows[tr],rows[te]
            assert not np.isin(g[t],g[q]).any()
            for name,x in [('mean_logit',mean),('individual_profiles',np.column_stack([mean,spread])),
                           ('sorted_profiles',np.column_stack([mean,np.sort(spread,axis=1)])),
                           ('all_126_individual_profiles',expanded_logits)]:
                values[name][q]=fit(x[t],y[t],x[q])
    scopes={'awake_paid_127_unanimous_negative':unanimous,'awake_paid_ten_unanimous_negative':active & same & ~parent}
    report=dict(input_names=names,added_original_names=original_names,counts=dict(selected_under_all_127=len(np.flatnonzero(unanimous)),
                 majority_correct_in_selected=int((~y&unanimous).sum())),
                label_isolation='All supervised inputs use the same outer booking fold. The three-way second level trains on other orders in that same outer fold. The 127-model subgroup and ten-way parent enter only final stratification. EXAONE and all queue meta-predictions are excluded from calibration features, routing labels and sample selection.',
                scopes={})
    with np.load(ART/'nov26_shared_purchase_exposure/scores.npz') as z:
        for key,expected in [('ids',raw.PassengerId),('fold',f),('y',y)]:
            np.testing.assert_array_equal(z[key],expected)
        exposure=z['shared_amount']>=.5
    for name,mask in scopes.items():
        report['scopes'][name]={}
        for method,p in values.items():
            top=np.zeros(len(y),bool)
            for k in np.unique(f):
                rows=np.flatnonzero(mask & (f==k))
                ranked=rows[np.lexsort((rows,-p[rows]))]
                top[ranked[:int(np.ceil(.1*len(rows)))]]=True
            report['scopes'][name][method]=scorecard(p,mask,y,g,f,top)
            if mask is unanimous and method=='equal_soft_vote':
                report['paired_charge_shared_amount_vs_original_27_top_tenth']={}
                for label,rule in [('soft_only',top&~exposure),('exposure_only',exposure&~top),('both',top&exposure)]:
                    report['paired_charge_shared_amount_vs_original_27_top_tenth'][label]=dict(
                        corrected=int((mask&rule&y).sum()),harmed=int((mask&rule&~y).sum()))
    known_count=8532
    report['perfect_picker_ceiling_gain_pp']=100*int(y[unanimous].sum())/known_count
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(report['scopes']['awake_paid_127_unanimous_negative'])


if __name__=='__main__':
    with threadpool_limits(1):
        run()
