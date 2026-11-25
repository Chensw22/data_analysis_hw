"""Separate recognizing a risky pair from identifying which passenger needs a correction."""
import json

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from conflict_pair_ordering_n26 import CACHE,REPORT as RANK_REPORT
from cross_order_conflict_cover_n26 import relations
from exhaustive_family_quorums_n26 import inputs
from full_roster_booking_validation_n23 import ROOT,data
from kaggle_workflow import sha256
from outside_panel_coverage_n26 import archives

REPORT = ROOT/'reports/nov23_matched_learning_curve/conflict_pair_trigger.json'


def match(a,b,order,rows,n):
    available = np.zeros(n,bool)
    available[rows] = True
    subset = order[available[a[order]]&available[b[order]]]
    pairs = []
    for e in subset:
        i,j = a[e],b[e]
        if available[i] and available[j]:
            available[[i,j]] = False
            pairs.append((i,j))
    return np.asarray(pairs,int).reshape(-1,2)


def train(x,wrong,t,train_pairs,q):
    a,b = train_pairs.T
    i,j = q.T
    def symmetric(u,v):
        return np.column_stack([(x[u]+x[v])/2, abs(x[u]-x[v])])
    count = wrong[a].astype(int)+wrong[b]
    multi = LogisticRegression(C=.05,max_iter=400).fit(symmetric(a,b),count)
    mass = np.zeros((len(q),3))
    mass[:,multi.classes_] = multi.predict_proba(symmetric(i,j))
    mixed = count==1
    delta = x[a[mixed]]-x[b[mixed]]
    positive = wrong[a[mixed]]
    rank = LogisticRegression(C=.05,max_iter=400,fit_intercept=False)
    rank.fit(np.vstack([delta,-delta]),np.r_[positive,~positive])
    choose_left = rank.predict_proba(x[i]-x[j])[:,1]
    model = LogisticRegression(C=.05,max_iter=400).fit(x[t],wrong[t])
    risk = model.predict_proba(x[np.r_[i,j]])[:,1]
    return mass,choose_left,np.reshape(risk,(2,-1)).T


def measure(mass,rank,base,two_errors,query,keep):
    mass,rank,base,two_errors,query = [z[keep] for z in (mass,rank,base,two_errors,query)]
    count = two_errors.sum(1)
    which = np.where(rank>=.5,0,1)
    ids = np.arange(len(rank))
    right = two_errors[ids,which]
    any_error = count>0
    score = mass[:,1:].sum(1)
    expected = mass[:,2]+mass[:,1]*np.maximum(rank,1-rank)
    def stats(select,correct):
        return dict(selected=int(select.sum()),fixed=int(correct[select].sum()),
                    harmed=int((~correct[select]).sum()),net=int(np.where(correct[select],1,-1).sum()))
    def single_risk(risk):
        who = np.argmax(risk,axis=1)
        return stats(risk[ids,who]>.5,two_errors[ids,who])
    order = np.argsort(-score,kind='stable')
    return dict(pairs=len(keep.nonzero()[0]),passengers=query.size,errors=int(count.sum()),
                zero_one_two_errors=np.bincount(count,minlength=3).tolist(),
                pair_selection_auc=float(roc_auc_score(any_error,score)),ranking_correct_if_one=int(right[count==1].sum()),
                ranking_accuracy_if_one=float(right[count==1].mean()),top_fifth_any_error_rate=float(any_error[order[:max(1,len(order)//5)]].mean()),
                individual=single_risk(base),both_learned=stats(expected>.5,right),
                learned_selection_perfect_ordering=stats(score>.5,any_error),
                perfect_selection_learned_ordering=stats(any_error,right),
                both_perfect=stats(any_error,any_error))


def run():
    raw,y,g,f = data()
    _,cur,_ = inputs()
    archive,_ = archives(raw,y,g,f,True)
    names = ['Mitra2_adapted','LimiX2','TabDPT','mask_net_joint','latent_class_mixture','conditional_bill_flow']
    probability = np.column_stack([archive[key] for key in names])
    path = CACHE/'held.npz'
    assert sha256(path)==json.loads(RANK_REPORT.read_text())['cached_predictions_sha256']
    with np.load(path) as z:
        for key,expected in [('passenger',raw.PassengerId.to_numpy(str)),('y',y),('fold',f)]:
            np.testing.assert_array_equal(z[key],expected)
        ranks = z['scores']
    side = probability[:,0]>=.5
    wrong = y!=side
    direction = np.where(side,-1,1)
    x = np.column_stack([logit(probability.clip(.004,.996))*direction[:,None],ranks*direction[:,None],side])
    a,b,d = relations(raw,g,False)
    same = (side[a]==side[b])&(f[a]==f[b])
    a,b,d = a[same],b[same],d[same]
    gap = np.abs(x[a,0]-x[b,0])
    report = {}
    for name,order in [('nearest',np.lexsort((b,a,d))),('confidence_gap',np.lexsort((b,a,d,-gap)))]:
        chunks = []
        for k in np.unique(f):
            rows = np.flatnonzero(f==k)
            for tix,qix in GroupKFold(3,shuffle=True,random_state=153).split(rows,groups=g[rows]):
                t,q = rows[tix],rows[qix]
                assert not np.isin(g[t],g[q]).any()
                query = match(a,b,order,q,len(y))
                # Equalize query and training pool size to avoid tighter training matches.
                train_pairs = []
                for _,part in GroupKFold(2,shuffle=True,random_state=972).split(t,groups=g[t]):
                    train_pairs.append(match(a,b,order,t[part],len(y)))
                examples = np.concatenate(train_pairs)
                assert len(np.unique(examples))==examples.size and len(np.unique(query))==query.size
                mass,rank,base = train(x,wrong,t,examples,query)
                chunks.append((query,mass,rank,base))
        query,mass,rank,base = [np.concatenate([c[j] for c in chunks]) for j in range(4)]
        assert len(np.unique(query))==query.size and (side[query]==side[query][:,:1]).all()
        applies = (side==cur['parent'])&cur['known']
        report[name] = {}
        for scope,eligible in [('agree_with_anchor',applies),('all_eleven_agree',applies&cur['common'])]:
            keep = eligible[query].all(1)
            report[name][scope] = measure(mass,rank,base,wrong[query],query,keep)
        print(name,report[name]['all_eleven_agree'],flush=True)
    report['scope'] = ('Both members must share the Mitra prediction. A pair is formed without either outcome. '
       'All secondary training orders lie outside the query, within the same outer model fold. Counts are '
       'errors relative to the shared prediction. Each policy can change at most one passenger in each pair. '
       'Perfect selection and perfect ordering use held outcomes only for diagnostic ceilings.')
    REPORT.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
