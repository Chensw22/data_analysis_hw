"""Compare independent evidence with a class-conditional tree of observed fields and expert votes."""
import json
from collections import Counter

import numpy as np
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from full_roster_conditional_pool_n23 import BILLS, ROOT, banks, data, latest_reference
from full_roster_gate_probability_n23 import gate_probabilities
from full_roster_same_model_calibration_n23 import learn


def attributes(raw):
    cabin = raw.Cabin.str.split('/',expand=True)
    table = {}
    for k,col in [('planet',raw.HomePlanet),('destination',raw.Destination),('sleep',raw.CryoSleep),
                  ('vip',raw.VIP),('deck',cabin[0]),('side',cabin[2])]:
        table[k] = np.unique(col.astype('string').fillna('missing'),return_inverse=True)[1]
    def binned(col, cut):
        return np.where(col.isna(),0,1+np.digitize(col.fillna(0),cut))
    table['age'] = binned(raw.Age,[13,18,25,40,60])
    for k in BILLS:
        table[k] = binned(raw[k],[.5,100,500,1500])
    return np.column_stack(list(table.values())), list(table)


def evidence(train, target, query, levels):
    n,p = train.shape
    vertices = []
    prior = (target.sum()+1)/(n+2)
    unary = np.full(len(query),logit(prior))
    for j, size in enumerate(levels):
        total = np.bincount(train[:,j],minlength=size)+1
        prob = []
        for label in [False,True]:
            counts = np.bincount(train[target==label,j],minlength=size) + 8*total/total.sum()
            prob.append(counts/counts.sum())
        vertices.append(prob)
        unary += np.log(prob[1][query[:,j]]/prob[0][query[:,j]])
    # Use one maximum spanning tree for the two outcomes. Its edge weights
    # are mutual information conditional on the actual outcome in training.
    weights = np.zeros((p,p))
    pair_tables = {}
    for i in range(p):
        for j in range(i+1,p):
            dims = (levels[i],levels[j])
            flat = train[:,i]*levels[j]+train[:,j]
            tables = []
            for k in (0,1):
                select = target==k
                counts = np.bincount(flat[select],minlength=dims[0]*dims[1]).reshape(dims)
                smoothed = counts+8*np.outer(vertices[i][k],vertices[j][k])
                prob = smoothed/smoothed.sum()
                ratio = prob/(prob.sum(0)[None,:]*prob.sum(1)[:,None])
                weights[i,j] += select.mean()*(prob*np.log(ratio)).sum()
                tables.append(counts)
            weights[j,i] = weights[i,j]
            pair_tables[i,j] = tables
    used = [0]
    tree = []
    joint = np.full(len(query),logit(prior))
    joint += np.log(vertices[0][1][query[:,0]]/vertices[0][0][query[:,0]])
    while len(used)<p:
        unseen = np.setdiff1d(np.arange(p),used)
        a,b = np.unravel_index(np.argmax(weights[np.ix_(used,unseen)]),(len(used),len(unseen)))
        parent, child = used[a], unseen[b]
        used.append(child)
        tree.append((min(parent,child),max(parent,child)))
        tables = pair_tables[tree[-1]]
        for k,sign in [(0,-1),(1,1)]:
            counts = tables[k] if parent<child else tables[k].T
            conditional = (counts+8*vertices[child][k])/(counts.sum(1)[:,None]+8)
            joint += sign*np.log(conditional[query[:,parent],query[:,child]])
    assert np.isfinite(joint).all() and np.isfinite(unary).all()
    return unary,joint,tree


def run():
    raw,y,g,fold = data()
    names,half,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,half,full)
    gate,_ = gate_probabilities(names,half,full,y,g,fold)
    np.testing.assert_array_equal(gate>=.5,ref)
    anchor = logit(gate.clip(.001,.999))
    x, field_names = attributes(raw)
    cuts = [.15,.4,.5,.6,.85]
    hs,fs = np.digitize(half,cuts),np.digitize(full,cuts)
    agree = ((full>=.5)==(full[:,0:1]>=.5)).all(1)
    shared = agree & (ref!=y)
    with np.load(ROOT/'artifacts/run_002/split_indices.npz') as saved:
        additional = np.ones(len(y),bool)
        additional[saved['development']] = False
    report = {}
    for view,a,b,labels,all_training in [('fields',x,x,field_names,False),('fields_full',x,x,field_names,True),
                    ('experts',hs,fs,names,False),('fields_experts',np.column_stack([x,hs]),np.column_stack([x,fs]),field_names+names,False)]:
        levels = np.maximum(a.max(0),b.max(0))+1
        likelihoods = np.zeros((len(y),2))
        edge_counts = Counter()
        for f in np.unique(fold):
            t = fold!=f
            if not all_training:
                t &= fold%2 == f%2
            q = fold==f
            assert not np.isin(g[t],g[q]).any()
            unary,joint,tree = evidence(a[t],y[t],b[q],levels)
            likelihoods[q] = np.column_stack([unary,joint])
            edge_counts.update((labels[j],labels[k]) for j,k in tree)
        plans = {}
        for col,kind in enumerate(['independent','tree']):
            gap = likelihoods[:,col]-anchor
            p = gate.copy()
            for f in np.unique(fold):
                outer = np.flatnonzero(fold==f)
                for aa,bb in GroupKFold(4,shuffle=True,random_state=462).split(outer,groups=g[outer]):
                    t,q = outer[aa],outer[bb]
                    norm = max(gap[t].std(),1e-6)
                    w = learn(gap[t,None]/norm,y[t],anchor[t],np.zeros(1))
                    p[q] = expit(anchor[q]+gap[q]*w[0]/norm)
            for policy,pick in [('direct',likelihoods[:,col]>=0),('combined',p>=.5)]:
                d = (pick==y).astype(int)-(ref==y)
                plans[kind+'_'+policy] = dict(accuracy=float((pick==y).mean()),fixed=int((d==1).sum()),
                         harmed=int((d==-1).sum()),common_fixed=int((d[shared]==1).sum()),
                         unanimous_harmed=int((d[agree]==-1).sum()),additional_net=int(d[additional].sum()),
                         fold_net=[int(d[fold==f].sum()) for f in np.unique(fold)],ci_pp=interval(d,g))
        report[view] = dict(methods=plans,frequent_edges=[list(k)+[v] for k,v in edge_counts.most_common(15)])
        print(view,[(k,v['fixed'],v['harmed'],v['common_fixed']) for k,v in plans.items()],flush=True)
    path = ROOT/'reports/nov23_matched_learning_curve/evidence_tree.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
