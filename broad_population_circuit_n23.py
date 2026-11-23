"""Fit normalized hierarchical mixtures of products over raw passenger fields."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import logsumexp, expit, logit, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import panel
from broad_class_transport_n23 import encode
from broad_error_scope_balance_n23 import archived_votes
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from broad_release_miss_coverage_n23 import summarize

REPORT = ROOT / "reports/nov23_population_circuit"


def field_block(x, cols, widths):
    dims = widths[cols]
    starts = np.r_[0, dims.cumsum()[:-1]]
    indices = (x[:, cols] + starts).ravel()
    ptr = np.arange(len(x)+1) * len(cols)
    values = csr_matrix((np.ones(len(indices)), indices, ptr), shape=(len(x), dims.sum()))
    return values, np.repeat(np.arange(len(cols)), dims), starts


def normalize_fields(count, starts, field):
    return count / np.add.reduceat(count, starts, axis=0)[field]


def fit_circuit(blocks, train, query, roots, local, seed, return_components=False, return_parameters=False):
    # Each root state mixes a separate collection of local states per block.
    # Smoothing uses the class-specific training marginal of that field.
    if return_parameters and (local != 1 or return_components):
        raise ValueError('Root and leaf parameters are available for flat mixtures only.')
    rng = np.random.default_rng(seed)
    layouts, theta, trans = [], [], []
    for data, field, starts in blocks:
        a, b = data[train], data[query]
        prior = normalize_fields(np.asarray(a.sum(0)).T + .5, starts, field)
        params = prior * rng.lognormal(0, .6, (prior.shape[0], roots*local))
        theta.append(np.log(normalize_fields(params, starts, field)))
        layouts.append((a, b, field, starts, prior))
        trans.append(np.full((roots, local), -np.log(local)))
    root_prior = np.full(roots, -np.log(roots))
    for step in range(31):
        emissions, factors = [], []
        root_log = np.tile(root_prior, (len(query) if step == 30 else len(train), 1))
        for (a,b,*_), logtheta, weight in zip(layouts, theta, trans):
            x = b if step == 30 else a
            z = (x @ logtheta).reshape(len(root_log), roots, local) + weight
            norm = logsumexp(z, axis=2)
            root_log += norm
            factors.append(norm)
            emissions.append(z)
        ll = logsumexp(root_log, axis=1)
        if step == 30:
            if return_parameters:
                return root_prior, theta
            return root_log if return_components else ll
        resp = np.exp(root_log - ll[:, None])
        root_count = resp.sum(0) + 1.
        root_prior = np.log(root_count / root_count.sum())
        for idx, ((a,_,field,starts,prior),z,v) in enumerate(zip(layouts, emissions, factors)):
            sub = resp[:,:,None] * np.exp(z-v[:,:,None])
            counts = a.T @ sub.reshape(len(train), -1) + 6*prior
            theta[idx] = np.log(normalize_fields(counts, starts, field))
            state_counts = sub.sum(0) + 1.
            trans[idx] = np.log(state_counts / state_counts.sum(1,keepdims=True))


def main():
    row, y, frame, g, _ = load()
    x = encode(frame)
    widths = x.max(0)+1
    fields = [["HomePlanet","Destination","VIP"],["Deck","Side","CabinNumber","BookingSize"],
              ["Age","CryoSleep"],["FoodCourt","ShoppingMall","Spa","VRDeck","RoomService"]]
    semantic = [frame.columns.get_indexer(b) for b in fields]
    order = np.concatenate(semantic)
    assert sorted(order.tolist()) == list(range(x.shape[1]))
    perm = np.random.default_rng(828).permutation(order)
    shuffled = np.split(perm,np.cumsum([len(b) for b in fields])[:-1])
    specs = {"one_subtype": ([order], 4, 1), "flat_twelve": ([order], 12, 1),
             "hierarchy": (semantic, 4, 3), "random_blocks": (shuffled, 4, 3)}
    models = {k:([field_block(x,c,widths) for c in layout],r,l) for k,(layout,r,l) in specs.items()}
    ids, labels, base_panel, archive, *_ = panel()
    np.testing.assert_array_equal(ids,row)
    np.testing.assert_array_equal(labels,y)
    votes = archived_votes(row,y,base_panel,archive)
    stable_prior = (votes==y[None,:,None]).all(0).any(1)
    baseline = (base_panel>=.5)==y
    scopes = dict(ever=~baseline.all(0),repeat=~baseline.any(0))
    history = {}
    scope_audit = []
    out = ART / "nov23_population_circuit"
    out.mkdir(exist_ok=True)
    for i,seed in enumerate(SEEDS):
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            np.testing.assert_array_equal(z["row"],row)
            np.testing.assert_array_equal(z["y"],y)
            fold,base=z["fold"],z["base"]
            np.testing.assert_array_equal(base,base_panel[i])
        preds = {}
        component = np.empty((len(y),2,4))
        for key,(blocks,r,l) in models.items():
            score = np.empty(len(y))
            for f in np.unique(fold):
                t,q = np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
                assert not np.intersect1d(g[t],g[q]).size
                ll=[]
                for cls in (False, True):
                    tr = t[y[t]==cls]
                    if key == "one_subtype":
                        lp = fit_circuit(blocks,tr,q,r,l,262,return_components=True)
                        component[q,int(cls)] = softmax(lp,axis=1)
                        likelihood = logsumexp(lp,axis=1)
                    else:
                        likelihood = fit_circuit(blocks,tr,q,r,l,262)
                    ll.append(likelihood+np.log(len(tr)/len(t)))
                score[q] = ll[1]-ll[0]
            preds[key]=expit(score)
            combined=np.empty(len(y))
            design=np.c_[logit(base.clip(.001,.999)),score]
            for f in np.unique(fold):
                ix=np.flatnonzero(fold==f)
                for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ix,groups=g[ix]):
                    t,q=ix[a],ix[b]
                    assert not np.intersect1d(g[t],g[q]).size
                    combined[q]=LogisticRegression(C=.5).fit(design[t],y[t]).predict_proba(design[q])[:,1]
            preds["combined_"+key]=combined
        # Component identities are shared only within each first-stage fold.
        # Learn on other bookings whether subtype membership predicts which
        # opinion wins when the weaker model disagrees with the incumbent.
        raw = preds["one_subtype"]
        opposite = (base>=.5) != (raw>=.5)
        for control in (False,True):
            probability = base.copy()
            for f in np.unique(fold):
                ix = np.flatnonzero((fold==f)&opposite)
                probvec = component[ix].reshape(len(ix),-1)
                if control:
                    rng = np.random.default_rng(643)
                    cell = np.digitize(base[ix],np.quantile(base[ix],np.arange(1,9)/9))
                    for c in np.unique(cell):
                        mask = cell==c
                        probvec[mask]=rng.permutation(probvec[mask])
                strength = logit(base[ix].clip(.001,.999))
                weak = logit(raw[ix].clip(.001,.999))
                direction = np.sign(strength)
                design = np.c_[strength,weak,probvec,probvec*direction[:,None],probvec*np.tanh(weak[:,None])]
                for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ix,groups=g[ix]):
                    assert not np.intersect1d(g[ix[a]],g[ix[b]]).size
                    probability[ix[b]] = LogisticRegression(C=.5).fit(design[a],y[ix[a]]).predict_proba(design[b])[:,1]
            preds["subtype_shuffled" if control else "subtype_arbiter"]=probability
        scope_audit.append(opposite)
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base,component=component,**preds)
        for k,p in preds.items():
            assert np.isfinite(p).all()
            history.setdefault(k,[]).append((p>=.5)==y)
        print(seed,[(k,round(h[-1].mean(),4)) for k,h in history.items()],flush=True)
    result={k:summarize(np.stack(h),baseline,scopes) for k,h in history.items()}
    coverage={k:int((np.logical_and.reduce(h)&scopes["repeat"]&~stable_prior).sum()) for k,h in history.items()}
    target = np.logical_and.reduce(history["one_subtype"]) & scopes["repeat"] & ~stable_prior
    old_hit = votes==y[None,:,None]
    opposition = np.logical_and.reduce(scope_audit)
    audit = dict(novel=int(target.sum()), never_old=int((target&~old_hit.any((0,2))).sum()),
                 novel_transported=int((target&y).sum()), always_disagree=int(opposition.sum()),
                 incumbent_always_right=int((opposition&baseline.all(0)).sum()),
                 intermittent=int((opposition&~baseline.all(0)&~scopes["repeat"]).sum()))
    write(REPORT/"results.json",dict(results=result,new_beyond_67_on_repeat=coverage,audit=audit,
                                     semantic_fields=fields,random_fields=[frame.columns[c].tolist() for c in shuffled]))


if __name__=="__main__":
    with threadpool_limits(2):
        main()
