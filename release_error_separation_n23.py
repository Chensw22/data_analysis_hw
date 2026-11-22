"""Match opposite outcomes on bill profiles and baseline risk, then audit relative scores."""
import numpy as np
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ROOT, panel, predictions
from broad_bill_exception_n23 import BILLS, load, strata
from broad_partition_evidence_n23 import EXTRA, SOURCES, SEEDS
from broad_release_transfer_n23 import write
from released_correction_audit_n23 import development

REPORT = ROOT / "reports/nov23_error_separation"


def match(left, right, code, amount, base, group):
    """Maximum-cardinality, then minimum-distance, one-to-one constrained assignment."""
    lefts, rights = [], []
    for pattern in np.intersect1d(code[left], code[right]):
        a, b = left[code[left] == pattern], right[code[right] == pattern]
        risk = abs(base[a,None] - base[b])
        bill = abs(amount[a,None,:] - amount[b]).max(2)
        allowed = (risk <= .05) & (bill <= 1.) & (group[a,None] != group[b])
        cost = risk / .05 + bill
        dummy = 3. * (len(a) + 1)
        cost = np.where(allowed, cost, dummy * 2)
        cost = np.column_stack([cost, np.full((len(a),len(a)),dummy)])
        i,j = linear_sum_assignment(cost)
        valid = j < len(b)
        lefts.extend(a[i[valid]])
        rights.extend(b[j[valid]])
    a,b = np.asarray(lefts, int), np.asarray(rights, int)
    assert len(np.unique(a)) == len(a) == len(np.unique(b))
    return a,b


def fit(x, offset, y, t, positive, negative, paired):
    delta = x[positive] - x[negative]
    gap = offset[positive] - offset[negative]
    def objective(w):
        s = offset[t] + x[t] @ w
        loss = np.logaddexp(0,s).mean() - y[t] @ s / len(t)
        grad = x[t].T @ (expit(s)-y[t]) / len(t)
        if paired:
            d = gap + delta @ w
            loss += np.logaddexp(0,-d).mean()
            grad -= delta.T @ expit(-d) / len(d)
        return loss + .004 * (w@w), grad + .008*w
    sol = minimize(objective,np.zeros(x.shape[1]),jac=True,method="L-BFGS-B",
                   options={"maxiter":400,"ftol":1e-12})
    assert sol.success
    return sol.x


def run():
    row,y,base,p,_,names,_ = panel()
    ids,target,frame,g,_ = load()
    np.testing.assert_array_equal(row,ids)
    np.testing.assert_array_equal(y,target)
    targets,releases,_ = development()
    np.testing.assert_array_equal(targets,y)
    unanimous = np.logical_and.reduce([pred != y for pred in releases.values()])
    state = strata(frame)
    cohorts = {"paid":unanimous & y & (state==2), "zero":unanimous & ~y & (state==0),
               "all_missed":unanimous, "all":np.ones(len(y),bool)}
    assert [v.sum() for v in cohorts.values()] == [415,281,941,6954]
    extra = {**EXTRA,**dict(list(SOURCES.items())[:4])}
    panels, folds = [], []
    for seed in SEEDS:
        _,values,fold = predictions(row,y,seed,extra)
        panels.append(values)
        folds.append(fold)
    models = np.concatenate([p,np.stack(panels)],axis=2)
    members = ["archive/"+k for k in names]+["additional/"+k for k in extra]
    bill = frame[BILLS].to_numpy()
    code = np.where(np.isnan(bill),2,(bill>0).astype(int)) @ (3**np.arange(bill.shape[1]))
    amount = np.log1p(np.nan_to_num(bill))
    # Sensitivity diagnostic on each whole outer fold: preserve bill state and whether
    # all bills are observed, then bound total spending and the five service shares.
    total = np.nan_to_num(bill).sum(1)
    shares = np.sqrt(np.nan_to_num(bill) / np.maximum(total[:, None],1))
    coarse_amount = np.c_[np.log1p(total)/np.log(2), shares/.4]
    coarse_code = state*2 + np.isnan(bill).any(1)
    wide, records, histories = {}, {}, {}
    joint_hits, matched_all = [], []
    for s,fold in enumerate(folds):
        b = base[s]
        offset = logit(b.clip(.002,.998))
        feat = logit(models[s].clip(.002,.998)) - offset[:,None]
        feat /= np.sqrt(feat.shape[1])
        x = np.c_[np.ones(len(y)),state==0,state==2,offset,feat]
        scores = {k:np.full(len(y),np.nan) for k in ("point","pair_and_point")}
        # Every evaluated pair goes through the same fitted stack. Neither booking
        # supplies outcomes for the base models, the ranking fit or calibration.
        pair_cases, pair_controls, training_counts = [],[],[]
        for f in np.unique(fold):
            outer = np.flatnonzero(fold==f)
            for ia,ib in GroupKFold(3,shuffle=True,random_state=925).split(outer,groups=g[outer]):
                t,q = outer[ia],outer[ib]
                assert not np.intersect1d(g[t],g[q]).size
                pos,neg = match(t[y[t]],t[~y[t]],code,amount,b,g)
                training_counts.append(len(pos))
                for key in scores:
                    w = fit(x,offset,y,t,pos,neg,key=="pair_and_point")
                    scores[key][q] = expit(offset[q]+x[q]@w)
                for label in (False,True):
                    cases = q[unanimous[q] & (y[q]==label)]
                    controls = q[~unanimous[q] & (y[q]!=label)]
                    a,c = match(cases,controls,code,amount,b,g)
                    pair_cases.extend(a)
                    pair_controls.extend(c)
        a,c = np.asarray(pair_cases),np.asarray(pair_controls)
        assert len(np.unique(c)) == len(a) == len(np.unique(a))
        banks = {"base":b,**scores,**dict(zip(members,models[s].T))}
        rr = {}
        for key,z in banks.items():
            assert np.isfinite(z).all()
            hit = (z>=.5)==y
            rec = {"accuracy":float(hit.mean()), "fixed":int((hit&((b>=.5)!=y)).sum()),
                   "harmed":int((~hit&((b>=.5)==y)).sum()), "pairs":{}}
            for k,mask in cohorts.items():
                if k=="all":continue
                aa,cc = a[mask[a]],c[mask[a]]
                directed = (z[aa]-z[cc])*(2*y[aa].astype(float)-1)
                rec["pairs"][k] = {"n":len(aa),"wins":int((directed>0).sum()),"ties":int((directed==0).sum()),
                        "both_correct":int((hit[aa]&hit[cc]).sum()),"case_correct":int(hit[aa].sum()),
                        "control_wrong":int((~hit[cc]).sum()),
                        "median_abs_base_gap":float(np.median(abs(b[aa]-b[cc]))),
                        "median_max_logbill_gap":float(np.median(abs(amount[aa]-amount[cc]).max(1)))}
            rr[key] = rec
            histories.setdefault(key,[]).append(hit)
        wide_pairs = []
        for f in np.unique(fold):
            pool = np.flatnonzero(fold==f)
            for label in (True,False):
                aa,cc = match(pool[unanimous[pool] & (y[pool]==label)],
                              pool[~unanimous[pool] & (y[pool]!=label)],
                              coarse_code,coarse_amount,b,g)
                wide_pairs.extend(zip(aa,cc))
        aa,cc = np.asarray(wide_pairs).T
        assert len(np.unique(aa)) == len(np.unique(cc)) == len(aa)
        paired_hits = np.zeros((len(y),len(members)),bool)
        case_hits = (models[s][aa]>=.5)==y[aa,None]
        control_hits = (models[s][cc]>=.5)==y[cc,None]
        paired_hits[aa] = case_hits & control_hits
        joint_hits.append(paired_hits)
        matched_all.append(np.isin(np.arange(len(y)),aa))
        wide[str(SEEDS[s])] = {}
        for name,mask in cohorts.items():
            if name=="all": continue
            ac,bc = aa[mask[aa]],cc[mask[aa]]
            scores_raw = np.column_stack([b,models[s]])
            dif = (scores_raw[ac] - scores_raw[bc])*(2*y[ac,None].astype(float)-1)
            good = (scores_raw[ac]>=.5)==y[ac,None]
            intact = (scores_raw[bc]>=.5)==y[bc,None]
            wide[str(SEEDS[s])][name] = dict(n=len(ac), median_base_gap=float(np.median(abs(b[ac]-b[bc]))),
                median_abs_log_total_gap=float(np.median(abs(np.log1p(total[ac])-np.log1p(total[bc])))),
                win_fraction=dict(zip(["base",*members],np.mean((dif>0)+.5*(dif==0),axis=0).tolist())),
                both_correct=dict(zip(["base",*members],(good&intact).sum(0).tolist())))
        records[str(SEEDS[s])] = dict(methods=rr,train_pairs=training_counts)
    stable = {k:{c:int((np.logical_and.reduce(v)&mask).sum()) for c,mask in cohorts.items()}
              for k,v in histories.items()}
    present = np.logical_and.reduce(matched_all)
    stable_case = ((models>=.5)==y[None,:,None]).all(0).any(1)
    stable_both = np.logical_and.reduce(joint_hits).any(1)
    linked = {k:dict(same_cases_all_three=int((present&mask).sum()),
                     stable_case_given_match=int((present&mask&stable_case).sum()),
                     stable_both_given_match=int((present&mask&stable_both).sum()))
              for k,mask in cohorts.items() if k!="all"}
    write(REPORT/"results.json",dict(plans=records,stable=stable,whole_fold_matching=wide,linked=linked))
    print({k:stable[k] for k in ("base","point","pair_and_point")})
    for s,r in records.items():
        for k in ("base","point","pair_and_point"):
            print(s,k,r["methods"][k])


if __name__=="__main__":
    with threadpool_limits(2):
        run()
