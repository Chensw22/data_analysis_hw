"""Attribute the change from an amount-ablated passenger model by refitting all 32 coalitions."""
from math import comb

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from threadpoolctl import threadpool_limits

from exaone_validation_data import CATEGORICAL, compact_features
from grouped_five_way_coverage_n23 import verify
from population_rescue_map_n23 import ART, ROOT, SEEDS, read_panel
from relational_error_corrector import write
from spend_simplex_likelihood_n23 import BILLS, design
from transported_subclasses_n23 import stats


def attribute(p):
    scores = logit(p.clip(.00001,.99999))
    deltas = np.zeros((len(p),len(BILLS)))
    for field in range(len(BILLS)):
        bit = 1<<field
        for subset in range(1<<len(BILLS)):
            if not subset & bit:
                weight = 1 / (len(BILLS)*comb(len(BILLS)-1,subset.bit_count()))
                deltas[:,field] += weight*(scores[:,subset|bit]-scores[:,subset])
    np.testing.assert_allclose(deltas.sum(1),scores[:,-1]-scores[:,0],atol=1e-8)
    return deltas


def matched_summary(delta, population, y, base, spend, solo):
    target = np.flatnonzero(population)
    negative = (base < .5) & ~y
    cells = pd.DataFrame(dict(risk=(base*20).astype(int), size=solo, services=(spend>0).sum(1),
                              budget=np.digitize(np.nansum(spend,axis=1),[1,500,1500,3000])))
    actual, reference = [],[]
    for rows in cells.groupby(list(cells.columns)).indices.values():
        pos=rows[population[rows]]
        control=rows[negative[rows]]
        if len(pos) and len(control):
            actual.extend(delta[pos])
            reference.extend(np.repeat(delta[control].mean(0)[None,:],len(pos),axis=0))
    assert len(actual) > 0
    return dict(targets=len(target),matched=len(actual),mean_log_odds_contribution=np.mean(actual,axis=0).tolist(),
                matched_true_negative_contribution=np.mean(reference,axis=0).tolist())


def run(from_cache=False):
    row = pd.read_csv(ART/"nov23_global_error_inventory.csv").row.to_numpy()
    raw = pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in {"Name","Transported"})
    booking = raw.PassengerId.str.split("_").str[0]
    solo = booking.map(booking.value_counts()).to_numpy()[row] == 1
    group = booking.to_numpy()[row]
    x = compact_features(raw.iloc[row]).reset_index(drop=True)
    background = design(x)[0]
    numeric = background.drop(columns=CATEGORICAL).to_numpy(float)
    raw_bills = x[BILLS].to_numpy(float)
    y, older, others = read_panel()
    persistent = ~older.any(0)
    hard = persistent & ~others.any(axis=(0,1))
    folder = ART/"nov23_bill_coalitions"
    folder.mkdir(exist_ok=True)
    result = {}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            fold,base=z["fold"],z["base"]
            verify(z,row,y,fold)
        scores = np.full((len(row),32),np.nan)
        if from_cache:
            with np.load(folder/f"{seed}.npz") as cache:
                verify(cache,row,y,fold)
                np.testing.assert_array_equal(cache["base"],base)
                scores=cache["coalitions"]
                assert scores.shape == (len(y),32)
        for k in ([] if from_cache else np.unique(fold)):
            t,q = np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            assert not np.intersect1d(group[t],group[q]).size
            enc = OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1).fit(x.iloc[t][CATEGORICAL])
            covariates=np.column_stack([enc.transform(x[CATEGORICAL]),numeric])
            for subset in range(32):
                use=[j for j in range(5) if subset & (1<<j)]
                matrix=np.column_stack([covariates,raw_bills[:,use]])
                cat=np.arange(matrix.shape[1])<len(CATEGORICAL)
                model=HistGradientBoostingClassifier(max_iter=140, max_leaf_nodes=15,learning_rate=.065,
                    min_samples_leaf=24,l2_regularization=3,categorical_features=cat,early_stopping=False,
                    random_state=437)
                model.fit(matrix[t],y[t])
                scores[q,subset]=model.predict_proba(matrix[q])[:,1]
            print("finished",seed,"fold",k,flush=True)
        assert np.isfinite(scores).all()
        delta=attribute(scores)
        if not from_cache:
            np.savez_compressed(folder/f"{seed}.npz",row=row,y=y,fold=fold,base=base,coalitions=scores,
                                log_odds_contribution=delta)
        report={}
        for name,score in [("without_individual_amounts",scores[:,0]),("all_amounts",scores[:,-1]),
                           ("average_all_subsets",scores.mean(1))]:
            for variant,p in [(name,score),(name+"_blend",.75*base+.25*score)]:
                report[variant]=stats(p>=.5,base>=.5,y,persistent,hard,solo)
        report["contribution_on_false_negatives"]={name:matched_summary(delta,mask,y,base,raw_bills,solo)
            for name,mask in [("all_in_split",y & (base<.5)),("persistent",y & persistent),
                              ("eight_families",y & hard)]}
        no_amount = (scores[:,0]>=.5)==y
        all_amount = (scores[:,-1]>=.5)==y
        report["amount_information"]={}
        for n,group_mask in [("all",np.ones(len(y),bool)),("up_to_one_positive",(raw_bills>0).sum(1)<=1),
                             ("multiple_positive",(raw_bills>0).sum(1)>=2)]:
            counts={}
            for label,scope in [("all",np.ones(len(y),bool)),("persistent_fn",persistent&y),
                                ("persistent_fp",persistent&~y),("eight_family_fn",hard&y)]:
                mask=scope&group_mask
                counts[label]=dict(rows=int(mask.sum()),fixed_by_adding_amounts=int((mask&~no_amount&all_amount).sum()),
                                   harmed_by_adding_amounts=int((mask&no_amount&~all_amount).sum()),
                                   predicted_probability_decreases=int((mask&(scores[:,-1]<scores[:,0])).sum()))
            report["amount_information"][n]=counts
        result[seed]=report
        write(ROOT/"reports/nov23_bill_coalitions/result.json",dict(bills=BILLS,results=result))
        print(seed,report["average_all_subsets_blend"],flush=True)


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-cache",action="store_true")
    with threadpool_limits(2):
        run(parser.parse_args().from_cache)
