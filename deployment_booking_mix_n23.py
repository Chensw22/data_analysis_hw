"""Predict passengers in entirely unseen bookings from sliced distributions of their party."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from exaone_validation_data import CATEGORICAL, compact_features
from grouped_five_way_coverage_n23 import verify
from population_error_accounting_n23 import ADDITIONS, ART, BILLS, FAMILIES, ROOT, SEEDS
from relational_error_corrector import write


def passenger_vectors(roster):
    frame = compact_features(roster)
    indicators = pd.get_dummies(frame[CATEGORICAL], dtype=float).to_numpy()
    bills = frame[BILLS].to_numpy(float)
    dense = np.column_stack([indicators, frame.Age.fillna(30)/35, frame.Age.isna(),
                            frame.CabinNumber.fillna(800)/1000, frame.CabinNumber.isna(),
                            np.nan_to_num(np.log1p(bills))/6, np.isnan(bills), bills > 0])
    return dense


def booking_features(x, booking):
    """Fixed mixed-field random projections; each sorted party is a probability measure."""
    rng = np.random.default_rng(61391)
    projection = rng.choice([-1., 1.], size=(x.shape[1], 28))/np.sqrt(x.shape[1])
    directions = x @ projection
    low = np.empty_like(directions)
    med = np.empty_like(directions)
    high = np.empty_like(directions)
    size = np.empty(len(x))
    for rows in booking.groupby(booking, sort=False).indices.values():
        size[rows] = len(rows)
        ranks = np.sort(directions[rows], axis=0)
        low[rows] = ranks[0]
        med[rows] = np.median(ranks, axis=0)
        high[rows] = ranks[-1]
    return size, np.column_stack([med-directions, high-low, high+low-2*med])


def describe(p, base, y, hard, multiple):
    correct = (p >= .5) == y
    initial = (base >= .5) == y
    def stats(mask):
        return dict(rows=int(mask.sum()), correct=int(correct[mask].sum()),
                    fixed=int((correct & ~initial & mask).sum()),
                    harmed=int((~correct & initial & mask).sum()))
    return {key:stats(mask) for key,mask in {
        "all":np.ones(len(y),bool), "multiple":multiple, "single":~multiple,
        "never_recovered_365":hard}.items()}


def run():
    dataset = pd.read_csv(ART/"nov23_global_error_inventory.csv")
    rows, y = dataset.row.to_numpy(), dataset.target.to_numpy(bool)
    train, test = [pd.read_csv(ROOT/f"data/raw/{name}.csv", usecols=lambda c:c not in ["Transported","Name"])
                   for name in ["train","test"]]
    roster = pd.concat([train,test],ignore_index=True)
    party = roster.PassengerId.str.split("_").str[0]
    train_party, test_party = party[:len(train)], party[len(train):]
    overlap = set(train_party) & set(test_party)
    assert not overlap
    x = passenger_vectors(roster)
    size, extra = booking_features(x,party)
    bookings = party.iloc[rows].to_numpy()
    individual = np.column_stack([x[rows], np.log(size[rows])])
    features = {"passenger":individual, "party_distribution":np.column_stack([individual,extra[rows]])}
    sources = {**FAMILIES, "budget":("simplex_likelihood","budget_prior"), **ADDITIONS}
    ever = np.zeros(len(y),bool)
    base_mean = np.zeros(len(y))
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as ref:
            base_mean += ref["base"]/len(SEEDS)
            fold = ref["fold"]
        for folder,key in sources.values():
            with np.load(ART/f"nov23_{folder}/{seed}.npz") as cache:
                verify(cache, rows,y,fold)
                ever |= (cache[key]>=.5)==y
    hard = ~ever & ((base_mean>=.5)!=y)
    assert hard.sum()==365
    results={}
    out=ART/"nov23_deployment_booking_mix"
    out.mkdir(exist_ok=True)
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as ref:
            base,fold=ref["base"],ref["fold"]
            verify(ref,rows,y,fold)
        models={}
        for name,design in features.items():
            p=np.empty(len(y))
            for k in np.unique(fold):
                t,q=fold!=k,fold==k
                assert not set(bookings[t]) & set(bookings[q])
                model=HistGradientBoostingClassifier(max_iter=120,learning_rate=.065,l2_regularization=6,
                    max_leaf_nodes=15,min_samples_leaf=32,early_stopping=False,random_state=622)
                model.fit(design[t],y[t])
                p[q]=model.predict_proba(design[q])[:,1]
            models[name]=p
            models[name+"_quarter"]=.75*base+.25*p
        results[str(seed)]={k:describe(p,base,y,hard,size[rows]>1) for k,p in models.items()}
        np.savez_compressed(out/f"{seed}.npz",row=rows,y=y,fold=fold,base=base,**models)
        print(seed,[(k,v["all"]) for k,v in results[str(seed)].items()],flush=True)
    write(ROOT/"reports/nov23_deployment_booking_mix/result.json",dict(
        test_rows=len(test),test_groups=int(test_party.nunique()),train_groups=int(train_party.nunique()),
        test_rows_with_training_group=int(test_party.isin(train_party).sum()),
        test_multiperson=int((size[len(train):]>1).sum()),development_multiperson=int((size[rows]>1).sum()),results=results))


if __name__=="__main__":
    with threadpool_limits(limits=2):
        run()
