"""Learn passenger attribute co-occurrence from the joint unlabeled manifest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART,ROOT,SEEDS,panel,verify
from deployment_booking_mix_n23 import passenger_vectors
from exaone_validation_data import CATEGORICAL,compact_features
from population_error_accounting_n23 import BILLS
from relational_error_corrector import write


def fields(raw, party):
    x = compact_features(raw)
    blocks = [pd.get_dummies(x[c].fillna("missing").astype(str)).to_numpy(float) for c in CATEGORICAL]
    for c in ["Age", "CabinNumber", *BILLS]:
        v = x[c].to_numpy()
        valid = np.isfinite(v)
        amount = valid & (v > 0) if c in BILLS else valid
        cuts = np.unique(np.quantile(v[amount],np.linspace(0,1,13)[1:-1]))
        code = np.where(valid, 1, 0)
        code[amount] = 2 + np.searchsorted(cuts,v[amount])
        blocks.append(pd.get_dummies(code).to_numpy(float))
    counts = party.map(party.value_counts()).clip(upper=5)
    blocks.append(pd.get_dummies(counts).to_numpy(float))
    return blocks


def transformed(blocks, train, query, shuffle):
    # Multiple correspondence analysis on the full disjunctive table. Use the
    # same real rows for projection under the destroyed-co-occurrence control.
    rng = np.random.default_rng(210)
    learn = np.column_stack([v[rng.permutation(train) if shuffle else train] for v in blocks])
    inputs = np.column_stack([v[query] for v in blocks])
    marginal = learn.mean(0)
    scale = np.sqrt(marginal.clip(.0001))
    z = (learn-marginal)/scale
    _,vectors = np.linalg.eigh(z.T@z/len(train))
    basis = vectors[:,-14:]
    projected = ((inputs-marginal)/scale)@basis
    offset = 0
    features = [projected]
    nll = []
    for block in blocks:
        cols = slice(offset,offset+block.shape[1])
        centered = (inputs[:,cols]-marginal[cols])/scale[cols]
        # Remove this field before predicting its value using only the other
        # fields. Aggregate reconstruction evidence is an input to classification.
        masked = projected - centered@basis[cols]
        reconstructed = (masked@basis[cols].T)*scale[cols]+marginal[cols]
        probabilities = np.maximum(reconstructed,.001)
        probabilities /= probabilities.sum(axis=1,keepdims=True)
        observed = np.sum(inputs[:,cols]*probabilities,axis=1).clip(.0001,1.)
        background = np.sum(inputs[:,cols]*marginal[cols],axis=1).clip(.0001,1.)
        features.append(np.column_stack([np.log(observed/background), np.max(probabilities,axis=1)]))
        nll.append(float(np.mean(-np.log(observed))))
        offset+=block.shape[1]
    return np.column_stack(features),nll


def run():
    dev,y,b,p,_,_,_=panel()
    base_hits=(b>=.5)==y
    other=(p>=.5)==y[None,:,None]
    near=(other.mean(1)>=base_hits.mean(1)[:,None]-.01).all(0)
    weak=(~base_hits.any(0)) & other.all(0).any(1) & ~other.all(0)[:,near].any(1)
    deep=~base_hits.any(0) & ~other.any(axis=(0,2))
    assert (int(weak.sum()),int(deep.sum()))==(335,252)
    roster=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda c:c not in {"Name","Transported"})
                      for s in ("train","test")],ignore_index=True)
    party=roster.PassengerId.str.split("_").str[0]
    group=party.to_numpy()[dev]
    blocks=fields(roster,party)
    personal=np.column_stack([passenger_vectors(roster)[dev], np.log1p(party.map(party.value_counts()).to_numpy()[dev])])
    rng=np.random.default_rng(721)
    hold=party.isin(rng.choice(party.unique(),size=len(party.unique())//5,replace=False)).to_numpy()
    learn,query=np.flatnonzero(~hold),np.flatnonzero(hold)
    assert not np.intersect1d(party.iloc[learn].unique(),party.iloc[query].unique()).size
    inputs={"personal":personal}
    selftest={}
    for name,shuffle in [("manifest",False),("shuffled",True)]:
        phi,_=transformed(blocks,np.arange(len(roster)),dev,shuffle)
        inputs[name]=np.column_stack([personal,phi])
        _,selftest[name]=transformed(blocks,learn,query,shuffle)
    out=ART/"nov23_unlabelled_matrix_views"
    out.mkdir(exist_ok=True)
    result={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as ref:
            fold,base=ref["fold"],ref["base"]
            verify(ref,dev,y,fold)
        models={}
        summary={}
        for name,x in inputs.items():
            pred=np.full(len(y),np.nan)
            for f in np.unique(fold):
                t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
                assert not np.intersect1d(group[t],group[q]).size
                model=HistGradientBoostingClassifier(max_iter=185,max_leaf_nodes=15,min_samples_leaf=22,
                    learning_rate=.06,l2_regularization=3,early_stopping=False,random_state=82)
                model.fit(x[t],y[t])
                pred[q]=model.predict_proba(x[q])[:,1]
            assert np.isfinite(pred).all()
            models[name]=pred
            for key,s in [(name,pred),(name+"_quarter",.25*pred+.75*base)]:
                ok=(s>=.5)==y
                old=(base>=.5)==y
                summary[key]=dict(accuracy=float(ok.mean()),fixed=int((ok&~old).sum()),harmed=int((~ok&old).sum()),
                                  weak335=int(ok[weak].sum()),deep252=int(ok[deep].sum()))
        np.savez_compressed(out/f"{seed}.npz",row=dev,y=y,fold=fold,base=base,**models)
        result[seed]=summary
        print(seed,summary,flush=True)
    write(ROOT/"reports/nov23_unlabelled_matrix_views/result.json",dict(
        roster=len(roster),selftest_rows=len(query),fields=[*CATEGORICAL,"Age","CabinNumber",*BILLS,"BookingSize"],
        held_booking_reconstruction_nll=selftest,results=result))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
