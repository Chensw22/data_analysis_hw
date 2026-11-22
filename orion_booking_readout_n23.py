"""Relearn the readout of a frozen Orion representation on independent bookings."""
import json
import sys
import numpy as np
import pandas as pd
import torch
from scipy.special import logit,softmax
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from pretrained_recheck_n23 import ART, ROOT
from orion_bix_probe import VENDOR,CHECKPOINT,WEIGHT_SHA,MANIFEST_SHA,sha


def representations(model, frame, query, n_support):
    chunks=[]
    native=[]
    layer=model.model_.icl_predictor.decoder[-1]
    for part in np.array_split(query,6):
        features=[]
        logits=[]
        def capture(module, args, output):
            x=args[0]
            assert x.ndim==3 and x.shape[:2]==(1,n_support+len(part))
            features.append(x[0,n_support:].detach().cpu().numpy().copy())
            logits.append(output[0,n_support:,:2].detach().cpu().numpy().copy())
        handle=layer.register_forward_hook(capture)
        try:
            scores=model.predict_proba(frame.iloc[part])
        finally:
            handle.remove()
        shift=[int(k) for v in model.ensemble_generator_.class_shift_offsets_.values() for k in v]
        assert len(shift)==len(features)==2
        ordered=np.stack([np.roll(output,-k,axis=1) for output,k in zip(logits,shift)])
        if model.average_logits:
            rebuilt=softmax(ordered.mean(axis=0)/model.softmax_temperature,axis=1)
        else:
            rebuilt=softmax(ordered/model.softmax_temperature,axis=2).mean(axis=0)
        np.testing.assert_allclose(rebuilt,scores,atol=1e-5)
        chunks.append(np.column_stack(features))
        native.append(scores[:,1])
    return np.vstack(chunks),np.concatenate(native)


def estimate(states,native,y,groups):
    estimates={name:np.full(len(y),np.nan) for name in ("native_only","representation","shuffled")}
    for t,q in GroupKFold(3,shuffle=True,random_state=216).split(y,groups=groups):
        assert not np.intersect1d(groups[t],groups[q]).size
        pca=PCA(n_components=32,whiten=True,svd_solver="randomized",random_state=120)
        pca.fit(states[t])
        features=pca.transform(states).clip(-6,6)
        # Permute the hidden activations between passengers with similar native
        # scores, independently in fit and query. Keep their true score fixed.
        null=features.copy()
        rng=np.random.default_rng(145)
        for part in (t,q):
            band=(native[part]*12).astype(int)
            for j in np.unique(band):
                ids=part[band==j]
                null[ids]=features[rng.permutation(ids)]
        raw=logit(native.clip(.0005,.9995))[:,None]
        for name,x in [("native_only",raw),("representation",np.column_stack([raw,features])),
                       ("shuffled",np.column_stack([raw,null]))]:
            clf=LogisticRegression(C=.05,solver="lbfgs",max_iter=300).fit(x[t],y[t])
            estimates[name][q]=clf.predict_proba(x[q])[:,1]
    return estimates


def run():
    torch.set_num_threads(2)
    assert sha(CHECKPOINT)==WEIGHT_SHA and sha(VENDOR/"SOURCE.json")==MANIFEST_SHA
    manifest=json.loads((VENDOR/"SOURCE.json").read_text())
    assert all(sha(VENDOR/k)==v for k,v in manifest["vendor_files_sha256"].items())
    sys.path.insert(0,str(VENDOR/"src"))
    from orion_bix.sklearn.classifier import OrionBixClassifier
    out=ART/"nov23_orion_readout"
    out.mkdir(exist_ok=True)
    rep=ROOT/"reports/nov23_orion_readout"
    rep.mkdir(exist_ok=True)
    results={}
    for seed in (20261112,20261115,20261116):
        with np.load(ART/f"nov23_pretrained_recheck/{seed}.npz") as ref:
            row,y,base,fold,expected=(ref[k] for k in ("row","y","base","fold","orion"))
        with np.load(ART/"nov23_archive_recovery/masks.npz") as masks:
            np.testing.assert_array_equal(masks["row"],row)
            hard=masks["still_unresolved"]
        frame=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ("Name","Transported")).iloc[row].copy()
        group=frame.pop("PassengerId").str.split("_").str[0].to_numpy()
        cabin=frame.pop("Cabin").str.split("/",expand=True)
        frame["Deck"],frame["CabinNumber"],frame["Side"]=cabin[0],pd.to_numeric(cabin[1]),cabin[2]
        for c in ("VIP","CryoSleep"):
            frame[c]=frame[c].map({True:"yes",False:"no"})
        probs={}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            with np.load(ART/f"nov23_pretrained_recheck/{seed}_fold{f}.npz") as rec:
                np.testing.assert_array_equal(row[q],rec["query"])
                support=np.searchsorted(row,rec["training"])
                np.testing.assert_array_equal(row[support],rec["training"])
            assert not np.intersect1d(group[support],group[q]).size
            model=OrionBixClassifier(n_estimators=2,batch_size=1,use_amp=False,device="cpu",n_jobs=2,
                    model_path=str(CHECKPOINT),allow_auto_download=False,checkpoint_version=CHECKPOINT.name,random_state=121)
            model.fit(frame.iloc[support],y[support].astype(int))
            states,native=representations(model,frame,q,len(support))
            np.testing.assert_allclose(expected[q],native,atol=1e-5)
            estimates=estimate(states,native,y[q],group[q])
            for name,p in estimates.items():
                probs.setdefault(name,np.full(len(y),np.nan))[q]=p
            print(seed,"fold",f,"width",states.shape[1],flush=True)
        scores={}
        for name,p in probs.items():
            assert np.isfinite(p).all()
            for key,s in [(name,p),(name+"_quarter",.75*base+.25*p)]:
                ok=(s>=.5)==y
                original=(base>=.5)==y
                scores[key]=dict(accuracy=float(ok.mean()),fixed=int((ok&~original).sum()),harmed=int((~ok&original).sum()),
                                 deep252=int(ok[hard].sum()),logloss=float(-np.mean(np.log(np.where(y,s,1-s).clip(1e-9)))))
        results[seed]=scores
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**probs)
        (rep/"result.json").write_text(json.dumps(dict(results=results),indent=2)+"\n")
        print(seed,scores["representation"],scores["representation_quarter"],flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
