"""Use surname-consistent observed HomePlanet attributes, never target labels."""
import argparse
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
import iterate
import contrast_features

BASE_FIT=contrast_features.BASE_FIT
iterate.CONFIGS={
    iterate.BASELINE:dict(context=False,model='catboost',**iterate.CAT),
    'v25_spatial_blend':dict(context=False,model='spatial',knots=5,C=.1,weight=.5),
    'v34_contrasts_spatial':dict(context=False,model='spatial',knots=5,C=.1,weight=.5),
    'v42_surname_planet':dict(context=False,model='surname',base=iterate.BASELINE),
    'v43_surname_contrasts_spatial':dict(context=False,model='surname',base='v34_contrasts_spatial'),
}
iterate.BLEND=[]
CACHE=None


def lookup(roster):
    roster=roster.drop(columns='Transported',errors='ignore')
    surname=roster.Name.str.split().str[-1]
    stats=roster.HomePlanet.groupby(surname).agg(['nunique','first','count'])
    return stats['first'].where(stats['nunique'].eq(1)).dropna().to_dict(),stats


def augment(raw,x,mapping):
    x=x.copy()
    surname=raw.Name.str.split().str[-1]
    fill=surname.map(mapping)
    missing=raw.HomePlanet.isna() & fill.notna()
    x.loc[missing,'HomePlanet']=fill.loc[missing].astype(str)
    return x


class SurnamePredictor:
    def __init__(self,model,mapping):
        self.model=model;self.mapping=mapping

    def predict_proba(self,prepared_features):
        # Prepared-feature API; imputation is required before this method.
        return self.model.predict_proba(prepared_features)

    def predict_proba_raw(self,raw):
        x,_=iterate.base.features(raw,'engineered_no_ids')
        return self.model.predict_proba(augment(raw,x,self.mapping))


def fit(name,matrices,train_idx,y,predict_idx,seed):
    global CACHE
    if name not in ['v42_surname_planet','v43_surname_contrasts_spatial']:
        return contrast_features.fit(name,matrices,train_idx,y,predict_idx,seed)
    if CACHE is None:
        train,test,_=iterate.base.load_data()
        raw=pd.concat([train.drop(columns='Transported'),test],ignore_index=True)
        mapping,_=lookup(raw)
        x,cats=matrices[False]
        CACHE=(augment(raw,x,mapping),cats,mapping)
    x,cats,mapping=CACHE
    base=iterate.CONFIGS[name]['base']
    p,model=contrast_features.fit(base,{False:(x,cats)},train_idx,y,predict_idx,seed)
    return p,SurnamePredictor(model,mapping)


iterate.fit=fit

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['screen','confirm','export'])
    parser.add_argument('--run',default='run_017');args=parser.parse_args()
    out=iterate.ROOT/'artifacts'/args.run
    {'screen':iterate.screen,'confirm':iterate.confirm,'export':iterate.export}[args.stage](out)
    if args.stage=='screen':
        protocol=json.loads((out/'protocol.json').read_text())
        for name in ['surname_impute.py','contrast_features.py','spatial_expert.py']:
            path=iterate.ROOT/name;shutil.copy2(path,out/name)
            protocol['sources'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
        protocol['surname_policy']='HomePlanet only: fill missing value if observed passengers sharing a real surname all have the same planet. Full unlabeled train+test roster, Transported removed. Preserve original missing flags; no Deck rules or surname target encoding.'
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        tr,te,_=iterate.base.load_data();raw=pd.concat([tr.drop(columns='Transported'),te],ignore_index=True)
        mapping,stats=lookup(raw);s=raw.Name.str.split().str[-1]
        filled=raw.HomePlanet.isna() & s.map(mapping).notna()
        support=s[filled].map(stats['count'])
        audit=dict(conflicting_surnames=int(stats['nunique'].gt(1).sum()),
            filled_train_rows=int(filled.iloc[:len(tr)].sum()),filled_test_rows=int(filled.iloc[len(tr):].sum()),
            mapping_support_for_filled_rows={str(k):int(v) for k,v in support.value_counts().sort_index().items()},
            note='Zero observed conflicts is not a guarantee for unseen people. Input-attribute imputation, not Transported propagation.')
        (out/'attribute_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
