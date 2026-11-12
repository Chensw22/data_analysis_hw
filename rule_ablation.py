"""One-factor, no-ID comparisons of attribute-only missing-value rules."""
import argparse
import hashlib
import json
import shutil
import pandas as pd
import iterate

ORIGINAL_FIT=iterate.fit
iterate.CONFIGS={
    iterate.BASELINE:dict(context=False,model='catboost',**iterate.CAT),
    'v26_logic_only':dict(context=False,model='rules',logic=True,peers=False),
    'v27_unanimous_only':dict(context=False,model='rules',logic=False,peers=True),
    'v28_logic_unanimous':dict(context=False,model='rules',logic=True,peers=True),
}
iterate.BLEND=[]


def rule_features(roster,logic,peers):
    original=roster.drop(columns='Transported',errors='ignore').reset_index(drop=True)
    clean=original.copy()
    if logic:
        zero=clean.CryoSleep.eq(True) | clean.Age.lt(13)
        clean.loc[zero,iterate.base.SPEND]=clean.loc[zero,iterate.base.SPEND].fillna(0)
        paid=clean[iterate.base.SPEND].gt(0).any(axis=1)
        clean.loc[clean.CryoSleep.isna() & paid,'CryoSleep']=False
    x,cats=iterate.base.features(clean,'engineered_no_ids')
    # Preserve evidence of original missing values after deterministic fill.
    old,_=iterate.base.features(original,'engineered_no_ids')
    for col in ['MissingCount','SpendMissingCount','Age_missing']+[s+'_missing' for s in iterate.base.SPEND]:
        x[col]=old[col]
    if peers:
        group=original.PassengerId.str.split('_').str[0]
        values={c:original[c] for c in ['HomePlanet','Destination']}
        values['Deck']=original.Cabin.str.split('/').str[0]
        values['Side']=original.Cabin.str.split('/').str[-1]
        for col,series in values.items():
            grouped=series.groupby(group)
            fill=group.map(grouped.first().where(grouped.nunique().eq(1)))
            x[col]=series.fillna(fill).fillna('__MISSING__').astype(str)
    return x,cats


MATRICES={}


def fit(name,matrices,train_idx,y,predict_idx,seed):
    if name==iterate.BASELINE:
        return ORIGINAL_FIT(name,matrices,train_idx,y,predict_idx,seed)
    if name not in MATRICES:
        train,test,_=iterate.base.load_data()
        roster=pd.concat([train.drop(columns='Transported'),test],ignore_index=True)
        config=iterate.CONFIGS[name]
        MATRICES[name]=rule_features(roster,config['logic'],config['peers'])
    # Keep baseline model hyperparameters identical; change feature matrix only.
    return ORIGINAL_FIT(iterate.BASELINE,{False:MATRICES[name]},train_idx,y,predict_idx,seed)


iterate.fit=fit

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['screen','confirm','export'])
    parser.add_argument('--run',default='run_009');args=parser.parse_args()
    out=iterate.ROOT/'artifacts'/args.run
    {'screen':iterate.screen,'confirm':iterate.confirm,'export':iterate.export}[args.stage](out)
    if args.stage=='screen':
        path=iterate.ROOT/'rule_ablation.py';shutil.copy2(path,out/'rule_ablation_source.py')
        protocol=json.loads((out/'protocol.json').read_text())
        protocol['sources']['rule_ablation.py']=hashlib.sha256(path.read_bytes()).hexdigest()
        protocol['rules']='Fill missing spend with zero for CryoSleep True or Age<13; missing CryoSleep False if positive spend; optionally unanimous group HomePlanet/Destination/Deck/Side. Original missing indicators retained. No labels used.'
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
