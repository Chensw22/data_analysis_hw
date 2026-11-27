"""Trace recoverable mistakes and collateral errors among passengers on whom seven scores agree."""
import json
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from dissent_likelihood_transfer_n27 import splits
from full_roster_booking_validation_n23 import ROOT, data
from repeated_archive_misses_n26 import ART, cohort, count
from price_implied_deck_n26 import FEES

OUT = ART / 'nov27_unanimous_error_reach'
REPORT = ROOT / 'reports/nov23_matched_learning_curve/unanimous_error_reach.json'


def inputs():
    raw, y, g, f = data()
    frame = pd.concat([raw, pd.read_csv(ROOT/'data/raw/test.csv', usecols=raw.columns)],ignore_index=True)
    ids = frame.PassengerId.to_numpy(str)
    def get(folder, field, *keys):
        with np.load(ART/folder/field) as z:
            np.testing.assert_array_equal(z['ids' if 'ids' in z else 'passenger'], ids)
            np.testing.assert_array_equal(z['y'], y)
            return [z[k] for k in keys]
    first, random, ordered = get('nov27_dissent_likelihood_transfer','aligned.npz','first','random_median','contiguous_median')
    net,dpt = get('nov27_outsider_directional','aligned.npz','tabnet','dpt')
    p, = get('nov26_latent_passenger_difficulty','panel.npz','scores')
    wide = np.column_stack([np.median(p[:,:18],axis=1),p[:,-3:],np.median(p[:,-3:],axis=1),dpt,np.median(net,axis=1)])
    gen, = get('nov27_manifest_context_coding','aligned.npz','prob')
    dist, = get('nov27_local_hulls','aligned.npz','dist')
    geom = np.log((dist[:,0]+.005)/(dist[:,1]+.005))
    tree,aug = get('nov27_partial_receipt_deck','aligned.npz','base','augmented')
    # Ordinary/robust/stratified-null validation models use the same random booking split as the other inputs.
    robust = np.empty((len(ids), 3))
    for k in range(9):
        q = np.flatnonzero(f==k) if k<8 else np.arange(len(y),len(ids))
        path = ART/'nov27_manifest_group_dro'/(f'random_{k}.npz' if k<8 else 'test.npz')
        with np.load(path) as z:
            np.testing.assert_array_equal(z['q'],q)
            np.testing.assert_array_equal(z['ids'],ids[q])
            robust[q] = np.column_stack([z[key] for key in ('ordinary','observable','shuffled')])
    methods = {'bit_context':gen[:,1], 'bit_independent':gen[:,0], 'same_input_tree':gen[:,2],
        'hull':expit(geom[:,2]), 'nearest':expit(geom[:,0]), 'plain_boost':robust[:,0], 'robust_boost':robust[:,1],
        'random_environment_boost':robust[:,2], 'receipt_tree':tree,'augmented_tree':aug}
    return raw, frame, y,g,f,first,random,ordered,wide,methods


def run():
    raw,frame,y,g,f,first,random,ordered,wide,methods = inputs()
    n = len(y)
    _,blind,known,fresh=cohort(raw,y,f)
    paid=frame[FEES].gt(0).any(axis=1).to_numpy()
    asleep=frame.CryoSleep.eq(True).to_numpy()
    awake=frame.CryoSleep.eq(False).to_numpy()
    report={}
    output={}
    odds=lambda p:logit(np.clip(p,.003,.997))
    core=np.median(odds(wide),axis=1)
    info=np.column_stack([odds(s) for s in methods.values()])
    features=np.column_stack([core,np.std(odds(wide),axis=1),paid,asleep,(frame.Cabin.str.startswith('G/').fillna(False)).to_numpy()])
    for scheme,protected in [('random',random),('contiguous',ordered)]:
        anchor=first^protected
        gate=((wide>=.5)==first[:,None]).all(1)
        agreement=gate&~protected
        masks={'all':known&agreement[:n], 'old_unanimous':blind&agreement[:n],
               'awake_paying':known&(awake&paid&agreement)[:n], 'sleep_unpaid':known&(asleep&~paid&agreement)[:n]}
        err=first[:n]!=y
        diss=np.column_stack([(p>=.5)!=first for p in methods.values()])
        record={}
        for area,m in masks.items():
            record[area]={'eligible':int(m.sum()),'wrong':int((m&err).sum()),
                          'any':count(diss[:n].any(1),m,err),'all':count(diss[:n].all(1),m,err),
                          'experts':{name:count(diss[:n,i],m,err) for i,name in enumerate(methods)}}
        # Matched calibrator on the original seven scores; add evidence about amounts and local geometry.
        # Calibrate only on held predictions in the same outer fold, then judge untouched orders.
        part=splits(g,f,scheme)
        flags={}
        for name,x in [('ordinary_scores',features),('alternative_scores',np.column_stack([features,info]))]:
            oof=np.zeros(len(frame),bool)
            votes=[]
            for k in range(8):
                for j in (0,1,2,3,None):
                    tr=np.flatnonzero((f==k)&gate[:n]&known&((part!=j) if j is not None else True))
                    q=np.flatnonzero((f==k)&(part==j)) if j is not None else np.arange(n,len(frame))
                    model=LogisticRegression(C=.15,max_iter=250).fit(x[tr],y[tr])
                    v=(model.predict(x[q])!=first[q]) & agreement[q]
                    if j is None:votes.append(v)
                    else:oof[q]=v
            oof[n:]=np.mean(votes,axis=0)>.5
            flags[name]=oof
        flags['additional']=flags['alternative_scores']&~flags['ordinary_scores']
        record['calibration']={}
        for k,flag in flags.items():
            output[scheme+'_'+k]=flag
            delta=flag[:n]*np.where(err,1,-1)
            record['calibration'][k]=dict(all=count(flag[:n],known,err),fresh=count(flag[:n],known&fresh,err),
                old=count(flag[:n],blind,err),test=int(flag[n:].sum()),folds=[int(delta[known&(f==j)].sum()) for j in range(8)])
        report[scheme]=record
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'aligned.npz',passenger=frame.PassengerId.to_numpy(str),y=y,fold=f,**output)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    for k,r in report.items():
        print(k,r['all'],r['calibration'])


if __name__=='__main__':
    run()
