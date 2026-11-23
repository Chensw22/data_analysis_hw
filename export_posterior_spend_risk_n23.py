"""Replay, fit and export groupwise posterior error probabilities above the deployed v131."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import strata
from broad_full_basket_policy_n23 import evidence
from broad_geographic_transfer_n23 import load
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from broad_service_transfer_n23 import prepare
from exaone_validation_data import compact_features
from export_budget_quarter_n23 import source
from kaggle_workflow import sha256, validate_submission

VERSION='n23_posterior_spending_risk'
NAME='proposal_'+VERSION


def profile(frame,base,ref):
    dominant=prepare(frame)[0]
    basket=np.where(dominant>=0,dominant,np.where(strata(frame)==0,5,6))
    return pd.DataFrame(dict(base=base,ref=ref,cell=2*basket+ref.astype(int)))


def probabilities(training, query):
    base=query.base.to_numpy()
    ref=query.ref.to_numpy()
    before=training.base.to_numpy()
    old=training.ref.to_numpy()
    err=training.error.to_numpy()
    risk=np.where(old,1-before,before).clip(.001,.999)
    p=np.where(ref,1-base,base).clip(.001,.999)
    for cell in np.unique(query.cell):
        t=(training.cell.to_numpy()==cell)&(old==(before>=.5))
        q=(query.cell.to_numpy()==cell)&(ref==(base>=.5))
        if t.sum()>=2 and q.any():
            p[q]=evidence(before[t,None],base[q,None],risk[t],p[q],err[t])
    score=np.where(ref,1-p,p)
    score=np.where((p==.5)&~ref,np.nextafter(.5,0),score)
    return np.where(ref==(base>=.5),score,ref.astype(float))


def main():
    row,y,frame,g,_=load()
    with np.load(ART/'nov23_release_vote_information/panel.npz') as z:
        np.testing.assert_array_equal(row,z['row'])
        np.testing.assert_array_equal(y,z['y'])
        wrong=(z['votes']!=y[None,:,None]).all((0,2))
        right=(z['votes']==y[None,:,None]).all((0,2))
    plans,history={},[]
    expected=[11,37,8,3]
    for i,k in enumerate([*map(str,SEEDS),'v131']):
        with np.load(ART/f'nov23_broad_partition_evidence/{k}.npz') as z:
            np.testing.assert_array_equal(row,z['row'])
            np.testing.assert_array_equal(y,z['y'])
            ref,base,fold=z['ref'],z['base'],z['fold']
        train=profile(frame,base,ref).assign(error=ref!=y)
        pred=np.full(len(y),np.nan)
        for f in np.unique(fold):
            ids=np.flatnonzero(fold==f)
            for a,b in GroupKFold(3,shuffle=True,random_state=911).split(ids,groups=g[ids]):
                t,q=ids[a],ids[b]
                assert not np.intersect1d(g[t],g[q]).size
                pred[q]=probabilities(train.iloc[t],train.iloc[q].drop(columns='error'))
        assert np.isfinite(pred).all()
        hits=(pred>=.5)==y
        plans[k]=paired(hits,ref==y)
        plans[k]['fold_net']=[int((hits[fold==f].astype(int)-(ref==y)[fold==f]).sum()) for f in np.unique(fold)]
        assert plans[k]['fixed']-plans[k]['harmed']==expected[i]
        if k!='v131':history.append(hits)
    out=ART/NAME
    out.mkdir(exist_ok=True)
    mfile=out/'calibration.npz'
    np.savez_compressed(mfile,row=row,**{c:train[c].to_numpy() for c in train})
    test=pd.read_csv(ROOT/'data/raw/test.csv')
    base,_,bh=source('proposal_v108_party_bills',test)
    _,ref,rh=source('proposal_v131_party_consensus',test)
    query=profile(compact_features(test),base,ref)
    p=probabilities(train,query)
    with np.load(mfile) as saved:
        np.testing.assert_array_equal(saved['row'],row)
        reload=probabilities(pd.DataFrame({c:saved[c] for c in train}),query)
    np.testing.assert_array_equal(p,reload)
    pd.DataFrame(dict(PassengerId=test.PassengerId,probability=p,reload_probability=reload)).to_csv(out/'test_probabilities.csv',index=False)
    csv=out/'submission.csv'
    pd.DataFrame(dict(PassengerId=test.PassengerId,Transported=p>=.5)).to_csv(csv,index=False)
    check=validate_submission(csv,ROOT/'data/raw/test.csv')
    gaps={}
    for rec in (ROOT/'reports').glob('*/online_attempts/*.json'):
        r=json.loads(rec.read_text())
        name=rec.parents[1].name
        if name==NAME:continue
        assert r.get('file_sha256')!=check['sha256']
        path=ART/name/'submission.csv'
        if path.is_file() and sha256(path)==r.get('file_sha256'):
            old=pd.read_csv(path)
            np.testing.assert_array_equal(old.PassengerId,test.PassengerId)
            gaps[name]=int((old.Transported!=(p>=.5)).sum())
    nearest=min(gaps.values())
    assert nearest>0
    info=dict(**check,version=VERSION,model_sha256=sha256(mfile),reload_difference=0,calibration_rows=len(train),
              v108_sha256=bh,v131_sha256=rh,changed_from_v131=int(((p>=.5)!=ref).sum()),
              nearest_prior={n:v for n,v in gaps.items() if v==nearest},development=plans,
              stable_common=int((np.all(history,axis=0)&wrong).sum()),stable_common_right_missed=int((~np.any(history,axis=0)&right).sum()))
    for folder in [out,ROOT/'reports'/NAME]:
        write(folder/'manifest.json',info)
        write(folder/'selection.json',dict(version=VERSION,sha256=check['sha256']))
    print(info,flush=True)


if __name__=='__main__':
    with threadpool_limits(1):main()
