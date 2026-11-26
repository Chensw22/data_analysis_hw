"""Combine one vote per model family and compare against matched versions of all three families."""
import json
import numpy as np
import pandas as pd

from exhaustive_family_quorums_n26 import summary
from latent_passenger_difficulty_n26 import OUT as PANEL,ROOT
from matched_state_context_n26 import inputs

ART=ROOT/'artifacts'
OUT=ART/'nov26_three_family_corroboration'
REPORT=ROOT/'reports/nov23_matched_learning_curve/three_family_corroboration.json'


def decisions(base,new,old):
    hard=new>=.5
    prior=old>=.5
    challenges=hard!=base[:,None]
    baseline=prior!=base[:,None]
    moves=(new-old)*np.where(base,-1.,1.)[:,None]
    count=challenges.sum(1)
    return dict(ordinary_majority=baseline.sum(1)>=2, majority=count>=2, unanimous=count==3,
                two_strengthened=(challenges&(moves>0)).sum(1)>=2, two_new=(challenges&~baseline).sum(1)>=2,
                expanded_majority=(count>=2)&(count>baseline.sum(1)), newly_unanimous=(count==3)&~baseline.all(1))


def run():
    raw,y,g,f,masks=inputs()
    ids=raw.PassengerId.to_numpy(str)
    n=len(y)
    with np.load(PANEL/'panel.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        base,known,common,fresh=[z[k] for k in ('anchor','known','common','fresh')]
        assert z['names'][-2]=='TabICL fare'
        global_tab=z['scores'][:,-2]
    def read(folder,file,keys):
        with np.load(ART/folder/file) as z:
            np.testing.assert_array_equal(z['passenger'],ids)
            np.testing.assert_array_equal(z['fold'],f)
            np.testing.assert_array_equal(z['y'],y)
            return [z[k] for k in keys]
    with np.load(ART/'nov26_lopez_numeric_clusters/aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids[:n])
        np.testing.assert_array_equal(z['test_ids'],ids[n:])
        np.testing.assert_array_equal(z['reference161'],base[:n])
        np.testing.assert_array_equal(z['test_reference161'],base[n:])
        plain,cluster,_=np.vstack([z['oof'],z['test_scores']]).T
    total,weighted=read('nov26_weighted_bill_ridges','scores.npz',['total','weighted'])
    adjusted,uniform=read('nov26_surprise_sampled_foundation','aligned.npz',['adjusted','uniform'])
    expert,=read('nov26_full_state_context','aligned.npz',['p'])
    ordinal,indicator,rotated=read('nov26_nori_nominal_geometry','aligned.npz',['ordinal','indicator','rotated'])
    a=np.isfinite(expert)
    new=np.column_stack([.5*(cluster+weighted),np.where(a,.5*(expert+adjusted),adjusted),.5*(indicator+rotated)])
    old=np.column_stack([.5*(plain+total),np.where(a,.5*(global_tab+uniform),uniform),ordinal])
    assert np.isfinite(new).all() and np.isfinite(old).all()
    with np.load(ART/'nov26_retained_order_mechanisms/aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids[:n])
        blind=z['eligible']&((z['pairs']>=.5)==base[:n,None,None]).all(axis=(1,2))&((z['original_panel']>=.5)==base[:n,None]).all(axis=1)
    assert (blind&(base[:n]!=y)).sum()==266
    size=pd.Series(g[:n]).groupby(g[:n]).transform('size').to_numpy()
    scopes=dict(all=known,common=common,fresh=known&fresh,all31_agree=blind,solo=known&(size==1),pair=known&(size==2),
                larger=known&(size>=3),hard_awake=common&masks['awake_paid'][:n]&~base[:n],hard_sleep=common&masks['sleep_unpaid'][:n]&base[:n])
    uncertain=pd.read_csv(ART/'nov26_clean_fare_unknown_branches/paired_rows.csv')
    np.testing.assert_array_equal(uncertain.PassengerId,ids[:n][~known])
    deltas={}
    for col in ('161_low','161_high'):
        reference=base[:n].copy()
        reference[~known]=uncertain[col].to_numpy(bool)
        for k,v in decisions(reference,new[:n],old[:n]).items():
            deltas.setdefault(k,[]).append(np.where(v,np.where(reference!=y,1,-1),0))
    result={}
    rules=decisions(base,new,old)
    for k,v in rules.items():
        lo,hi=np.min(deltas[k],axis=0),np.max(deltas[k],axis=0)
        np.testing.assert_array_equal(lo[known],hi[known])
        result[k]=dict(test_changes=int(v[n:].sum()),all_train_net_range=[int(lo.sum()),int(hi.sum())],
                       scopes={s:summary(v[:n],base[:n]!=y,m,f) for s,m in scopes.items()})
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'aligned.npz',passenger=ids,y=y,fold=f,base=base,new=new,old=old,**rules)
    REPORT.write_text(json.dumps(result,indent=2)+'\n')
    print([(k,r['scopes']['all']['net'],r['scopes']['fresh']['net'],r['all_train_net_range']) for k,r in result.items()])


if __name__=='__main__':
    run()
