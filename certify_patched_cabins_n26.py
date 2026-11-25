"""Propagate exact repaired cabin predictions through party offsets and one-direction production gates."""
import argparse
import json

import numpy as np
import pandas as pd

from certified_tree_release_n25 import structural_inputs
from disjoint_tree_fields_n25 import ART,DEST
from family_identity_transfer_n26 import panels_and_reference
from full_roster_booking_validation_n23 import ROOT,data
from grouped_peer_release_replay_n25 import limits
from grouped_peer_teacher_n25 import OUT as TEACHER
from joint_candidate_coverage_n26 import branch_decisions

PATCH=ART/'nov26_patched_cabins'


def run(folds):
    raw,y,g,f=data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())
    patch,peer,sleeping,party=structural_inputs(pd.concat([raw,test],ignore_index=True))
    _,_,_,rows,names,p,_,native,_,ref=panels_and_reference()
    reference=ref.copy()
    n=len(y)
    with np.load(ART/'nov25_production_anchor_cpu/oof.npz') as z:
        w,family=z['witness'],z['family']
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId)
    records={}
    for k in folds:
        t,q=np.flatnonzero(f!=k),np.flatnonzero(f==k)
        path=PATCH/f'fold{k}.npz'
        with np.load(path) as z,np.load(TEACHER/f'release{k}.npz') as old:
            for key,expected in [('t',t),('q',q),('train_y',y[t]),('native',native[q]),('patch_ids',q[patch[q]])]:
                np.testing.assert_array_equal(z[key],expected)
            repaired=z['native'].copy()
            repaired[patch[q]]=z['patch_p']
            p108,cert,prior=limits(repaired,np.zeros(len(q),bool),peer[q],old['shift'],sleeping[q],w[q],party[q],family[q])
            np.testing.assert_array_equal(p108[~patch[q]],old['p108'][~patch[q]])
            np.testing.assert_array_equal(cert[~patch[q]],old['v131'][~patch[q]]>=0)
            np.testing.assert_array_equal(prior[old['v131']>=0],old['v131'][old['v131']>=0])
            reference[0,q[cert]]=prior[cert]
            records[str(k)]={'patched':int(patch[q].sum()),'newly_determined':int((patch[q]&cert).sum()),
                             'v106_changes':int((patch[q]&((repaired>=.5)!=(native[q]>=.5))).sum())}
    out={'folds':records}
    for tag,r in [('old',ref),('updated',reference)]:
        a=branch_decisions(np.arange(n),f,p,names,native,*r,raw.CryoSleep.eq(False).to_numpy(),'full',891)
        known=r[0]>=0
        out[tag]={}
        for comparator in ('nine','band','fields_awake'):
            d=(a['all_three']==y).astype(int)-(a[comparator]==y)
            np.testing.assert_array_equal(d[0,known],d[1,known])
            summary={}
            for key,use in [('all',np.ones(n,bool)),('new',~np.isin(g,g[rows]))]:
                summary[key]=dict(resolved=int((known&use).sum()),fixed=int(((d[0]>0)&known&use).sum()),
                    harmed=int(((d[0]<0)&known&use).sum()),net_bounds=[int(op(d[:,use],axis=0).sum()) for op in (np.min,np.max)])
            if comparator=='nine':
                out[tag].update(summary)
            else:
                out[tag]['relative_to_'+comparator]=summary
    np.savez_compressed(PATCH/('known_'+'_'.join(map(str,folds))+'.npz'),row=np.arange(n),fold=f,prior=reference[0],original_prior=ref[0])
    path=DEST.with_name('certified_patched_cabins.json') if len(folds)==8 else PATCH/'progress.json'
    path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--folds',type=int,nargs='+',default=list(range(8)))
    run(parser.parse_args().folds)
