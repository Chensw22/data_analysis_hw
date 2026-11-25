"""Measure corroborated changes from the same cabin evidence in three separately trained models."""
import json

import numpy as np

from disjoint_tree_fields_n25 import ART, DEST, stats
from family_identity_transfer_n26 import panels_and_reference
from full_roster_booking_validation_n23 import data
from global_cabin_assignment_n26 import OUT
from outstanding_error_families_n26 import observed


def run():
    raw,y,g,f=data()
    _,_,_,rows,names,p,h,native,old_native,ref=panels_and_reference()
    with np.load(ART/'nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['original_prior'],ref[0])
        ref[0]=z['prior']
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        hf,old,par=z['fold'],z['nine'],z['v131']
    awake=raw.CryoSleep.eq(False).to_numpy()
    hist=observed(rows,hf,h,names,old_native,par,old,old,awake,'historical')
    current=observed(np.arange(len(y)),f,p,names,native,*ref,awake,'full')
    results={}
    sets={}
    for tag,ids,fold,view in [('historical',rows,hf,hist),('full',np.arange(len(y)),f,current)]:
        clean=~np.isin(g[ids],g[rows])
        with np.load(OUT/f'geometry_{tag}.npz') as z:
            np.testing.assert_array_equal(z['q'],ids)
            previous=[z['plain']>=.5]
            new=[z['geometry']>=.5]
        scores=np.empty((len(ids),4),bool)
        for k in np.unique(fold):
            loc=fold==k
            q=ids[loc]
            if tag=='historical':
                clean[loc]=~np.isin(g[q],g[ids[~loc]])
            with np.load(OUT/f'tabicl_{tag}_{k}.npz') as z:
                np.testing.assert_array_equal(z['q'],q)
                scores[loc,0]=z['base']>=.5
                scores[loc,1]=z['p']>=.5
            with np.load(ART/f'nov26_constrained_track_odds/{tag}_{k}.npz') as z:
                np.testing.assert_array_equal(z['q'],q)
                scores[loc,2]=z['prior']>=0
                scores[loc,3]=(z['prior']+z['real'])>=0
        previous=np.column_stack(previous+[scores[:,0],scores[:,2]])
        new=np.column_stack(new+[scores[:,1],scores[:,3]])
        anchor=view['anchor']
        updated=(new!=previous)&(new!=anchor[:,None])
        tests={'any':updated.any(1),'two':updated.sum(1)>=2,'three':updated.all(1),
               'two_forward':updated[:,:2].all(1),'plain_two':(previous!=anchor[:,None]).sum(1)>=2,
               'informed_two':(new!=anchor[:,None]).sum(1)>=2}
        sets[tag]=tests
        families=view['eligible']&(view['votes']==anchor[:,None]).all(1)
        data_set={}
        for method,flip in tests.items():
            delta=flip*np.where(anchor==y[ids],-1,1)
            table={}
            for label,mask in [('all',view['known']),('new',view['known']&clean),
                               ('families',families),('new_families',families&clean)]:
                table[label]=stats(delta,mask,fold,g[ids])
            data_set[method]=table
        results[tag]=data_set
    eligible=hist['known']&current['known'][rows]&(hist['anchor']==current['anchor'][rows])
    agreement={}
    for key,historical_edit in sets['historical'].items():
        repeat=historical_edit&sets['full'][key][rows]
        delta=repeat*np.where(hist['anchor']==y[rows],-1,1)
        agreement[key]=stats(delta,eligible,hf,g[rows])
    results['paired']=agreement
    DEST.with_name('cross_model_cabin_corrections.json').write_text(json.dumps(results,indent=2)+'\n')
    for key in sets['full']:
        print(key,[(tag,results[tag][key]['all']['fixed'],results[tag][key]['all']['harmed']) for tag in ('full','historical')],
              agreement[key]['fixed'],agreement[key]['harmed'])


if __name__=='__main__':
    run()
