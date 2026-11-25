"""Measure the effect of restoring exact bill amounts inside bins and save a passenger-level audit."""
import json
import numpy as np
import pandas as pd

from disjoint_tree_fields_n25 import ART,DEST
from full_roster_booking_validation_n23 import data
from full_roster_six_failure_archive_n24 import prediction_sets
from grouped_peer_teacher_n25 import OUT as REF
from precise_source_views_n25 import GROUPS,OUT as TAB

BASE=ART/'nov25_passenger_joint_likelihood'
OUT=ART/'nov25_within_bin_bill_precision'
COLS=['direct',*['omit_'+s for s in GROUPS]]
MODES=('quantized','exact_bill_position','shuffled_bill_position')


def run():
    raw,y,g,fold=data()
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        rows,hf=z['rows'],z['fold']
    _,panels=prediction_sets(raw,y,g,fold,rows,hf)
    votes=(np.column_stack(panels[0])>=.5).sum(1)
    ref=votes>34
    common=(votes==0)|(votes==68)
    errors=(ref!=y)
    probs={mode:np.empty((len(y),len(COLS))) for mode in MODES}
    all_names=['outcome_only','direct','chain_0','chain_1',*[name+'_'+s for s in GROUPS for name in ('omit','restore')]]
    original_all=np.empty((len(y),len(all_names)),bool)
    tab=np.empty((len(y),len(GROUPS)),bool)
    nine=np.full(len(y),-1)
    for k in range(8):
        q=np.flatnonzero(fold==k)
        with np.load(REF/f'release{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            nine[q]=z['nine']
        with np.load(BASE/f'fold{k}.npz') as z:
            original_all[q]=np.column_stack([z[n] for n in all_names])>=.5
        for mode in MODES:
            path=BASE/f'fold{k}.npz' if mode=='quantized' else OUT/f'{k}_{mode}.npz'
            with np.load(path) as z:
                np.testing.assert_array_equal(z['q'],q)
                np.testing.assert_array_equal(z['passenger'],raw.PassengerId.iloc[q])
                probs[mode][q]=np.column_stack([z[n] for n in COLS])
        for j,name in enumerate(GROUPS):
            with np.load(TAB/f'{k}_{name}.npz') as z:
                np.testing.assert_array_equal(z['q'],q)
                tab[q,j]=z['p']>=.5
    def pair(h,anchor,mask):
        return dict(fixed=int((mask&(anchor!=y)&(h==y)).sum()),harmed=int((mask&(anchor==y)&(h!=y)).sum()))
    def exposure(mask):
        return dict(reachable=int((mask&common&errors).sum()),false_alarms=int((mask&common&~errors).sum()))
    report={}
    flags={}
    quant=probs['quantized']>=.5
    for mode in MODES:
        h=probs[mode]>=.5
        flags[mode]=(h!=ref[:,None])
        opportunity=flags[mode].any(1)
        report[mode]=dict(unanimous_opportunity=exposure(opportunity),
            net_direct_relative_to_quantized=pair(h[:,0],quant[:,0],np.ones(len(y),bool)),
            direct_vs_known_nine=pair(h[:,0],nine,nine>=0),
            by_source={n:dict(vs_quantized=pair(h[:,j],quant[:,j],np.ones(len(y),bool)),
                              unanimous=pair(h[:,j],ref,common)) for j,n in enumerate(COLS)})
    for a,b in [('quantized','exact_bill_position'),('exact_bill_position','shuffled_bill_position')]:
        report[a+'_and_'+b]=dict(same_view=exposure((flags[a]&flags[b]).any(1)),
                                   either_view=exposure(flags[a].any(1)|flags[b].any(1)))
    source=original_all!=ref[:,None]
    raw_hits=tab!=ref[:,None]
    def names_for(mask,names):
        return ['|'.join(n for n,on in zip(names,row,strict=True) if on) for row in mask]
    table=pd.DataFrame(dict(PassengerId=raw.PassengerId,booking_fold=fold,
              transported=y,original_votes_true=votes,original_majority=ref,
              unanimous=common,majority_wrong=errors,reference_nine=pd.array(np.where(nine<0,None,nine==1),dtype='boolean'),
              new_order=~np.isin(g,g[rows]),binned_dissent=names_for(source,all_names),
              exact_position_dissent=names_for(flags['exact_bill_position'],COLS),
              raw_tabicl_dissent=names_for(raw_hits,list(GROUPS))))
    table.to_csv(OUT/'passenger_error_audit.csv',index=False)
    check=pd.read_csv(OUT/'passenger_error_audit.csv',dtype={'PassengerId':str})
    np.testing.assert_array_equal(check.PassengerId,raw.PassengerId)
    assert ((check.unanimous)&(check.majority_wrong)&check.binned_dissent.notna()).sum()==48
    DEST.with_name('within_bin_bill_precision.json').write_text(json.dumps(report,indent=2)+'\n')
    for k in MODES:print(k,report[k]['unanimous_opportunity'],report[k]['net_direct_relative_to_quantized'])
    print('cross-view',report['quantized_and_exact_bill_position'])
    print('audit',len(table))


if __name__=='__main__':
    run()
