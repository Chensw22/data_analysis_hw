"""Replay the field-required correction after retraining on historic whole-booking splits."""
import json
import numpy as np
from threadpoolctl import threadpool_limits

from evaluate_nested_release_n26 import DECK,REVIEW,panel_predictions,stages
from held_expert_portfolios_n26 import sources,splits
from frozen_queue_shared_errors_n26 import paired
from omit_surprising_passenger_field_n26 import CACHE,compact_features,fit_split
from tabnet_attention_transfer_n26 import collect,RESULT
from tabnet_history_test_n26 import historical_indices
from target_mix_expert_routing_n26 import ticker
from worst_booking_group_ensemble_n26 import first_stages


def compare(scores,base,y,scope):
    direct,selected,sham=(scores[:,j]>=.5 for j in range(3))
    base_corrected=(direct!=base)&(selected==base)
    return {name:ticker(scope,base!=y,act) for name,act in dict(field_required=base_corrected,
              required_only_with_actual_choice=base_corrected&(sham!=base)).items()}


def run():
    raw,y,g,f,known,versions=paired()
    frame=compact_features(raw)
    frame['OrderSize']=raw.groupby(g).PassengerId.transform('size')
    panels=sources()[-1]
    v=panels['historical']
    h=v['ids']
    hist=np.full((len(y),3),np.nan)
    for k in range(3):
        t,q,_=historical_indices(raw,y,g,k)
        pred,selected,sham=fit_split(frame,y,g,t,q,k)
        file=CACHE/f'historical_fold{k}.npz'
        np.savez_compressed(file,t=t,q=q,passenger=raw.PassengerId.iloc[q].to_numpy(str),training_y=y[t],
                            p=pred,selected=selected,sham=sham)
        with np.load(file) as z:
            for n,x in [('q',q),('t',t),('training_y',y[t]),('passenger',raw.PassengerId.iloc[q])]:
                np.testing.assert_array_equal(z[n],x)
            hist[q]=z['p']
        print('historical',k,flush=True)
    # Check the extracted training function against a complete frozen eight-fold packet.
    with np.load(CACHE/'scores.npz') as z:
        np.testing.assert_array_equal(z['ids'],raw.PassengerId)
        np.testing.assert_array_equal(z['fold'],f)
        full=z['p']
        t,q=np.flatnonzero(f!=0),np.flatnonzero(f==0)
        check,which,sham=fit_split(frame,y,g,t,q,0)
        np.testing.assert_allclose(full[q],check,rtol=0,atol=1.e-8)
        np.testing.assert_array_equal(which,z['selected'][q])
        np.testing.assert_array_equal(sham,z['sham'][q])
    bank=panel_predictions(v,'historical')
    first=first_stages(raw,panels,versions)['historical']
    with np.load(DECK/'deck_estimates.npz') as z:
        enabled=z['eligible']
    common=np.zeros(len(h),bool)
    base=np.zeros(len(h),bool)
    for key,_,_,_,iq in splits(panels,g):
        if not key.startswith('historical_') or key=='historical_to_new':
            continue
        with np.load(REVIEW/f'pred_{key}.npz') as z:
            np.testing.assert_array_equal(z['q'],h[iq])
            risk=np.column_stack([z['identities_error'],z['identities_outcome']])
        anchor,review,fare,now=stages(v,bank,iq,enabled[h[iq]],risk)
        np.testing.assert_array_equal(anchor,first[iq,-1])
        cold=~np.isin(g[h[iq]],g[h[v['fold']!=v['fold'][iq[0]]]])
        common[iq]=cold&(first[iq]==now[:,None]).all(1)&(review==now)&(fare==now)
        base[iq]=now
    report={'historical_cold_before_tabnet':compare(hist[h],base,y[h],common)}
    runs=collect()[0]
    for tag,items in runs.items():
        q,same,now=(np.concatenate([i[j] for i in items]) for j in range(3))
        tabnet=np.concatenate([i[3]['actual_and_scores'] for i in items])
        report[tag]=compare(full[q],now,y[q],same&~tabnet)
    (RESULT.parent/'repeat_field_evidence.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report,flush=True)


if __name__=='__main__':
    with threadpool_limits(1):
        run()
