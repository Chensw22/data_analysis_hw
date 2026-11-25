"""Check repeated reference assignments and a smaller deployable panel under the fixed rule."""
import json

import numpy as np
from threadpoolctl import threadpool_limits

from conflict_case_control_banks_n26 import fit_banks,inputs_for_banks
from exhaustive_family_quorums_n26 import inputs,summary
from full_roster_booking_validation_n23 import ROOT,data

SEEDS = [214,607,1904]
REPORT = ROOT/'reports/nov23_matched_learning_curve/replicate_case_control_banks.json'


def run():
    raw,y,g,f = data()
    _,cur,_ = inputs()
    values,side,marks,edges,dist,regions = inputs_for_banks(raw,y,g,f)
    actual = cur['known']&(side==cur['parent'])
    scopes = {'all':actual,'common':actual&cur['common'],'new':actual&cur['new']}
    initial = json.loads((REPORT.parent/'conflict_case_control_banks.json').read_text())
    report = {}
    cache = ROOT/'artifacts/nov26_replicate_case_control_banks'
    cache.mkdir(exist_ok=True)
    for name,fields in [('all_signals',values),('mitra_and_pair_rankers',values[:,[0,6,7,8,9]])]:
        actions = {}
        probabilities = []
        supports = []
        for seed in SEEDS:
            raw_prob,covered = fit_banks(fields,side,marks,edges,dist,regions,g,f,seed)
            p = raw_prob.mean(1)
            supported = covered.all(1)
            act = (p[:,1]>.5)&supported
            actions[str(seed)] = act
            probabilities.append(p)
            supports.append(supported)
            print(name,seed,summary(act,marks,scopes['common'],f),flush=True)
        probabilities = np.stack(probabilities)
        supports = np.stack(supports)
        votes = np.array(list(actions.values())).sum(0)
        actions.update(majority=votes>=2,unanimous=votes==3,
                       average=(probabilities[:,:,1].mean(0)>.5)&(supports.sum(0)>=2))
        report[name] = {s:{key:summary(v,marks,mask,f) for key,v in actions.items()} for s,mask in scopes.items()}
        report[name]['pairwise_overlap'] = {str(a)+'_'+str(b):int((actions[str(a)]&actions[str(b)]&scopes['common']).sum())
                         for i,a in enumerate(SEEDS) for b in SEEDS[i+1:]}
        if name=='all_signals':
            for s,old in [('all','all'),('common','eleven_agree'),('new','new_orders')]:
                assert report[name][s]['214']==initial[old]['supported']
        np.savez_compressed(cache/(name+'.npz'),passenger=raw.PassengerId.to_numpy(str),truth=y,fold=f,
                            side=side,seeds=SEEDS,probabilities=probabilities,supports=supports)
    report['scope'] = ('The first split reproduces the original result. Two additional whole-order splits '
                       'rotate both independently labelled reference orders and calibration orders. The reduced '
                       'model uses only an adapted Mitra probability and four raw-feature point/ranking models.')
    REPORT.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
