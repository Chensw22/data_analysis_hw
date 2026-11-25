"""Measure reproducible NODE dissent for the same passenger under two whole-booking training protocols."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from distinct_challenger_error_reach_n26 import population
from node_countervote_transfer_n26 import scores
from solo_archive_recovery_n26 import ticker
from tabnet_attention_transfer_n26 import RESULT


def run():
    raw,y,g,f,parent,common,reach,_,_,wrong,_,_=population()
    bank=scores()
    sets=np.isfinite(bank['historical']).all(1)
    assert sets.sum()==6954 and np.isfinite(bank['full']).all()
    # The baseline and enrichment scopes are evaluated using the same frozen eight-fold releases.
    a=(bank['full'][:,0]>=.5)!=parent
    b=(bank['historical'][:,0]>=.5)!=parent
    old_a=(bank['full'][:,1]>=.5)!=parent
    old_b=(bank['historical'][:,1]>=.5)!=parent
    report={}
    for name,scope in dict(all_matched=common&sets,with_old_challenger=common&sets&(reach>0),
                           without_old_challenger=common&sets&(reach==0)).items():
        r={}
        for label,act in dict(full=a,historical=b,both=a&b,only_full=a&~b,
                              both_node_both_ordinary=a&b&old_a&old_b,
                              both_node_neither_ordinary=a&b&~old_a&~old_b).items():
            s=ticker(scope,wrong,act)
            s['net_by_fold']=[int((scope*act*np.where(wrong,1,-1)*(f==k)).sum()) for k in range(8)]
            r[label]=s
        report[name]=r
        print(name,{k:(s['fixed'],s['harmed']) for k,s in r.items()},flush=True)
    (RESULT.parent/'repeated_node_countervotes.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
