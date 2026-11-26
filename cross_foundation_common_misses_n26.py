"""Measure complementary evidence from nominal feature geometry and cohort-specific support on the common errors."""
import json
import numpy as np
import pandas as pd

from exhaustive_family_quorums_n26 import summary
from latent_passenger_difficulty_n26 import OUT as PANEL,ROOT
from matched_state_context_n26 import OUT as CONTEXT
from nori_nominal_geometry_n26 import OUT as NOMINAL
from nori_fare_party_n26 import FEES


def run():
    with np.load(PANEL/'panel.npz') as z:
        ids,known,common,fresh,y,f,base=[z[k] for k in ('passenger','known','common','fresh','y','fold','anchor')]
        assert z['names'][-2]=='TabICL fare'
        ordinary_tab=z['scores'][:,-2]>=.5
    n=len(y)
    with np.load(CONTEXT/'aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        np.testing.assert_array_equal(z['y'],y)
        spec=z['specific']
        available=np.isfinite(spec)
        support=available&((spec>=.5)!=base)
        ordinary=available&((z['matched']>=.5)!=base)
    with np.load(NOMINAL/'aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        np.testing.assert_array_equal(z['fold'],f)
        geometry=((z['indicator']>=.5)!=base)&((z['rotated']>=.5)!=base)
        original=(z['ordinal']>=.5)!=base
    with np.load(ROOT/'artifacts/nov26_retained_order_mechanisms/aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids[:n])
        np.testing.assert_array_equal(z['reference161'],base[:n])
        group=z['eligible']
        unanimous=group&((z['pairs']>=.5)==base[:n,None,None]).all(axis=(1,2))&((z['original_panel']>=.5)==base[:n,None]).all(axis=1)
        assert unanimous.sum()==3861 and ((base[:n]!=y)&unanimous).sum()==266
    raw=pd.read_csv(ROOT/'data/raw/train.csv')
    np.testing.assert_array_equal(raw.PassengerId,ids[:n])
    paid=raw[FEES].gt(0).any(axis=1).to_numpy()
    scopes=dict(all=known,common=common,retained=group,unanimous_31=unanimous,fresh=known&fresh,
                awake_paid_negative=common&raw.CryoSleep.eq(False).to_numpy()&paid&~base[:n],
                sleep_no_positive=common&raw.CryoSleep.eq(True).to_numpy()&~paid&base[:n])
    rules=dict(both=support&geometry, both_new=support&geometry&~ordinary&~original&(ordinary_tab==base),
               support_only=support&~geometry, geometry_only=geometry&~support,
               controls=ordinary&original, incumbent_foundations=(ordinary_tab!=base)&original)
    result={k:dict(test_changes=int(a[n:].sum()),
                   scopes={s:summary(a[:n],base[:n]!=y,m,f) for s,m in scopes.items()}) for k,a in rules.items()}
    path=ROOT/'reports/nov23_matched_learning_curve/cross_foundation_common_misses.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print([(k,row['scopes']['all']['net'],row['scopes']['unanimous_31']['fixed'],row['scopes']['unanimous_31']['harmed']) for k,row in result.items()])


if __name__=='__main__':
    run()
