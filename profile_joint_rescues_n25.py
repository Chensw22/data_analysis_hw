"""Describe the true and false alarms among archive-unanimous cases challenged by the manifest network."""
import json
import numpy as np

from disjoint_tree_fields_n25 import ART,DEST
from full_roster_booking_validation_n23 import data
from full_roster_six_failure_archive_n24 import prediction_sets
from screen_joint_disagreements_n25 import collect

BILLS=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']


def run():
    raw,y,g,f=data()
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        rows,hf=z['rows'],z['fold']
    _,panels=prediction_sets(raw,y,g,f,rows,hf)
    original=np.column_stack(panels[0])>=.5
    ref=original[:,0]
    unanimous=(original==ref[:,None]).all(1)
    alternate=collect(np.arange(len(y)),f,y,g,'fold')>=.5
    any_dissent=(alternate!=ref[:,None]).any(1)
    wrong=ref!=y
    awake=raw.CryoSleep.eq(False).to_numpy()
    asleep=raw.CryoSleep.eq(True).to_numpy()
    n_paid=raw[BILLS].gt(0).sum(axis=1).to_numpy()
    solo=~raw.PassengerId.str.split('_').str[0].duplicated(keep=False).to_numpy()
    masks={'all':np.ones(len(y),bool),'predicted_false':~ref,'predicted_true':ref,
           'awake':awake,'asleep':asleep,'sleep_missing':~(awake|asleep),
           'solo':solo,'multi_person_order':~solo,'no_recorded_payment':n_paid==0,
           'one_service':n_paid==1,'two_or_more_services':n_paid>=2,
           'awake_solo_predicted_not_transported':awake&solo&~ref,
           'all_personal_fields_observed':raw.notna().all(axis=1).to_numpy()}
    result={}
    for name,mask in masks.items():
        def score(use):
            at=use&mask&unanimous
            failures=at&wrong
            fixes=failures&any_dissent
            false=at&~wrong&any_dissent
            return dict(total=int(at.sum()),errors=int(failures.sum()),reachable=int(fixes.sum()),false_alarms=int(false.sum()))
        result[name]={'all':score(np.ones(len(y),bool)),'new_order':score(~np.isin(g,g[rows]))}
    DEST.with_name('profile_joint_rescues.json').write_text(json.dumps(result,indent=2)+'\n')
    for name in masks:
        print(name,result[name]['all'],flush=True)


if __name__=='__main__':
    run()
