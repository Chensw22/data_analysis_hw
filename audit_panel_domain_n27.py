"""Separate raw-manifest drift from differences between fold-trained and test model outputs."""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from booking_mix_confidence_transfer_n26 import partition
from exaone_validation_data import compact_features, CATEGORICAL
from repeated_archive_misses_n26 import ART
from full_roster_booking_validation_n23 import ROOT
from unanimous_error_reach_n27 import inputs

NAMES=['history','mitra','tabicl','nori','foundation_median','tabdpt','tabnet']


def run():
    raw,frame,y,g,f,first,random,ordered,wide,_=inputs()
    n=len(y)
    domain=np.arange(len(frame))>=n
    groups=frame.PassengerId.str.split('_').str[0].to_numpy()
    size=pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy()
    split=partition(groups,'score_density_estimation',4)
    masks={'all':np.ones(len(frame),bool)}
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        masks['known']=np.r_[z['known'],np.ones(domain.sum(),bool)]
    raw_design=pd.get_dummies(compact_features(frame),columns=CATEGORICAL,dtype=float).to_numpy()
    base=np.column_stack([size,frame.CryoSleep.eq(True),first])
    views={'raw':raw_design,'booking_sleep_decision':base,'raw_with_booking':np.column_stack([raw_design,size]),
           'scores':wide,'fundamentals':wide[:,1:4],'without_history':wide[:,1:],'without_tabnet':wide[:,:-1]}
    views['scores_same_float32']=wide.astype(np.float32).astype(float)
    views['mitra_same_float32']=wide[:,1:2].astype(np.float32).astype(float)
    views['scores_three_decimals']=np.round(wide,3)
    views['scores_no_mitra_tabnet']=wide[:,[0,2,3,5]]
    views.update({k:v[:,None] for k,v in zip(NAMES,wide.T)})
    views.update({k+'_with_decision':np.column_stack([v,first]) for k,v in zip(NAMES,wide.T)})
    views['scores_and_meta']=np.column_stack([wide, np.std(wide,axis=1), np.mean(wide>=.5,axis=1),base])
    report={}
    for name,x in views.items():
        pred=np.zeros(len(frame))
        for j in range(4):
            t=split!=j
            model=HistGradientBoostingClassifier(max_iter=90,max_leaf_nodes=8,l2_regularization=10,early_stopping=False,random_state=9)
            pred[~t]=model.fit(x[t],domain[t]).predict_proba(x[~t])[:,1]
        report[name]={scope:float(roc_auc_score(domain[mask],pred[mask])) for scope,mask in masks.items()}
    (ROOT/'reports/nov23_matched_learning_curve/panel_domain_diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(sorted(report.items(),key=lambda r:-r[1]['all']))


if __name__=='__main__':
    with threadpool_limits(1):
        run()
