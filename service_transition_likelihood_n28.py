"""Estimate two transitions from a registration-only latent state when services are used."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from conditional_shortcut_boost_n28 import experiment_inputs
from price_implied_deck_n26 import FEES
from repeated_archive_misses_n26 import ART, count


def design(frame):
    cabin = frame.Cabin.str.split('/', expand=True)
    group = frame.PassengerId.str[:4]
    age = frame.Age.fillna(28).to_numpy()
    category = pd.DataFrame({
        'planet': frame.HomePlanet, 'destination': frame.Destination, 'sleep': frame.CryoSleep,
        'vip': frame.VIP, 'deck': cabin[0], 'side': cabin[2],
        'party': group.map(group.value_counts()).clip(upper=5),
    }).astype('string').fillna('?')
    category['planet_sleep'] = category.planet + '/' + category.sleep
    category['deck_sleep'] = category.deck + '/' + category.sleep
    reg = np.column_stack([np.ones(len(frame)), pd.get_dummies(category).to_numpy(float),
                           frame.Age.isna(), age/30, np.maximum(0, age[:, None]-[12, 18, 40])/30])
    amount = frame[FEES].fillna(0).to_numpy()
    x = np.log1p(amount)
    tiers = np.column_stack([x/5, amount>0, np.maximum(x-5, 0)/2, np.maximum(x-7, 0),
                             frame[FEES].isna().to_numpy()])
    flows = np.column_stack([np.ones(len(frame)), tiers, *[
        tiers*(category.planet.to_numpy()==name)[:, None] for name in ['Earth','Europa','Mars']]])
    return reg, flows, np.any(amount>0, axis=1)


def transition(reg, flows, active, train, query, y):
    x, b, event = reg[train], flows[train], active[train]
    width, terms = x.shape[1], b.shape[1]
    start = np.zeros(width+terms*2)
    start[[width, width+terms]] = -2.
    def predict(params, context, actions, paid):
        v, u, w = np.split(params, [width, width+terms])
        initial = expit(context@v)
        gain = expit(actions@u)*paid
        loss = expit(actions@w)*paid
        chance = (initial*(1-loss)+(1-initial)*gain).clip(1.e-10, 1-1.e-10)
        return initial, gain, loss, chance
    def objective(params):
        initial, gain, loss, chance = predict(params, x, b, event)
        residual = (chance-y[train])/(chance*(1-chance))
        likelihood = -np.where(y[train], np.log(chance), np.log1p(-chance)).sum()
        gradient = np.r_[x.T@(residual*(1-gain-loss)*initial*(1-initial)),
                         b.T@(residual*(1-initial)*gain*(1-gain)),
                         b.T@(-residual*initial*loss*(1-loss))]
        delta = params-start
        return likelihood+np.dot(delta,delta), gradient+2*delta
    fit = minimize(objective, start, method='L-BFGS-B', jac=True,
                   options={'maxiter':650,'ftol':1.e-9})
    if not fit.success:
        raise RuntimeError(fit.message)
    return logit(predict(fit.x, reg[query], flows[query], active[query])[3])


def run():
    frame,y,fold,parent,g,_,_,known,_,wrong,masks,ordered,anchor = experiment_inputs()
    n=len(y)
    reg, flows, paid = design(frame)
    additive=np.column_stack([reg,flows*paid[:,None]])
    report, arrays = {}, {}
    for tag, partition in [('random',fold),('ordered',ordered)]:
        scores=np.empty((len(frame),2))
        for k in range(9):
            train=np.flatnonzero(partition!=k)
            query=np.flatnonzero(partition==k) if k<8 else np.arange(n,len(frame))
            assert not np.isin(g[train],g[query]).any()
            if k==8 and tag=='ordered':
                scores[query]=arrays['random'][query]
                continue
            plain=LogisticRegression(fit_intercept=False,C=.5,solver='newton-cholesky',tol=1.e-7)
            scores[query,0]=plain.fit(additive[train],y[train]).decision_function(additive[query])
            scores[query,1]=transition(reg,flows,paid,train,query,y)
        old,new=scores.T
        opp=(new>=0)!=parent
        rules={'direct':opp,'distinct':opp&((old>=0)==parent),
               'increment':opp&((anchor+new-old>=0)!=parent)}
        report[tag]={'native':np.mean((scores[:n]>=0)==y[:,None],axis=0).tolist(),'rules':{}}
        for key,flag in rules.items():
            report[tag]['rules'][key]=dict(scopes={s:count(flag[:n],m,wrong) for s,m in masks.items()},
                accuracy=float(((parent[:n]^flag[:n])[known]==y[known]).mean()),test=int(flag[n:].sum()))
        arrays[tag]=scores
        print(tag,report[tag],flush=True)
    dest=ART/'nov28_service_transition_likelihood'
    dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'aligned.npz',passenger=frame.PassengerId.to_numpy(str),y=y,fold=fold,**arrays)
    path=Path(__file__).resolve().parent/'reports/nov23_matched_learning_curve/service_transition_likelihood.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
