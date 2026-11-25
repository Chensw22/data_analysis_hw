"""Test one name-free spending/location challenger against two anchors on identical solo passengers."""
import hashlib
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import BILLS
from export_budget_quarter_n23 import source
from export_selected_tree_pair_n25 import NINE
from full_roster_adaptive_bill_prior_n23 import EVIDENCE
from full_roster_booking_validation_n23 import FAMILIES
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT, data, banks
from full_roster_selected_tree_replay_n24 import prepared_vote, original_prediction
from kaggle_workflow import validate_submission


def basis(values, knots):
    x=np.asarray(values,float)
    knots=np.asarray(knots,float)
    p=np.interp(np.nan_to_num(x,nan=knots[0]),knots,np.arange(len(knots)))
    h=(1-abs(p[:,None]-np.arange(len(knots)))).clip(0,1)
    h[~np.isfinite(x)]=0
    return np.c_[h, ~np.isfinite(x)]


def design(frame):
    cat=lambda x: x.astype('string').fillna('?')
    oh=lambda x: pd.get_dummies(x,dtype=float).to_numpy()
    cab=frame.Cabin.str.split('/',expand=True)
    planet,deck,side=map(cat,[frame.HomePlanet,cab[0],cab[2]])
    cryo=cat(frame.CryoSleep)
    party=frame.PassengerId.str.split('_').str[0]
    alone=~party.duplicated(keep=False)
    bills=frame[BILLS].to_numpy(float)
    age=basis(frame.Age,[0,6,12,16,22,30,45,65,90])
    position=basis(pd.to_numeric(cab[1]),np.arange(0,2301,125))
    pos=np.nan_to_num(bills)>0
    total=np.nansum(bills,axis=1)
    # Common physical units across services; bases are fixed independently of the targets.
    knots=np.log1p([0,1,20,100,300,600,900,1500,3000,8000,30000])
    amounts=np.column_stack([basis(np.log1p(s),knots) for s in [*bills.T,total]])
    patterns=oh(pd.Series([''.join(map(str,p.astype(int))) for p in pos]))
    ctx=np.c_[oh(planet),oh(cryo),alone,oh(deck)]
    registration=oh(planet+'/'+deck+'/'+side)
    cross=lambda a,b:(a[:,:,None]*b[:,None,:]).reshape(len(frame),-1)
    features=np.column_stack([ctx,registration,oh(cat(frame.Destination)),oh(cat(frame.VIP)),
        age,position,amounts,patterns,cross(age,oh(planet)),cross(position,oh(deck+'/'+side)),
        cross(amounts,ctx),cross(patterns,oh(planet)),cross(age,oh(cryo))])
    return sparse.csr_matrix(features),alone.to_numpy()


def result(ref,mask,y,group,fold,new):
    gain=np.where(mask,np.where(ref!=y,1,-1),0)
    tally=lambda s: [int((gain[s]==v).sum()) for v in (1,-1)]
    return dict(fixed_harmed=tally(np.ones(len(y),bool)),new_order_fixed_harmed=tally(new),
                fold_net=[int(gain[fold==f].sum()) for f in np.unique(fold)])


def run():
    train,y,g,f=data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=lambda c:c!='Name')
    x,solo=design(pd.concat([train,test],ignore_index=True))
    tg=test.PassengerId.str.split('_').str[0].to_numpy()
    assert not np.intersect1d(g,tg).size
    tf=np.array([int.from_bytes(hashlib.sha256(('12551/'+s).encode()).digest()[:8],'big')%8 for s in tg])
    assert len(np.unique(tf))==len(np.unique(f))==8
    n=len(y); p=np.full(len(solo),np.nan); solo_fit=np.full(len(solo),np.nan)
    for j in np.unique(f):
        t=np.flatnonzero(f!=j); q=np.r_[np.flatnonzero(f==j),n+np.flatnonzero(tf==j)]
        model=LogisticRegression(C=.6,solver='lbfgs',max_iter=650).fit(x[t],y[t])
        p[q]=model.predict_proba(x[q])[:,1]
        only=t[solo[t]]
        specialist=LogisticRegression(C=.6,solver='lbfgs',max_iter=650).fit(x[only],y[only])
        solo_fit[q]=specialist.predict_proba(x[q])[:,1]
    assert np.isfinite(p).all()
    names,_,full=banks(train,y,f)
    _,oy,og,row,of,_,historical,_,_,_,_,old,_=aligned()
    rep=full[:,names.index('mitra96')]
    newbase=rep>=.5
    nine=full[:,[names.index(k) for k in (*FAMILIES,'mitra96',*EVIDENCE)]]
    tree=original_prediction(np.arange(n),f,g,y,'full')
    historical_tree=original_prediction(row,of,g,y,'historical')
    # Preserve even possible historical nine-vote edits: these three necessary votes
    # cover every all-nine agreement. Exact historical nine-vote decisions may be unavailable.
    old_lock=(prepared_vote(historical_tree,historical,old)!=historical)
    new_lock=prepared_vote(tree,newbase,nine)!=newbase
    true_test_anchor,_,_=source('proposal_online_first_counts15',test)
    _,old_test,_=source('proposal_v131_party_consensus',test)
    _,best,_=source(NINE,test)
    newgroups=~np.isin(g,og)
    folder=ROOT/'artifacts/nov25_solo_conditional_splines'
    folder.mkdir(exist_ok=True)
    with np.load(ROOT/'artifacts/nov25_production_anchor_cpu/certified.npz') as certified:
        for k,value in [('row',np.arange(n)),('fold',f),('y',y),('passenger',train.PassengerId.to_numpy(str)),('v96',rep)]:
            np.testing.assert_array_equal(certified[k],value)
        exact=certified['certified_v131_equals_v96']
    assert exact.sum()==4111
    report={}; outputs={}
    for key,w,rival in [('standalone',1.,p),('equal_log_odds',.5,p),('solo_trained',1.,solo_fit),('solo_trained_equal_odds',.5,solo_fit)]:
        sides={}
        for tag,ix,ref,anchor,lock,ff,actual,grp,new in [
            ('solo_restored',np.arange(n),newbase,rep,new_lock,f,y,g,newgroups),
            ('historical_exact_v131',row,historical,old[:,0],old_lock,of,oy,og,newgroups[row]),
            ('test',np.arange(n,n+len(test)),best,true_test_anchor,best!=old_test,tf,None,tg,np.ones(len(test),bool))]:
            vote=w*logit(rival[ix].clip(.001,.999))+(1-w)*logit(anchor.clip(.001,.999))>=0
            eligible=solo[ix] & ~lock & (ref==(anchor>=.5))
            mask=eligible & (ref!=vote)
            if actual is None:
                sides[tag]=dict(solos=int(solo[ix].sum()),edits=int(mask.sum()),preserved_nine=int((best!=old_test).sum()))
                outputs[key]=ref^mask
            else:
                sides[tag]=result(ref,mask,actual,grp,ff,new)
                sides[tag].update(solos=int(solo[ix].sum()),baseline_solo_errors=int((solo[ix]&(ref!=actual)).sum()))
                if tag=='solo_restored':
                    fixed=(mask&(ref!=actual))[row]; harmed=(mask&(ref==actual))[row]
                    with_nine=prepared_vote(tree,newbase,nine)
                    assert np.all(~mask[new_lock])
                    certified_metrics=result(with_nine[exact],mask[exact],actual[exact],grp[exact],ff[exact],new[exact])
                    certified_metrics.update(rows=int(exact.sum()),solos=int((exact&solo[:n]).sum()),
                        baseline_errors=int((exact&(actual!=with_nine)).sum()),nine_edits=int((new_lock&exact).sum()),
                        solo_errors=int((exact&solo[:n]&(actual!=with_nine)).sum()))
                    sides['certified_v131_plus_nine']=certified_metrics
                else: sides['both']=dict(fixed=int((fixed&mask&(ref!=actual)).sum()),harmed=int((harmed&mask&(ref==actual)).sum()))
        report[key]=sides
    selected='equal_log_odds'
    csv=folder/'submission.csv'
    pd.DataFrame(dict(PassengerId=test.PassengerId,Transported=outputs[selected])).to_csv(csv,index=False)
    report['selected']=dict(method=selected,**validate_submission(csv,ROOT/'data/raw/test.csv'))
    report['validation']=dict(train=len(y),unseen_orders_passengers=int(newgroups.sum()),retrospective_rows=len(row),
                              historical_possible_nine_protected=int((old_lock&solo[row]).sum()),
                              eight_nine_protected=int((new_lock&solo[:n]).sum()))
    np.savez_compressed(folder/'scores.npz',passenger=np.r_[train.PassengerId.to_numpy(str),test.PassengerId.to_numpy(str)],folds=np.r_[f,tf],p=p,solo_fit=solo_fit)
    out_paths=[folder/'results.json',ROOT/'reports/nov23_matched_learning_curve/solo_conditional_splines.json']
    for path in out_paths:
        path.write_text(json.dumps(report,indent=2)+'\n')
    assert out_paths[0].read_bytes()==out_paths[1].read_bytes()
    print(report,flush=True)


if __name__=='__main__':
    with threadpool_limits(2):run()
