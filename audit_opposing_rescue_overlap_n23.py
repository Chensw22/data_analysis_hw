"""Count new stable rescues of the opposite-label-pair classifier and their observable controls."""
import numpy as np

from audit_dependence_scores_n23 import match, intervals
from broad_bill_neighbor_decision_n23 import opinions
from broad_geographic_transfer_n23 import BILLS, load
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from broad_tree_field_dividends_n23 import hard_cohorts

REPORT=ROOT/'reports/nov23_opposing_rescue_overlap'
KEYS=('matched_conditional','shuffled_conditional','matched','binary')


def describe(mask,frame,groups,state,confidence,vote):
    bills=frame[BILLS]
    sleep=frame.CryoSleep.eq('True').to_numpy()
    def n(flag):return int((mask&flag).sum())
    def quartiles(column):
        x=np.asarray(column)[mask]
        if not np.isfinite(x).any():return []
        return np.nanquantile(x,[.25,.5,.75]).tolist()
    return dict(passengers=int(mask.sum()),bookings=int(np.unique(groups[mask]).size),solo=n(frame.BookingSize==1),
                predicted_transported=n(vote),paid_spa_room_vr=n(state==2),paid_only_store_food=n(state==1),
                no_positive_spending=n(state==0),any_missing_bill=n(bills.isna().any(axis=1)),sleep_recorded_true=n(sleep),
                home_planet={k:n(frame.HomePlanet.eq(k)) for k in frame.HomePlanet.unique()},
                age_quartiles=quartiles(frame.Age),observed_spending_quartiles=quartiles(bills.sum(axis=1,min_count=1)),
                confidence_quartiles=quartiles(confidence))


def main():
    row,y,frame,g,_=load()
    _,_,hard,correct=hard_cohorts(row,y)
    with np.load(ART/'nov23_release_vote_information/panel.npz') as z:
        np.testing.assert_array_equal(row,z['row'])
        np.testing.assert_array_equal(y,z['y'])
        vote=z['votes'][0,:,0]
    assert not (vote[hard]==y[hard]).any() and (vote[correct]==y[correct]).all()
    old,gen,risk,challenger,original=[],[],[],{},{}
    for seed in [*map(str,SEEDS),'v131']:
        fold,base,ref,bills,neighbor=opinions(seed,row,y,frame.BookingSize.to_numpy()>1)
        def load_verified(folder,keys):
            with np.load(ART/f'nov23_{folder}/{seed}.npz') as z:
                for k,v in [('row',row),('y',y),('fold',fold),('ref',ref),('base',base)]:
                    np.testing.assert_array_equal(z[k],v)
                return {k:z[k] for k in keys}
        result=load_verified('opposing_passengers',KEYS)
        if seed=='v131':
            original={k:(v>=.5)!=vote for k,v in result.items()}
            break
        selected=load_verified('abstaining_fields',['personal_first','bill_first','averaged_order'])
        history=np.column_stack([bills,neighbor,*selected.values()])
        old.append(((history>=.5)!=ref[:,None])&(ref==(base>=.5))[:,None])
        models=load_verified('dependence_anchors',['independent','single','double'])
        gen.append(np.column_stack([np.where(vote,1-p,p) for p in models.values()])>.5)
        risk.append(np.where(vote,base,1-base))
        for k,p in result.items(): challenger.setdefault(k,[]).append(p)
    old41=np.all(old,axis=0).any(1)
    old44=old41|np.all(gen,axis=0).any(1)
    assert np.sum(hard&old41)==41 and np.sum(hard&old44)==44
    assert all(not np.any(np.asarray(p)[:,hard|correct]==.5) for p in challenger.values())
    stable={k:np.all((np.asarray(p)>=.5)!=vote[None,:],axis=0) for k,p in challenger.items()}
    contested=stable['matched_conditional']
    assert (hard&contested).sum()==34
    states=np.where(frame[['Spa','VRDeck','RoomService']].gt(0).any(axis=1),2,
                    np.where(frame[BILLS].gt(0).any(axis=1),1,0))
    counts={}
    for name,flag in {'old41':old41,'old44':old44,**stable,
                      'matched_and_prior':contested&old44,'matched_only':contested&~old44,
                      'prior_only':old44&~contested,'union':contested|old44,
                      'matched_and_shuffled':contested&stable['shuffled_conditional'],
                      'prior_new_matched_and_shuffled':~old44&contested&stable['shuffled_conditional'],
                      'prior_new_matched_only_real':~old44&contested&~stable['shuffled_conditional']}.items():
        counts[name]=dict(rescues=int((flag&hard).sum()),harms=int((flag&correct).sum()))
    confidence=np.mean(risk,0)
    summary={}
    for name,m in [('newly_rescued',hard&contested&~old44),('newly_harmed',correct&contested&~old44)]:
        features=describe(m,frame,g,states,confidence,vote)
        features['again_on_original_split']=int((m&original['matched_conditional']).sum())
        features['by_spending_and_prior_vote']={f'{i}_{int(v)}':int((m&(states==i)&(vote==v)).sum()) for i in range(3) for v in (False,True)}
        summary[name]=features
    def matched_summary(stratum):
        a,b,meta=match(hard&~old44,correct&~old44,stratum,g,np.asarray(risk))
        ci=intervals(a,b,g)
        wins={}
        for key in KEYS:
            p=np.stack(challenger[key])
            against=np.where(vote,1-p,p)
            delta=against[:,a]-against[:,b]
            win=((delta>0)+.5*(delta==0)).mean(0)
            wins[key]=win
            meta[key]=dict(concordance=float(win.mean()),ci=ci(win))
        for key in KEYS[1:]:
            diff=wins['matched_conditional']-wins[key]
            meta['gain_over_'+key]=dict(difference=float(diff.mean()),ci=ci(diff))
        meta['paired_stable_disagreement']=dict(case_only=int((contested[a]&~contested[b]).sum()),
                                                control_only=int((contested[b]&~contested[a]).sum()),
                                                both=int((contested[a]&contested[b]).sum()))
        return meta
    _,planet=np.unique(frame.HomePlanet,return_inverse=True)
    raw=matched_summary(vote)
    by_planet=matched_summary(2*planet+vote.astype(int))
    by_spend_planet=matched_summary(4*planet+2*(states>0)+vote.astype(int))
    write(REPORT/'overlap.json',dict(counts=counts,novel_groups=summary,matched_outside_prior=raw,
                                    matched_same_planet_outside_prior=by_planet,
                                    matched_same_planet_and_spending_outside_prior=by_spend_planet))
    print(by_spend_planet,flush=True)


if __name__=='__main__':main()
