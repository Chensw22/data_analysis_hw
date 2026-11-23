"""Use changes under plausible alternate field blocks to predict full-population errors."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import strata
from broad_booking_canonical_n23 import REPORT
from broad_boosting_path_n23 import calibrated
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART, SEEDS, historical, write
from broad_release_miss_coverage_n23 import summarize
from peer_attribute_surprise_n23 import encode
from released_correction_audit_n23 import development


def run_fold(blocks,y,g,t,q):
    assert not np.intersect1d(g[t],g[q]).size
    x = np.column_stack(blocks)
    groups = np.split(np.arange(x.shape[1]),np.cumsum([b.shape[1] for b in blocks])[:-1])
    model = HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=17,min_samples_leaf=25,
                                          learning_rate=.075,l2_regularization=2,early_stopping=False,
                                          random_state=277).fit(x[t],y[t])
    ordinary = model.predict_proba(x[q])[:,1]
    arms = {'local':[], 'random':[]}
    for cols in groups:
        remaining = np.setdiff1d(np.arange(x.shape[1]),cols)
        rng = np.random.default_rng(913)
        # Retain the measured values of every other block. Break distance ties randomly.
        perm = rng.permutation(t)
        order = np.argsort(cdist(x[q][:,remaining],x[perm][:,remaining],'sqeuclidean'),axis=1,kind='stable')[:,:40]
        positions = np.argsort(rng.random(order.shape),axis=1)[:,:8]
        near = perm[np.take_along_axis(order, positions, axis=1)]
        anywhere = rng.choice(t,size=near.shape)
        for name, donors in [('local',near),('random',anywhere)]:
            copies = np.repeat(x[q],donors.shape[1],axis=0)
            copies[:,cols] = x[donors.ravel()][:,cols]
            p = model.predict_proba(copies)[:,1].reshape(donors.shape)
            arms[name].append(p.mean(1))
    return ordinary,{k:np.column_stack(v) for k,v in arms.items()}


def main():
    row,y,frame,g,_ = load()
    ids,labels,masks = historical()
    np.testing.assert_array_equal(row,ids)
    np.testing.assert_array_equal(y,labels)
    _, releases,_ = development()
    blocks = encode(frame)
    # Size and cabin form one observable location/booking block.
    blocks[2] = np.column_stack([blocks[2],np.log(frame.BookingSize)])
    state = strata(frame)
    numeric = ['Age','CabinNumber','Spa','VRDeck','RoomService','ShoppingMall','FoodCourt']
    masks = {**masks,'paid_transported':masks['repeat'] & y & (state==2),
             'zero_stayed':masks['repeat'] & ~y & (state==0),
             'complete_repeat':masks['repeat'] & frame[numeric].notna().all(axis=1).to_numpy()}
    out = ART/'nov23_plausible_replacements'
    out.mkdir(exist_ok=True)
    history,refs,plans = {},[],{}
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as z:
            for k,v in [('row',row),('y',y)]:np.testing.assert_array_equal(z[k],v)
            fold,ref,base=z['fold'],z['ref'],z['base']
        np.testing.assert_array_equal(ref,releases[131] if seed=='v131' else base>=.5)
        ordinary=np.zeros(len(y))
        arms={k:np.zeros((len(y),len(blocks))) for k in ['local','random']}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            ordinary[q],values=run_fold(blocks,y,g,t,q)
            for k,v in values.items():arms[k][q]=v
        def odds(p):
            return logit(p.clip(.002,.998))
        straight=np.c_[odds(base),odds(ordinary)]
        votes={'ordinary':ordinary>=.5}
        for mode in ['baseline','local','random']:
            x=straight
            if mode in arms:
                shift=odds(arms[mode])-odds(ordinary)[:,None]
                x=np.c_[straight,shift,shift*np.where(ref,1.,-1.)[:,None]]
                votes[mode+'_average']=arms[mode].mean(1)>=.5
                votes[mode+'_bill_replaced']=arms[mode][:,-1]>=.5
            vote=ref.copy()
            for f in np.unique(fold):
                ix=np.flatnonzero((fold==f)&(ref==(base>=.5)))
                for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ix,groups=g[ix]):
                    t,q=ix[a],ix[b]
                    vote[q]=calibrated(x[t],x[q],y,t)>=.5
            votes['combined_'+mode]=vote
        stats={}
        anchor=ref==y
        if seed!='v131':refs.append(anchor)
        for k,v in votes.items():
            hit=v==y
            stats[k]=dict(accuracy=float(hit.mean()),fixed=int((hit&~anchor).sum()),harmed=int((~hit&anchor).sum()))
            if seed!='v131':history.setdefault(k,[]).append(hit)
        plans[seed]=stats
        np.savez_compressed(out/f'{seed}.npz',row=row,y=y,base=base,ref=ref,fold=fold,ordinary=ordinary,**arms)
        print(seed,stats['combined_local'],flush=True)
    write(REPORT/'replacements.json',dict(plans=plans,grouped={
        k:summarize(np.stack(h),np.stack(refs),masks) for k,h in history.items()}))


if __name__ == '__main__':
    with threadpool_limits(2):
        main()
