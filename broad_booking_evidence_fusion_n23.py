"""Combine party evidence and neighborhood evidence, with the same training protocol for controls."""
import numpy as np
from scipy.special import logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import strata
from broad_booking_canonical_n23 import REPORT
from broad_boosting_path_n23 import calibrated
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART, SEEDS, historical, write
from broad_release_miss_coverage_n23 import summarize


def odds(p):
    return logit(np.clip(p,.002,.998))


def cluster_interval(group, differences):
    # Repeated splits concern the same passengers. Resample whole bookings with all splits together.
    _, ids = np.unique(group, return_inverse=True)
    change = np.bincount(ids, weights=np.mean(differences,axis=0))
    size = np.bincount(ids)
    rng = np.random.default_rng(867)
    samples = []
    for _ in range(20):
        take = rng.integers(len(size),size=(100,len(size)))
        samples.extend(100*change[take].sum(1)/size[take].sum(1))
    return np.quantile(samples,[.025,.5,.975]).tolist()


def main():
    row,y,frame,g,_ = load()
    rows,labels,masks = historical()
    np.testing.assert_array_equal(row,rows)
    np.testing.assert_array_equal(y,labels)
    state = strata(frame)
    masks = {**masks,'paid_transported':masks['repeat'] & (state==2) & y,
             'zero_stayed':masks['repeat'] & (state==0) & ~y,
             'solo_repeat':masks['repeat'] & (frame.BookingSize == 1).to_numpy(),
             'never_paid_transported':masks['never'] & (state==2) & y}
    plans,history,refs = {},{},[]
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as cache:
            base,fold,ref = cache['base'],cache['fold'],cache['ref']
            np.testing.assert_array_equal(cache['row'],row)
            np.testing.assert_array_equal(cache['y'],y)
        def read(name, keys):
            with np.load(ART/f'nov23_{name}/{seed}.npz') as z:
                for k,v in [('y',y),('row',row),('ref',ref),('base',base),('fold',fold)]:
                    np.testing.assert_array_equal(z[k],v)
                return [z[k] for k in keys]
        tree, = read('class_cost_path',['logits'])
        icl, = read('tabicl_prior_context',['sides'])
        ordinary, real, fake = read('booking_coordinates',['individual','real','shuffled'])
        spherical, oriented, rotated = read('local_class_shapes',['spherical','oriented','rotated'])
        own,direct,ratio,sham = read('booking_size_likelihood',['own','direct','true','sham'])
        fixed = np.c_[odds(base),tree[:,1],icl[:,:2].mean(1),odds(ordinary),spherical,own,direct]
        actual = np.c_[odds(real),oriented,ratio]
        swapped = np.c_[odds(fake),rotated,sham]
        sign = np.where(ref,1.,-1.)[:,None]
        designs = dict(ordinary=fixed, real=np.c_[fixed,actual,actual*sign],
                       sham=np.c_[fixed,swapped,swapped*sign])
        outputs = {name:ref.copy() for name in designs}
        for f in np.unique(fold):
            ix = np.flatnonzero((fold == f) & (ref == (base >= .5)))
            for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ix,groups=g[ix]):
                t,q = ix[a],ix[b]
                assert not np.intersect1d(g[t],g[q]).size
                for name,x in designs.items():
                    outputs[name][q] = calibrated(x[t],x[q],y,t) >= .5
        if seed != 'v131':
            refs.append(ref == y)
        plans[seed] = {}
        for k,v in outputs.items():
            h = v == y
            plans[seed][k] = dict(accuracy=float(h.mean()),fixed=int((h & (ref != y)).sum()),harmed=int((~h & (ref == y)).sum()))
            if seed != 'v131':
                history.setdefault(k,[]).append(h)
    hits = {k:np.stack(v) for k,v in history.items()}
    coverage = {}
    for name,mask in masks.items():
        real,control = hits['real'].all(0),hits['ordinary'].all(0)
        coverage[name] = dict(total=int(mask.sum()),real_only=int((mask&real&~control).sum()),
                              ordinary_only=int((mask&control&~real).sum()),both=int((mask&control&real).sum()))
    uncertainty = {k:cluster_interval(g,hits['real'].astype(float)-v) for k,v in
                   [('ordinary',hits['ordinary']),('sham',hits['sham']),('incumbent',np.stack(refs))]}
    write(REPORT/'fusion.json',dict(plans=plans,grouped={k:summarize(h,np.stack(refs),masks) for k,h in hits.items()},
                                    coverage=coverage,paired_gain_pp=uncertainty))
    print(plans,uncertainty,flush=True)


if __name__ == '__main__':
    with threadpool_limits(2):
        main()
