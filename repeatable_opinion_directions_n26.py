"""Use paired forecasts for the same passengers to separate reproducible disputes from training-set sensitivity."""
import json

import numpy as np
from scipy.linalg import eigh
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits

from adjudicate_amount_dissent_n25 import counts
from disjoint_tree_fields_n25 import DEST
from family_identity_transfer_n26 import fit,panels_and_reference
from grouped_peer_release_replay_n25 import edit_effect_bounds


def residual_view(p):
    z=logit(p.clip(.005,.995))
    center=np.median(z,axis=1)
    return z-center[:,None],center


def paired_basis(a,b,mode):
    center=(a.mean(0)+b.mean(0))/2
    mean=(a+b)/2-center
    diff=(a-b)/2
    noise=(diff.T@diff)/len(diff)
    signal=(mean.T@mean)/len(mean)
    if mode=='variance':
        _,axes=eigh(signal)
    else:
        regularizer=max(.01,np.trace(noise)/len(noise)*.35)
        _,axes=eigh(signal,noise+regularizer*np.eye(len(noise)))
    axes=axes[:,-8:]
    scale=np.std(np.vstack([a,b])@axes,axis=0).clip(.12)
    return center,axes/scale


def design(view,center,anchor,bias,axes):
    projection=(view-bias)@axes
    return np.column_stack([np.ones(len(view)),anchor,center-anchor,projection])


def run():
    y,g,f,rows,_,p,h,native,old,reference=panels_and_reference()
    full,mid=residual_view(p)
    hist,old_mid=residual_view(h)
    anchor=logit(native.clip(.002,.998))
    old_anchor=logit(old.clip(.002,.998))
    known=reference[0]>=0
    unseen=~np.isin(g,g[rows])
    minor=(p>=.5).sum(1)
    minor=np.minimum(minor,68-minor)
    results={}
    for k in range(8):
        # Every current-model probability used for training or evaluating the correction
        # comes from a common outer model that omitted the whole fold. The old archive
        # never had access to labels from the query bookings.
        index=np.flatnonzero(f[rows]==k)
        t=rows[index]
        q=np.flatnonzero(unseen&(f==k))
        assert not np.isin(g[q],g[t]).any() and not np.isin(g[q],g[rows]).any()
        a,b=full[t],hist[index]
        for mode in ('confidence','variance','repeatable','scrambled'):
            paired=b.copy()
            if mode=='scrambled':
                # Retain the broad transport tendency while breaking matching passenger identities.
                order=np.argsort(mid[t])
                rng=np.random.default_rng(418+k)
                for block in np.array_split(order,10):
                    paired[block]=b[rng.permutation(block)]
            bias,axes=paired_basis(a,paired,mode)
            if mode=='confidence':
                axes=axes[:,:0]
            xx=design(a,mid[t],anchor[t],bias,axes)
            zz=design(b,old_mid[index],old_anchor[index],bias,axes)
            test=design(full[q],mid[q],anchor[q],bias,axes)
            # Stack two views with weight one-half per matched passenger.
            # Doubling the data without changing the ridge prior would reduce shrinkage;
            # the sqrt(2) scaling gives the same penalty as one observation per passenger.
            beta=fit(np.vstack([xx,zz])/np.sqrt(2),np.r_[anchor[t],old_anchor[index]],np.tile(y[t],2))
            prediction=expit(anchor[q]+test@beta/np.sqrt(2))>=.5
            d,e=edit_effect_bounds(prediction,native[q]>=.5,*reference[:,q],y[q])
            np.testing.assert_array_equal(d[known[q]],e[known[q]])
            results.setdefault(mode,[]).append((q,d,e))
    report={}
    for mode,parts in results.items():
        q,d,e=(np.concatenate([part[i] for part in parts]) for i in range(3))
        np.testing.assert_array_equal(np.sort(q),np.flatnonzero(unseen))
        r={'all_net_bounds':[int(d.sum()),int(e.sum())]}
        for tag,mask in [('known',known[q]),('unanimous',known[q]&(minor[q]==0)),
                         ('four_plus',known[q]&(minor[q]>=4))]:
            r[tag]=counts(d,mask,f[q])
        report[mode]=r
        print(mode,r['all_net_bounds'],r['known']['fixed'],r['known']['harmed'],r['unanimous']['fixed'])
    DEST.with_name('repeatable_opinion_directions.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
