"""Fit exact local Laplace kernels in regions split using gradients on unseen training mistakes."""
import numpy as np
from scipy.linalg import eigh, solve
from threadpoolctl import threadpool_limits

from full_roster_recursive_metric_n24 import derivative, kernel, learned_metric, run

MODES = ('random', 'all_gradients', 'mistake_gradients', 'matched_random_gradients')


def bandwidth(x, metric, rng):
    pair = rng.integers(len(x),size=(2,2048))
    diff = x[pair[0]]-x[pair[1]]
    return max(float(np.median(np.sqrt(np.maximum(((diff@metric)*diff).sum(1),0)))),.01)


def exact(x,y,metric,rng):
    width = bandwidth(x,metric,rng)
    k = kernel(x,x,metric,width)
    np.fill_diagonal(k,1.75)
    mean = y.mean()
    alpha = solve(k,y-mean,assume_a='pos')
    return width,alpha,mean


def split_direction(x,y,g,mode,rng):
    width = x.shape[1]
    if mode == 'random':
        return rng.normal(size=width)
    # Entire source orders are removed before deciding which observations were missed.
    groups = rng.permutation(np.unique(g))
    fit = np.isin(g,groups[:int(np.ceil(len(groups)*.45))])
    t = np.flatnonzero(fit)
    t = rng.choice(t,min(len(t),384),replace=False)
    hold = np.flatnonzero(~fit)
    assert not np.isin(g[hold],g[t]).any()
    identity = np.eye(width)
    band,coef,mean = exact(x[t],y[t],identity,rng)
    wrong = (mean+kernel(x[hold],x[t],identity,band)@coef>=.5)!=y[hold]
    if mode == 'mistake_gradients':
        select = hold[wrong]
    elif mode == 'matched_random_gradients':
        # Match the number of missed outcomes in each true class, without using error identity.
        select = np.concatenate([rng.choice(hold[y[hold]==side],int((wrong & (y[hold]==side)).sum()),replace=False)
                                 for side in (False,True)])
    else:
        select = hold
    if not len(select):
        select = hold
    grad = derivative(x[select],x[t],identity,band,coef)
    _,vec = eigh(grad.T@grad,subset_by_index=[width-1,width-1])
    return vec[:,-1]


def leaf(x,y,t,q,rng):
    eye = np.eye(x.shape[1])
    first,coef,_ = exact(x[t],y[t],eye,rng)
    sub = rng.choice(t,min(len(t),768),replace=False)
    grad = derivative(x[sub],x[t],eye,first,coef)
    metric = learned_metric(grad,False)
    band,coef,mean = exact(x[t],y[t],metric,rng)
    return (mean+kernel(x[q],x[t],metric,band)@coef).clip(0,1)


def tree(x,y,g,t,q,mode,depth,node):
    rng = np.random.default_rng(605+node)
    if depth == 0 or len(t) < 150:
        return leaf(x,y,t,q,rng)
    axis = split_direction(x[t],y[t],g[t],mode,rng)
    train_axis,query_axis = x[t]@axis,x[q]@axis
    threshold = np.median(train_axis)
    left,right = train_axis<=threshold,query_axis<=threshold
    if min(left.sum(),(~left).sum())<60:
        return leaf(x,y,t,q,rng)
    result = np.empty(len(q))
    for j, (source,dest) in enumerate([(left,right),(~left,~right)]):
        if dest.any():
            result[dest] = tree(x,y,g,t[source],q[dest],mode,depth-1,2*node+j+1)
    return result


def score(x,y,g,t,q):
    z = ((x-x[t].mean(0))/x[t].std(0).clip(.2)).clip(-7,7)
    return np.column_stack([tree(z,y,g,t,q,mode,2,0) for mode in MODES])


if __name__=='__main__':
    with threadpool_limits(2):
        run(score,MODES,'error_partition_kernel')
