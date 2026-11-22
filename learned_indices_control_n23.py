"""Check learned directional derivatives and response to rotated nonlinear signals."""
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits
from broad_learned_indices_n23 import REPORT,forward,train,write


def main():
    rng=np.random.default_rng(27)
    x=rng.normal(size=(18,5))
    w=rng.normal(size=5)
    axes=rng.normal(size=(5,3))
    coef=rng.normal(size=(3,6))
    direction=rng.normal(size=(5,3))
    _,_,dz=forward(x,w,axes,coef)
    exact=np.einsum("ij,ij->i",x@direction,dz)
    numerical=(forward(x,w,axes+1e-6*direction,coef)[0]-forward(x,w,axes-1e-6*direction,coef)[0])/2e-6
    np.testing.assert_allclose(exact,numerical,rtol=1e-6,atol=3e-9)
    x=rng.normal(size=(7000,18))
    directions=np.linalg.qr(rng.normal(size=(18,3)))[0]
    u=x@directions
    prob=expit(2*np.sin(2*u[:,0])+1.8*np.cos(2*u[:,1])+1.6*np.sin(1.5*u[:,2]))
    y=rng.random(len(x))<prob
    t,q=np.arange(4500),np.arange(4500,len(x))
    scores=train(x,y,t,q)
    acc={k:float(((v>=.5)==y[q]).mean()) for k,v in scores.items()}
    write(REPORT/"control.json",dict(accuracy=acc,generating_probability_accuracy=float(((prob[q]>=.5)==y[q]).mean()),
              max_gradient_error=float(np.max(np.abs(exact-numerical)))))


if __name__=="__main__":
    with threadpool_limits(1):
        main()
