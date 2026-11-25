"""Hold network, bin boundaries and training rows fixed while restoring precision within each bill bin."""
import numpy as np
import torch

from full_roster_booking_validation_n23 import data
from passenger_joint_likelihood_n25 import BLOCKS,OUT as BASE,TARGET,encode,train_model

OUT=BASE.parent/'nov25_within_bin_bill_precision'
BILLS=('RoomService','FoodCourt','ShoppingMall','Spa','VRDeck')


def coordinates(raw,encoded,t,q,k):
    shape=(len(raw),TARGET+1)
    pos=np.zeros(shape,dtype=np.float32)
    placebo=np.zeros_like(pos)
    rng=np.random.default_rng(318+k)
    for j,name in zip(BLOCKS['bills'],BILLS,strict=True):
        values=raw[name].to_numpy(float)
        for cell in np.unique(encoded[t,j]):
            if cell<2:continue
            learn=t[encoded[t,j]==cell]
            low,high=values[learn].min(),values[learn].max()
            if low==high:continue
            for ids in (t,q):
                chosen=ids[encoded[ids,j]==cell]
                v=np.clip((values[chosen]-low)/(high-low),0,1)-.5
                pos[chosen,j]=v
                placebo[chosen,j]=v[rng.permutation(len(v))]
    return torch.as_tensor(pos),torch.as_tensor(placebo)


@torch.inference_mode()
def predictions(model,query,position):
    out={}
    for name,drop in [('direct',()),*BLOCKS.items()]:
        mask=torch.ones_like(query,dtype=torch.bool)
        mask[:,TARGET]=False
        if drop:mask[:,list(drop)]=False
        scores=model(query,mask,position)[:,TARGET,:2].softmax(1)[:,1].numpy()
        out[name if not drop else 'omit_'+name]=scores
    return out


def run():
    torch.set_num_threads(2)
    raw,y,g,fold=data()
    OUT.mkdir(exist_ok=True)
    for k in range(8):
        q,t=np.flatnonzero(fold==k),np.flatnonzero(fold!=k)
        encoded,cards=encode(raw,t)
        x=torch.as_tensor(np.column_stack([encoded,y.astype(int)]))
        v=x[q].clone()
        v[:,TARGET]=0
        pos,control=coordinates(raw,encoded,t,q,k)
        with np.load(BASE/f'fold{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            np.testing.assert_array_equal(z['t'],t)
            if k==0:
                pure=train_model(x,cards,t,1241+k,True)
                for name,p in predictions(pure,v,torch.zeros_like(pos[q])).items():
                    np.testing.assert_array_equal(p,z[name])
        for name,positions in [('exact_bill_position',pos),('shuffled_bill_position',control)]:
            path=OUT/f'{k}_{name}.npz'
            if not path.exists():
                fitted=train_model(x,cards,t,1241+k,True,positions)
                p=predictions(fitted,v,positions[q])
                with path.with_suffix('.tmp').open('wb') as stream:
                    np.savez_compressed(stream,t=t,q=q,labels=y[t],passenger=raw.PassengerId.iloc[q].to_numpy(str),**p)
                path.with_suffix('.tmp').replace(path)
            print(k,name,flush=True)


if __name__=='__main__':
    run()
