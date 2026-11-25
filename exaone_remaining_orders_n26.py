"""Evaluate the historical grouped EXAONE recipe on every passenger outside its cached roster."""
import numpy as np

from grouped_exaone_veto_n26 import ART, core, inputs, predict

OUT = ART/'nov26_exaone_remaining_orders'


def run():
    frame,y,g,_,_,_,_ = inputs()
    reference = ART/'exaone_group_20261117/EXAONE_v91_CAST_default8_seed20261112.npz'
    with np.load(reference) as z:
        old = z['dev_ids']
    q = np.flatnonzero(~np.isin(np.arange(len(y)),old))
    t = old[~np.isin(g[old],g[q])]
    assert len(q)==1739 and not np.isin(g[t],g[q]).any()
    OUT.mkdir(exist_ok=True)
    path=OUT/'scores.npz'
    if not path.exists():
        assert core.torch.backends.mps.is_available()
        core.torch.set_num_threads(4)
        core.torch.mps.set_per_process_memory_fraction(.65)
        p = predict(frame,y,t,q)
        with path.with_suffix('.tmp').open('wb') as h:
            np.savez_compressed(h,t=t,q=q,train_y=y[t],p=p,checkpoint=core.WEIGHT_SHA)
        path.with_suffix('.tmp').replace(path)
    with np.load(path) as z:
        np.testing.assert_array_equal(z['t'],t)
        np.testing.assert_array_equal(z['q'],q)
        np.testing.assert_array_equal(z['train_y'],y[t])
        assert z['checkpoint']==core.WEIGHT_SHA and np.isfinite(z['p']).all()
    print(len(t),len(q))


if __name__=='__main__':
    run()
