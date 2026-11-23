"""Repeat the partner-specific neural model with independent initialization and batch order."""
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from broad_partner_specific_fields_n23 import REPORT, encode, train_pair
from broad_opposing_passengers_n23 import evaluate


def run_fold(frame,groups,y,t,q):
    assert not np.intersect1d(groups[t],groups[q]).size
    x=encode(frame,t)
    p=train_pair(x,torch.from_numpy(y.astype(np.float32)),t,q,True,seed=817)
    return dict(replication=p),dict(train=len(t),query=len(q),network_seed=817)


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(2):
        evaluate('repeat_partner_specific_fields',run_fold,REPORT)
