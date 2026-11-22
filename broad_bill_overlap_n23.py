"""Count the passengers reached exclusively by each independent correction approach."""
import numpy as np

from broad_bill_exception_n23 import ART, REPORT, SEEDS, historical, load, strata, write
from broad_cross_family_rescue_n23 import read


def main():
    row,y,frame,_,_=load()
    actual,labels,masks=historical()
    np.testing.assert_array_equal(actual,row)
    np.testing.assert_array_equal(labels,y)
    stages,specialist=[],[]
    for seed in SEEDS:
        base,ref,fold,bank=read(str(seed),row,y)
        with np.load(ART/f"nov23_broad_bill_exception/{seed}.npz") as z:
            for k,v in (("row",row),("y",y),("base",base),("ref",ref),("fold",fold)):
                np.testing.assert_array_equal(z[k],v)
            specialist.append((z["neutral"]>=.5)==y)
        stages.append((np.column_stack(list(bank.values()))>=.5)==y[:,None])
    old=np.logical_and.reduce(stages).any(1)
    new=np.logical_and.reduce(specialist)
    overlap={}
    for k,m in masks.items():
        overlap[k]=dict(both=int((m&old&new).sum()),new_only=int((m&new&~old).sum()),
                        path_only=int((m&old&~new).sum()),neither=int((m&~old&~new).sum()))
        assert sum(overlap[k].values())==int(m.sum())
    state=strata(frame)
    sleep={}
    for k,m in (("paid",y&(state==2)),("zero",~y&(state==0))):
        sel=m&masks["repeat"]
        sleep[k]={str(s):int((sel&(frame.CryoSleep==s)).sum()) for s in frame.CryoSleep.unique()}
    write(REPORT/"overlap.json",dict(overlap=overlap,registered_sleep=sleep))


if __name__=="__main__":
    main()
