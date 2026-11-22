"""Calibrate the apparent rescue coverage against comparable volumes of contrary votes."""
import numpy as np
from paid_archive_direction_n23 import panel,predictions,EXTRA,SOURCES,MORE
from broad_observed_agreement_n23 import ART,REPORT,SEEDS,historical,load,strata,witnesses,write


def reference_distribution(cells,selection,scope):
    # Permuting a binary selection without replacement within each cell has an
    # exact hypergeometric marginal for its overlap with any prespecified scope.
    rng=np.random.default_rng(523)
    draws=np.zeros(4000,int)
    movable=0
    expectation=0.
    for ids in cells:
        n=len(ids)
        count=int(selection[ids].sum())
        target=int(scope[ids].sum())
        expectation+=count*target/n
        if not target or not count:continue
        if count==n:
            draws+=target
        else:
            draws+=rng.hypergeometric(target,n-target,count,size=len(draws))
            movable+=target
    return dict(observed=int((selection&scope).sum()),conditional_mean=float(expectation),
                central95=np.quantile(draws,[.025,.975]).tolist(),scope_in_mixed_selection_cells=movable)


def main():
    row,y,frame,_,_=load()
    orig,labels,bases,old,_,names,_=panel()
    np.testing.assert_array_equal(row,orig)
    np.testing.assert_array_equal(y,labels)
    _,_,masks=historical()
    cases=list(witnesses(frame))
    signature=np.column_stack([v for _,v,_ in cases]).astype(int)@(2**np.arange(len(cases)))
    extended={**EXTRA,**dict(list(SOURCES.items())[:4])}
    oldvotes=[];newvotes=[]
    for i,seed in enumerate(SEEDS):
        b,v,fold=predictions(row,y,seed,extended)
        np.testing.assert_array_equal(b,bases[i])
        oldvotes.append(np.column_stack([old[i],v,predictions(row,y,seed,MORE)[1]])>=.5)
        with np.load(ART/f"nov23_observed_agreement/{seed}.npz") as z:
            for k,value in (("row",row),("fold",fold),("y",y)):
                np.testing.assert_array_equal(z[k],value)
            np.testing.assert_allclose(z["base"],bases[i],atol=.00011,rtol=0)
            newvotes.append(np.column_stack([z[k+"_agreement"]>=.5 for k,_,_ in cases]))
    votes=np.stack(oldvotes)
    new=np.stack(newvotes)
    bv=bases>=.5
    assert votes.shape==(3,len(row),65)
    old_flip=(votes!=bv[:,:,None]).all(0)
    new_flip=(new!=bv[:,:,None]).all(0).any(1)
    old_reach=old_flip.any(1)
    stable=(bv==bv[0]).all(0)
    persistent=masks["repeat"]
    correct=stable&(bv[0]==y)
    assert persistent.sum()==1070 and correct.sum()==5436
    conf=np.abs(bases.mean(0)-.5)
    density=(votes!=bv[:,:,None]).mean((0,2))
    def band(a,nb):
        return np.digitize(a,np.quantile(a[stable],np.arange(1,nb)/nb))
    basic=np.column_stack([bv[0],band(conf,10),band(density,4),old_reach])
    scope=dict(repeat=persistent,always_correct=correct,unresolved=masks["unresolved"],never=masks["never"],
               solo=persistent&(frame.BookingSize==1).to_numpy(),paid_missed=persistent&(strata(frame)==2)&y,
               missed_by_65=persistent&~old_reach,correct_untouched_by_65=correct&~old_reach)
    controls={}
    for k,design in (("risk_archive",basic),("also_match_visible_markers",np.c_[basic,signature])):
        ids=np.flatnonzero(stable)
        _,cell=np.unique(design[ids],axis=0,return_inverse=True)
        blocks=[ids[cell==j] for j in np.unique(cell)]
        controls[k]={s:reference_distribution(blocks,new_flip,mask) for s,mask in scope.items()}
    census={s:dict(total=int(mask.sum()),archived=int((mask&old_reach).sum()),
              added=int((mask&new_flip&~old_reach).sum()),new=int((mask&new_flip).sum()),
              shared=int((mask&new_flip&old_reach).sum())) for s,mask in scope.items()}
    write(REPORT/"coverage.json",dict(census=census,controls=controls))


if __name__=="__main__":
    main()
