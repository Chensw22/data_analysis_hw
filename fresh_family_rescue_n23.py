"""Check if new model families recover the same historical errors before combining their votes."""
import numpy as np
from archive_error_recovery_n23 import ART,ROOT,SEEDS,panel,predictions
from relational_error_corrector import write

# These five families were not used when the original 335 and 252 were defined.
NEW = {
    "orion":("pretrained_recheck","orion"),
    "iltm_concat":("iltm_concat_grouped","p"),
    "variational_gp":("variational_gp","p"),
    "berth_activity":("berth_activity","anchored"),
    "manifest_reconstruction":("unlabelled_matrix_views","manifest"),
}


def run():
    row,y,base,old,_,_,_=panel()
    before=(base>=.5)==y
    hits=(old>=.5)==y[None,:,None]
    strong=(hits.mean(1)>=before.mean(1)[:,None]-.01).all(0)
    repeated=~before.any(0)
    weak=repeated&hits.all(0).any(1)&~hits.all(0)[:,strong].any(1)
    deep=repeated&~hits.any(axis=(0,2))
    assert (weak.sum(),deep.sum())==(335,252)
    new_bank=[]
    scores={}
    directory=ART/"nov23_fresh_family_rescue"
    directory.mkdir(exist_ok=True)
    for i,seed in enumerate(SEEDS):
        b,p,fold=predictions(row,y,seed,NEW)
        np.testing.assert_array_equal(b,base[i])
        new_bank.append(p)
        votes=p>=.5
        opposite=(votes!=(b>=.5)[:,None]).sum(1)
        rows_by_dissent=[]
        for dissent in range(p.shape[1]+1):
            mask=opposite==dissent
            rows_by_dissent.append(dict(dissent=dissent,rows=int(mask.sum()),base_wrong=int((mask&~before[i]).sum()),
                                        weak335=int((mask&weak).sum()),deep252=int((mask&deep).sum())))
        direct=p.mean(axis=1)
        policies={"mean_probabilities":direct>=.5,"quarter_mean":.75*b+.25*direct>=.5,
                  "majority_vote":votes.sum(1)>=3,"unanimous_reversal":(b>=.5)^(opposite==5)}
        performance={}
        for name,d in policies.items():
            ok=d==y
            performance[name]=dict(accuracy=float(ok.mean()),fixed=int((ok&~before[i]).sum()),harmed=int((~ok&before[i]).sum()),
                                   weak335=int((ok&weak).sum()),deep252=int((ok&deep).sum()))
        np.savez_compressed(directory/f"{seed}.npz",row=row,y=y,fold=fold,base=b,**policies)
        scores[seed]=dict(vote_distribution=rows_by_dissent,policies=performance)
    all_hits=(np.array(new_bank)>=.5)==y[None,:,None]
    coverage={}
    for name,cohort in [("ever_wrong",~before.all(0)),("repeat_wrong",repeated),("weak335",weak),("deep252",deep)]:
        bank=all_hits[:,cohort]
        reliable=bank.all(axis=0)
        coverage[name]=dict(rows=int(cohort.sum()),at_least_once=int(bank.any(axis=(0,2)).sum()),
                            one_stable=int(reliable.any(axis=1).sum()),two_stable=int((reliable.sum(1)>=2).sum()),
                            same_correct_majority_each_split=int((bank.sum(2)>=3).all(axis=0).sum()),
                            stable_by_family={k:int(reliable[:,j].sum()) for j,k in enumerate(NEW)})
    print(coverage,scores,flush=True)
    write(ROOT/"reports/nov23_fresh_family_rescue/result.json",dict(members=NEW,coverage=coverage,results=scores))


if __name__=="__main__":
    run()
