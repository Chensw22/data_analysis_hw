"""Check whether official test bookings cluster in passenger-ID order and affect model comparisons."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from archive_error_recovery_n23 import panel, predictions
from broad_bill_exception_n23 import load, strata
from broad_partition_evidence_n23 import EXTRA, ROOT, SEEDS, SOURCES, write
from released_correction_audit_n23 import development

REPORT = ROOT / "reports/nov23_split_order"
WINDOWS = (32, 128, 512)


def signatures(bits):
    n = len(bits)
    p = bits.mean()
    prefix = np.r_[0, np.cumsum(bits)]
    values = []
    for width in WINDOWS:
        counts = prefix[width:] - prefix[:-width]
        sd = np.sqrt(width*p*(1-p)*(n-width)/(n-1))
        values.append(np.max(np.abs(counts-width*p))/sd)
    change = np.flatnonzero(np.diff(np.r_[0, bits, 0]))
    max_run = np.diff(change.reshape(-1,2),axis=1).max()
    # Count adjacent test pairs and record both excesses and deficits.
    expected = bits.sum()*(bits.sum()-1)/n
    adjacent = np.count_nonzero(bits[:-1]&bits[1:])
    return np.r_[max(values), max_run, abs(adjacent-expected)], values


def run():
    raw = [pd.read_csv(ROOT/f"data/raw/{split}.csv", usecols=["PassengerId"]).PassengerId
           for split in ("train", "test")]
    group = [col.str.split("_").str[0].astype(int).to_numpy() for col in raw]
    assert not np.intersect1d(*group).size
    identifiers, inverse, size = np.unique(np.concatenate(group), return_inverse=True, return_counts=True)
    it, iq = np.split(inverse, [len(group[0])])
    bits = np.zeros(len(identifiers), bool)
    bits[iq] = True
    n = len(bits)
    actual, scan = signatures(bits)
    parts = [np.flatnonzero(size==s) for s in np.unique(size)]
    rng = np.random.default_rng(443)
    rep = np.empty((3000,3))
    for i in range(len(rep)):
        drawn = bits.copy()
        for ix in parts:
            drawn[ix] = rng.permutation(bits[ix])
        rep[i], _ = signatures(drawn)
    scan_stats = {}
    for j,key in enumerate(("max_standardized_scan","longest_test_run","adjacent_pair_deviation")):
        scan_stats[key] = dict(value=float(actual[j]), randomization_p=float((1+np.sum(rep[:,j]>=actual[j]))/(len(rep)+1)),
                               null_95=np.quantile(rep[:,j],[.025,.975]).tolist())
    window = {}
    prefix = np.r_[0,np.cumsum(bits)]
    for width,stat in zip(WINDOWS,scan):
        counts = prefix[width:]-prefix[:-width]
        def span(j):
            return [int(identifiers[j]),int(identifiers[j+width-1]),int(counts[j])]
        window[width] = dict(minimum=span(np.argmin(counts)),maximum=span(np.argmax(counts)),standardized=float(stat))
    row,y,frame,g,_=load()
    np.testing.assert_array_equal(g,group[0][row])
    rows,labels,base,old,_,names,_=panel()
    np.testing.assert_array_equal(rows,row)
    np.testing.assert_array_equal(labels,y)
    complete,release,_=development()
    np.testing.assert_array_equal(complete,y)
    missed=np.logical_and.reduce([p!=y for p in release.values()])
    state=strata(frame)
    core=missed & (((state==2)&y)|((state==0)&~y))
    assert [missed.sum(),core.sum()]==[941,696]
    extra={**EXTRA,**dict(list(SOURCES.items())[:4])}
    late=np.stack([predictions(row,y,s,extra)[1] for s in SEEDS])
    hit=(np.concatenate([old,late],axis=2)>=.5)==y[None,:,None]
    assert hit.shape==(3,len(y),50)
    names=list(names)+list(extra)
    observed=(base>=.5)!=y
    devgroup=it[row]
    post={}
    for width in WINDOWS:
        code = np.arange(n)//width
        # Calibrate passenger weights to the observed official test counts in
        # contiguous intervals. Merge the final short interval when necessary.
        if n%width and n%width<width/2:
            code[code==code.max()]-=1
        src=code[devgroup]
        target=np.bincount(code[iq],minlength=code.max()+1)
        dev=np.bincount(src,minlength=len(target))
        assert dev.min()>0
        weights=(target/target.sum())/(dev/dev.sum())
        w=weights[src]
        assert abs(w.mean()-1)<1e-10
        weighted=np.einsum("sij,i->sj",hit,w)/len(y)
        original=hit.mean(1)
        rawavg=original.mean(0)
        avg=weighted.mean(0)
        error=observed.mean(0)
        post[width]=dict(weight_range=[float(w.min()),float(w.max())],effective_rows=float(w.sum()**2/(w@w)),
            reference_accuracy=(1-(w@error)/len(y)),unweighted_reference_accuracy=float(1-error.mean()),
            cohort_prevalence_pp_shift={k:float(100*(w@m/len(y)-m.mean())) for k,m in (("historical941",missed),("core696",core))},
            rank_correlation=float(spearmanr(rawavg,avg).statistic),highest_original=names[int(rawavg.argmax())],
            highest_adjusted=names[int(avg.argmax())],best_adjusted=float(avg.max()),
            max_abs_accuracy_shift_pp=float(100*np.max(abs(rawavg-avg))),
            max_abs_pair_gap_shift_pp=float(100*np.ptp(avg-rawavg)))
    # Leave the focal booking out when estimating the local test density.
    i=np.arange(n)
    left=np.maximum(0,i-64);right=np.minimum(n,i+65)
    density=(prefix[right]-prefix[left]-bits)/(right-left-1)
    rates={}
    for key,value in (("reference_error",observed.mean(0)),("historic941",missed),("core696",core)):
        # Aggregate outcomes within booking and randomize booking-mean errors
        # within exact roster size and number of evaluated development members.
        used,ix,num=np.unique(devgroup,return_inverse=True,return_counts=True)
        response=np.bincount(ix,weights=value)/num
        domain=density[used]
        cell=size[used]*10+num
        residual=domain.copy()
        for v in np.unique(cell):
            loc=cell==v
            residual[loc]-=domain[loc].mean()
        scale=np.sqrt(np.sum(residual**2)*np.sum((response-response.mean())**2))
        val=float((residual@response)/scale)
        null=np.zeros(2000)
        for v in np.unique(cell):
            loc=np.flatnonzero(cell==v)
            for start in range(0,len(null),100):
                shuffled=rng.permuted(np.tile(response[loc],(100,1)),axis=1)
                null[start:start+100]+=shuffled@residual[loc]
        null/=scale
        rates[key]=dict(within_size_density_covariance_scaled=val,
                        permutation_p=float((1+np.sum(abs(null)>=abs(val)))/(1+len(null))))
    write(REPORT/"results.json",dict(groups=n,test=int(bits.sum()),train=int((~bits).sum()),windows=window,
          permutation_tests=scan_stats,poststratified=post,local_error=rates))
    print(scan_stats,window,post,rates,flush=True)


if __name__=="__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(2):
        run()
