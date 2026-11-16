"""Fixed two-shape MPS resource/persistence proof; synthetic inputs and labels only."""
import argparse
import gc
import hashlib
from pathlib import Path
import shutil
import time

import joblib
import numpy as np

import mitra_missing_flags_candidate as wrapper

core,base=wrapper.core,wrapper.base
ROOT=wrapper.ROOT
OUTPUT=ROOT/'artifacts/probes/mitra_missing_flags'
REPORT=ROOT/'reports/mitra_missing_flags_resources'


def synthetic(n,seed):
    x=base.synthetic_frame(n,seed)
    # All seven numeric indicators vary, so this is a genuine20-token resource proof.
    for index,name in enumerate(core.NUMERIC):x.loc[np.arange(index,n,19+index),name]=np.nan
    return x


def same_predictions(actual,expected):
    core.valid_probabilities(actual,len(expected));core.valid_probabilities(expected,len(expected))
    difference=float(np.max(np.abs(actual-expected)))
    if difference>2e-6 or not np.array_equal(actual[:,1]>=.5,expected[:,1]>=.5):
        raise RuntimeError('Same-backend subset/reload probability or threshold mismatch.')
    return difference


def case(rows,out):
    started=time.monotonic();model=restored=None;path=out/f'portable_{rows}.joblib'
    if path.exists():raise RuntimeError('Portable probe artifact exists; no overwrite.')
    x,q=synthetic(rows,42),synthetic(2318,43);y=np.arange(rows,dtype=np.int64)%2
    result=dict(status='incomplete',support_rows=rows,query_rows=2318,full_support=True,
        synthetic_attributes=True,synthetic_alternating_labels=True,no_accuracy_computed=True)
    try:
        model=core.fit_missing_flags(x,y,np.arange(rows),42,'mps')
        schema=model.preprocessor.schema();support=model.support.copy();query=model.preprocessor.transform(q)
        if len(support)!=rows or support.shape[1]!=20 or query.shape!=(2318,20):
            raise RuntimeError('Complete support with20 actual features required.')
        for array in [support,query]:
            transferred=base.torch.as_tensor(array,device='mps').cpu().numpy()
            np.testing.assert_array_equal(transferred,array)
        result.update(cpu_mps_input_exact=True,actual_support_rows=len(support),actual_retained_features=20,
            schema=schema,transformed_support_sha256=hashlib.sha256(support.tobytes()).hexdigest(),
            transformed_query_sha256=hashlib.sha256(query.tobytes()).hexdigest(),resources_before=base.snapshot('mps'))
        tick=time.monotonic();p=core.valid_probabilities(model.predict_proba(q),len(q));result['full_forward_seconds']=time.monotonic()-tick
        small=core.valid_probabilities(model.predict_proba(q.iloc[:128]),128)
        result.update(subset_max_difference=same_predictions(small,p[:128]),subset_passed=True,resources_after=base.snapshot('mps'))
        joblib.dump(model,path);model.release();model=None;gc.collect();base.release_memory()
        restored=joblib.load(path)
        if restored.device!='cpu' or restored._engine is not None or any(t.device.type!='cpu' for t in restored.state.values()):
            raise RuntimeError('Portable state must contain only CPU weights and no engine.')
        if restored.preprocessor.schema()!=schema:raise RuntimeError('Persisted flags/schema changed.')
        np.testing.assert_array_equal(restored.support,support)
        np.testing.assert_array_equal(restored.labels,y)
        np.testing.assert_array_equal(restored.preprocessor.transform(q),query)
        if len(restored.support)!=rows or restored.query_chunk!=2318:raise RuntimeError('Reload changed full context/chunk.')
        restored.device='mps';tick=time.monotonic()
        reloaded=core.valid_probabilities(restored.predict_proba(q),len(q))
        result.update(reload_seconds=time.monotonic()-tick,reload_max_difference=same_predictions(reloaded,p),
            reload_passed=True,schema_reload_exact=True,resources_after_reload=base.snapshot('mps'))
        base.check_deadline();result.update(status='complete',seconds=time.monotonic()-started)
        return result
    finally:
        if model is not None:model.release()
        if restored is not None:restored.release()
        model=restored=None;base.release_memory();path.unlink(missing_ok=True)


def probe(session_file):
    if OUTPUT.exists() or REPORT.exists():raise RuntimeError('Resource destination exists; no overwrite/resume.')
    wrapper.check_quota(session_file)
    with wrapper.deadline(1200):
        contract,lock=wrapper.freeze_contract();wrapper.runtime()
        OUTPUT.mkdir(parents=True);REPORT.mkdir(parents=True)
        result=dict(contract,status='incomplete',device='mps',checkpoint_sha256=base.WEIGHT_SHA,
            budget_seconds=1200,no_accuracy_computed=True,cases=[],
            precision_disclosure='CPU preprocessing is copied exactly to MPS; model uses existing FP32 weights/input and bf16 MPS autocast. No CPU-vs-MPS probability equality claim.')
        (OUTPUT/'environment.lock.txt').write_text(lock);(REPORT/'environment.lock.txt').write_text(lock)
        for name in contract['sources']:
            target=OUTPUT/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        started=time.monotonic()
        try:
            for rows in [4636,8693]:
                wrapper.assert_frozen(contract);base.check_deadline()
                completed=case(rows,OUTPUT);result['cases'].append(completed)
                wrapper.write(OUTPUT/f'case_{rows}.json',completed);wrapper.write(REPORT/f'case_{rows}.json',completed)
                wrapper.write(OUTPUT/'result.json',result);wrapper.write(REPORT/'result.json',result)
                print(f'Completed synthetic resource shape {rows}/2318',flush=True)
            wrapper.assert_frozen(contract);base.check_deadline()
            result.update(status='complete',seconds=time.monotonic()-started)
            return result
        except BaseException as error:
            result.update(status='failed_inconclusive',error_type=type(error).__name__,error=str(error),seconds=time.monotonic()-started)
            raise
        finally:
            wrapper.write(OUTPUT/'result.json',result);wrapper.write(REPORT/'result.json',result)
            base.release_memory()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--session-file',type=Path,required=True)
    probe(parser.parse_args().session_file)
