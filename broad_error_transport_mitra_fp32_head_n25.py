"""Replay the original BF16 Mitra forward while measuring a centered FP32 final head."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'FALSE'
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import mitra_candidate as mitra
from mitra_counts_fixed15_online_first import COLUMNS, CountsPredictor, SEED
from relational_error_corrector import write

ROOT = Path(__file__).resolve().parent
V96 = 'proposal_online_first_counts15'
PACK = ROOT/'artifacts/v108_exact_oof'
OUT = ROOT/'artifacts/nov25_mitra_fp32_head'
REPORT = ROOT/'reports/nov23_matched_learning_curve/mitra_fp32_head.json'
SOURCE = ['mitra_candidate.py','mitra_counts_fixed15_online_first.py','third_party/mitra/_internal/models/tab2d.py',
          'third_party/mitra/_internal/models/embedding.py','third_party/mitra/_internal/data/preprocessor.py']
read = lambda p: json.loads(p.read_text())


def centered(module,x):
    """Both class weights are FP32. Only token zero supplies the classification logits."""
    assert x.ndim==4 and x.shape[0]==1 and module.weight.dtype==module.bias.dtype==torch.float32
    with torch.autocast(device_type=x.device.type,enabled=False):
        h = x[0,:,0,:].float()
        w = module.weight[1]-module.weight[0]
        b = module.bias[1]-module.bias[0]
        delta = F.linear(h,w[None,:],b[None]).flatten()
    assert delta.dtype==torch.float32
    return delta


def specs():
    release = read(ROOT/'reports'/V96/'manifest.json')
    history = read(ROOT/'reports/v108_exact_oof/result.json')
    source_hashes = {k:mitra.sha256(ROOT/k) for k in SOURCE}
    assert all(release['source_hashes'][k]==v for k,v in source_hashes.items())
    assert all(history['source_hashes'][k]==v for k,v in source_hashes.items() if k in history['source_hashes'])
    assert mitra.sha256(PACK/'oof.npz')==history['oof_sha256']
    train = pd.read_csv(ROOT/'data/raw/train.csv')
    test = pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId'])
    for part in ('train','test'):
        path = 'data/raw/'+part+'.csv'
        assert mitra.sha256(ROOT/path)==release['source_hashes'][path]
    with np.load(PACK/'oof.npz') as z:
        dev,fold,expected = z['dev_ids'],z['fold_id'],z['v96']
        np.testing.assert_array_equal(train.PassengerId.iloc[dev],z['PassengerId'])
        np.testing.assert_array_equal(train.Transported.iloc[dev],z['true'])
    entries = []
    for i in range(3):
        info = read(ROOT/'reports/v108_exact_oof'/f'fold{i}.json')
        comp = history['model_components'][i]
        assert comp['fold']==i and comp['base_sha256']==info['model_sha256']
        packet = PACK/f'fold{i}_packet.json'
        assert comp['query_patch_packet_sha256']==mitra.sha256(packet)
        t,q = [np.array(read(packet)[key]) for key in ('train_ids','query_ids')]
        np.testing.assert_array_equal(q,dev[fold==i])
        assert not np.intersect1d(t,q).size
        entries.append((f'fold{i}',PACK/f'fold{i}_v96_model.joblib',info['model_sha256'],
                        t,q,expected[fold==i].astype(np.float32),info['support_sha256']))
    path = ROOT/'artifacts'/V96
    prev = pd.read_csv(path/'test_probabilities.csv',dtype={'probability':'float32','reload_probability':'float32'})
    np.testing.assert_array_equal(prev.PassengerId,test.PassengerId)
    np.testing.assert_array_equal(prev.probability,prev.reload_probability)
    assert mitra.sha256(path/'submission.csv')==release['sha256']
    np.testing.assert_array_equal(pd.read_csv(path/'submission.csv').Transported,prev.probability>=.5)
    entries.append(('test',path/'model.joblib',release['model_sha256'],np.arange(len(train)),
                    np.arange(len(train),len(train)+len(test)),prev.probability.to_numpy(),None))
    return entries,train,pd.concat([train.PassengerId,test.PassengerId],ignore_index=True),source_hashes


def load(entry,train,ids):
    tag,path,digest,t,q,old,support_digest = entry
    assert mitra.sha256(path)==digest
    model = joblib.load(path)
    assert isinstance(model,CountsPredictor) and model.device=='cpu' and model._engine is None
    assert model.query_chunk==2318 and model.support.shape==(len(t),15)
    assert list(model.preprocessor.active_columns)==list(COLUMNS)
    assert model.metadata['training_ids_sha256']==mitra.ids_hash(t)
    np.testing.assert_array_equal(model.labels,train.Transported.iloc[t].astype(int))
    roster = model.preprocessor.roster
    np.testing.assert_array_equal(roster.passenger_ids,ids)
    query_ids = ids.iloc[q].to_numpy(str)
    frame = roster.frame_by_ids(query_ids)
    np.testing.assert_array_equal(frame.index,q)
    if support_digest:
        assert hashlib.sha256(model.support.tobytes()).hexdigest()==support_digest
    assert model.preprocessor.transform(frame).shape==(len(q),15)
    return model,frame


def historical_champions(q,fold,ids,truth,native):
    with np.load(ROOT/'artifacts/nov23_incumbent_budget_transfer/oof.npz') as z:
        np.testing.assert_array_equal(z['row'],q)
        np.testing.assert_array_equal(z['y'],truth)
        v131 = z['v131'].astype(bool)
    path = ROOT/'artifacts/nov25_historical_nine/historical.npz'
    meta = read(ROOT/'reports/nov23_matched_learning_curve/historical_nine_release.json')
    declared = read(ROOT/'reports/proposal_n24_selected_dissent/manifest.json')
    assert meta['verified_test_rows']==4277 and meta['scored_csv_sha256']==declared['sha256']
    with np.load(path) as z:
        for key,v in [('rows',q),('fold',fold),('passenger',ids.iloc[q]),('y',truth),('v131',v131)]:
            np.testing.assert_array_equal(z[key],v)
        np.testing.assert_array_equal(z['scores'][:,meta['votes'].index('mitra96')],native)
        nine = np.where(((z['scores']>=.5)==v131[:,None]).all(1),z['tree']>=.5,v131)
        np.testing.assert_array_equal(nine,z['nine'])
    return v131,nine


def cpu_test():
    entries,train,ids,sources = specs()
    torch.manual_seed(778)
    evidence = {}
    for entry in entries:
        model,_ = load(entry,train,ids)
        weight,bias = model.state['final_layer.weight'],model.state['final_layer.bias']
        layer = torch.nn.Linear(weight.shape[1],len(weight))
        layer.load_state_dict(dict(weight=weight,bias=bias))
        h = torch.randn(1,16,2,weight.shape[1]).bfloat16()
        with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
            reference = F.linear(h.double(),weight.double(),bias.double())[0,:,0,:2]
            diff = centered(layer,h)
            torch.testing.assert_close(diff.double(),reference[:,1]-reference[:,0],atol=2e-5,rtol=1e-5)
        evidence[entry[0]] = dict(model_sha256=entry[2],support=len(entry[3]),query=len(entry[4]),raw_ties=int((entry[5]==.5).sum()))
        del model,layer,weight,bias
        gc.collect()
    with np.load(PACK/'oof.npz') as z:
        v131,nine = historical_champions(z['dev_ids'],z['fold_id'],ids,z['true'],z['v96'])
        free_ties = (z['v96']==.5) & (v131==(z['v96']>=.5)) & (nine==v131)
    print(json.dumps(dict(sources=sources,models=evidence,eligible_nine_ties=int(free_ties.sum()))),flush=True)


def forward(entry,model,frame,ids):
    tag,_,digest,_,q,old,_ = entry
    assert torch.backends.mps.is_available()
    model.device='mps'
    chunks = []
    def capture(module,args,output):
        x, = args
        assert output.dtype==torch.bfloat16 and output.ndim==4 and not module.training
        assert x.shape==(1,output.shape[1],1+len(COLUMNS),512)
        delta = centered(module,x)
        with torch.autocast(device_type=x.device.type,enabled=False):
            difference = module.weight[1]-module.weight[0]
            bias = module.bias[1]-module.bias[0]
            # Projection is a dot product, without a per-token bias. Save the classifier intercept separately.
            projection = F.linear(x[0].float(),difference[None,:]).squeeze(-1)
            torch.testing.assert_close(projection[:,0]+bias,delta,atol=2e-5,rtol=1e-4)
        # Independently check the direct difference against float64 on CPU without another backbone pass.
        n = min(16,len(delta))
        cpu = x[0,:n,0,:].detach().cpu().double()
        weight,bias = module.weight[:2].detach().cpu().double(),module.bias[:2].detach().cpu().double()
        compare = F.linear(cpu,weight,bias)
        torch.testing.assert_close(delta[:n].detach().cpu().double(),compare[:,1]-compare[:,0],rtol=1e-4,atol=2e-5)
        chunks.append((output[0,:,0,:2].float().softmax(1)[:,1].detach().cpu().numpy(),
                       delta.detach().cpu().numpy(),str(x.dtype),x[0,:,0,:].detach().cpu().half().numpy(),
                       projection.detach().cpu().numpy()))
    handle = model.engine().final_layer.register_forward_hook(capture)
    try:
        regular = model.predict_proba(frame)[:,1]
    finally:
        handle.remove()
    expected_chunks = [min(2318,len(q)-i) for i in range(0,len(q),2318)]
    assert [len(s[0]) for s in chunks]==expected_chunks
    np.testing.assert_array_equal(regular,old)
    np.testing.assert_allclose(regular,np.concatenate([s[0] for s in chunks]),rtol=0,atol=1e-7)
    delta = np.concatenate([s[1] for s in chunks])
    p = torch.from_numpy(delta).sigmoid().numpy()
    assert np.isfinite(delta).all() and np.isfinite(p).all()
    hidden = np.vstack([s[3] for s in chunks])
    projections = np.vstack([s[4] for s in chunks])
    assert hidden.shape==(len(q),512) and projections.shape==(len(q),16)
    assert np.isfinite(hidden).all() and np.isfinite(projections).all()
    weight = model.state['final_layer.weight']
    bias = model.state['final_layer.bias']
    out = OUT/f'{tag}.npz'
    with out.with_suffix('.tmp').open('wb') as f:
        np.savez_compressed(f,row=q,passenger=ids.iloc[q].to_numpy(str),native=regular,
             fp32=p,fp32_margin=delta,model_sha256=digest,head_input_dtype=np.array([s[2] for s in chunks]),
             label_hidden=hidden,feature_projection=projections[:,1:],label_projection=projections[:,0],
             projection_weight=(weight[1]-weight[0]).numpy(),projection_bias=float(bias[1]-bias[0]),
             feature_names=np.array(COLUMNS))
    out.with_suffix('.tmp').replace(out)


def summarize(entries,train,ids,sources):
    results = {}
    tags = {}
    fold_features = {k:[] for k in ('label_hidden','feature_projection','label_projection',
                                     'projection_weight','projection_bias')}
    for tag,_,digest,_,q,old,_ in entries:
        path = OUT/f'{tag}.npz'
        with np.load(path) as z:
            np.testing.assert_array_equal(z['row'],q)
            np.testing.assert_array_equal(z['passenger'],ids.iloc[q])
            np.testing.assert_array_equal(z['native'],old)
            assert str(z['model_sha256'])==digest
            p,delta = z['fp32'],z['fp32_margin']
            np.testing.assert_array_equal(z['feature_names'],COLUMNS)
            np.testing.assert_allclose(z['label_projection']+z['projection_bias'],delta,atol=2e-5,rtol=1e-4)
            compressed = z['label_hidden'].astype(np.float64)@z['projection_weight'].astype(np.float64)
            result = dict(rows=len(q),native_replay_max_diff=0.,ties=int((old==.5).sum()),
                     ties_to_negative=int(((old==.5)&(p<.5)).sum()),fp32_ties=int((p==.5).sum()),
                     off_tie_flips=int(((old!=.5)&((p>=.5)!=(old>=.5))).sum()),
                     cache=str(path.relative_to(ROOT)),cache_sha256=mitra.sha256(path),model_sha256=digest,
                     head_input_dtype=z['head_input_dtype'].tolist(),
                     hidden_shape=list(z['label_hidden'].shape),feature_projection_shape=list(z['feature_projection'].shape),
                     fp16_hidden_projection_max_abs_diff=float(np.max(np.abs(compressed-z['label_projection']))))
            tags[tag]=(q,old,p,delta)
            if tag!='test':
                for k,values in fold_features.items():
                    values.append(z[k])
        results[tag]=result
    q,native,p,delta = (np.concatenate([tags[f'fold{k}'][j] for k in range(3)]) for j in range(4))
    order = np.argsort(q)
    q,native,p,delta = [a[order] for a in (q,native,p,delta)]
    folds = np.concatenate([np.full(len(tags[f'fold{k}'][0]),k,dtype=int) for k in range(3)])[order]
    with np.load(PACK/'oof.npz') as archived:
        for column,values in [('dev_ids',q),('fold_id',folds),('v96',native),('PassengerId',ids.iloc[q])]:
            np.testing.assert_array_equal(archived[column],values)
    combined = OUT/'oof.npz'
    representations = {k:np.concatenate(values)[order] for k,values in fold_features.items()
                       if k in ('label_hidden','feature_projection','label_projection')}
    representations.update(fold_projection_weight=np.stack(fold_features['projection_weight']),
                           fold_projection_bias=np.array(fold_features['projection_bias']),feature_names=np.array(COLUMNS))
    with combined.with_suffix('.tmp').open('wb') as stream:
        np.savez_compressed(stream,row=q,fold=folds,passenger=ids.iloc[q].to_numpy(str),
                            native=native,fp32=p,fp32_margin=delta,
                            model_sha256=np.array([results[f'fold{k}']['model_sha256'] for k in range(3)]),
                            **representations)
    combined.with_suffix('.tmp').replace(combined)
    history = {}
    y = train.Transported.iloc[q].to_numpy()
    champ,nine = historical_champions(q,folds,ids,y,native)
    free = np.ones(len(y),bool)
    previous = native>=.5
    for key,baseline in [('v96',previous),('historical_v131',champ),('historical_nine',nine)]:
        free &= baseline==previous
        previous = baseline
        for mode,mask in [('all',free),('ties',free & (native==.5))]:
            pred = np.where(mask,p>=.5,baseline)
            history[key+'_'+mode]=dict(corrected=int(((baseline!=y)&(pred==y)).sum()),
                harmed=int(((baseline==y)&(pred!=y)).sum()),total_correct=int((pred==y).sum()),
                eligible=int(mask.sum()),protected=int((~free).sum()),
                fold_net=[int(((pred==y)&(folds==k)).sum()-((baseline==y)&(folds==k)).sum()) for k in range(3)])
    q,old,p,delta = tags['test']
    tests = {}
    free = np.ones(len(q),bool)
    previous = old>=.5
    for name in ['proposal_v131_party_consensus','proposal_n24_selected_dissent']:
        path=ROOT/'artifacts'/name/'submission.csv'
        assert mitra.sha256(path)==read(ROOT/'reports'/name/'manifest.json')['sha256']
        df = pd.read_csv(path)
        np.testing.assert_array_equal(df.PassengerId,ids.iloc[q])
        champion=df.Transported.to_numpy(bool)
        free &= previous==champion
        previous = champion
        for mode,mask in [('all',free),('ties',free&(old==.5))]:
            pred = np.where(mask,p>=.5,champion)
            tests[name+'_'+mode]=dict(changed=int((pred!=champion).sum()),eligible=int(mask.sum()),protected=int((~free).sum()),
                                      to_true=int((pred&~champion).sum()),to_false=int((~pred&champion).sum()))
    write(REPORT,dict(models=results,source_hashes=sources,historical=history,test_release=tests,
         oof_cache=str(combined.relative_to(ROOT)),oof_sha256=mitra.sha256(combined),
         covered='Three original v96 historical folds (6,954 unique passengers) and the frozen 4,277 passenger test model.'))


def run():
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.65)
    entries,train,ids,sources = specs()
    OUT.mkdir(exist_ok=True)
    for entry in entries:
        path = OUT/f'{entry[0]}.npz'
        if not path.exists():
            mitra.seed_everything(SEED,'mps')
            model,frame = load(entry,train,ids)
            try:
                forward(entry,model,frame,ids)
            finally:
                model.release()
                del model
                gc.collect()
                torch.mps.empty_cache()
        print(entry[0],flush=True)
    summarize(entries,train,ids,sources)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-check',action='store_true')
    args = parser.parse_args()
    if args.cpu_check:
        cpu_test()
    else:
        run()
