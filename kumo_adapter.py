"""Fixed Kumo Medium eight-view adapter using the resource-probed official recipe."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ROOT = next(p for p in [BASE,*BASE.parents] if (p/'iterate.py').is_file())
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))
import kumo_resource_probe as backend

NAME='v83_Kumo_medium8'


def array_sha(value):
    value=np.ascontiguousarray(value)
    h=hashlib.sha256();h.update(str((value.dtype,value.shape)).encode());h.update(value.tobytes())
    return h.hexdigest()


def table_sha(table):
    if table.device.type!='cpu':raise RuntimeError('Persistent context must remain on CPU.')
    frame=table.to_pandas()
    h=hashlib.sha256();h.update(str(tuple(frame.columns)).encode())
    h.update(str(tuple(str(d) for d in frame.dtypes)).encode())
    h.update(pd.util.hash_pandas_object(frame,index=False).to_numpy().tobytes())
    return h.hexdigest()


def source_hashes():
    sources=backend.source_hashes()
    for name in ('kumo_adapter.py','test_kumo_adapter.py'):
        sources[name]=backend.sha(BASE/name)
    return sources


def verify_sources():
    record=json.loads((backend.VENDOR/'UPSTREAM.json').read_text())
    if record['commit']!=backend.CODE_COMMIT:
        raise RuntimeError('Kumo source revision changed.')
    for name,module in tuple(sys.modules.items()):
        if name=='sdm' or name.startswith('sdm.'):
            file=getattr(module,'__file__',None)
            if file is None or not Path(file).resolve().is_relative_to((backend.VENDOR/'sdm').resolve()):
                raise RuntimeError('Imported sdm is outside the pinned vendor directory.')
    return source_hashes()


class KumoContextPredictor(backend.KumoPredictor):
    def __init__(self,x,y,ids,seed,device='mps'):
        yy=np.asarray(y);ii=np.asarray(ids)
        if (yy.shape!=(len(x),) or not np.isin(yy,[0,1]).all() or len(np.unique(yy))!=2
                or ii.shape!=yy.shape or not np.issubdtype(ii.dtype,np.integer)
                or len(np.unique(ii))!=len(ii)):
            raise ValueError('Aligned binary labels and unique integer training IDs required.')
        if device not in ('cpu','mps'):raise ValueError('Explicit CPU or MPS required.')
        self.sources=verify_sources()
        super().__init__(x,yy,device=device,estimators=8,seed=seed)
        self.labels=yy.astype(np.int64,copy=True);self.ids=ii.astype(np.int64,copy=True)
        self.classes_=np.array([0,1],dtype=np.int64)
        self.metadata=dict(candidate=NAME,training_rows=len(yy),feature_count=13,seed=int(seed),
            checkpoint_sha256=backend.WEIGHT_SHA,source_commit=backend.CODE_COMMIT,
            training_ids_sha256=array_sha(self.ids),labels_sha256=array_sha(self.labels),
            context_sha256=table_sha(self.x),target_table_sha256=table_sha(self.y),
            pretrained_state_sha256=self.state_sha,settings=copy.deepcopy(backend.SETTINGS),
            full_training_context=True,estimators=8,prediction_calls=0,
            persistence='CPU pretrained weights and raw context; original CPU recipe/FP32 device callback')

    def assert_context(self):
        if (self.sources!=verify_sources() or self.checkpoint_sha!=backend.WEIGHT_SHA
                or self.metadata['checkpoint_sha256']!=backend.WEIGHT_SHA
                or self.metadata['settings']!=backend.SETTINGS or self.estimators!=8
                or self.seed!=self.metadata['seed'] or self.device not in ('cpu','mps')
                or not np.array_equal(self.classes_,[0,1])
                or self.state_sha!=self.metadata['pretrained_state_sha256']
                or backend.state_hash(self.state)!=self.state_sha):
            raise RuntimeError('Frozen source, weights, seed or inference recipe changed.')
        if (len(self.labels)!=self.metadata['training_rows'] or len(self.x)!=len(self.labels)
                or len(self.y)!=len(self.labels) or len(np.unique(self.ids))!=len(self.ids)
                or array_sha(self.ids)!=self.metadata['training_ids_sha256']
                or array_sha(self.labels)!=self.metadata['labels_sha256']
                or table_sha(self.x)!=self.metadata['context_sha256']
                or table_sha(self.y)!=self.metadata['target_table_sha256']):
            raise RuntimeError('Persisted full training context changed.')

    def predict_proba(self,x):
        self.assert_context()
        if not len(x):raise ValueError('At least one query row required.')
        if self.device=='mps' and not backend.torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable; no CPU fallback.')
        try:
            result=super().predict_proba(x)
            contexts=self.last_forward_contexts
            if (result.shape!=(len(x),2) or not np.isfinite(result).all()
                    or len(contexts)!=8 or any(row['rows']!=len(self.labels) for row in contexts)):
                raise RuntimeError('Incomplete full-context eight-view prediction.')
            self.metadata.update(prediction_calls=self.metadata['prediction_calls']+1,
                last_query_rows=len(x),last_forward_contexts=copy.deepcopy(contexts),class_order=[0,1])
            return result
        finally:
            self.release()

    def release(self):
        self._engine=None
        backend.cleanup(self.device)

    def __getstate__(self):
        self.assert_context()
        return super().__getstate__()


def fit_context(x,y,ids,seed,device='mps'):
    return KumoContextPredictor(x,y,ids,seed,device)
