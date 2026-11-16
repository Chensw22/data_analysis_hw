"""Fixed raw Mitra-v2 compact13 plus seven pre-imputation numeric missing flags."""
import copy
import hashlib
import time

import numpy as np

import mitra_candidate as base

NAME='v89_Mitra2_missing_flags'
NUMERIC=['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','CabinNumber']
FLAGS=[name+'_missing' for name in NUMERIC]
COLUMNS=list(base.COMPACT)+FLAGS
SETTINGS=dict(columns=COLUMNS,flags=FLAGS,query_chunk=2318,device='mps',threads=4,
    weights_and_input_dtype='float32',mps_autocast_dtype='bfloat16',mps_memory_fraction=.65,
    sdpa_batch_chunk=1,sdpa_long_sequence_threshold=512,full_support=True,
    optimizer_steps=0,n_estimators=1,threshold=.5,checkpoint_sha256=base.WEIGHT_SHA)


def missing_flags(x):
    # Only the raw compact numeric cells define flags. Caller-supplied flags are ignored.
    return x[NUMERIC].isna().to_numpy(dtype=np.float32)


class MissingFlagsPreprocessor(base.MitraPreprocessor):
    def fit(self,x,y,seed=42):
        y=base.binary_labels(y,len(x))
        self.control=base.MitraPreprocessor().fit(x,y,seed)
        self.columns=list(COLUMNS);self.categories=list(base.CATEGORICAL)
        self.vocab=copy.deepcopy(self.control.vocab)
        array=self.encode(x)
        self.all_missing=np.isnan(array).all(axis=0)
        array[:,self.all_missing]=0.
        # Same upstream implementation/options and random seed, without altering the control.
        self.upstream=copy.deepcopy(self.control.upstream)
        previous=np.random.get_state()
        try:
            np.random.seed(seed)
            self.upstream.fit(array.copy(),y.copy())
        finally:np.random.set_state(previous)
        self.active_columns=[c for c,drop in zip(COLUMNS,self.upstream.singular_features) if not drop]
        self.control_active_columns=[c for c,drop in zip(base.COMPACT,self.control.upstream.singular_features) if not drop]
        self.control_positions=[self.active_columns.index(c) for c in self.control_active_columns]
        self.n_features=len(self.active_columns)
        np.testing.assert_array_equal(self.all_missing[:13],self.control.all_missing)
        np.testing.assert_array_equal(self.upstream.singular_features[:13],self.control.upstream.singular_features)
        np.testing.assert_array_equal(self.upstream.pre_nan_mean[:13],self.control.upstream.pre_nan_mean)
        np.testing.assert_array_equal(self.upstream.mirror[:,self.control_positions],self.control.upstream.mirror)
        self.transform(x)  # Enforce the original13 invariant for every fit and later query.
        return self

    def encode(self,x):
        return np.ascontiguousarray(np.concatenate([self.control.encode(x),missing_flags(x)],axis=1),dtype=np.float32)

    def transform(self,x):
        transformed=super().transform(x)
        np.testing.assert_array_equal(transformed[:,self.control_positions],self.control.transform(x))
        return transformed

    def schema(self):
        return dict(input_columns=self.columns,flags=FLAGS,actual_retained_features=self.n_features,
            active_columns=self.active_columns,control_active_columns=self.control_active_columns,
            control_positions=self.control_positions,mirror_mask=self.upstream.mirror.reshape(-1).tolist(),
            original13_exact=True)


def fit_missing_flags(x,y,ids,seed,device='mps'):
    ids=np.asarray(ids,dtype=np.int64);y=base.binary_labels(y,len(x))
    if ids.shape!=(len(x),) or len(np.unique(ids))!=len(ids):raise ValueError('Unique aligned training IDs required.')
    base.check_deadline();started=time.monotonic();base.seed_everything(seed,device)
    prep=MissingFlagsPreprocessor().fit(x,y,seed)
    config=base.verify_checkpoint(base.CHECKPOINT_DIR)
    state=base.load_file(str(base.CHECKPOINT_DIR/'model.safetensors'),device='cpu')
    try:
        support=prep.transform(x)
        metadata=dict(status='fit_complete',candidate=NAME,settings=SETTINGS,seed=int(seed),
            training_rows=len(ids),training_ids_sha256=base.ids_hash(ids),labels_sha256=base.ids_hash(y),
            full_prediction_support=True,checkpoint_sha256=base.WEIGHT_SHA,optimizer_steps=0,
            schema=prep.schema(),transformed_support_sha256=hashlib.sha256(support.tobytes()).hexdigest(),
            preprocessing_seconds=time.monotonic()-started)
        return base.MitraPredictor(prep,config,state,support,y,device,metadata,query_chunk=2318)
    finally:del state


def valid_probabilities(p,rows):
    p=np.asarray(p)
    if p.shape!=(rows,2) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Complete finite binary probabilities required.')
    if not np.allclose(p.sum(axis=1),1.,rtol=0,atol=1e-6):raise ValueError('Binary probability rows must sum to one.')
    return p
