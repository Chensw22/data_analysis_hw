"""Portable train-context-only adapter for pinned official TabDPT v1.3.

Encoding/model construction/inference are shared with the synthetic resource
probe. Every prediction rebuilds the official estimator from pinned weights;
serialized artifacts contain CPU arrays and category maps, never an engine.
"""
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

import tabdpt_resource_probe as upstream

ROOT = Path(__file__).resolve().parent
COMPACT, CATEGORICAL = list(upstream.COMPACT), list(upstream.CATEGORICAL)
CODE_COMMIT = '3cef828a0bbfe8818bcdd6d855940e1949f2b4cd'
WEIGHT_SHA = '97dc3b60bfad6b42ec1a07b7e121d86b0fc7c9fd67c8d2eaac3d815da197eacb'
PREDICT = dict(n_ensembles=8, temperature=1., context_size=None, batch_size=4277,
               permute_classes=True)
ADAPTER_VERSION = 'tabdpt-v1.3-compact13-context-v1'
CompactEncoder = upstream.CompactEncoder


def array_sha(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def verify_sources():
    manifest_path = upstream.VENDOR/'SOURCE.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest['code_commit'] != CODE_COMMIT or manifest['model_sha256'] != WEIGHT_SHA
            or manifest['model_version'] != '1.3' or upstream.CODE_COMMIT != CODE_COMMIT
            or upstream.WEIGHT_SHA != WEIGHT_SHA or upstream.PREDICT != PREDICT):
        raise RuntimeError('Pinned TabDPT source/model/prediction recipe changed.')
    for relative, expected in manifest['unmodified_vendored_files'].items():
        if upstream.sha(upstream.VENDOR/relative) != expected:
            raise RuntimeError('Pinned vendor source changed: '+relative)
    # A previously imported site-package must not override the vendored module.
    expected_root = (upstream.VENDOR/'tabdpt').resolve()
    for name, module in tuple(sys.modules.items()):
        if name == 'tabdpt' or name.startswith('tabdpt.'):
            source = getattr(module, '__file__', None)
            if source is None or not Path(source).resolve().is_relative_to(expected_root):
                raise RuntimeError('TabDPT was imported outside the pinned vendor directory.')
    return dict(source_commit=CODE_COMMIT, model_version='1.3', checkpoint_sha256=WEIGHT_SHA,
        manifest_sha256=upstream.sha(manifest_path),
        adapter_sha256=upstream.sha(Path(__file__)),
        resource_helper_sha256=upstream.sha(ROOT/'tabdpt_resource_probe.py'),
        vendor_sources=copy.deepcopy(manifest['unmodified_vendored_files']))


def binary_labels(y, rows):
    labels = np.asarray(y)
    if labels.shape != (rows,) or not np.isin(labels, [0,1]).all():
        raise ValueError('One binary training label per context row is required.')
    labels = labels.astype(np.int64, copy=True)
    if not np.array_equal(np.unique(labels), [0,1]):
        raise ValueError('The fixed binary classifier requires both training classes.')
    return labels


def check_encoded(array, rows):
    if array.shape != (rows,13) or array.dtype != np.float32 or np.isinf(array).any():
        raise ValueError('Expected thirteen FP32 features with NaN allowed, infinity rejected.')


class TabDPTContextPredictor:
    def __init__(self, encoder, support, labels, ids, seed, device, provenance):
        self.preprocessor = encoder
        self.support = np.ascontiguousarray(support, dtype=np.float32).copy()
        self.labels = np.asarray(labels, dtype=np.int64).copy()
        self.ids = np.asarray(ids, dtype=np.int64).copy()
        self.seed, self.device = int(seed), device
        self.classes_ = np.array([0,1], dtype=np.int64)
        self.provenance = copy.deepcopy(provenance)
        self.predict_settings = copy.deepcopy(PREDICT)
        self.metadata = dict(adapter_version=ADAPTER_VERSION, seed=self.seed,
            training_rows=len(labels), feature_count=13, class_order=[0,1],
            full_training_context=True, checkpoint_sha256=WEIGHT_SHA,
            source_commit=CODE_COMMIT, context_sha256=array_sha(self.support),
            labels_sha256=array_sha(self.labels), training_ids_sha256=array_sha(self.ids),
            encoder_sha256=json_sha(dict(columns=encoder.columns, maps=encoder.maps)),
            prediction_settings=copy.deepcopy(PREDICT), model_rebuilds=0, prediction_calls=0,
            serialization='CPU raw encoded context/labels/maps; reconstruct official estimator each prediction')

    def assert_context(self):
        if self.device not in ('cpu','mps'):
            raise ValueError('Only explicit CPU or MPS execution is supported.')
        if (self.provenance != verify_sources() or self.predict_settings != PREDICT
                or self.metadata['checkpoint_sha256'] != WEIGHT_SHA
                or self.metadata['source_commit'] != CODE_COMMIT
                or not np.array_equal(self.classes_, [0,1])):
            raise RuntimeError('Pinned artifact source/model/settings/class order changed.')
        check_encoded(self.support, len(self.labels))
        binary_labels(self.labels, len(self.support))
        if (self.ids.shape != self.labels.shape or len(np.unique(self.ids)) != len(self.ids)
                or array_sha(self.support) != self.metadata['context_sha256']
                or array_sha(self.labels) != self.metadata['labels_sha256']
                or array_sha(self.ids) != self.metadata['training_ids_sha256']
                or self.seed != self.metadata['seed']
                or json_sha(dict(columns=self.preprocessor.columns,maps=self.preprocessor.maps))
                    != self.metadata['encoder_sha256']):
            raise RuntimeError('Stored training context or encoding changed.')

    def predict_proba(self, x):
        self.assert_context()
        query = self.preprocessor.transform(x)
        check_encoded(query, len(x))
        if not len(query):
            raise ValueError('Prediction requires at least one query row.')
        if self.device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('Requested MPS backend unavailable; no CPU fallback.')
        engine = handle = None
        observed = []
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        if self.device == 'mps': torch.mps.manual_seed(self.seed)
        self.metadata['model_rebuilds'] += 1
        try:
            engine = upstream.new_model(self.device)  # Verifies exact local weight SHA; no download.
            engine.fit(self.support.copy(), self.labels.copy())
            if (engine.n_instances != len(self.labels) or len(engine.X_train) != len(self.labels)
                    or engine.num_classes != 2 or not np.array_equal(engine.y_train,self.labels)
                    or (hasattr(engine,'classes_') and not np.array_equal(engine.classes_,self.classes_))):
                raise RuntimeError('Official fit changed training support or binary class order.')
            def capture(_module, args, kwargs):
                observed.append(int(kwargs['y_src'].shape[1]))
            handle = engine.model.register_forward_pre_hook(capture, with_kwargs=True)
            probability = upstream.probabilities(engine, query, ensembles=8, seed=self.seed)
            if not observed or set(observed) != {len(self.labels)}:
                raise RuntimeError('Actual forward omitted or reduced the full training context.')
            self.metadata.update(prediction_calls=self.metadata['prediction_calls']+1,
                last_query_rows=len(query), actual_forward_support_rows=sorted(set(observed)),
                last_forward_calls=len(observed), class_order_verified=True)
            self.assert_context()
            return probability
        finally:
            if handle is not None: handle.remove()
            upstream.release(engine, self.device)
            engine = None
            upstream.release(None, self.device)

    def release(self):
        # No estimator/tensor is retained on this object between predictions.
        upstream.release(None, self.device)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['device'] = 'cpu'
        return state


def fit_context(x, y, ids, seed, device='mps'):
    """Fit only an encoder and CPU training context; never receives query labels."""
    if device not in ('mps','cpu'):
        raise ValueError('Only explicit CPU or MPS execution is supported.')
    labels = binary_labels(y, len(x))
    raw_ids = np.asarray(ids)
    if (raw_ids.shape != labels.shape or not np.issubdtype(raw_ids.dtype,np.integer)
            or len(np.unique(raw_ids)) != len(raw_ids)):
        raise ValueError('Unique integer training IDs aligned with labels are required.')
    provenance = verify_sources()
    upstream.verify_checkpoint()
    encoder = CompactEncoder().fit(x)
    support = encoder.transform(x)
    check_encoded(support, len(x))
    return TabDPTContextPredictor(encoder, support, labels, raw_ids, seed, device, provenance)
