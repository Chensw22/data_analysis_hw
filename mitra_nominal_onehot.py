"""Raw Mitra-v2 with exactly two train-only nominal one-hot fields; no optimizer."""
import copy
import hashlib
import time

import numpy as np

import mitra_candidate as base

NAME = 'v94_Mitra2_nominal_onehot'
REPLACE = ['HomePlanet', 'Destination']
KEEP = [name for name in base.COMPACT if name not in REPLACE]
MIRROR_RULE = 'Original13 active-name signs; every nominal child inherits its parent sign; no expanded-dimension RNG draw.'
SETTINGS = dict(replace_fields=REPLACE, keep_fields=KEEP, encoding='train-only known one-hot plus missing; unknown all-zero',
    mirror_rule=MIRROR_RULE, query_chunk=2318, weights_and_input_dtype='float32', mps_autocast_dtype='bfloat16',
    mps_memory_fraction=.65, threads=4, sdpa_batch_chunk=1, sdpa_long_sequence_threshold=512,
    full_support=True, optimizer_steps=0, n_estimators=1, threshold=.5, checkpoint_sha256=base.WEIGHT_SHA)


class NominalOneHotPreprocessor(base.MitraPreprocessor):
    def fit(self, x, y, seed=42):
        y = base.binary_labels(y, len(x))
        self.control = base.MitraPreprocessor().fit(x, y, seed)
        self.vocab = copy.deepcopy(self.control.vocab)
        self.vocabularies = {name:list(self.control.vocab[name]) for name in REPLACE}
        self.columns, self.parent_by_column, self.child_specs = [], {}, {}
        for parent in base.COMPACT:
            if parent not in REPLACE:
                self.columns.append(parent); self.parent_by_column[parent] = parent
                continue
            for index, category in enumerate(self.vocabularies[parent]):
                name = f'{parent}__known_{index}'
                self.columns.append(name); self.parent_by_column[name] = parent
                self.child_specs[name] = dict(parent=parent, kind='known', category=category)
            name = parent+'__missing'
            self.columns.append(name); self.parent_by_column[name] = parent
            self.child_specs[name] = dict(parent=parent, kind='missing', category=None)
        self.categories = [name for name in base.CATEGORICAL if name not in REPLACE]
        array = self.encode(x)
        self.all_missing = np.isnan(array).all(axis=0); array[:, self.all_missing] = 0.
        self.upstream = copy.deepcopy(self.control.upstream)
        # Fit ordinary mean/constant filtering, without drawing a new mirror mask.
        self.upstream.random_mirror_x = False
        self.upstream.fit(array.copy(), y.copy())
        self.active_columns = [name for name, drop in zip(self.columns, self.upstream.singular_features) if not drop]
        control_active = [name for name, drop in zip(base.COMPACT, self.control.upstream.singular_features) if not drop]
        self.control_active_columns = control_active
        self.parent_mirror_signs = {name:int(sign) for name, sign in zip(control_active, self.control.upstream.mirror[0])}
        signs = []
        for name in self.active_columns:
            parent = self.parent_by_column[name]
            if parent not in self.parent_mirror_signs:
                raise RuntimeError('Nonconstant child of a constant parent; no invented mirror sign.')
            signs.append(self.parent_mirror_signs[parent])
        self.upstream.mirror = np.asarray([signs], dtype=self.control.upstream.mirror.dtype)
        self.upstream.random_mirror_x = self.control.upstream.random_mirror_x
        self.unchanged_active_columns = [name for name in control_active if name in KEEP]
        self.unchanged_positions = [self.active_columns.index(name) for name in self.unchanged_active_columns]
        self.control_positions = [control_active.index(name) for name in self.unchanged_active_columns]
        self.n_features = len(self.active_columns)
        for name in KEEP:
            new, old = self.columns.index(name), base.COMPACT.index(name)
            if (self.all_missing[new] != self.control.all_missing[old]
                    or self.upstream.singular_features[new] != self.control.upstream.singular_features[old]
                    or self.upstream.pre_nan_mean[new] != self.control.upstream.pre_nan_mean[old]):
                raise RuntimeError('Unchanged field mean/missing/constant filtering differs: '+name)
        self.transform(x)
        return self

    def encode(self, x):
        control = self.control.encode(x)
        missing = {name:base.MitraPreprocessor.category_missing(x[name]).to_numpy(bool) for name in REPLACE}
        strings = {name:x[name].astype(str).to_numpy() for name in REPLACE}
        columns = []
        for name in self.columns:
            if name in KEEP:
                columns.append(control[:, base.COMPACT.index(name)])
            else:
                spec = self.child_specs[name]; parent = spec['parent']
                values = missing[parent] if spec['kind'] == 'missing' else (~missing[parent] & (strings[parent] == spec['category']))
                columns.append(values.astype(np.float32))
        return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)

    def transform(self, x):
        result = super().transform(x)
        unchanged = np.ascontiguousarray(result[:, self.unchanged_positions])
        control = np.ascontiguousarray(self.control.transform(x)[:, self.control_positions])
        if unchanged.dtype != control.dtype or unchanged.shape != control.shape or unchanged.tobytes() != control.tobytes():
            raise RuntimeError('Unchanged active fields must be bitwise identical to original13 preprocessing.')
        return result

    def schema(self):
        return dict(columns=self.columns, active_columns=self.active_columns, vocabularies=self.vocabularies,
            child_specs=self.child_specs, parent_by_column=self.parent_by_column,
            control_active_columns=self.control_active_columns, parent_mirror_signs=self.parent_mirror_signs,
            derived_mirror_mask=self.upstream.mirror.reshape(-1).tolist(), mirror_rule=MIRROR_RULE,
            unchanged_active_columns=self.unchanged_active_columns, unchanged_positions=self.unchanged_positions,
            control_positions=self.control_positions, original11_exact=True, actual_retained_features=self.n_features)


def fit_nominal_onehot(x, y, ids, seed, device='mps'):
    ids = np.asarray(ids, dtype=np.int64); y = base.binary_labels(y, len(x))
    if ids.shape != (len(x),) or len(np.unique(ids)) != len(ids):
        raise ValueError('Unique aligned training IDs required.')
    base.check_deadline(); started = time.monotonic(); base.seed_everything(seed, device)
    prep = NominalOneHotPreprocessor().fit(x, y, seed)
    config = base.verify_checkpoint(base.CHECKPOINT_DIR)
    state = base.load_file(str(base.CHECKPOINT_DIR/'model.safetensors'), device='cpu')
    try:
        support = prep.transform(x)
        metadata = dict(status='fit_complete', candidate=NAME, settings=SETTINGS, seed=int(seed),
            training_rows=len(ids), training_ids_sha256=base.ids_hash(ids), labels_sha256=base.ids_hash(y),
            full_prediction_support=True, checkpoint_sha256=base.WEIGHT_SHA, optimizer_steps=0,
            schema=prep.schema(), transformed_support_sha256=hashlib.sha256(support.tobytes()).hexdigest(),
            preprocessing_seconds=time.monotonic()-started)
        return base.MitraPredictor(prep, config, state, support, y, device, metadata, query_chunk=2318)
    finally:
        del state


def valid_probabilities(p, rows):
    p = np.asarray(p)
    if (p.shape != (rows, 2) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
            or not np.allclose(p.sum(axis=1), 1., rtol=0, atol=1e-6)):
        raise ValueError('Complete finite normalized binary probabilities required.')
    return p
