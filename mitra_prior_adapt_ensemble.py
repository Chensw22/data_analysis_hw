"""Fixed equal probability mean of original and adapted Mitra, sequential inference."""
import numpy as np

import mitra_candidate as mitra

NAME = 'v86_Mitra_prior_adapt_equal2'
COMPONENTS = ('v71_Mitra2_zeroshot', 'v81_Mitra_full_episode50')


def probabilities(value, rows):
    value = np.asarray(value, dtype=np.float64)
    if (value.shape != (rows, 2) or not np.isfinite(value).all()
            or np.any((value < 0) | (value > 1))
            or not np.allclose(value.sum(axis=1), 1, atol=2e-6, rtol=0)):
        raise ValueError('Complete normalized two-class probabilities required.')
    return value


class PriorAdaptMeanPredictor:
    def __init__(self, models, metadata=None):
        if len(models) != 2:
            raise ValueError('Exactly the original and adapted Mitra models required.')
        if not np.array_equal(models[0].labels, models[1].labels):
            raise ValueError('Components must share aligned full-training labels.')
        self.models = list(models)
        self.labels = np.array(models[0].labels, copy=True)
        self.metadata = dict(metadata or {})
        self.device = 'cpu'
        self.release()

    def predict_proba(self, x):
        total = None
        for model in self.models:
            mitra.check_deadline()
            model.device = self.device
            try:
                p = probabilities(model.predict_proba(x), len(x))
                total = p.copy() if total is None else total + p
            finally:
                model.release()  # Never retain both MPS engines concurrently.
        return total / 2

    def release(self):
        for model in self.models:
            model.release()

    def __getstate__(self):
        state = dict(self.__dict__)
        state['device'] = 'cpu'
        return state
