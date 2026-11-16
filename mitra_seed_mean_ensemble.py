"""Fixed two-full50-seed Mitra mean; only one inference engine resident at a time."""
import numpy as np

from mitra_prior_adapt_ensemble import PriorAdaptMeanPredictor, probabilities, mitra

NAME = 'v88_Mitra_full_episode50_two_seed_mean'
COMPONENTS = ('v81_Mitra_full_episode50', 'v87_Mitra_full_episode50_seed100')
SEEDS = (20261112, 20261212)


class SeedMeanPredictor(PriorAdaptMeanPredictor):
    def __init__(self, models, metadata=None):
        if len(models)!=2 or tuple(m.metadata.get('seed') for m in models)!=SEEDS:
            raise ValueError('Exactly the original full50 and new seed20261212 models required, in order.')
        if any(m.metadata.get('recipe',{}).get('max_steps')!=50 for m in models):
            raise ValueError('Both models must use the original full50 training program.')
        super().__init__(models,metadata)

    def predict_proba(self,x):
        if tuple(m.metadata.get('seed') for m in self.models)!=SEEDS:
            raise RuntimeError('Persisted component seeds changed.')
        if not np.array_equal(self.models[0].labels,self.models[1].labels):
            raise RuntimeError('Persisted full-training labels differ.')
        return super().predict_proba(x)
