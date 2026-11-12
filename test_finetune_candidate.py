import unittest
from unittest.mock import patch

import numpy as np
import torch
from tabpfn.finetuning.finetuned_base import EvalResult

import finetune_candidate as candidate


class FinetuneTests(unittest.TestCase):
    def test_epoch_zero_remains_valid_and_min_delta_matches_official(self):
        history = [dict(epoch=0, primary=-.4, log_loss=.4),
                   dict(epoch=1, primary=-.5, log_loss=.5),
                   dict(epoch=2, primary=-.39995, log_loss=.39995)]
        self.assertEqual(candidate.selected_epoch(history)['epoch'], 0)
        history.append(dict(epoch=3, primary=-.39, log_loss=.39))
        self.assertEqual(candidate.selected_epoch(history)['epoch'], 3)

    def test_swallowed_validation_nan_is_rejected(self):
        tuner = candidate.make_tuner(42, [])
        with patch.object(candidate.FinetunedTabPFNClassifier, '_evaluate_model',
                          return_value=EvalResult(primary=np.nan, secondary={'log_loss': np.nan})):
            with self.assertRaisesRegex(RuntimeError, 'nonfinite'):
                tuner._evaluate_model()

    def test_control_and_tuner_match_inference_preprocessing(self):
        cats = ['HomePlanet', 'CryoSleep', 'Side']
        control = candidate.frozen_model(42, cats)
        tuner = candidate.make_tuner(42, cats)
        cfg = tuner._build_eval_config(tuner.extra_classifier_kwargs, 1)
        self.assertEqual(cfg['inference_config'], control.inference_config)
        self.assertEqual(cfg['categorical_features_indices'], control.categorical_features_indices)
        self.assertEqual(control.n_estimators, tuner.n_estimators_final_inference)
        self.assertEqual(tuner.finetune_model_version, candidate.ModelVersion.V2)

    def test_checkpoint_integrity_rejected_before_loading(self):
        predictor = candidate.FinetunePredictor.__new__(candidate.FinetunePredictor)
        with self.assertRaisesRegex(RuntimeError, 'SHA256'):
            predictor.__setstate__({'checkpoint_bytes': b'changed', 'checkpoint_sha256': 'bad'})

    def test_parameter_change_requires_finite_actual_update(self):
        model = torch.nn.Linear(2, 1)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        self.assertFalse(candidate.weight_change(model, before)[0])
        with torch.no_grad():
            model.weight[0, 0] += .01
        self.assertTrue(candidate.weight_change(model, before)[0])
        with torch.no_grad():
            model.weight[0, 0] = float('nan')
        with self.assertRaisesRegex(RuntimeError, 'Nonfinite'):
            candidate.weight_change(model, before)


if __name__ == '__main__':
    unittest.main()
