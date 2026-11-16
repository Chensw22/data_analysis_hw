import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import torch

import tabdpt_adapter as adapter


def frame(rows=16):
    data={column:np.arange(rows,dtype=float) for column in adapter.COMPACT}
    for column in adapter.CATEGORICAL:
        data[column]=['b' if i%2 else 'a' for i in range(rows)]
    result=pd.DataFrame(data)
    result.loc[0,'HomePlanet']='__MISSING__'
    result.loc[1,'Age']=np.nan
    return result


class TinyForward(torch.nn.Module):
    def forward(self, x_src, y_src, is_cls=True):
        query=x_src[:,y_src.shape[1]:,:]
        logit=torch.nan_to_num(query[...,0],nan=0.).reshape(-1)*.03
        p=torch.sigmoid(logit)
        return torch.stack([1-p,p],dim=1)


class FakeClassifier:
    def __init__(self):
        self.model=TinyForward()
        self.calls=[];self.fit_calls=[];self.moves=[]
        self.reverse_classes=False;self.shrink_context=False;self.bad_probability=False
    def fit(self,x,y):
        self.X_train=x.copy();self.y_train=y.copy()
        self.n_instances=len(x);self.num_classes=2
        self.fit_calls.append((x.copy(),y.copy()))
        if self.reverse_classes:self.classes_=np.array([1,0])
    def ensemble_predict_proba(self,x,**kwargs):
        self.calls.append((x.copy(),copy.deepcopy(kwargs)))
        y=self.y_train[:-1] if self.shrink_context else self.y_train
        support=self.X_train[:len(y)]
        tensors=torch.from_numpy(np.concatenate([support,x])[None])
        labels=torch.from_numpy(y[None])
        result=np.mean([self.model(x_src=tensors,y_src=labels,is_cls=True).numpy()
                        for _ in range(kwargs['n_ensembles'])],axis=0)
        if self.bad_probability:result[0,0]=np.nan
        return result
    def to(self,device):
        self.moves.append(device);self.model.to(device);return self


class AdapterTests(unittest.TestCase):
    def make(self,x=None,seed=123):
        x=frame() if x is None else x
        with patch.object(adapter.upstream,'verify_checkpoint'):
            return adapter.fit_context(x,np.arange(len(x))%2,np.arange(len(x))+100,seed,'cpu')

    def test_train_only_encoding_missing_and_unknown_are_distinct(self):
        x=frame();model=self.make(x)
        query=x.iloc[:3].copy()
        query['HomePlanet']=['unseen','__MISSING__',None]
        before=copy.deepcopy(model.preprocessor.maps)
        encoded=model.preprocessor.transform(query)
        self.assertEqual(encoded[0,0],-1)
        self.assertTrue(np.isnan(encoded[1:,0]).all())
        self.assertTrue(np.isnan(model.support[1,adapter.COMPACT.index('Age')]))
        self.assertNotIn('__MISSING__',model.preprocessor.maps['HomePlanet'])
        self.assertNotIn('unseen',model.preprocessor.maps['HomePlanet'])
        self.assertEqual(before,model.preprocessor.maps)
        self.assertEqual(encoded.shape,(3,13))
        self.assertEqual(encoded.dtype,np.float32)

    def test_fixed_official_ensemble_arguments_full_support_and_no_query_labels(self):
        model=self.make(seed=91);query=frame(5000)
        engine=FakeClassifier()
        with patch.object(adapter.upstream,'new_model',return_value=engine) as factory:
            p=model.predict_proba(query)
        factory.assert_called_once_with('cpu')
        self.assertEqual(p.shape,(5000,2))
        self.assertEqual(engine.calls[0][1],dict(adapter.PREDICT,seed=91))
        np.testing.assert_array_equal(engine.fit_calls[0][1],model.labels)
        np.testing.assert_array_equal(engine.fit_calls[0][0],model.support)
        self.assertEqual(len(engine.fit_calls),1)
        self.assertEqual(model.metadata['actual_forward_support_rows'],[16])
        self.assertEqual(model.metadata['last_forward_calls'],8)
        self.assertEqual(engine.moves,['cpu'])
        self.assertFalse(any(isinstance(v,torch.Tensor) for v in model.__dict__.values()))

    def test_joblib_restores_cpu_context_maps_seed_and_exact_predictions(self):
        model=self.make(seed=19);engines=[]
        def factory(device):
            self.assertEqual(device,'cpu');engine=FakeClassifier();engines.append(engine);return engine
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(adapter.upstream,'new_model',side_effect=factory):
            first=model.predict_proba(frame(7))
            model.device='mps'  # Serializing this string cannot save an accelerator engine.
            path=Path(folder)/'context.joblib';joblib.dump(model,path)
            restored=joblib.load(path)
            self.assertEqual(restored.device,'cpu')
            self.assertEqual(restored.seed,19)
            self.assertEqual(restored.preprocessor.maps,model.preprocessor.maps)
            np.testing.assert_array_equal(restored.support,model.support)
            np.testing.assert_array_equal(restored.labels,model.labels)
            second=restored.predict_proba(frame(7))
            np.testing.assert_array_equal(first,second)
            self.assertEqual(len(engines),2)
            self.assertEqual(restored.metadata['prediction_calls'],2)
            self.assertEqual(restored.metadata['model_rebuilds'],2)

    def test_invalid_class_order_reduced_context_and_nonfinite_release_engine(self):
        for defect in ('reverse_classes','shrink_context','bad_probability'):
            with self.subTest(defect=defect):
                model=self.make();engine=FakeClassifier();setattr(engine,defect,True)
                with patch.object(adapter.upstream,'new_model',return_value=engine), self.assertRaises(RuntimeError):
                    model.predict_proba(frame(5))
                self.assertEqual(engine.moves,['cpu'])
                self.assertEqual(len(engine.model._forward_pre_hooks),0)
                self.assertEqual(model.metadata['prediction_calls'],0)

    def test_changed_artifact_source_weight_or_context_is_rejected_before_model(self):
        for defect in ('weight','source','context','seed','settings'):
            with self.subTest(defect=defect):
                model=self.make()
                if defect=='weight':model.metadata['checkpoint_sha256']='invalid'
                if defect=='source':model.provenance['source_commit']='invalid'
                if defect=='context':model.support[1,0]=42
                if defect=='seed':model.seed+=1
                if defect=='settings':model.predict_settings['batch_size']=128
                with patch.object(adapter.upstream,'new_model') as factory,self.assertRaises(RuntimeError):
                    model.predict_proba(frame(5))
                factory.assert_not_called()
        with patch.object(adapter,'CODE_COMMIT','invalid'),self.assertRaises(RuntimeError):
            adapter.verify_sources()
        with patch.object(adapter.upstream,'verify_checkpoint',side_effect=RuntimeError('bad weight')):
            with self.assertRaisesRegex(RuntimeError,'bad weight'):
                adapter.fit_context(frame(),np.arange(16)%2,np.arange(16),1,'cpu')

    def test_bad_training_labels_ids_and_infinite_inputs_rejected(self):
        with patch.object(adapter.upstream,'verify_checkpoint'):
            for labels,ids in [(np.zeros(16),np.arange(16)),(np.arange(16)%2,np.zeros(16,dtype=int)),
                               (np.arange(15)%2,np.arange(16)),(np.arange(16)%2,np.arange(16)+.5)]:
                with self.subTest(labels=len(labels),ids=ids[:2]),self.assertRaises(ValueError):
                    adapter.fit_context(frame(),labels,ids,1,'cpu')
            invalid=frame();invalid.loc[0,'Age']=np.inf
            with self.assertRaises(ValueError):self.make(invalid)


if __name__=='__main__':
    unittest.main()
