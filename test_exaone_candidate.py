from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import joblib
import numpy as np

import exaone_candidate as core
from exaone_resource_probe import fixture


class CandidateTests(unittest.TestCase):
    def test_train_only_numeric_categories_missing_and_schema_roundtrip(self):
        x,y,q=fixture(32,17);x['Spa']=np.nan;x['VIP']='constant'
        encoder=core.CompactEncoder().fit(x);a=encoder.transform(x);b=encoder.transform(q)
        self.assertEqual(a.dtype,np.float64);self.assertEqual(a.shape,(32,13))
        self.assertTrue(np.isnan(a[:,core.COMPACT.index('Spa')]).all())
        self.assertTrue(np.isnan(b[0,0]));self.assertNotIn('unseen_query_category',encoder.vocab['HomePlanet'])
        q.loc[1,'HomePlanet']='__MISSING__';self.assertTrue(np.isnan(encoder.transform(q)[1,0]))
        predictor=core.EXAONEPredictor(x,y,np.arange(len(x)))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'context.joblib';joblib.dump(predictor,path);loaded=joblib.load(path)
        self.assertIsNone(loaded._engine);self.assertEqual(loaded.device,'cpu')
        np.testing.assert_array_equal(loaded.encoder.transform(q),predictor.encoder.transform(q))
        np.testing.assert_array_equal(loaded.support,predictor.support)
        np.testing.assert_array_equal(loaded.labels,y)
        self.assertEqual(loaded.metadata,predictor.metadata)
        with self.assertRaises(ValueError):core.EXAONEPredictor(x,y,np.arange(len(x)),seed=1)
        with self.assertRaises(ValueError):core.EXAONEPredictor(x,y,np.zeros(len(x)))

    def test_loader_uses_explicit_digest_verification_and_local_class_only(self):
        class TinyModel:
            def to(self,device):self.device=device;return self
        class TinyClassifier:
            def __init__(self,manifest,device):self.manifest=manifest;self.device=device;self.model=TinyModel()
        manifest=replace(core.released_manifest('classification'),checkpoint_sha256=core.WEIGHT_SHA)
        with patch.object(core,'verify_checkpoint',return_value=manifest), \
                patch.object(core,'EXAONELocalClassifier',TinyClassifier), \
                patch.object(core,'load_classifier_checkpoint') as load, \
                patch.object(core.compat,'install_local',return_value={'local':True}):
            model=core.new_classifier(device='cpu')
            load.assert_called_once_with(core.CHECKPOINT,model.model,manifest,verify_checksum=True)
            self.assertEqual(model.device,'cpu');self.assertEqual(model.model.device,'cpu')
        with self.assertRaises(ValueError):core.new_classifier(device='cuda')

    def test_exact_native_executor_plan_cpu_inputs_and_all_view_counts(self):
        manifest=core.released_manifest('classification')
        class Tiny(core.torch.nn.Module):
            def forward(self,x,y,q,**kwargs):return core.torch.zeros((len(x),q.shape[1],10),dtype=core.torch.float16)
        clf=core.EXAONELocalClassifier(manifest,device='cpu',model=Tiny())
        clf._fitted_state={'support_x':np.zeros((7,13))}
        support=core.torch.zeros((8,7,13));labels=core.torch.zeros((8,7));query=core.torch.zeros((8,2318,13))
        out=clf._forward_chunked(support,labels,query)
        self.assertEqual(out.shape,(8,2318,10));self.assertEqual(out.device.type,'cpu')
        self.assertEqual(len(clf.forward_records),24)
        self.assertEqual([v['query_rows'] for v in clf.forward_records[:3]],[1024,1024,270])
        self.assertTrue(all(v['support_rows']==7 and v['views']==1 for v in clf.forward_records))
        with self.assertRaises(RuntimeError):clf._forward_chunked(support[:7],labels[:7],query[:7])
        with self.assertRaises(RuntimeError):clf._forward_chunked(support.double(),labels,query)


if __name__=='__main__':unittest.main()
