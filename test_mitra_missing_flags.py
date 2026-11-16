import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import joblib
import numpy as np

import mitra_missing_flags as core


class MissingFlagsTests(unittest.TestCase):
    def frames(self):
        x=core.base.synthetic_frame(61,1);q=core.base.synthetic_frame(17,2)
        for i,col in enumerate(core.NUMERIC):x.loc[np.arange(i,61,11),col]=np.nan
        q.loc[0,'HomePlanet']='never_seen';q.loc[1,'Destination']='__MISSING__'
        return x,q

    def test_original13_exact_with_train_only_vocabulary_means_and_unseen_query(self):
        x,q=self.frames();y=np.arange(len(x))%2
        prep=core.MissingFlagsPreprocessor().fit(x,y,20261112)
        control=core.base.MitraPreprocessor().fit(x,y,20261112)
        self.assertEqual(prep.n_features,20)
        np.testing.assert_array_equal(prep.encode(x)[:,:13],control.encode(x))
        np.testing.assert_array_equal(prep.encode(q)[:,:13],control.encode(q))
        for frame in [x,q]:np.testing.assert_array_equal(prep.transform(frame)[:,prep.control_positions],control.transform(frame))
        self.assertEqual(prep.vocab,control.vocab)
        self.assertNotIn('never_seen',prep.vocab['HomePlanet'])
        self.assertEqual(prep.encode(q)[0,0],0)
        self.assertEqual(prep.encode(q)[1,2],1)
        np.testing.assert_array_equal(prep.upstream.pre_nan_mean[:13],control.upstream.pre_nan_mean)
        np.testing.assert_array_equal(prep.encode(x)[:,13:],x[core.NUMERIC].isna().to_numpy(dtype=np.float32))
        malicious=q.copy();malicious[core.FLAGS]=0
        np.testing.assert_array_equal(prep.transform(malicious),prep.transform(q))

    def test_constant_and_allmissing_active_mapping_preserves_original_mirror(self):
        x,q=self.frames();x['Age']=np.nan;x['HomePlanet']='Earth';x['Spa']=7.
        prep=core.MissingFlagsPreprocessor().fit(x,np.arange(len(x))%2,20261113)
        self.assertNotIn('Age',prep.active_columns);self.assertNotIn('Age_missing',prep.active_columns)
        self.assertNotIn('HomePlanet',prep.active_columns);self.assertNotIn('Spa_missing',prep.active_columns)
        self.assertLess(prep.n_features,20)
        np.testing.assert_array_equal(prep.upstream.mirror[:,prep.control_positions],prep.control.upstream.mirror)
        np.testing.assert_array_equal(prep.transform(q)[:,prep.control_positions],prep.control.transform(q))
        self.assertTrue(np.isfinite(prep.transform(q)).all())

    def test_schema_context_and_flags_survive_cpu_portable_roundtrip(self):
        x,q=self.frames();y=np.arange(len(x))%2
        with patch.object(core.base,'verify_checkpoint',return_value={'fixture':True}), \
                patch.object(core.base,'load_file',return_value={'fixture':core.base.torch.ones(1)}):
            predictor=core.fit_missing_flags(x,y,np.arange(len(x)),20261114,'cpu')
        self.assertEqual(predictor.query_chunk,2318);self.assertEqual(len(predictor.support),len(x))
        self.assertEqual(predictor.metadata['optimizer_steps'],0)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.joblib';joblib.dump(predictor,path);restored=joblib.load(path)
        self.assertIsNone(restored._engine);self.assertEqual(restored.device,'cpu')
        self.assertEqual(restored.preprocessor.schema(),predictor.preprocessor.schema())
        np.testing.assert_array_equal(restored.support,predictor.support)
        np.testing.assert_array_equal(restored.labels,y)
        np.testing.assert_array_equal(restored.preprocessor.transform(q),predictor.preprocessor.transform(q))
        with self.assertRaises(ValueError):core.fit_missing_flags(x,y,np.zeros(len(x)),42,'cpu')
        with self.assertRaises(ValueError):core.valid_probabilities(np.zeros((3,2)),3)


if __name__=='__main__':unittest.main()
