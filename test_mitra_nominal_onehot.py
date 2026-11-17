import io
import unittest
from unittest.mock import patch

import joblib
import numpy as np

import mitra_nominal_onehot as core


class NominalTests(unittest.TestCase):
    def frames(self):
        x=core.base.synthetic_frame(97,11);q=core.base.synthetic_frame(17,12)
        x.loc[::17,'Destination']='__MISSING__'
        q.loc[0,['HomePlanet','Destination']]='new_unseen'
        q.loc[1,['HomePlanet','Destination']]='__MISSING__'
        return x,q

    def test_known_unknown_missing_and_unchanged_eleven_bitwise(self):
        x,q=self.frames();y=np.arange(len(x))%2
        prep=core.NominalOneHotPreprocessor().fit(x,y,20261112)
        self.assertEqual(prep.n_features,19)
        for frame in [x,q]:
            np.testing.assert_array_equal(prep.transform(frame)[:,prep.unchanged_positions],prep.control.transform(frame)[:,prep.control_positions])
        for parent in core.REPLACE:
            positions=[prep.columns.index(c) for c in prep.columns if prep.parent_by_column[c]==parent]
            self.assertEqual(prep.encode(q)[0,positions].sum(),0)
            self.assertEqual(prep.encode(q)[1,positions].sum(),1)
            self.assertEqual(prep.encode(q)[1,prep.columns.index(parent+'__missing')],1)
            self.assertNotIn('new_unseen',prep.vocabularies[parent])
        for col,sign in zip(prep.active_columns,prep.upstream.mirror[0]):
            self.assertEqual(sign,prep.parent_mirror_signs[prep.parent_by_column[col]])
        self.assertEqual(prep.schema()['mirror_rule'],core.MIRROR_RULE)

    def test_only_original_mirror_draw_and_rng_restored(self):
        x,_=self.frames();y=np.arange(len(x))%2
        original=core.base.Preprocessor.determine_mirror;calls=[]
        def record(obj,values):calls.append(values.shape[1]);return original(obj,values)
        np.random.seed(199);before=np.random.get_state()
        with patch.object(core.base.Preprocessor,'determine_mirror',record):
            prep=core.NominalOneHotPreprocessor().fit(x,y,20261113)
        self.assertEqual(calls,[13])
        after=np.random.get_state()
        for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
        self.assertEqual(prep.upstream.mirror.shape,(1,19))

    def test_constant_allmissing_missing_category_and_single_label(self):
        x,q=self.frames();x['Age']=np.nan;x['VIP']='constant';x['HomePlanet']='Earth';x['Destination']=None
        prep=core.NominalOneHotPreprocessor().fit(x,np.zeros(len(x),dtype=int),20261114)
        self.assertEqual(prep.vocabularies,{'HomePlanet':['Earth'],'Destination':[]})
        self.assertFalse(any(prep.parent_by_column[c] in core.REPLACE for c in prep.active_columns))
        self.assertNotIn('Age',prep.active_columns);self.assertNotIn('VIP',prep.active_columns)
        np.testing.assert_array_equal(prep.transform(q)[:,prep.unchanged_positions],prep.control.transform(q)[:,prep.control_positions])
        x,_=self.frames();x.loc[x.HomePlanet=='level_2','HomePlanet']='level_0'
        prep=core.NominalOneHotPreprocessor().fit(x,np.ones(len(x),dtype=int),42)
        self.assertEqual(len(prep.vocabularies['HomePlanet']),2)
        self.assertNotIn('level_2',prep.vocabularies['HomePlanet'])

    def test_renaming_categories_only_reorders_children(self):
        x,q=self.frames();mapping={'level_0':'z','level_1':'a','level_2':'m'}
        xr,qr=x.copy(),q.copy()
        for field in core.REPLACE:
            xr[field]=xr[field].replace(mapping);qr[field]=qr[field].replace(mapping)
        a=core.NominalOneHotPreprocessor().fit(x,np.arange(len(x))%2,42)
        b=core.NominalOneHotPreprocessor().fit(xr,np.arange(len(xr))%2,42)
        positions=[]
        for col in a.active_columns:
            if col in core.KEEP:target=col
            else:
                spec=a.child_specs[col]
                category=mapping.get(spec['category'],spec['category'])
                target=next(c for c,s in b.child_specs.items() if s=={**spec,'category':category})
            positions.append(b.active_columns.index(target))
        for left,right in [(x,xr),(q,qr)]:np.testing.assert_array_equal(a.transform(left),b.transform(right)[:,positions])
        self.assertEqual(a.parent_mirror_signs,b.parent_mirror_signs)

    def test_portable_full_schema_context_and_no_optimizer(self):
        x,q=self.frames();y=np.arange(len(x))%2
        with patch.object(core.base,'verify_checkpoint',return_value={'tiny':True}),patch.object(core.base,'load_file',return_value={'w':core.base.torch.ones(1)}):
            model=core.fit_nominal_onehot(x,y,np.arange(len(x)),42,'cpu')
        stream=io.BytesIO();joblib.dump(model,stream);stream.seek(0);restored=joblib.load(stream)
        self.assertEqual(restored.device,'cpu');self.assertIsNone(restored._engine)
        self.assertEqual(restored.metadata['optimizer_steps'],0);self.assertEqual(restored.query_chunk,2318)
        self.assertEqual(restored.preprocessor.schema(),model.preprocessor.schema())
        np.testing.assert_array_equal(restored.support,model.support)
        np.testing.assert_array_equal(restored.preprocessor.transform(q),model.preprocessor.transform(q))
        with self.assertRaises(ValueError):core.fit_nominal_onehot(x,y,np.zeros(len(x)),42,'cpu')


if __name__=='__main__':unittest.main()
