import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import kaggle_gpu_readiness as runtime
import prepare_kaggle_gpu_readiness as preparation


class ReadinessTests(unittest.TestCase):
    def test_refuses_unrequested_hardware(self):
        runtime.require_requested_gpu(['NVIDIA L4'])
        for names in [[], ['Tesla T4'], ['NVIDIA L40'], ['Tesla T4','NVIDIA L4']]:
            with self.subTest(names=names), self.assertRaises(RuntimeError):
                runtime.require_requested_gpu(names)

    def test_private_preparation_preserves_code_and_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in preparation.SOURCES:
                (root/name).write_text('# fixture\n')
            out=root/'artifacts/probe'
            with patch.object(preparation,'ROOT',root), \
                 patch.object(preparation,'committed_sources',return_value='a'*40), \
                 patch.object(preparation,'official_username',return_value='fixtureowner'):
                record=preparation.prepare(out)
                metadata=json.loads((out/'kernel-metadata.json').read_text())
                self.assertTrue(metadata['is_private'])
                self.assertTrue(metadata['enable_gpu'])
                self.assertFalse(metadata['enable_internet'])
                self.assertEqual(metadata['machine_shape'],'NvidiaL4')
                for key in ['dataset_sources','competition_sources','kernel_sources','model_sources']:
                    self.assertEqual(metadata[key],[])
                self.assertEqual(record['code_sha256'],record['source_hashes'][preparation.SOURCES[0]])
                self.assertEqual(record['commands']['push'][-4:],['--accelerator','NvidiaL4','--timeout','180'])
                self.assertFalse(record['upload_attempted'])
                with self.assertRaises(RuntimeError):preparation.prepare(out)
                with self.assertRaises(ValueError):preparation.prepare(root/'reports/probe')

    def test_uncommitted_sources_stop_before_account_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(preparation,'ROOT',root), \
                 patch.object(preparation,'committed_sources',side_effect=RuntimeError('uncommitted')), \
                 patch.object(preparation,'official_username') as account:
                with self.assertRaises(RuntimeError):preparation.prepare(root/'artifacts/probe')
                account.assert_not_called()
                self.assertFalse((root/'artifacts/probe').exists())


if __name__=='__main__':unittest.main()
