import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import kaggle_gpu_readiness_v2 as runtime
import prepare_kaggle_gpu_readiness_v2 as preparation


class ReadinessV2Tests(unittest.TestCase):
    def test_refuses_hardware_substitution(self):
        runtime.require_requested_gpu(['Tesla T4', 'Tesla T4'])
        for names in [[], ['NVIDIA L4'], ['Tesla T40'], ['Tesla T4', 'NVIDIA L4']]:
            with self.subTest(names=names), self.assertRaises(RuntimeError):
                runtime.require_requested_gpu(names)

    def test_driver_evidence_survives_missing_tools_without_torch(self):
        with patch.object(runtime.subprocess, 'run', side_effect=FileNotFoundError('not installed')) as command, \
             patch.object(runtime.ctypes, 'CDLL', side_effect=OSError('no libcuda')):
            result = runtime.driver_diagnostics()
        self.assertEqual(command.call_args.kwargs['timeout'], 10)
        self.assertEqual(result['nvidia_smi']['error_type'], 'FileNotFoundError')
        self.assertEqual(result['libcuda']['error_type'], 'OSError')

    def test_cpu_wheel_keeps_driver_evidence_and_never_computes(self):
        torch = MagicMock()
        torch.__version__ = 'fixture+cpu'
        torch.version.cuda = None
        torch.backends.cuda.is_built.return_value = False
        torch.cuda.is_available.return_value = False
        evidence = {'libcuda': {'loaded': True, 'cuInit': 0, 'device_count': 2}}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(runtime, 'OUTPUT', Path(directory)/'result.json'), \
             patch.object(runtime, 'driver_diagnostics', return_value=evidence), \
             patch.dict(sys.modules, {'torch': torch}), patch('builtins.print'):
            result = runtime.run()
        self.assertEqual(result['status'], 'failed_readiness')
        self.assertEqual(result['driver'], evidence)
        self.assertFalse(result['torch_cuda_built'])
        torch.randn.assert_not_called()

    def test_draft_commit_finalization_and_snapshot_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in preparation.SOURCES:
                (root/name).write_text('# fixture\n')
            out = root/'artifacts/probe'
            with patch.object(preparation, 'ROOT', root), \
                 patch.object(preparation, 'current_commit', return_value='a'*40), \
                 patch.object(preparation, 'committed_sources', return_value='b'*40), \
                 patch.object(preparation, 'official_username', return_value='fixtureowner'), \
                 patch.object(preparation.subprocess, 'run', return_value=subprocess.CompletedProcess([],0)):
                draft = preparation.prepare(out)
                self.assertNotIn('commands', draft)
                self.assertIsNone(draft['prepared_from_commit'])
                metadata = json.loads((out/'kernel-metadata.json').read_text())
                self.assertEqual(metadata['machine_shape'], 'NvidiaTeslaT4')
                self.assertEqual(metadata['docker_image_pinning_type'], 'latest')
                self.assertNotIn('docker_image', metadata)
                self.assertTrue(metadata['is_private'])
                self.assertTrue(metadata['enable_gpu'])
                self.assertFalse(metadata['enable_internet'])
                self.assertFalse(metadata['enable_tpu'])
                for key in ['dataset_sources','competition_sources','kernel_sources','model_sources']:
                    self.assertEqual(metadata[key], [])
                (out/preparation.SOURCES[0]).write_text('# tampered\n')
                with self.assertRaises(RuntimeError): preparation.prepare(out, True)
                (out/preparation.SOURCES[0]).write_bytes((root/preparation.SOURCES[0]).read_bytes())
                final = preparation.prepare(out, True)
                self.assertEqual(final['prepared_from_commit'], 'b'*40)
                self.assertEqual(final['code_sha256'], final['source_hashes'][preparation.SOURCES[0]])
                self.assertEqual(final['commands']['push'][-4:], ['--accelerator','NvidiaTeslaT4','--timeout','180'])
                self.assertFalse(final['upload_attempted'])
                # Exercise installed official request construction with the entire
                # network client replaced. No real save_kernel call is possible.
                from kaggle.api.kaggle_api_extended import KaggleApi
                client = MagicMock()
                api = KaggleApi.__new__(KaggleApi)
                with patch.object(api, 'build_kaggle_client') as build:
                    build.return_value.__enter__.return_value = client
                    api.kernels_push(str(out), timeout='180', acc='NvidiaTeslaT4')
                request = client.kernels.kernels_api_client.save_kernel.call_args.args[0]
                self.assertEqual(request.machine_shape, 'NvidiaTeslaT4')
                self.assertEqual(request.docker_image_pinning_type, 'latest')
                self.assertEqual(request.session_timeout_seconds, 180)
                self.assertTrue(request.is_private)
                self.assertFalse(request.enable_internet)
                self.assertEqual(request.text, (out/preparation.SOURCES[0]).read_text())
                with self.assertRaises(RuntimeError): preparation.prepare(out, True)
                with self.assertRaises(ValueError): preparation.prepare(root/'reports/probe')

    def test_uncommitted_or_unignored_files_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(preparation, 'ROOT', root), \
                 patch.object(preparation.subprocess, 'run', return_value=subprocess.CompletedProcess([],0)), \
                 patch.object(preparation, 'committed_sources', side_effect=RuntimeError('uncommitted')):
                with self.assertRaises(RuntimeError): preparation.prepare(root/'artifacts/probe', True)
            with patch.object(preparation, 'ROOT', root), \
                 patch.object(preparation.subprocess, 'run', return_value=subprocess.CompletedProcess([],1)), \
                 patch.object(preparation, 'official_username') as account:
                with self.assertRaises(ValueError): preparation.prepare(root/'artifacts/probe')
                account.assert_not_called()


if __name__ == '__main__':
    unittest.main()
