"""Synthetic CPU and process supervision tests; no weights, targets or MPS."""
import copy
import hashlib
import multiprocessing
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

import mitra_long_context_probe as probe


def sleeping_child(ready):
    ready.set()
    time.sleep(30)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(3, 4), torch.nn.Linear(4, 10)])

    def forward(self, query):
        return self.layers[1](torch.tanh(self.layers[0](query)))


def successful_worker_record():
    updates = []
    for k in range(1, 5):
        updates.append(dict(step=k, lr=probe.base.literal_learning_rate(k), support_rows=3072,
            query_rows=256, disjoint_ids=True, query_equal_v81=True, support_prefix_equal_v81=True,
            rng_state_equal_v81=True, parameters_with_finite_grad=probe.PARAMETERS,
            optimizer_state_tensor_bytes=probe.OPTIMIZER_BYTES, optimizer_state_steps=[k],
            updated_blocks={f'layers.{i}':True for i in range(12)}, seconds=float(10+k)))
    validations = [dict(support_rows=n, query_rows=q, optimizer_state_steps=[step],
        optimizer_state_tensor_bytes=probe.OPTIMIZER_BYTES, full_support=True, gradients_cleared=True,
        moments_unchanged=True, probabilities_finite_normalized=True, seconds=float(k+1))
        for k, (n, q, step) in enumerate([(3708, 928, 2), (6954, 1739, 3)])]
    return dict(status='worker_complete', recipe=probe.RECIPE, updates=updates, validations=validations,
                snapshots=[], resources_after_release={}, release_complete=True)


class FakeProcess:
    def __init__(self, out, record=None, hang=False):
        self.out, self.record, self.hang = out, record or successful_worker_record(), hang
        self.live = False
        self.exitcode = 0
        self.pid = 12345
        self.killed = self.closed = False

    def start(self):
        probe.write(self.out/'partial.json', {'preserve':'partial synthetic evidence'})
        if self.hang:
            self.live = True
        else:
            probe.write(self.out/'worker_result.json', self.record)

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return self.live

    def kill(self):
        self.killed = True; self.live = False; self.exitcode = -9

    def close(self):
        self.closed = True


class SamplerTests(unittest.TestCase):
    def test_800_paired_permutations_prefix_query_and_rng(self):
        for n in [3708, 4636, 6954, 8693]:
            for seed in [42, 20261112, 20261113, 20261114]:
                rng = np.random.default_rng(seed)
                reference = np.random.default_rng(seed)
                independent = np.random.default_rng(seed)
                for _ in range(50):
                    support, query = probe.paired_episode(n, rng, reference)
                    old, old_query = probe.original.sample_episode(n, independent)
                    np.testing.assert_array_equal(old, support[:1024])
                    np.testing.assert_array_equal(old_query, query)
                    self.assertEqual(len(np.unique(np.r_[support, query])), 3328)
                    self.assertEqual(rng.bit_generator.state, independent.bit_generator.state)

    def test_shortage_fails_before_rng_consumption_and_bad_reference_fails(self):
        rng, ref = np.random.default_rng(42), np.random.default_rng(42)
        state = copy.deepcopy(rng.bit_generator.state)
        with self.assertRaises(ValueError):probe.paired_episode(3327, rng, ref)
        self.assertEqual(rng.bit_generator.state, state)
        with self.assertRaises(RuntimeError):probe.paired_episode(3708, rng, np.random.default_rng(99))


class ResourceMathTests(unittest.TestCase):
    def test_real_tiny_cpu_four_adamw_updates_survive_both_validations(self):
        torch.set_num_threads(4); torch.manual_seed(5)
        model = Tiny()
        data, labels, query = torch.randn(8693, 3), torch.arange(8693) % 2, torch.randn(1739, 3)
        optimizer = probe.base.optimizer_for(model)
        stages, actual_calls = [], []
        def forward(net, support, y, x):
            actual_calls.append((len(support), len(x), net.training))
            return net(x+support.mean(dim=0)*.01)
        with patch.object(probe.base, 'model_forward', forward):
            updates, validations = probe.sequence(model, optimizer, data, labels, query,
                lambda stage, snap:stages.append(stage), lambda *args:None)
        self.assertEqual([r['optimizer_state_steps'] for r in updates], [[1], [2], [3], [4]])
        self.assertEqual([r['optimizer_state_steps'] for r in validations], [[2], [3]])
        self.assertEqual(actual_calls, [(3072,256,True), (3072,256,True), (3708,928,False),
                                      (3072,256,True), (6954,1739,False), (3072,256,True)])
        self.assertEqual(len({r['optimizer_state_tensor_bytes'] for r in updates+validations}), 1)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertLess(stages.index('validation3708x928:after'), stages.index('step3:before_forward'))
        self.assertLess(stages.index('validation6954x1739:after'), stages.index('step4:before_forward'))
        self.assertTrue(all(all(r['updated_blocks'].values()) for r in updates))

    def test_native_head_shape_and_nonfinite_are_rejected(self):
        model = Tiny(); data = torch.randn(3328, 3); y = torch.arange(3328) % 2
        for bad in [torch.zeros(256, 2), torch.full((256, 10), float('nan'))]:
            with patch.object(probe.base, 'model_forward', return_value=bad):
                with self.assertRaises((RuntimeError, FloatingPointError)):
                    probe.update(model, probe.base.optimizer_for(model), data, y,
                                 np.random.default_rng(42), np.random.default_rng(42), 1, lambda *a:None)

    def test_cost_uses_slowest_warm_step_and_rejects_partial(self):
        record = successful_worker_record()
        record['updates'][0]['seconds'] = 999  # Cold first step is deliberately excluded.
        record['updates'][2]['seconds'] = 100
        cost = probe.qualify(record)
        self.assertEqual(cost['warm_update_max_seconds'], 100)
        self.assertEqual(cost['screen_projection_seconds'], 1.25*(300*100+153*1+600))
        self.assertEqual(cost['full_export_projection_seconds'], 1.25*(100*100+51*2+600))
        self.assertTrue(cost['passed']); self.assertFalse(cost['automatic_cv_authorization'])
        record['updates'][3]['seconds'] = 150
        self.assertFalse(probe.qualify(record)['passed'])
        for mutation in ['partial', 'moments', 'release', 'pairing']:
            bad = successful_worker_record()
            if mutation == 'partial':bad['updates'].pop()
            if mutation == 'moments':bad['validations'][0]['optimizer_state_steps'] = [1]
            if mutation == 'release':bad['release_complete'] = False
            if mutation == 'pairing':bad['updates'][2]['query_equal_v81'] = False
            with self.assertRaises(RuntimeError):probe.qualify(bad)


class GuardAndSupervisionTests(unittest.TestCase):
    def test_existing_output_preserved_before_quota_or_runtime(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(probe, 'quota_guard') as quota:
            p = Path(folder)/'result.json'; p.write_text('historical failure')
            with self.assertRaises(FileExistsError):probe.run(folder, 'unused')
            self.assertEqual(p.read_text(), 'historical failure'); quota.assert_not_called()

    def test_fresh_quota_denial_creates_no_probe(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'new'
            with patch.object(probe, 'quota_guard', side_effect=RuntimeError('guard stopped')), patch.object(probe, 'contract') as contract:
                with self.assertRaises(RuntimeError):probe.run(out, 'PRIVATE_SESSION')
            self.assertFalse(out.exists()); contract.assert_not_called()

    def test_uncommitted_source_cannot_spawn(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'new'
            with patch.object(probe, 'quota_guard'), patch.object(probe, 'contract', side_effect=RuntimeError('not committed')), patch.object(probe.multiprocessing, 'get_context') as spawn:
                with self.assertRaises(RuntimeError):probe.run(out, 'PRIVATE_SESSION')
            spawn.assert_not_called()
            result = probe.read(out/'result.json')
            self.assertFalse(result['resource_qualification'])
            self.assertNotIn('PRIVATE_SESSION', (out/'result.json').read_text())

    def test_source_and_environment_changes_fail_before_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root/'source.py').write_text('original')
            frozen = dict(sources={'source.py':probe.base.sha256(root/'source.py')},
                          environment_sha256=hashlib.sha256(b'pin\n').hexdigest(),
                          checkpoint_sha256=probe.base.WEIGHT_SHA, backend=probe.base.SETTINGS)
            with patch.object(probe, 'ROOT', root), patch.object(probe.base, 'environment_lock', return_value='pin\n'), patch.object(probe.base, 'verify_checkpoint') as verify:
                probe.check_frozen(frozen); verify.assert_called_once()
                (root/'source.py').write_text('changed')
                with self.assertRaises(RuntimeError):probe.check_frozen(frozen)
                verify.assert_called_once()
            (root/'source.py').write_text('original')
            with patch.object(probe, 'ROOT', root), patch.object(probe.base, 'environment_lock', return_value='changed\n'), patch.object(probe.base, 'verify_checkpoint') as verify:
                with self.assertRaises(RuntimeError):probe.check_frozen(frozen)
                verify.assert_not_called()

    def run_fake(self, out, process):
        context = Mock(); context.Process.return_value = process
        with patch.object(probe, 'quota_guard'), patch.object(probe, 'contract', return_value=({'sources':{}}, 'fixture\n')), patch.object(probe, 'check_frozen'), patch.object(probe, 'require_committed'), patch.object(probe.multiprocessing, 'get_context', return_value=context):
            return probe.run(out, 'PRIVATE_SESSION')

    def test_complete_and_cost_failure_never_authorize_cv(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'complete'; process = FakeProcess(out)
            result = self.run_fake(out, process)
            self.assertTrue(result['resource_qualification'])
            self.assertFalse(result['candidate_authorized'])
            self.assertFalse(result['cost_gate']['automatic_cv_authorization'])
            self.assertTrue(process.closed)
            self.assertEqual(probe.read(out/'result.json')['status'], 'complete')
            out = Path(folder)/'slow'; record = successful_worker_record()
            record['updates'][3]['seconds'] = 150
            result = self.run_fake(out, FakeProcess(out, record))
            self.assertTrue(result['resource_qualification'])
            self.assertFalse(result['cost_qualified_for_root_review'])

    def test_external_timeout_kills_owned_child_and_keeps_partial(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'timeout'; process = FakeProcess(out, hang=True)
            with self.assertRaises(TimeoutError):self.run_fake(out, process)
            self.assertTrue(process.killed); self.assertTrue(process.closed)
            self.assertEqual(probe.read(out/'partial.json'), {'preserve':'partial synthetic evidence'})
            self.assertEqual(probe.read(out/'result.json')['status'], 'failed_inconclusive')
            self.assertFalse(probe.read(out/'result.json')['resource_qualification'])
            self.assertTrue(probe.read(out/'external_timeout.json')['child_reaped'])
            self.assertFalse((out/'worker_result.json').exists())

    def test_actual_spawned_blocked_cpu_child_is_killed(self):
        context = multiprocessing.get_context('spawn')
        ready = context.Event()
        process = context.Process(target=sleeping_child, args=(ready,))
        process.start()
        try:
            self.assertTrue(ready.wait(timeout=15))
            started = time.monotonic()
            with self.assertRaises(TimeoutError):probe.supervise(process, started+.05)
            self.assertLess(time.monotonic()-started, 5)
            self.assertFalse(process.is_alive()); self.assertEqual(process.exitcode, -9)
        finally:
            if process.is_alive():process.kill(); process.join(timeout=5)
            process.close()

    def test_worker_failure_cannot_be_promoted_to_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'failed'
            record = successful_worker_record(); record.update(status='failed_inconclusive', error='synthetic OOM',
                error_type='OutOfMemoryError', current_stage='step3:after_forward')
            with self.assertRaises(RuntimeError):self.run_fake(out, FakeProcess(out, record))
            self.assertEqual(probe.read(out/'worker_result.json')['error'], 'synthetic OOM')
            result = probe.read(out/'result.json')
            self.assertFalse(result['resource_qualification'])
            self.assertEqual(result['child_error_type'], 'OutOfMemoryError')
            self.assertEqual(result['child_error'], 'synthetic OOM')
            self.assertEqual(result['child_current_stage'], 'step3:after_forward')


if __name__ == '__main__':
    unittest.main()
