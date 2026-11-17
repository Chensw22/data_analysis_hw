"""Pure stdlib/AST/mock checks; never construct or deserialize native models."""
import ast
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import iltm_r128bn_cv_false as candidate

ROOT=Path(__file__).resolve().parent


def functions(name):
    tree=ast.parse((ROOT/name).read_text())
    return {n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}


def normalized(node):
    return ast.dump(node,include_attributes=False)


def false_guard(stage='runtime'):
    if os.environ.get('KMP_DUPLICATE_LIB_OK')!='FALSE':raise RuntimeError('FALSE missing: '+stage)


class FalseMigrationTests(unittest.TestCase):
    def test_original_math_packet_splits_and_metrics_ast_equal(self):
        old,new=functions('iltm_r128bn_cv.py'),functions('iltm_r128bn_cv_false.py')
        for name in ['deadline_check','verify_files','validate_resource','validate_splits','save_packet',
                     'load_packet','validate_native_fold','read_ids','validate_worker','assess','paired','random_gate']:
            self.assertEqual(normalized(old[name]),normalized(new[name]),name)

    def test_exact_native_calls_and_fit_keywords_unchanged(self):
        old,new=functions('iltm_r128bn_cv.py')['worker'],functions('iltm_r128bn_cv_false.py')['worker']
        def calls(tree):
            result=[]
            for n in ast.walk(tree):
                if not isinstance(n,ast.Call):continue
                name=ast.unparse(n.func)
                if name in ['iLTMClassifier','model.fit','resource.probabilities','resource.expected_parameters',
                            'resource.check_model','model._release_training_model',
                            'torch.set_num_threads','torch.set_num_interop_threads']:
                    result.append(normalized(n))
            return result
        self.assertEqual(calls(old),calls(new))
        fitting=[n for n in ast.walk(new) if isinstance(n,ast.Call) and ast.unparse(n.func)=='model.fit']
        self.assertEqual(len(fitting),1)
        self.assertEqual({k.arg:ast.literal_eval(k.value) for k in fitting[0].keywords},
                         {'eval_set':None,'fit_max_time':None,'return_partial_on_timeout':False})

    def test_parent_freezes_env_before_false_and_does_not_rebuild(self):
        snapshot={'duplicate_runtime_flag':None,'numeric_modules_already_loaded':[]}
        supplied={'KMP_DUPLICATE_LIB_OK':'FALSE','OMP_NUM_THREADS':'4'}
        def entry():
            self.assertNotIn('KMP_DUPLICATE_LIB_OK',os.environ)
            return supplied
        with patch.dict(os.environ,{},clear=True), patch.object(candidate,'ENTRY_SNAPSHOT',snapshot), \
             patch.object(candidate.bridge,'entry_environment',side_effect=entry,create=True) as call, \
             patch.object(candidate.bridge,'require_false',side_effect=false_guard,create=True):
            env=candidate.parent_environment()
            self.assertEqual(os.environ['KMP_DUPLICATE_LIB_OK'],'FALSE')
            call.assert_called_once()
            supplied['OMP_NUM_THREADS']='99'
            self.assertEqual(env['OMP_NUM_THREADS'],'4')
        run=functions('iltm_r128bn_cv_false.py')['run']
        self.assertEqual(ast.unparse(run.body[0]),'child_env = parent_environment()')
        self.assertEqual(sum(isinstance(n,ast.Call) and ast.unparse(n.func)=='parent_environment' for n in ast.walk(run)),1)
        popen=[n for n in ast.walk(run) if isinstance(n,ast.Call) and ast.unparse(n.func)=='subprocess.Popen']
        self.assertEqual(len(popen),1)
        self.assertEqual(ast.unparse(next(k.value for k in popen[0].keywords if k.arg=='env')),'child_env')

    def test_any_flag_or_preloaded_numeric_entry_rejected(self):
        for flag in ['TRUE','FALSE','0','']:
            with patch.object(candidate,'ENTRY_SNAPSHOT',{'duplicate_runtime_flag':flag,'numeric_modules_already_loaded':[]}), \
                 patch.object(candidate.bridge,'entry_environment',create=True) as entry:
                with self.assertRaises(RuntimeError):candidate.parent_environment()
                entry.assert_not_called()
        with patch.object(candidate,'ENTRY_SNAPSHOT',{'duplicate_runtime_flag':None,'numeric_modules_already_loaded':['numpy']}), \
             patch.object(candidate.bridge,'entry_environment',create=True) as entry:
            with self.assertRaises(RuntimeError):candidate.parent_environment()
            entry.assert_not_called()

    def test_bridge_failure_blocks_old_resource_and_data(self):
        with patch.dict(os.environ,{'KMP_DUPLICATE_LIB_OK':'FALSE'}), \
             patch.object(candidate.bridge,'require_false',side_effect=false_guard,create=True), \
             patch.object(candidate.bridge,'resource_gate',side_effect=RuntimeError('bridge not complete'),create=True), \
             patch.object(candidate.resource,'require_committed') as historical:
            with self.assertRaisesRegex(RuntimeError,'bridge not complete'):candidate.resource_gate()
            historical.assert_not_called()

    def test_mutated_flag_rejected_before_gate(self):
        with patch.dict(os.environ,{'KMP_DUPLICATE_LIB_OK':'TRUE'}), \
             patch.object(candidate.bridge,'require_false',side_effect=false_guard,create=True), \
             patch.object(candidate.bridge,'resource_gate',create=True) as bridge_gate:
            with self.assertRaisesRegex(RuntimeError,'FALSE missing'):candidate.resource_gate()
            bridge_gate.assert_not_called()

    def test_missing_failed_or_wrong_bridge_root_audit_rejected(self):
        valid={'status':'verified_complete_FALSE_runtime_bridge','bridge_result_sha256':'expected-result-sha',
               'single_FALSE_training_completed':True,
               'historical_eight_FALSE_inference_and_reload_completed':True,
               'eight_FALSE_training_repeated':False}
        cases=[None,{**valid,'status':'failed'},
               {**valid,'bridge_result_sha256':'wrong'},
               {**valid,'single_FALSE_training_completed':False},
               {**valid,'historical_eight_FALSE_inference_and_reload_completed':False},
               {**valid,'eight_FALSE_training_repeated':True}]
        for value in cases:
            with self.subTest(audit=value), patch.dict(os.environ,{'KMP_DUPLICATE_LIB_OK':'FALSE'}), \
                 patch.object(candidate.bridge,'require_false',side_effect=false_guard), \
                 patch.object(candidate.bridge,'resource_gate',return_value={}), \
                 patch.object(candidate,'validate_migration'), \
                 patch.object(candidate.resource,'require_committed') as committed, \
                 patch.object(candidate,'sha',return_value='expected-result-sha'), \
                 patch.object(candidate,'read',side_effect=FileNotFoundError('root audit missing') if value is None else None,
                              return_value=value):
                with self.assertRaises((FileNotFoundError,RuntimeError)):candidate.resource_gate()
                committed.assert_called_once_with(['reports/iltm_false_runtime_bridge/root_audit.json',
                                                   'reports/iltm_false_runtime_bridge/result.json'])

    def test_bridge_root_audit_must_be_committed(self):
        with patch.dict(os.environ,{'KMP_DUPLICATE_LIB_OK':'FALSE'}), \
             patch.object(candidate.bridge,'require_false',side_effect=false_guard), \
             patch.object(candidate.bridge,'resource_gate',return_value={}), \
             patch.object(candidate,'validate_migration'), \
             patch.object(candidate.resource,'require_committed',side_effect=RuntimeError('uncommitted root audit')), \
             patch.object(candidate,'read') as read:
            with self.assertRaisesRegex(RuntimeError,'uncommitted root audit'):candidate.resource_gate()
            read.assert_not_called()

    def test_worker_guard_before_packets_and_each_native_progress(self):
        worker=functions('iltm_r128bn_cv_false.py')['worker']
        source=ast.unparse(worker)
        self.assertLess(source.index("flag_check('worker_before_numeric_import')"),source.index('load_packet(out)'))
        self.assertIn("flag_check('after_native_import')",source)
        self.assertIn("flag_check('after_release')",source)
        observer=[n for n in ast.walk(worker) if isinstance(n,ast.FunctionDef) and n.name=='observer_progress'][0]
        self.assertEqual(ast.unparse(observer.body[0]),"flag_check('native_observer_progress')")
        calls=[n for n in ast.walk(worker) if isinstance(n,ast.Call) and ast.unparse(n.func)=='resource.NativeObservation']
        self.assertEqual(ast.unparse(calls[0].args[1]),'observer_progress')

    def test_swallowed_observer_flag_error_aborts_immediately(self):
        class OriginalObserver:
            error=None
            def __call__(self, frame, event, arg):
                # Original observer catches Exception from its progress hook.
                try:false_guard('native_progress')
                except Exception as error:self.error=str(error)
        with patch.dict(os.environ,{'KMP_DUPLICATE_LIB_OK':'TRUE'}):
            with self.assertRaisesRegex(RuntimeError,'FALSE missing'):
                candidate.FailClosedObservation(OriginalObserver())(None,'return',None)

    def test_migration_retains_old_recipe_hash_and_new_paths(self):
        candidate.validate_migration()
        self.assertEqual(candidate.OUTPUT.name,'iltm_r128bn_cv_false')
        self.assertEqual(candidate.REPORT.name,'iltm_r128bn_cv_false')
        self.assertEqual(candidate.BUDGET,14400)
        self.assertEqual(candidate.resource.LIMIT,10*1024**3)


if __name__=='__main__':unittest.main()
