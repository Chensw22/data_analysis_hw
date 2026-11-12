import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import top5_adaptation_evaluate as candidate


class AdaptationPhaseTests(unittest.TestCase):
    def test_policy_unchanged_gates_and_explicit_existing_screen_disclosure(self):
        policy=json.loads((candidate.ROOT/candidate.POLICY).read_text())
        old=json.loads((candidate.ROOT/candidate.previous.POLICY).read_text())
        for section in ['A_research_online_gate','S_stable_local_superiority_gate','bootstrap_recipe']:
            self.assertEqual(policy[section],old[section])
        self.assertEqual(policy['fixed_scope_before_new_scores']['candidates'],['v76_Mitra_lastblock100'])
        self.assertEqual(policy['fixed_scope_before_new_scores']['group_split_seeds'],candidate.SEEDS)
        self.assertEqual(policy['resource_limit']['candidate_total_seconds'],21600)
        self.assertEqual(policy['resource_limit']['weekly_reserve_percent'],12)
        self.assertFalse(policy['batch_spec']['new_screen_training_allowed'])
        self.assertIn('observed before',policy['protocol_relation']['prescreen_disclosure'])

    def test_fresh_process_old_helpers_and_run042_sources_are_unchanged(self):
        code='''
import hashlib,json
from pathlib import Path
import top5_group_evaluate as group
import top5_generation_evaluate as generation
before=(group.PHASE,dict(group.CANDIDATES),generation.PHASE,dict(generation.CANDIDATES))
files=[Path(group.__file__),Path(generation.__file__),Path('artifacts/run_042/protocol.json')]
hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
import top5_adaptation_evaluate as new
assert before==(group.PHASE,dict(group.CANDIDATES),generation.PHASE,dict(generation.CANDIDATES))
assert list(new._g._impl.CANDIDATES)==['adapt']
assert new._g._impl.PHASE==new.PHASE
assert hashes==[hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
protocol=json.loads(Path('artifacts/run_042/protocol.json').read_text())
for relative,expected in protocol['sources'].items():
    assert hashlib.sha256(Path(relative).read_bytes()).hexdigest()==expected,relative
print('fresh_isolation_original_sources_ok')
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=candidate.ROOT,
            capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('fresh_isolation_original_sources_ok',result.stdout)

    def test_all9folds_share_one21600second_budget_and_only_training_labels(self):
        dev=np.arange(12);y=dev%2;groups=np.repeat(np.arange(6).astype(str),2)
        passengers=np.array([f'{g}_{i}' for i,g in enumerate(groups)])
        assignment=np.repeat([0,1,2,0,1,2],2).astype(np.int8)
        matrices={False:(pd.DataFrame({'x':dev}),[])};calls=[]
        runtime=SimpleNamespace(DEADLINE=None)
        def configure(*a,**kw):runtime.DEADLINE=10**12
        def fit(name,m,a,target,b,seed):
            self.assertEqual(name,'v76_Mitra_lastblock100')
            np.testing.assert_array_equal(target,y[a]);self.assertFalse(set(a)&set(b))
            calls.append((seed,runtime.DEADLINE))
            return np.full(len(b),.4),SimpleNamespace(release=lambda:None)
        wrapper=SimpleNamespace(mitra=runtime,configure_runtime=configure,assert_frozen=lambda *a:None,fit=fit)
        impl=candidate._g._impl
        original_import=impl.importlib.import_module
        def importer(name,*a,**kw):
            if name=='mitra_lastblock100_candidate':return wrapper
            return original_import(name,*a,**kw)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);phase=root/'phase';phase.mkdir();(phase/'protocol.json').write_text('{}')
            hashes={}
            for seed in candidate.SEEDS:
                p=phase/f'folds_{seed}.npz'
                np.savez(p,fold_id=assignment,dev_ids=dev,group_ids=groups,passenger_ids=passengers)
                hashes[str(seed)]=candidate.digest(p)
            record=dict(budgets_seconds={'candidates':{'adapt':21600}},legacy_caches={},fold_hashes=hashes,
                data=candidate.previous.data_identity(matrices,dev,y,groups,passengers))
            with patch.object(impl,'ROOT',root),patch.object(impl,'PHASE',phase), \
                 patch.object(impl,'assert_phase',return_value=record), \
                 patch.object(impl.importlib,'import_module',side_effect=importer), \
                 patch.object(impl.audit,'load_audit_data',return_value=(matrices,dev,y,groups,passengers)), \
                 patch.object(impl.signal,'setitimer') as timer:
                candidate.candidate()
            self.assertEqual([s for s,_ in calls],[s+f for s in candidate.SEEDS for f in range(3)])
            self.assertEqual(len(set(d for _,d in calls)),1)
            self.assertLess(calls[0][1],10**12)
            self.assertEqual([call.args[1] for call in timer.call_args_list],[21600,0])

    def test_existing_gate_equivalence_and_no_incomplete_or_new_candidate(self):
        rows=[dict(seed=s,accuracy=.85,vs_mitra={'gain':.012},vs_v44={'gain':.011}) for s in candidate.SEEDS]
        ci={'new_mean_vs_v44':{'lower95':.001}}
        self.assertEqual(candidate.make_decision(.84,rows,ci),candidate.previous.make_decision(.84,rows,ci))
        with self.assertRaises(ValueError):candidate.candidate('max200')
        with tempfile.TemporaryDirectory() as directory,patch.object(candidate._g._impl,'PHASE',Path(directory)):
            with self.assertRaisesRegex(RuntimeError,'Entire fixed queue'):candidate.complete_queue({})


if __name__=='__main__':unittest.main()
