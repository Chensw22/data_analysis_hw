import copy
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

import top5_generation_evaluate as candidate


class GenerationPhaseTests(unittest.TestCase):
    def test_policy_changes_queue_but_keeps_all_gate_and_split_rules(self):
        new=json.loads((candidate.ROOT/candidate.POLICY).read_text())
        old=json.loads((candidate.ROOT/candidate.previous.POLICY).read_text())
        for key in ['A_research_online_gate','S_stable_local_superiority_gate','bootstrap_recipe']:
            self.assertEqual(new[key],old[key])
        for key in ['group_split_seeds','new_primary_seeds','folds_per_seed','splitter','group_key','fit_seed']:
            self.assertEqual(new['fixed_scope_before_new_scores'][key],old['fixed_scope_before_new_scores'][key])
        self.assertEqual(new['fixed_scope_before_new_scores']['candidates'],['v78_Mitra1_zeroshot','v79_Mitra12_mean'])
        self.assertFalse(new['batch_spec']['reference_training_allowed'])

    def test_fresh_process_keeps_original_namespace_and_old_data_untouched(self):
        script='''
import hashlib,json
from pathlib import Path
import top5_group_evaluate as old
snapshot=(old.PHASE,dict(old.CANDIDATES),old.POLICY,old.AUTH)
files=[Path(old.__file__),old.PHASE/'protocol.json',old.PHASE/'eligibility.json']
before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
import top5_generation_evaluate as new
assert snapshot==(old.PHASE,dict(old.CANDIDATES),old.POLICY,old.AUTH)
assert new._impl is not old and new._impl.PHASE!=old.PHASE
assert list(new._impl.CANDIDATES)==['v1','mean']
assert new._impl.SEEDS==old.SEEDS
rows=[dict(seed=s,accuracy=.85,vs_mitra=dict(gain=.012),vs_v44=dict(gain=.011)) for s in old.SEEDS]
ci={'new_mean_vs_v44':{'lower95':.001}}
assert new.make_decision(.84,rows,ci)==old.make_decision(.84,rows,ci)
assert before==[hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
print('fresh_namespace_and_rules_ok')
'''
        result=subprocess.run([sys.executable,'-c',script],cwd=candidate.ROOT,
            capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('fresh_namespace_and_rules_ok',result.stdout)

    def test_all_reference_seeds_copy_bitwise_without_any_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old=root/'old';new=root/'new';old.mkdir();new.mkdir()
            (new/'protocol.json').write_text('{}')
            refs={}
            for key,spec in candidate.REFERENCES.items():
                folder=old/'refs'/key;folder.mkdir(parents=True)
                rows=[]
                for seed in candidate.SEEDS:
                    path=folder/f'group_{seed}.npz'
                    np.savez(path,prob=np.array([.2,.8]),dev_ids=[1,2],fold_id=[0,1])
                    rows.append(dict(seed=seed,oof_path=str(path.relative_to(root)),
                        oof_sha256=candidate.digest(path),legacy_reused=False,legacy_evidence=None,
                        folds=[dict(fold=i) for i in range(3)]))
                refs[key]=dict(status='complete',kind='refs',key=key,version=spec['version'],results=rows)
                (folder/'summary.json').write_text(json.dumps(refs[key]))
            auth={'budgets_seconds':{'refs':{'mitra':1800,'v44':1800}}}
            with patch.object(candidate,'ROOT',root),patch.object(candidate,'PHASE',new), \
                 patch.object(candidate.previous,'PHASE',old), \
                 patch.object(candidate.previous.audit,'initialize_reference',side_effect=AssertionError('no reference training')):
                candidate.copy_references(refs,auth)
            for key in candidate.REFERENCES:
                saved=json.loads((new/'refs'/key/'summary.json').read_text())
                self.assertTrue(saved['reference_copied_without_training'])
                self.assertEqual(saved['total_seconds'],0)
                self.assertEqual([r['seed'] for r in saved['results']],candidate.SEEDS)
                for row in saved['results']:
                    self.assertTrue(row['reference_copied_without_training'])
                    self.assertEqual(candidate.digest(root/row['oof_path']),candidate.digest(root/row['original_oof_path']))

    def test_candidate_version_not_shared_module_NAME_and_all_three_splits_fit(self):
        dev=np.arange(12,dtype=np.int64);y=dev%2
        groups=np.repeat(np.arange(6).astype(str),2)
        passengers=np.array([f'{g}_{i}' for i,g in enumerate(groups)])
        assignment=np.repeat([0,1,2,0,1,2],2).astype(np.int8)
        frame=pd.DataFrame({'x':dev.astype(float)});matrices={False:(frame,[])}
        names=[]
        def fit(name,matrix,a,target,b,seed):
            names.append(name)
            np.testing.assert_array_equal(target,y[a]);self.assertFalse(set(a)&set(b))
            return np.full(len(b),.4),SimpleNamespace(release=lambda:None)
        fake=SimpleNamespace(NAME='v79_Mitra12_mean',configure_runtime=lambda *a,**kw:None,
            assert_frozen=lambda *a:None,fit=fit)
        import iterate
        original_import=candidate._impl.importlib.import_module
        def importer(name,*a,**kw):
            if name=='mitra_generation_candidate':return fake
            return original_import(name,*a,**kw)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);phase=root/'phase';phase.mkdir();(phase/'protocol.json').write_text('{}')
            hashes={}
            for seed in candidate.SEEDS:
                path=phase/f'folds_{seed}.npz'
                np.savez(path,fold_id=assignment,dev_ids=dev,group_ids=groups,passenger_ids=passengers)
                hashes[str(seed)]=candidate.digest(path)
            record=dict(budgets_seconds={'candidates':{'v1':60,'mean':60}},legacy_caches={},fold_hashes=hashes,
                data=candidate.previous.data_identity(matrices,dev,y,groups,passengers))
            with patch.object(candidate._impl,'ROOT',root),patch.object(candidate._impl,'PHASE',phase), \
                 patch.object(candidate._impl,'assert_phase',return_value=record), \
                 patch.object(candidate._impl.importlib,'import_module',side_effect=importer), \
                 patch.object(candidate._impl.audit,'load_audit_data',return_value=(matrices,dev,y,groups,passengers)):
                candidate.candidate('v1');candidate.candidate('mean')
        self.assertEqual(names,['v78_Mitra1_zeroshot']*9+['v79_Mitra12_mean']*9)

    def test_incomplete_batch_never_produces_eligibility_and_unknown_recipe_rejected(self):
        with self.assertRaises(ValueError):candidate.candidate('rescue')
        with tempfile.TemporaryDirectory() as directory,patch.object(candidate._impl,'PHASE',Path(directory)):
            with self.assertRaisesRegex(RuntimeError,'Entire fixed queue'):
                candidate.complete_queue({})

    def test_copied_reference_hash_stays_bound_to_its_original_all_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);phase=root/'phase';manifest={}
            for key in candidate.REFERENCES:
                manifest[key]=[]
                for seed in candidate.SEEDS:
                    source=root/f'old_{key}_{seed}.npz';source.write_bytes(f'{key}:{seed}'.encode())
                    target=phase/'refs'/key/f'group_{seed}.npz';target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(source.read_bytes())
                    manifest[key].append(dict(seed=seed,original_oof_path=source.name,oof_sha256=candidate.digest(source)))
            with patch.object(candidate,'ROOT',root),patch.object(candidate,'PHASE',phase), \
                 patch.object(candidate,'check_policy'), \
                 patch.object(candidate,'_base_assert_phase',return_value={'reference_copy_manifest':manifest}):
                candidate.assert_phase()
                (phase/'refs/mitra/group_20261115.npz').write_bytes(b'changed')
                with self.assertRaisesRegex(RuntimeError,'Copied reference differs'):
                    candidate.assert_phase()


if __name__=='__main__':unittest.main()
