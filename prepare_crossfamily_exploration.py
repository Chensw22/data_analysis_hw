"""Prepare the user's one explicit v80 exploratory upload; never upload here."""
import argparse
from datetime import datetime,timezone
import hashlib,json,subprocess
from pathlib import Path
import joblib,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/proposal_crossfamily_v80'
REPORT=ROOT/'reports/proposal_crossfamily_v80'
AUTH=ROOT/'reports/crossfamily_online_20261113/authorization.json'
SPECS={'v44':'proposal_batch_v44','mitra':'proposal_batch_mitra2'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def authorized():
 a=read(AUTH)
 assert a['version']=='v80_MitraCat_mean' and a['maximum_upload_attempts']==1
 for path,digest in a['inputs'].items():
  if sha(ROOT/path)!=digest:raise RuntimeError('Authorized input changed: '+path)
 return a

def component(key):
 a=authorized();out=OUT/key
 if out.exists():raise RuntimeError('Component exists; no overwrite.')
 out.mkdir(parents=True)
 import audit_top5_references as audit
 framework,wrapper,_,protocol,version=audit.initialize_reference(key,out)
 folder=ROOT/'artifacts'/SPECS[key];manifest=read(folder/'manifest.json')
 assert manifest['selection']['version']==version and manifest['fit_rows']==8693
 assert sha(folder/'model.joblib')==manifest['model_sha256']
 model=joblib.load(folder/'model.joblib')
 if key=='mitra':
  model.device='mps'
  assert len(model.support)==8693 and len(model.labels)==8693
 train,test,matrices,_=framework.dataset()
 assert len(train)==8693 and len(test)==4277
 p=np.asarray(model.predict_proba(matrices[False][0].iloc[np.arange(8693,12970)]))[:,1]
 expected=pd.read_csv(folder/'test_probabilities.csv')
 np.testing.assert_array_equal(expected.PassengerId.astype(str),test.PassengerId.astype(str))
 assert p.shape==(4277,) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
 np.testing.assert_allclose(p,expected.probability.to_numpy(),rtol=0,atol=2e-6 if key=='mitra' else 1e-12)
 old_submission=pd.read_csv(folder/'submission.csv')
 assert sha(folder/'submission.csv')==manifest['sha256']
 np.testing.assert_array_equal(old_submission.PassengerId.astype(str),test.PassengerId.astype(str))
 np.testing.assert_array_equal(p>=.5,old_submission.Transported.to_numpy())
 pd.DataFrame({'PassengerId':test.PassengerId,'probability':p}).to_csv(out/'probabilities.csv',index=False)
 if hasattr(model,'release'):model.release()
 authorized()
 write(out/'complete.json',dict(status='complete',version=version,rows=4277,full_fit_rows=8693,
  source_model_sha256=manifest['model_sha256'],source_manifest_sha256=sha(folder/'manifest.json'),
  probabilities_sha256=sha(out/'probabilities.csv'),full_reload_verified=True,
  max_difference=float(np.max(np.abs(p-expected.probability.to_numpy()))),
  source_environment_verified=True,authorization_sha256=sha(AUTH),script_sha256=sha(__file__)))
 print(key,'complete',flush=True)

def assemble():
 authorized()
 if (OUT/'manifest.json').exists():raise RuntimeError('Existing final proposal.')
 test=pd.read_csv(ROOT/'data/raw/test.csv');parts=[];meta={}
 for key in SPECS:
  d=read(OUT/key/'complete.json');p=pd.read_csv(OUT/key/'probabilities.csv')
  assert d['status']=='complete' and d['full_reload_verified']
  assert d['authorization_sha256']==sha(AUTH) and d['script_sha256']==sha(__file__)
  assert d['probabilities_sha256']==sha(OUT/key/'probabilities.csv')
  np.testing.assert_array_equal(p.PassengerId.astype(str),test.PassengerId.astype(str))
  parts.append(p.probability.to_numpy());meta[key]=d
 p=.5*parts[0]+.5*parts[1]
 pd.DataFrame({'PassengerId':test.PassengerId,'Transported':p>=.5}).to_csv(OUT/'submission.csv',index=False)
 pd.DataFrame({'PassengerId':test.PassengerId,'probability':p}).to_csv(OUT/'test_probabilities.csv',index=False)
 from kaggle_workflow import validate_submission
 check=validate_submission(OUT/'submission.csv',ROOT/'data/raw/test.csv')
 for path in (ROOT/'reports').glob('*/online_attempts/*.json'):
  if read(path).get('file_sha256')==check['sha256']:raise RuntimeError('Duplicate prior upload: '+str(path))
 eligibility=read(ROOT/'artifacts/top5_crossfamily_phase/eligibility.json')
 assert eligibility['A_eligible'] is False and eligibility['S_supported'] is False
 selection=dict(version='v80_MitraCat_mean',components=['v44_seed_ensemble','v71_Mitra2_zeroshot'],
  weights=[.5,.5],threshold=.5,stage='explicit_user_authorized_exploration',A_eligible=False,S_supported=False)
 write(OUT/'selection.json',selection)
 auth=dict(read(AUTH),file_sha256=check['sha256'],maximum_upload_attempts_for_this_file=1)
 write(OUT/'authorization.json',auth)
 manifest=dict(**check,status='complete',selection=selection,components=meta,
  authorization_sha256=sha(AUTH),original_eligibility_sha256=sha(ROOT/'artifacts/top5_crossfamily_phase/eligibility.json'),
  user_exception=True,old_failed_gate_unchanged=True,source_models_reused=True,new_model_fits=0,
  full_reload_verified=True,script_sha256=sha(__file__),
  prepared_from_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
  completed_at_utc=datetime.now(timezone.utc).isoformat())
 write(OUT/'manifest.json',manifest)
 for name in ['manifest.json','selection.json','authorization.json']:write(REPORT/name,read(OUT/name))
 print(json.dumps(check),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['v44','mitra','assemble'])
 args=parser.parse_args();assemble() if args.stage=='assemble' else component(args.stage)
