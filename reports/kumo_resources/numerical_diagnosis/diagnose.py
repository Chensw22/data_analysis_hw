import os,json,time,torch,numpy as np,sys
from pathlib import Path
import kumo_resource_probe as k
out=Path('artifacts/probes/kumo_matmul_diagnosis');mode=os.environ.get('DIAGNOSTIC_MODE','native');name=mode+'_metal'+os.environ.get('PYTORCH_MPS_PREFER_METAL','0');assert not (out/(name+'.json')).exists()
torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.25);torch.manual_seed(0)
a,b=torch.randn(5,400,96),torch.randn(5,96,128);ref=torch.matmul(a,b).transpose(0,1);am,bm=a.to('mps'),b.to('mps');r=dict(mode=mode,prefer_metal=os.environ.get('PYTORCH_MPS_PREFER_METAL','unset'),synthetic_only=True,accuracy_computed=False,minimal=[])
for _ in range(3):
 dest=torch.empty(400,5,128,device='mps');torch.matmul(am,bm,out=dest.transpose(0,1));bad=float((dest.cpu()-ref).abs().max());dest.transpose(0,1).copy_(torch.matmul(am,bm));good=float((dest.cpu()-ref).abs().max());r['minimal'].append(dict(native_strided_out_max_error=bad,allocated_then_copy_max_error=good))
original=torch.matmul
if mode=='safe_out':
 def safe_matmul(a,b,*,out=None):
  if out is not None and out.device.type=='mps' and not out.is_contiguous():return out.copy_(original(a,b))
  return original(a,b) if out is None else original(a,b,out=out)
 torch.matmul=safe_matmul
x,y,z=k.synthetic(128,17,k.SETTINGS['seed']);m=k.KumoPredictor(x,y,'mps',8,k.SETTINGS['seed']);p=m.predict_proba(z)[:,1];repeat=m.predict_proba(z)[:,1];parts=np.r_[m.predict_proba(z.iloc[:8])[:,1],m.predict_proba(z.iloc[8:])[:,1]];cpu=np.load('artifacts/probes/kumo_numerical_diagnosis/synthetic_probabilities.npz')['cpu_default'];r.update(repeat_difference=float(np.max(abs(p-repeat))),partition_difference=float(np.max(abs(p-parts))),cpu_difference=float(np.max(abs(p-cpu))));k.write(out/(name+'.json'),r);print(json.dumps(r));torch.matmul=original
