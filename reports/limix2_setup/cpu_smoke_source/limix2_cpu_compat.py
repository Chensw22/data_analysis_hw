"""Built with StableAI LimiX. CPU-only import adapter for pinned upstream norms.

Two modules are populated before native imports; original source is untouched.
Their non-CUDA forward equations and parameter schema match the official classes.
No Triton module is fabricated and no torch global function is patched.
Source license: artifacts/limix2_setup/source/LICENSE.txt (Apache-derived StableAI).
"""
import ast
import json
import sys
import types
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from limix2_setup import ROOT,WORK,REPORT,sha

class _NonCUDANorm(nn.Module):
    def __init__(self,hs,eps,elementwise_affine=True,device=None,dtype=torch.float32,recompute=False):
        super().__init__()
        self.hs=hs;self.eps=eps;self.elementwise_affine=elementwise_affine
        self.device=device;self.dtype=dtype;self.recompute=recompute
        weight=nn.Parameter(torch.ones(self.hs,dtype=self.dtype)) if self.elementwise_affine else None
        self.register_parameter('weight',weight)
    def forward(self,input,output=None):
        if input.device.type!='cpu':raise RuntimeError('This reviewed smoke adapter is CPU-only.')
        if not input.is_contiguous():input=input.contiguous()
        weight=self.weight
        if weight is not None:weight=weight.to(device=input.device,dtype=input.dtype)
        shape=(int(self.hs),) if isinstance(self.hs,int) else tuple(int(x) for x in self.hs)
        return F.rms_norm(input,shape,weight,self.eps)


def install():
    manifest=json.loads((REPORT/'download.json').read_text())
    installed=[]
    for module_name,class_name in [('triton_rmsnorm','TritonRMSNorm'),('triton_qknorm','TritonQKNorm')]:
        path=WORK/'source/model/v2_0/operators/rmsnorm'/f'{module_name}.py'
        if sha(path)!=manifest['source_files'][str(path.relative_to(ROOT))]:raise RuntimeError('Pinned official source changed.')
        fullname='model.v2_0.operators.rmsnorm.'+module_name
        if fullname in sys.modules:raise RuntimeError('Compatibility installation must precede model imports.')
        module=types.ModuleType(fullname);klass=type(class_name,(_NonCUDANorm,),{'__module__':fullname})
        setattr(module,class_name,klass);module.__file__=__file__;sys.modules[fullname]=module;installed.append(fullname)
    return installed


def check_native_equivalence():
    """Execute only audited original norm class after removing unreachable CUDA tail."""
    records=[]
    for module_name,class_name in [('triton_rmsnorm','TritonRMSNorm'),('triton_qknorm','TritonQKNorm')]:
        source=WORK/'source/model/v2_0/operators/rmsnorm'/f'{module_name}.py'
        tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name)
        assert [n.name for n in node.body if isinstance(n,ast.FunctionDef)]==['__init__','forward','extra_repr']
        forward=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        stop=next(i for i,n in enumerate(forward.body) if isinstance(n,ast.If) and ast.unparse(n.test)=='not input.is_cuda')
        assert len(forward.body[:stop])==1
        forward.body=forward.body[:stop+1]+[ast.Raise(exc=ast.Call(func=ast.Name(id='RuntimeError',ctx=ast.Load()),args=[ast.Constant('CUDA not permitted')],keywords=[]),cause=None)]
        namespace={'torch':torch,'nn':nn,'F':F};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),namespace)
        official=namespace[class_name]
        for affine in [True,False]:
            a=official(8,1e-5,elementwise_affine=affine);b=_NonCUDANorm(8,1e-5,elementwise_affine=affine)
            b.load_state_dict(a.state_dict(),strict=True)
            x=torch.arange(48,dtype=torch.float32).reshape(8,6).T
            torch.testing.assert_close(a(x),b(x),rtol=0,atol=0)
            records.append({'class':class_name,'affine':affine,'noncontiguous_cpu_output_bitwise_equal':True,'state_keys':list(a.state_dict())})
    return records
