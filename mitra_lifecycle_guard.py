"""A state-preserving lifecycle diagnostic, not a numerical or model change.

Hash observations copy each tensor to CPU and can perturb timing/allocator state.
Returned proofs contain plain metadata and hashes, never tensor/storage objects.
"""
import copy
import gc
import hashlib
import json
from pathlib import Path
import pickle
import random
import resource

import numpy as np
import torch

BARRIER = ['synchronize', 'gc.collect', 'empty_cache', 'synchronize']


def write(path, value):
    if path is None:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def synchronize(device):
    if device=='mps':torch.mps.synchronize()


def empty_cache(device):
    if device=='mps':torch.mps.empty_cache()


def observe(device):
    values=dict(max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,measurement='snapshots_not_peak')
    if device=='mps':
        values.update(mps_current_bytes=torch.mps.current_allocated_memory(),mps_driver_bytes=torch.mps.driver_allocated_memory(),
                      mps_recommended_max_bytes=torch.mps.recommended_max_memory())
    return values


def tensor_bytes(tensor):
    # Never clone on GPU; the temporary host view dies before this function returns.
    host=tensor.detach().to(device='cpu').contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(host.numpy().tobytes()).hexdigest()


def tensor_record(tensor):
    if tensor.layout!=torch.strided:raise RuntimeError('Dense strided state required for exact byte audit.')
    storage=tensor.untyped_storage()
    return dict(object_id=id(tensor),storage_id=int(storage._cdata),storage_data_ptr=int(storage.data_ptr()),
                storage_bytes=int(storage.nbytes()),storage_offset=int(tensor.storage_offset()),
                shape=list(tensor.shape),stride=list(tensor.stride()),dtype=str(tensor.dtype),device=str(tensor.device),
                requires_grad=bool(tensor.requires_grad),bytes_sha256=tensor_bytes(tensor))


def plain(value):
    if torch.is_tensor(value):return {'tensor':tensor_record(value)}
    if isinstance(value,np.ndarray):return dict(dtype=str(value.dtype),shape=list(value.shape),bytes_sha256=hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,(str,int,float,bool,type(None))):return value
    raise TypeError('Unsupported state value in exact lifecycle audit: '+str(type(value)))


def rng_state(rng, reference_rng, device):
    def digest(state):return hashlib.sha256(pickle.dumps(state,protocol=5)).hexdigest()
    def torch_rng(state):return dict(dtype=str(state.dtype),shape=list(state.shape),bytes_sha256=tensor_bytes(state))
    state=dict(python_complete_state_sha256=digest(random.getstate()),
               numpy_global_complete_state_sha256=digest(np.random.get_state()),
               sampler=copy.deepcopy(rng.bit_generator.state),counterfactual_sampler=copy.deepcopy(reference_rng.bit_generator.state),
               torch_cpu=torch_rng(torch.get_rng_state()))
    if device=='mps':state['torch_mps']=torch_rng(torch.mps.get_rng_state())
    return state


def capture_state(model, optimizer, data, targets, query, rng, reference_rng, device='mps'):
    parameters=list(model.named_parameters(remove_duplicate=False))
    if any(p.grad is not None for _,p in parameters):
        raise RuntimeError('Lifecycle barrier is permitted only with all gradients None.')
    names={id(parameter):name for name,parameter in parameters}
    moments={}
    for parameter,state in optimizer.state.items():
        if id(parameter) not in names:raise RuntimeError('Optimizer holds a parameter outside the model.')
        moments[names[id(parameter)]]=dict(parameter_object_id=id(parameter),state_object_id=id(state),values=plain(state))
    groups=[]
    for group in optimizer.param_groups:
        groups.append(dict(object_id=id(group),parameter_ids=[id(p) for p in group['params']],
                           options=plain({k:v for k,v in group.items() if k!='params'})))
    inputs={name:tensor_record(tensor) for name,tensor in [('data',data),('targets',targets),('query',query)]}
    if any(tensor.grad is not None for tensor in [data,targets,query]):
        raise RuntimeError('Lifecycle barrier requires all input gradients None.')
    result=dict(model_object_id=id(model),optimizer_object_id=id(optimizer),optimizer_type=type(optimizer).__qualname__,
        model_training=bool(model.training),module_training={name:bool(module.training) for name,module in model.named_modules()},
        parameters={name:tensor_record(parameter) for name,parameter in parameters},
        buffers={name:tensor_record(buffer) for name,buffer in model.named_buffers(remove_duplicate=False)},
        all_gradients_none=True,all_input_gradients_none=True,
        optimizer_steps={names[id(p)]:float(state['step'].item()) for p,state in optimizer.state.items()},
        optimizer_state_object_id=id(optimizer.state),optimizer_groups_object_id=id(optimizer.param_groups),
        optimizer_groups=groups,optimizer_state=moments,inputs=inputs,rng=rng_state(rng,reference_rng,device),
        grad_enabled=bool(torch.is_grad_enabled()))
    # Fail if a future edit accidentally returns a tensor or arbitrary object.
    json.dumps(result,allow_nan=False)
    return result


def lifecycle_barrier(model, optimizer, data, targets, query, rng, reference_rng, device='mps', log_path=None):
    if str(data.device)!=device or targets.device!=data.device or query.device!=data.device:
        raise RuntimeError('Barrier input devices must match the declared runtime.')
    result=dict(status='incomplete',barrier=BARRIER,device=device,observations=[],
                observation_caveat='Full state hashing copies to CPU and can perturb allocator/timing; snapshots are not peaks.')
    def snapshot(stage):result['observations'].append(dict(stage=stage,**observe(device)));write(log_path,result)
    try:
        synchronize(device);snapshot('before_pre_hash')
        result['before']=capture_state(model,optimizer,data,targets,query,rng,reference_rng,device)
        synchronize(device);snapshot('after_pre_hash')
        synchronize(device);snapshot('barrier_after_synchronize1')
        result['gc_collected']=int(gc.collect());snapshot('barrier_after_gc_collect')
        empty_cache(device);snapshot('barrier_after_empty_cache')
        synchronize(device);snapshot('barrier_after_synchronize2')
        snapshot('before_post_hash')
        result['after']=capture_state(model,optimizer,data,targets,query,rng,reference_rng,device)
        synchronize(device);snapshot('after_post_hash')
        if result['before']!=result['after']:
            result['different_sections']=[key for key in result['before'] if result['before'][key]!=result['after'][key]]
            raise RuntimeError('Lifecycle barrier changed live numerical/RNG/object state: '+','.join(result['different_sections']))
        result.update(status='complete',state_preserved=True)
        write(log_path,result)
        return result
    except BaseException as error:
        result.update(status='failed_state_preservation',state_preserved=False,error_type=type(error).__name__,error=str(error))
        write(log_path,result)
        raise
