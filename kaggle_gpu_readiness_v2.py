"""Final synthetic GPU diagnostic: requested T4/latest image, never CPU fallback."""
import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import signal
import subprocess
import time

TIMEOUT_SECONDS = 120
REQUESTED_ACCELERATOR = 'NvidiaTeslaT4'
OUTPUT = Path('/kaggle/working/gpu_readiness_v2.json')


def require_requested_gpu(names):
    if not names or not all(re.search(r'\bT4\b', name, re.I) for name in names):
        raise RuntimeError('Requested T4 was not assigned; no substitute or CPU computation.')


def driver_diagnostics():
    result = dict(device_nodes=sorted(p.name for p in Path('/dev').glob('nvidia*')),
                  proc_driver_version_exists=Path('/proc/driver/nvidia/version').exists())
    command = ['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
               '--format=csv,noheader,nounits']
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        result['nvidia_smi'] = dict(returncode=process.returncode,
                                  stdout=process.stdout[:4096], stderr=process.stderr[:4096])
    except (OSError, subprocess.TimeoutExpired) as error:
        result['nvidia_smi'] = dict(error_type=type(error).__name__, error=str(error))
    try:
        library = ctypes.CDLL('libcuda.so.1')
        library.cuInit.argtypes = [ctypes.c_uint]
        library.cuInit.restype = ctypes.c_int
        library.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        library.cuDeviceGetCount.restype = ctypes.c_int
        initialization = int(library.cuInit(0))
        result['libcuda'] = dict(loaded=True, cuInit=initialization)
        if initialization == 0:
            count = ctypes.c_int(-1)
            code = int(library.cuDeviceGetCount(ctypes.byref(count)))
            result['libcuda'].update(cuDeviceGetCount=code, device_count=count.value)
    except (OSError, AttributeError) as error:
        result['libcuda'] = dict(error_type=type(error).__name__, error=str(error))
    return result


def write_result(result):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2)+'\n')


def run():
    started = time.monotonic()
    result = dict(status='incomplete', requested_accelerator=REQUESTED_ACCELERATOR,
        requested_image_policy='latest', synthetic_only=True, data_read=False,
        network_used=False, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python_version=platform.python_version(), started_at_utc=datetime.now(timezone.utc).isoformat(),
        own_timeout_seconds=TIMEOUT_SECONDS, phase='driver_diagnostics')
    def timeout(*_):
        raise TimeoutError('Final GPU diagnostic exceeded its 120 second limit.')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, TIMEOUT_SECONDS)
    try:
        result['driver'] = driver_diagnostics()
        result['phase'] = 'torch_import'
        write_result(result)  # Preserve driver evidence even if import is killed externally.
        import torch
        torch.set_num_threads(2)
        result.update(torch_version=torch.__version__, cuda_runtime_version=torch.version.cuda,
            torch_cuda_built=bool(torch.backends.cuda.is_built()),
            cuda_available=bool(torch.cuda.is_available()), devices=[], phase='torch_check')
        if not result['cuda_available']:
            raise RuntimeError('Torch CUDA unavailable; driver evidence retained, no CPU fallback or package replacement.')
        for index in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(index)
            result['devices'].append(dict(index=index, name=device.name,
                total_memory_bytes=int(device.total_memory),
                compute_capability=[device.major, device.minor],
                native_bf16_supported=device.major >= 8,
                bf16_check='compute capability >=8; no BF16 computation'))
        require_requested_gpu([d['name'] for d in result['devices']])
        result['phase'] = 'synthetic_cuda'
        write_result(result)
        torch.cuda.set_device(0)
        torch.manual_seed(20261116)
        torch.cuda.manual_seed_all(20261116)
        torch.cuda.reset_peak_memory_stats(0)
        x = torch.randn(256, 256, device='cuda:0', dtype=torch.float32, requires_grad=True)
        weight = torch.randn(256, 256, device='cuda:0', dtype=torch.float32, requires_grad=True)
        prediction = x @ weight
        loss = prediction.square().mean()
        loss.backward()
        torch.cuda.synchronize(0)
        finite = bool(torch.isfinite(prediction).all() and torch.isfinite(loss)
                      and torch.isfinite(x.grad).all() and torch.isfinite(weight.grad).all())
        if not finite:
            raise FloatingPointError('Nonfinite synthetic forward/backward result.')
        result.update(status='complete', phase='complete', forward_backward_finite=True,
            matmul_shape=[256, 256], tested_dtype='float32', bf16_compute_tested=False,
            tested_device_index=0, cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
            cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(0)))
    except Exception as error:
        result.update(status='failed_readiness', error_type=type(error).__name__, error=str(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        result['elapsed_seconds'] = time.monotonic()-started
        write_result(result)
        print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    raise SystemExit(0 if run()['status'] == 'complete' else 1)
