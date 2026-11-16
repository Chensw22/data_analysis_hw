"""Standalone, synthetic CUDA readiness check; no data, network or credentials."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import signal
import time

TIMEOUT_SECONDS = 120
REQUESTED_ACCELERATOR = 'NvidiaL4'


def require_requested_gpu(names):
    if not names or not re.search(r'\bL4\b', names[0], flags=re.IGNORECASE):
        raise RuntimeError('Requested NvidiaL4 was not assigned; no substitute GPU computation.')


def run():
    started = time.monotonic()
    result = dict(status='incomplete', requested_accelerator=REQUESTED_ACCELERATOR,
        synthetic_only=True, data_read=False, network_used=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python_version=platform.python_version(),
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        own_timeout_seconds=TIMEOUT_SECONDS)
    def timeout(*_):
        raise TimeoutError('Readiness check exceeded its 120 second limit.')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, TIMEOUT_SECONDS)
    try:
        import torch
        torch.set_num_threads(2)
        result.update(torch_version=torch.__version__, cuda_runtime_version=torch.version.cuda,
                      cuda_available=torch.cuda.is_available(), devices=[])
        if not result['cuda_available']:
            raise RuntimeError('CUDA unavailable; no CPU fallback.')
        for index in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(index)
            with torch.cuda.device(index):
                try:
                    native_bf16 = bool(torch.cuda.is_bf16_supported(including_emulation=False))
                    bf16_method = 'torch.cuda.is_bf16_supported(including_emulation=False)'
                except TypeError:
                    native_bf16 = device.major >= 8
                    bf16_method = 'compute capability >=8 fallback; not an executed BF16 test'
            result['devices'].append(dict(index=index, name=device.name,
                total_memory_bytes=int(device.total_memory),
                compute_capability=[device.major, device.minor],
                native_bf16_supported=native_bf16, bf16_check=bf16_method))
        require_requested_gpu([d['name'] for d in result['devices']])
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
        result.update(status='complete', forward_backward_finite=True,
            matmul_shape=[256, 256], tested_dtype='float32', bf16_compute_tested=False,
            cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
            cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(0)))
    except Exception as error:
        result.update(status='failed_readiness', error_type=type(error).__name__, error=str(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        result['elapsed_seconds'] = time.monotonic()-started
        output = Path('/kaggle/working/gpu_readiness.json')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    raise SystemExit(0 if run()['status']=='complete' else 1)
