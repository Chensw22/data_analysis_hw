"""Measure larger query batches against a frozen Mitra implementation, without scores."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--context-rows', type=int, required=True)
    parser.add_argument('--queries', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Use a new probe directory.')
    sys.path.insert(0, str(args.source_dir.resolve()))
    candidate = importlib.import_module('mitra_candidate')
    candidate.freeze_sources(args.output)
    shutil.copy2(__file__, args.output/'batch_probe_source.py')
    result = dict(kind='synthetic_query_batch_resource_probe', accuracy_computed=False,
                  context_rows=args.context_rows, query_rows=args.queries,
                  query_chunk=args.queries, common_subset_rows=min(128, args.queries),
                  batch_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  common_subset_tolerance=dict(rtol=0., atol=.002),
                  status='incomplete')
    predictor = None
    started = time.monotonic()
    try:
        torch.mps.set_per_process_memory_fraction(.65)
        config = candidate.verify_checkpoint(args.checkpoint_dir)
        raw = candidate.synthetic_frame(args.context_rows, 42)
        query = candidate.synthetic_frame(args.queries, 43)
        labels = np.arange(len(raw), dtype=np.int64) % 2
        prep = candidate.MitraPreprocessor().fit(raw, labels)
        state = load_file(str(args.checkpoint_dir/'model.safetensors'), device='cpu')
        predictor = candidate.MitraPredictor(prep, config, state, prep.transform(raw), labels, 'mps')
        del state
        predictor.query_chunk = args.queries
        timings = []
        for repeat in range(2):
            tick = time.monotonic()
            probability = predictor.predict_proba(query)
            timings.append(time.monotonic()-tick)
            print(json.dumps(dict(repeat=repeat, seconds=timings[-1],
                                  resources=candidate.snapshot('mps'))), flush=True)
        common = predictor.predict_proba(query.iloc[:result['common_subset_rows']])
        difference = float(np.max(np.abs(common-probability[:len(common)])))
        np.testing.assert_allclose(common, probability[:len(common)], rtol=0., atol=.002)
        result.update(status='complete', inference_seconds=timings,
                      common_subset_max_abs_difference=difference,
                      full_context=len(predictor.support)==args.context_rows,
                      resources=candidate.snapshot('mps'),
                      precision='FP32 weights/input; MPS bf16 AMP; FP32 probabilities')
    except Exception as error:
        result.update(status='failed_inconclusive', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if predictor is not None:
            predictor.release()
        result['total_seconds'] = time.monotonic()-started
        (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
