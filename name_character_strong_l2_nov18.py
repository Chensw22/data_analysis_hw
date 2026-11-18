"""One disclosed follow-up: tenfold L2, all other Name residual operations reused."""
import argparse
import numpy as np
import name_character_residual_nov18 as base

L2 = 10.0
ORIGINAL_OBJECTIVE = base.objective
ORIGINAL_PROTOCOL = base.protocol


def objective(theta, x, y, offset):
    value, gradient = ORIGINAL_OBJECTIVE(theta, x, y, offset)
    w = theta[1:]
    value += .5 * (L2 - 1.) * np.dot(w, w)
    gradient[1:] += (L2 - 1.) * w
    return float(value), gradient


def protocol():
    plan = ORIGINAL_PROTOCOL()
    if plan['l2_multiplier'] != L2 or plan['single_factor_followup'] is not True:
        raise RuntimeError('Only the frozen tenfold regularization follow-up is supported.')
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    # Reuse the reviewed split, vocabulary, offsets, fitting, reporting and reload functions.
    base.objective = objective
    base.protocol = protocol
    base.PLAN = base.ROOT / 'reports/error_targeted_20261118/name_strong_l2/protocol.json'
    base.REPORT = base.PLAN.parent
    base.OUT = base.ROOT / 'artifacts/error_targeted_20261118/name_strong_l2'
    if args.check:
        base.check()
    else:
        base.run()


if __name__ == '__main__':
    main()
