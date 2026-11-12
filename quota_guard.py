"""Read only weekly rate-limit metadata from this Codex session's local log.

No credentials, credit balances, plan details, or conversation text are emitted.
This is a polling guard, not a server-enforced spending limit.
"""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path


def latest_weekly(path):
    latest=None
    with Path(path).open() as stream:
        for line in stream:
            try:
                event=json.loads(line)
            except json.JSONDecodeError:
                continue
            payload=event.get('payload',{})
            if not isinstance(payload,dict) or payload.get('type')!='token_count':
                continue
            limits=payload.get('rate_limits')
            if not isinstance(limits,dict):
                continue
            for key in ['primary','secondary']:
                window=limits.get(key)
                if isinstance(window,dict) and window.get('window_minutes')==10080:
                    used=window.get('used_percent')
                    if isinstance(used,(int,float)) and math.isfinite(used) and 0<=used<=100:
                        latest=dict(observed_at_utc=event.get('timestamp'),used_percent=float(used),
                            remaining_percent=100-float(used),window_minutes=10080,
                            resets_at_epoch=window.get('resets_at'))
    if latest is None:
        raise ValueError('No weekly usage metadata found for this session.')
    return latest


def assess(latest,reserve=12.,max_age=180,now=None):
    now=now or datetime.now(timezone.utc)
    timestamp=datetime.fromisoformat(latest['observed_at_utc'].replace('Z','+00:00'))
    age=(now-timestamp).total_seconds()
    result={**latest,'reserve_percent':reserve,'metadata_age_seconds':round(age,2)}
    if age>max_age or age < -60:
        result.update(status='unknown',can_start_more_work=False,reason='Rate-limit metadata is stale or its clock is inconsistent.')
    elif latest['remaining_percent']<=reserve:
        result.update(status='stop',can_start_more_work=False,reason='User-requested weekly reserve reached.')
    else:
        result.update(status='continue',can_start_more_work=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--session-file',type=Path,required=True)
    parser.add_argument('--reserve',type=float,default=12.)
    parser.add_argument('--max-age-seconds',type=int,default=180)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'artifacts/quota_status.json')
    args=parser.parse_args()
    if not 0<=args.reserve<100:
        parser.error('reserve must be in [0,100).')
    try:
        result=assess(latest_weekly(args.session_file),args.reserve,args.max_age_seconds)
    except (OSError,ValueError,TypeError,KeyError):
        result=dict(status='unknown',can_start_more_work=False,reason='Current weekly usage could not be read reliably.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    raise SystemExit(0 if result['status']=='continue' else 2 if result['status']=='stop' else 3)


if __name__=='__main__':
    main()
