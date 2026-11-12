import json
from pathlib import Path
import tempfile
import unittest
from datetime import datetime,timezone
from quota_guard import assess,latest_weekly


class QuotaGuardTests(unittest.TestCase):
    def test_reserve_boundary_stops_and_stale_data_does_not_allow_work(self):
        now=datetime(2026,11,12,8,tzinfo=timezone.utc)
        info=dict(observed_at_utc='2026-11-12T08:00:00Z',remaining_percent=15.)
        self.assertEqual(assess(info,now=now)['status'],'stop')
        self.assertTrue(assess({**info,'remaining_percent':16},now=now)['can_start_more_work'])
        self.assertFalse(assess({**info,'observed_at_utc':'2026-11-12T07:00:00Z','remaining_percent':90},now=now)['can_start_more_work'])

    def test_only_weekly_metadata_is_emitted(self):
        event=dict(timestamp='2026-11-12T08:00:00Z',payload=dict(type='token_count',rate_limits=dict(
            primary=dict(used_percent=1,window_minutes=300),
            secondary=dict(used_percent=47,window_minutes=10080,resets_at=123),
            credits=dict(balance='private'),plan_type='private')))
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'session.jsonl';p.write_text(json.dumps(event)+'\npartial trailing line')
            result=latest_weekly(p)
        self.assertEqual(result['remaining_percent'],53)
        self.assertNotIn('private',json.dumps(result))


if __name__=='__main__':
    unittest.main()
