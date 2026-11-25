from datetime import date, timedelta
import unittest

from await_half_roster_slot_n23 import reserve_available


class HistoricalReserveTest(unittest.TestCase):
    def test_only_a_same_day_unused_slot_can_be_filled(self):
        today = date(2026, 11, 26)
        skipped = dict(phase='members_below_reference', attempted=False, sha256='original')
        args = dict(union=skipped, expected_digest='original', official_match=False,
                    last_submission_date='2026-11-26 01:25:00', today=today, remaining=1)
        self.assertTrue(reserve_available(**args))
        for changes in [
            dict(union={}), dict(union={**skipped, 'phase': 'completed'}),
            dict(union={**skipped, 'attempted': True}), dict(expected_digest='different'),
            dict(official_match=True), dict(remaining=0), dict(today=today + timedelta(days=1))
        ]:
            with self.subTest(changes=changes):
                self.assertFalse(reserve_available(**{**args, **changes}))


if __name__ == '__main__':
    unittest.main()
