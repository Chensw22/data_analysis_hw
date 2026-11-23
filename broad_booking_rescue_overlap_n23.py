"""Audit stable changes on the exact same unanimously classified passengers."""
import numpy as np

from broad_geographic_transfer_n23 import load
from broad_link_authenticity_n23 import REPORT
from broad_matched_model_changes_n23 import read_bank
from broad_partition_evidence_n23 import ART, SEEDS, write
from broad_unanimous_dissent_value_n23 import SOURCES


def main():
    row, y, frame, _, _ = load()
    with np.load(ART / 'nov23_release_vote_information/panel.npz') as data:
        np.testing.assert_array_equal(data['row'], row)
        np.testing.assert_array_equal(data['y'], y)
        vote = data['votes'][0, :, 0]
        unanimous = (data['votes'] == vote[None, :, None]).all((0, 2))

    sources = dict(SOURCES)
    for method in ('owner', 'symmetrized', 'shuffled'):
        sources[method] = ('within_booking_bill_matching', method)
    stable = {method: np.ones(len(row), bool) for method in sources}
    many = frame.BookingSize.to_numpy() > 1
    for seed in map(str, SEEDS):
        with np.load(ART / f'nov23_joint_observed_state/{seed}.npz') as data:
            np.testing.assert_array_equal(data['row'], row)
            np.testing.assert_array_equal(data['y'], y)
            fold = data['fold']
            bank = read_bank(row, y, fold, seed).reshape(len(y), -1)
            extra = [data[k] for k in ('sleep_conditional', 'sleep_direct', 'basket_conditional', 'basket_direct')]
            unanimous &= ((np.column_stack([bank, *extra]) >= .5) == vote[:, None]).all(1)
        for method, (directory, key) in sources.items():
            with np.load(ART / f'nov23_{directory}/{seed}.npz') as data:
                np.testing.assert_array_equal(data['row'], row)
                np.testing.assert_array_equal(data['fold'], fold)
                change = (data[key] >= .5) != vote
                if directory == 'within_booking_bill_matching':
                    change &= many
                stable[method] &= change

    wrong = vote != y
    assert (unanimous & wrong).sum() == 372 and (unanimous & ~wrong).sum() == 3867
    previous = np.any([stable[k] for k in SOURCES], axis=0)
    owner = stable['owner']
    averaged = stable['symmetrized']
    billing = owner | averaged
    rules = {'previous_six': previous, 'owner': owner, 'averaged': averaged,
             'shuffled_control': stable['shuffled'], 'owner_and_averaged': owner & averaged,
             'billing_union': billing, 'billing_already_found': billing & previous,
             'billing_new': billing & ~previous, 'all_methods_union': previous | billing}
    results = {}
    for method, mask in rules.items():
        selected = mask & unanimous
        results[method] = {'fixed': int((selected & wrong).sum()),
                           'harmed': int((selected & ~wrong).sum()),
                           'solo_fixed': int((selected & wrong & ~many).sum())}
    assert results['previous_six']['fixed'] == 15
    write(REPORT / 'overlap.json', results)
    print(results)


if __name__ == '__main__':
    main()
