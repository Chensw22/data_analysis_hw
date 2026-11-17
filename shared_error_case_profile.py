"""Describe error directions and observable attributes; never fit a rule."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    path = ROOT / 'artifacts/error_rescue_20261117/shared_error_cases.csv'
    aligned_path = ROOT / 'artifacts/data_error_diagnosis_20261117/aligned_rows.csv'
    expected = json.loads((ROOT / 'reports/error_rescue_20261117/structure_summary.json').read_text())
    assert hashlib.sha256(aligned_path.read_bytes()).hexdigest() == expected['aligned_sha256']
    aligned = pd.read_csv(aligned_path)
    rows = pd.read_csv(path)
    anchor = aligned.loc[aligned.family_wrong_count.eq(5)].reset_index(drop=True)
    assert len(rows) == 914
    assert np.array_equal(rows.row_id, anchor.row_id) and np.array_equal(rows.y, anchor.y)
    names = ['v44', 'v81', 'TabDPT', 'Kumo', 'EXAONE']
    for name in names:
        assert np.allclose(rows['p_'+name], anchor['p_'+name], atol=1e-15, rtol=0)
    slices = {
        'all_shared_wrong': np.ones(len(rows), dtype=bool),
        'Earth_asleep': rows.HomePlanet.eq('Earth') & rows.CryoSleep.eq(True),
        'Earth_awake': rows.HomePlanet.eq('Earth') & rows.CryoSleep.eq(False),
        'Earth_sleep_unknown': rows.HomePlanet.eq('Earth') & rows.CryoSleep.isna(),
        'other_or_unknown_planet': ~rows.HomePlanet.eq('Earth'),
    }
    profiles = {}
    for name, mask in slices.items():
        s = rows.loc[mask]
        bills = s[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
        p = s[['p_'+n for n in names]]
        confidence = np.maximum(p, 1-p)
        profiles[name] = dict(
            rows=len(s),
            actual_transported_all_predict_not=int(s.y.eq(1).sum()),
            actual_not_transported_all_predict_yes=int(s.y.eq(0).sum()),
            all_five_bills_known_zero=int((bills.notna().all(axis=1) & bills.sum(axis=1).eq(0)).sum()),
            any_bill_missing=int(bills.isna().any(axis=1).sum()),
            median_total_spend_complete_bills=float(bills.dropna().sum(axis=1).median()),
            median_age=float(s.Age.median()),
            deck_counts=s.Cabin.str.split('/').str[0].fillna('missing').value_counts().to_dict(),
            all_five_confidence_at_least80=int((confidence >= .8).all(axis=1).sum()),
            all_five_confidence_at_least95=int((confidence >= .95).all(axis=1).sum()),
            median_v81_predicted_class_probability=float(np.maximum(s.p_v81, 1-s.p_v81).median()),
            solo_group=int(s.GroupSize.eq(1).sum()),
        )
    raw_path = ROOT / 'data/raw/train.csv'
    attributes = pd.read_csv(raw_path, usecols=['HomePlanet', 'CryoSleep']).iloc[aligned.row_id].reset_index(drop=True)
    earth_asleep = attributes.HomePlanet.eq('Earth') & attributes.CryoSleep.eq(True)
    y = aligned.loc[earth_asleep, 'y']
    prediction = aligned.loc[earth_asleep, 'p_v81'].ge(.5)
    earth_asleep_all = dict(
        rows=len(y), actual_transported=int(y.sum()), actual_not_transported=int(y.eq(0).sum()),
        v81_true_positive=int((prediction & y.eq(1)).sum()),
        v81_true_negative=int((~prediction & y.eq(0)).sum()),
        v81_false_positive=int((prediction & y.eq(0)).sum()),
        v81_false_negative=int((~prediction & y.eq(1)).sum()),
        always_transported_accuracy=float(y.mean()), v81_accuracy=float(prediction.eq(y).mean()),
    )
    result = dict(
        scope='Descriptive properties of the original five-family 914 shared OOF errors; no fit or selection.',
        profiles=profiles,
        all_development_Earth_asleep=earth_asleep_all,
        caveats=[
            'Model probability is not a calibrated probability that a prediction is correct.',
            'Attributes describe errors, not causal explanations or evidence of incorrect labels.',
            'These are reused development rows, not the hidden Kaggle test labels.',
        ],
        sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in [path, aligned_path, raw_path, Path(__file__)]},
    )
    target = ROOT / 'reports/error_rescue_20261117/case_profile.json'
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(profiles, indent=2))


if __name__ == '__main__':
    main()
