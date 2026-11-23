"""Compare passenger distributions of fee records using projected optimal matching."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import BILLS
from broad_opposing_passengers_n23 import evaluate
from broad_partition_evidence_n23 import ROOT
from exaone_validation_data import CATEGORICAL

LABEL = 'service_distribution_distance'
REPORT = ROOT / 'reports' / ('nov23_' + LABEL)


def squared_distances(left, right):
    values = (left**2).sum(1)[:, None] + (right**2).sum(1) - 2 * (left @ right.T)
    return values.clip(min=0.)


def service_coordinates(frame, train):
    raw = frame[BILLS].to_numpy(float)
    missing = np.isnan(raw)
    fees = np.log1p(np.nan_to_num(raw))
    fees /= np.quantile(fees[train][fees[train] > 0], .9)
    # Five equally weighted atoms per person. Unknown fees carry a separate flag.
    # In one dimension coupling ordered atoms realizes their exact Wasserstein distance.
    atoms = np.stack([fees, missing], axis=2)
    directions = np.random.default_rng(913).normal(size=(2 + len(BILLS), 32))
    directions /= np.linalg.norm(directions, axis=0)
    projected_fees = atoms @ directions[:2]
    identified = projected_fees + .65 * directions[2:][None, :, :]
    return {
        'unordered_fees': np.sort(projected_fees, axis=1).reshape(len(frame), -1),
        'sliced_identity': np.sort(identified, axis=1).reshape(len(frame), -1),
        'aligned_services': atoms.reshape(len(frame), -1),
    }


def personal_coordinates(frame, train):
    spend = frame[BILLS]
    numeric = frame[['Age', 'BookingSize', 'CabinNumber']].copy()
    numeric['total'] = np.log1p(spend.sum(axis=1))
    numeric['positive'] = spend.gt(0).sum(axis=1)
    numeric['missing'] = spend.isna().sum(axis=1)
    continuous = numeric.to_numpy(float)
    null = np.isnan(continuous)
    low, high = np.nanquantile(continuous[train], [.02, .98], axis=0)
    continuous = np.where(null, np.nanmedian(continuous[train], axis=0), continuous)
    continuous = ((continuous-low) / np.maximum(high-low, 1.)).clip(0., 1.)
    categorical = pd.get_dummies(frame[CATEGORICAL], dtype=float)
    return np.column_stack([continuous, null, categorical.to_numpy() / np.sqrt(2.)])


def distance_to_landmarks(x, train, anchors):
    distance = squared_distances(x, x[anchors])
    subset = distance[train]
    return distance / max(.00001, np.median(subset[subset > .000001]))


def fit_fold(frame, groups, labels, train, query):
    anchors = np.random.default_rng(231).choice(train, 384, replace=False)
    context = distance_to_landmarks(personal_coordinates(frame, train), train, anchors)
    spending = service_coordinates(frame, train)
    distances = {'budget': context}
    for name, coords in spending.items():
        distances[name] = context + distance_to_landmarks(coords, train, anchors)

    probabilities = {}
    for name, distance in distances.items():
        kernel = np.exp(-distance)
        # Whiten the same landmarks to approximate regularized kernel logistic regression.
        vals, vecs = np.linalg.eigh((kernel[anchors] + kernel[anchors].T) / 2.)
        design = (kernel @ vecs) / np.sqrt(vals.clip(.002))
        model = LogisticRegression(C=10., max_iter=300)
        model.fit(design[train], labels[train])
        probabilities[name] = model.predict_proba(design[query])[:, 1]
    return probabilities, dict(landmarks=len(anchors), directions=32, records=len(BILLS))


if __name__ == '__main__':
    with threadpool_limits(2):
        evaluate(LABEL, fit_fold, REPORT)
