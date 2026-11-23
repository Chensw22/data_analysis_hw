"""Compare local class intensities with covariance orientation around each unseen passenger."""
import numpy as np
from scipy.special import logit
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import strata
from broad_booking_canonical_n23 import REPORT
from broad_boosting_path_n23 import calibrated
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART, SEEDS, historical, write
from broad_release_miss_coverage_n23 import summarize
from class_analogue_experts_n23 import design
from released_correction_audit_n23 import development


def log_density(delta, covariance):
    sign, logdet = np.linalg.slogdet(covariance)
    assert (sign == 1).all()
    mahal = (delta * np.linalg.solve(covariance, delta[..., None])[..., 0]).sum(1)
    return -.5 * (mahal + logdet)


def local_evidence(points, train, q, y):
    # Keep the neighborhood count fixed in each class. Radius describes local intensity.
    # Centered shape terms describe how departures are oriented inside the same ball.
    rng = np.random.default_rng(739)
    rot, _ = np.linalg.qr(rng.normal(size=(points.shape[1], points.shape[1])))
    values = {k: [] for k in ['votes','radius','spherical','diagonal','oriented','rotated']}
    for query in np.array_split(q, int(np.ceil(len(q) / 160))):
        distances = cdist(points[query], points[train], 'sqeuclidean')
        radius = np.partition(distances, 63, axis=1)[:, 63]
        near = distances <= radius[:, None]
        values['votes'].extend(logit((near @ y[train].astype(float) + .5) / (near.sum(1) + 1)))
        terms = {}
        for label in [False, True]:
            mask = y[train] == label
            d = distances[:, mask]
            # Stable order resolves equal distances around exactly coincident profiles.
            idx = np.argsort(d, axis=1, kind='stable')[:, :64]
            local = points[train[mask][idx]]
            scalesq = np.take_along_axis(d, idx[:, -1:], axis=1).ravel().clip(1e-6)
            center = local.mean(1)
            deviation = local - center[:, None, :]
            cov = np.swapaxes(deviation, 1, 2) @ deviation / local.shape[1]
            dim = points.shape[1]
            eye = np.eye(dim)
            variance = np.trace(cov, axis1=1, axis2=2) / dim
            # Ridge is referenced to the shared training coordinate scale (unit variance).
            ridge = (.25 * variance + .025)[:, None, None] * eye
            cov = .75 * cov + ridge
            isotropic = (variance + .025)[:, None, None] * eye
            diag = np.diagonal(cov, axis1=1, axis2=2)[:, :, None] * eye
            diff = points[query] - center
            terms[label] = dict(radius=-.5 * dim * np.log(scalesq),
                spherical=log_density(diff, isotropic), diagonal=log_density(diff, diag),
                oriented=log_density(diff, cov), rotated=log_density(diff, rot.T @ cov @ rot))
        for name in terms[False]:
            values[name].extend(terms[True][name] - terms[False][name])
    return {name: np.asarray(v) for name, v in values.items()}


def main():
    row, y, frame, groups, _ = load()
    ids, labels, masks = historical()
    np.testing.assert_array_equal(row, ids)
    np.testing.assert_array_equal(y, labels)
    _, releases, _ = development()
    bills = strata(frame)
    masks = {**masks, 'paid_transported': masks['repeat'] & (bills == 2) & y,
             'zero_stayed': masks['repeat'] & (bills == 0) & ~y,
             'solo_repeat': masks['repeat'] & (frame.BookingSize.to_numpy() == 1)}
    out = ART / 'nov23_local_class_shapes'
    out.mkdir(exist_ok=True)
    history, refs, plans = {}, [], {}
    for seed in [*map(str, SEEDS), 'v131']:
        with np.load(ART / f'nov23_broad_partition_evidence/{seed}.npz') as z:
            for k, v in [('row', row), ('y', y)]:
                np.testing.assert_array_equal(z[k], v)
            base, ref, fold = z['base'], z['ref'], z['fold']
        np.testing.assert_array_equal(ref, releases[131] if seed == 'v131' else base >= .5)
        fields = {}
        for f in np.unique(fold):
            q = np.flatnonzero(fold == f)
            t = np.flatnonzero((fold != f) & ~np.isin(groups, groups[q]))
            x, _ = design(frame, t)
            x = np.column_stack([x, np.log(frame.BookingSize)])
            pca = PCA(n_components=14, whiten=True, svd_solver='full').fit(x[t])
            z = pca.transform(x)
            for k, score in local_evidence(z, t, q, y).items():
                fields.setdefault(k, np.zeros(len(y)))[q] = score
        predictions = {'native_' + k: s >= 0 for k, s in fields.items()}
        raw = ['votes', 'radius', 'spherical']
        models = {'intensity':raw, 'diagonal':raw + ['diagonal'],
                  'oriented':raw + ['diagonal','oriented'], 'rotated':raw + ['diagonal','rotated']}
        for k, names in models.items():
            design_matrix = np.column_stack([logit(base.clip(.002,.998)), *[fields[col] for col in names]])
            vote = ref.copy()
            for f in np.unique(fold):
                ix = np.flatnonzero((fold == f) & (ref == (base >= .5)))
                for a, b in GroupKFold(3, shuffle=True, random_state=925).split(ix, groups=groups[ix]):
                    t, q = ix[a], ix[b]
                    assert not np.intersect1d(groups[t], groups[q]).size
                    vote[q] = calibrated(design_matrix[t], design_matrix[q], y, t) >= .5
            predictions['combined_' + k] = vote
        metrics = {}
        anchor = ref == y
        if seed != 'v131':
            refs.append(anchor)
        for k, v in predictions.items():
            hit = v == y
            metrics[k] = dict(accuracy=float(hit.mean()), fixed=int((hit & ~anchor).sum()), harmed=int((~hit & anchor).sum()))
            if seed != 'v131':
                history.setdefault(k, []).append(hit)
        np.savez_compressed(out / f'{seed}.npz', row=row,y=y,base=base,ref=ref,fold=fold,**fields)
        plans[seed] = metrics
        print(seed, metrics['combined_oriented'], flush=True)
    write(REPORT / 'local_shapes.json', dict(plans=plans, grouped={
        k:summarize(np.stack(h), np.stack(refs), masks) for k,h in history.items()}))


if __name__ == '__main__':
    with threadpool_limits(2):
        main()
