import unittest
import time
from pathlib import Path

from analyzers import AnalysisError

from visual_baseline import extract_embeddings, linear_probabilities, model_scores, sample_times, select_candidates, temporal_features
from train_visual_baseline import choose_threshold, dev_samples, make_folds, metric_set, training_rows
from visual_baseline import canonical_hash


class VisualBaselineTests(unittest.TestCase):
    def test_temporal_features_preserve_order_and_all_frames(self):
        import numpy as np
        frames = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=float)
        features, centers = temporal_features(frames, [0, 2, 4])
        self.assertEqual(features.shape, (3, 6))
        self.assertEqual(list(centers), [0, 2, 4])
        self.assertTrue(np.isfinite(features).all())
        with self.assertRaises(ValueError):
            temporal_features(frames, [0, 0, 4])
        self.assertEqual(sample_times(5), [0, 2, 4])

    def test_positive_nonreference_windows_are_excluded(self):
        import numpy as np
        videos = [
            {"sample_id": "a", "expected_violation": True, "reference_time_sec": 4, "centers": np.asarray([0, 2, 4, 6, 8]), "features": np.arange(10).reshape(5, 2)},
            {"sample_id": "b", "expected_violation": False, "reference_time_sec": None, "centers": np.asarray([0, 2, 4]), "features": np.arange(6).reshape(3, 2) + 20},
        ]
        x, y, weights, groups = training_rows(videos, [0, 1])
        self.assertEqual(list(y), [1, 1, 1, 0, 0, 0])
        self.assertEqual(list(x[:3, 0]), [2, 4, 6])
        self.assertEqual(groups, ["a", "a", "a", "b", "b", "b"])
        self.assertAlmostEqual(float(weights[y == 1].sum()), float(weights[y == 0].sum()))

    def test_threshold_balances_both_classes(self):
        labels = [True, True, True, False, False]
        scores = [.9, .8, .4, .5, .1]
        threshold, metrics = choose_threshold(labels, scores)
        self.assertGreater(metrics["balanced_accuracy"], .5)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metric_set(labels, scores, 0)["balanced_accuracy"], .5)

    def test_candidate_suppression_keeps_strongest_neighbor(self):
        candidates = select_candidates([0, 2, 10], [.8, .9, .7], .75)
        self.assertEqual(candidates, [{"timecode": 2.0, "score": .9}])

    def test_exported_linear_transform_matches_expected_margin(self):
        import numpy as np
        arrays = {"scaler_mean": np.asarray([1., 2.]), "scaler_scale": np.asarray([2., 4.]),
                  "pca_mean": np.asarray([0., 0.]), "pca_components": np.asarray([[1., 0.]]),
                  "coef": np.asarray([[2.]]), "intercept": np.asarray([0.])}
        result = linear_probabilities([[1, 2], [3, 2]], arrays)
        self.assertAlmostEqual(result[0], .5)
        self.assertAlmostEqual(result[1], 1 / (1 + np.exp(-2)))

    def test_holdout_is_excluded_and_video_folds_never_overlap(self):
        rows = [{"sample_id": str(index), "video_sha256": str(index), "split": "dev" if index < 20 else "holdout", "expected_violation": bool(index % 2)} for index in range(24)]
        selected = dev_samples({"samples": rows, "split_fingerprint": canonical_hash(rows)})
        self.assertEqual(len(selected), 20)
        all_validation = []
        for train, validation in make_folds(selected):
            self.assertFalse(set(train) & set(validation))
            all_validation.extend(validation)
        self.assertEqual(sorted(all_validation), list(range(20)))

    def test_identical_content_is_grouped_despite_different_sample_ids(self):
        rows = [{"sample_id": str(index), "video_sha256": str(index), "expected_violation": bool(index % 2)} for index in range(20)]
        rows.append({"sample_id": "duplicate-alias", "video_sha256": "0", "expected_violation": False})
        for train, validation in make_folds(rows):
            self.assertFalse({rows[index]["video_sha256"] for index in train} & {rows[index]["video_sha256"] for index in validation})

    def test_multiple_reference_times_select_union_of_positive_windows(self):
        import numpy as np
        videos = [
            {"sample_id": "a", "expected_violation": True, "reference_time_sec": 2, "reference_times_sec": [2, 10], "centers": np.arange(0, 14, 2), "features": np.arange(14).reshape(7, 2)},
            {"sample_id": "b", "expected_violation": False, "reference_time_sec": None, "centers": np.asarray([0]), "features": np.asarray([[20, 21]])},
        ]
        x, y, _, _ = training_rows(videos, [0, 1])
        self.assertEqual(list(x[y == 1, 0]), [0, 2, 4, 8, 10, 12])

    def test_expired_deadline_stops_before_video_access(self):
        with self.assertRaises(AnalysisError) as raised:
            extract_embeddings(Path("does-not-exist.mov"), 10, None, deadline=time.monotonic() - 1)
        self.assertEqual(raised.exception.code, "analysis_timeout")

    def test_row_l2_linear_scoring(self):
        import numpy as np
        weights = {"coef": np.asarray([[1., 0.]]), "intercept": np.asarray([0.])}
        scores = model_scores([[3, 4], [0, 2]], weights, {"preprocessing": "row_l2", "estimator": "logistic"})
        self.assertAlmostEqual(scores[0], 1 / (1 + np.exp(-.6)))
        self.assertAlmostEqual(scores[1], .5)

    def test_exported_rbf_scores_match_sklearn(self):
        import numpy as np
        from sklearn.svm import SVC
        x = np.asarray([[1, 0], [.9, .1], [0, 1], [.1, .9]], dtype=float)
        normalized = x / np.linalg.norm(x, axis=1, keepdims=True)
        model = SVC(C=.1, gamma="scale", kernel="rbf").fit(normalized, [0, 0, 1, 1])
        arrays = {"support_vectors": model.support_vectors_, "dual_coef": model.dual_coef_,
                  "gamma": np.asarray([model._gamma]), "intercept": model.intercept_}
        expected = 1 / (1 + np.exp(-model.decision_function(normalized)))
        scores = model_scores(x, arrays, {"preprocessing": "row_l2", "estimator": "rbf_svm"})
        self.assertTrue(np.allclose(scores, expected, atol=1e-10))


if __name__ == "__main__":
    unittest.main()
