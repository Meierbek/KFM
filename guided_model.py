"""Local learned candidate selection followed by factual/legal visual checks.

The binary classifier can rank suspicious moments, but cannot assign a rule.
A positive classifier signal without visual confirmation remains unresolved.
"""
from __future__ import annotations

import math
import time

from analyzers import AnalysisError
from refined_model import RefinedVideoAnalyzer
from visual_sampling import remaining


class GuidedVideoAnalyzer(RefinedVideoAnalyzer):
    version = "guided-v1"

    def __init__(self, *, model=None, endpoint=None):
        super().__init__(model=model, endpoint=endpoint)
        from visual_baseline import VisualBaseline
        self.baseline = VisualBaseline()
        self.name = f"KFM · обученный детектор + {self.model}"
        self.proposals = []

    def prepare_windows(self, video_path, duration, deadline):
        from visual_baseline import select_candidates
        started = time.monotonic()
        centers, scores = self.baseline.score_windows(video_path, duration, deadline=deadline)
        if len(centers) != len(scores) or not len(scores) or any(not math.isfinite(float(score)) for score in scores):
            raise AnalysisError("invalid_candidate_scores", "Локальный детектор вернул некорректные оценки кадров.")
        # Fixed before holdout evaluation; these are scores, not calibrated
        # probabilities of a legal violation. Only the three best moments are
        # rechecked by the slower vision-language model on this 4 GB GPU.
        self.proposals = select_candidates(centers, scores, self.baseline.threshold, limit=3)
        self.last_diagnostics.update(candidate_frames=len(scores), proposal_windows=len(self.proposals),
                                     detector_seconds=round(time.monotonic()-started, 3))
        remaining(deadline)
        end = duration-min(.1, duration/2)
        windows = []
        for proposal in self.proposals:
            center = proposal["timecode"]
            windows.append(sorted({round(max(0, min(end, center+offset)), 3) for offset in (-8, -6, -4, -2, 0, 2, 4, 6, 8)}))
        return windows

    def finalize_result(self, result):
        if self.proposals and result["verdict"] == "no_violation":
            raise AnalysisError("uncertain_observations", "Обученный детектор отметил подозрительный фрагмент, но подробная проверка не смогла обосновать тип нарушения. Отрицательный вердикт не сформирован.")
        result["analysis"].update(sampling_strategy="learned_candidates", candidate_windows=len(self.proposals),
                                  detector_fingerprint=self.baseline.config["fingerprint"],
                                  detector_signal=bool(self.proposals))
        return result
