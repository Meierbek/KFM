"""Independent local object proposals assist, but never replace, rule checking."""
import time

from refined_model import RefinedVideoAnalyzer


class ObjectAssistedAnalyzer(RefinedVideoAnalyzer):
    version = "object-assisted-v1"

    def __init__(self, *, model=None, endpoint=None):
        super().__init__(model=model, endpoint=endpoint)
        from yolo_candidates import LocalYoloCandidates
        self.detector = LocalYoloCandidates(confidence=.3, threads=2, image_size=960)
        self.name = f"KFM · поиск предметов + {self.model}"
        self.object_proposals = []

    def prepare_windows(self, video_path, duration, deadline):
        started = time.monotonic()
        self.object_proposals = self.detector.propose_video(video_path, duration,
            output_dir=self.current_work / "objects", interval=4.0, limit=2, deadline=deadline)
        self.last_diagnostics.update(object_proposals=len(self.object_proposals),
            object_frames=self.detector.last_diagnostics.get("frames", 0),
            object_seconds=round(time.monotonic()-started, 3))
        return super().prepare_windows(video_path, duration, deadline)

    def augment_candidates(self, candidates):
        result = list(candidates)
        for proposal in self.object_proposals:
            candidate = {"kind": "personal_object", "stamp": proposal["timecode"],
                         "crop_path": proposal["crop_path"], "object_score": proposal["score"]}
            # A generic overview tag has no object identity/bounding box.
            # Temporal proximity alone cannot justify replacing it with a crop.
            result.append(candidate)
        return sorted(result, key=lambda item: (item["stamp"], item["kind"]))

    def finalize_result(self, result):
        result["analysis"].update(sampling_strategy="overview_with_object_details",
                                  object_proposals=len(self.object_proposals))
        return result
