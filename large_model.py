"""Local 8B candidate with the existing grounded verification pipeline.

The fixed batch size limits prompt-processing batches; it does not change the
inherited context, sampling, JSON gates, or original-frame evidence handling.
Ollama exposes num_batch as a Runner option:
https://github.com/ollama/ollama/blob/main/api/types.go
"""
from refined_model import RefinedVideoAnalyzer


class LargeVideoAnalyzer(RefinedVideoAnalyzer):
    version = "large-v1"

    def __init__(self, *, model=None, endpoint=None):
        super().__init__(model=model or "qwen3-vl:8b-instruct", endpoint=endpoint)
        self.name = f"{self.model} · экспериментальная проверка 8B"

    def request(self, path, body=None, timeout=10):
        if path == "/api/chat" and body is not None:
            body = dict(body)
            body["options"] = {**body.get("options", {}), "num_batch": 128}
        return super().request(path, body, timeout=timeout)

    def analyze(self, video_path, metadata, progress):
        try:
            return super().analyze(video_path, metadata, progress)
        finally:
            try:
                # Empty generation with keep_alive=0 only requests unloading.
                # Each subsequent video pays its own model-loading time.
                self.request("/api/generate", {"model": self.model, "keep_alive": 0}, timeout=5)
            except Exception:
                # Releasing memory must never replace a result or its error.
                pass
