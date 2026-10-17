"""Small local end-to-end check; exports only aggregate metrics to stdout."""
import json
from pathlib import Path
import time

from analyzers import AnalysisError
from local_model import OllamaVideoAnalyzer, ROOT
from server import read_metadata


def main():
    import openpyxl
    dataset = ROOT.parent / "open"
    rows = list(openpyxl.load_workbook(dataset / "open_labels.xlsx", data_only=True).active.values)[1:]
    candidates = []
    for name, timestamp, label in rows:
        path = dataset / "videos" / f"{name}.mov"
        metadata = read_metadata(path)
        candidates.append((metadata["duration_sec"] or 9999, path, metadata, bool(timestamp)))
    # Short examples let us establish feasibility on this laptop before a larger
    # evaluation. These are not a representative independent final test set.
    positive = sorted((x for x in candidates if x[3]), key=lambda x: x[0])[:2]
    negative = sorted((x for x in candidates if not x[3]), key=lambda x: x[0])[:3]
    model = OllamaVideoAnalyzer()
    results = []
    for index, (duration, path, metadata, expected) in enumerate(positive + negative):
        started = time.monotonic()
        record = {"filename": path.name, "expected_violation": expected, "duration_sec": duration}
        try:
            result = model.analyze(path, metadata, lambda *args: None)
            record.update(status="done", predicted_violation=result["verdict"] == "violation", event_count=len(result["violations"]))
        except AnalysisError as exc:
            record.update(status="failed", error=exc.code)
        record["seconds"] = round(time.monotonic() - started, 2)
        results.append(record)
        (ROOT / "local_review" / "benchmark.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"test": index + 1, "total": 5, "status": record["status"], "seconds": record["seconds"], "error": record.get("error")}), flush=True)
    completed = [item for item in results if item["status"] == "done"]
    summary = {"tested": len(results), "completed": len(completed), "correct": sum(item["predicted_violation"] == item["expected_violation"] for item in completed), "failed": len(results)-len(completed), "max_seconds": max(item["seconds"] for item in results)}
    (ROOT / "local_review" / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
