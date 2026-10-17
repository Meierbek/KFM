"""Create a local preliminary annotation table without sending media to a cloud.

The supplied labels select useful review timestamps; they are NOT passed to the
model and are never used by the live analyzer. This is annotation, not a blind
quality benchmark. Model descriptions are written to disk, never to stdout.
"""
import argparse
import csv
import html
import json
from pathlib import Path
import re
import time

from analyzers import AnalysisError
from local_model import CODES, OllamaVideoAnalyzer
from server import read_metadata


ROOT = Path(__file__).resolve().parent


def parse_time(value):
    if not value:
        return None
    text = str(value)
    minutes = re.search(r"(\d+)\s*мин", text)
    seconds = re.search(r"(\d+)\s*сек", text)
    return (int(minutes.group(1)) * 60 if minutes else 0) + (int(seconds.group(1)) if seconds else 0)


def write_reports(rows, target):
    columns = ["Видео", "Метка организаторов", "Таймкод разметки, сек.", "Кадры просмотрены на секундах", "Что происходит по оценке локальной модели", "Предполагаемый тип", "Правило для проверки", "Ограничения", "Статус"]
    records = []
    for row in rows:
        observation = row.get("observation", {})
        events = observation.get("events", [])
        def event_time(event):
            stamps = observation.get("timestamps", [])
            index = event.get("frame_index")
            if isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(stamps):
                seconds = stamps[index]
                return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"
            return "время не определено"
        types = "\n".join(dict.fromkeys(f"{event_time(event)} — {CODES.get(event.get('code'), 'Не определён')}" for event in events)) or "Локальная модель не выделила тип"
        rules = []
        conflicts = 0
        for event in events:
            rule = row.get("rules", {}).get(event.get("rule_id"))
            if rule and rule.get("code") == event.get("code"):
                rules.append(f"{event_time(event)} — {rule['title']}; {rule['section']}; {rule['url']} — применимость требует проверки")
            elif rule:
                conflicts += 1
        descriptions = [observation.get("summary", "")]
        descriptions += [f"{event_time(event)} — {event.get('observation', '')}" for event in events]
        records.append([
            row["filename"], row["label"], "" if row["label_time"] is None else str(row["label_time"]),
            ", ".join(str(stamp) for stamp in observation.get("timestamps", [])),
            "\n".join(item for item in descriptions if item), types,
            "\n".join(dict.fromkeys(rules)) or "Надёжное соответствие правилу не установлено",
            " ".join(filter(None, [observation.get("limitations", "") or row.get("error", ""), "Модель указала противоречивые тип и правило; такая ссылка исключена." if conflicts else ""])),
            "Предварительный автоматический разбор" if row.get("status") == "done" else "Ошибка локального анализа",
        ])
    with (target / "video_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(columns)
        # Keep model-authored text from becoming spreadsheet formulas.
        writer.writerows([["'" + str(value) if str(value).lstrip().startswith(("=", "+", "-", "@")) else value for value in record] for record in records])
    heads = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = ""
    for row, record in zip(rows, records):
        cells = "".join(f"<td>{html.escape(str(value)).replace(chr(10), '<br>')}</td>" for value in record)
        body += f"<tr>{cells}</tr>"
    document = """<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>KFM — автоматический разбор обучающих видео</title><style>body{font:14px/1.6 'Segoe UI',sans-serif;margin:30px;color:#292532;background:#f7f5f8}h1{font-size:28px}p{max-width:1050px;color:#74677d}.notice{padding:15px 20px;background:#fff4dd;border:1px solid #e6d1a2;border-radius:10px}.scroll{overflow:auto;background:white;border-radius:12px;border:1px solid #e6dde9;margin-top:25px}table{border-collapse:collapse;min-width:1900px;width:100%}th,td{text-align:left;vertical-align:top;padding:15px;border-bottom:1px solid #eae3ef}th{background:#852344;color:white;font-size:12px;position:sticky;top:0}td{font-size:12px;min-width:100px}td:nth-child(5){min-width:360px}td:nth-child(7){min-width:300px}tr:nth-child(even){background:#fbf9fc}</style><h1>KFM — разбор обучающих видео</h1><p class="notice">Предварительный результат локальной модели. Это не подтверждённая разметка и не юридическое заключение. Использованы выбранные кадры, а не непрерывный просмотр. Метка и таймкод организаторов приведены отдельно и не заменяются выводом модели. Для роликов с меткой «нарушение» кадры выбираются вокруг размеченного времени; это не независимый тест качества.</p><p>Все изображения и ответы обрабатывались на этом компьютере. Таблица обновляется по мере обработки. CSV-файл рядом можно открыть в Excel.</p><div class="scroll"><table><thead><tr>""" + heads + "</tr></thead><tbody>" + body + "</tbody></table></div></html>"
    (target / "video_review.html").write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dataset", default=str(ROOT.parent / "open"))
    args = parser.parse_args()
    import openpyxl
    dataset = Path(args.dataset).resolve()
    labels = list(openpyxl.load_workbook(dataset / "open_labels.xlsx", data_only=True).active.values)[1:]
    if args.limit:
        labels = labels[:args.limit]
    target = ROOT / "local_review"
    target.mkdir(exist_ok=True)
    model = OllamaVideoAnalyzer()
    if not model.ready():
        raise RuntimeError("Local model or FFmpeg is not ready")
    rows = []
    for index, (name, stamp, label) in enumerate(labels):
        path = dataset / "videos" / f"{name}.mov"
        output = target / ("video-" + str(index + 1).zfill(2))
        output.mkdir(exist_ok=True)
        cache = output / "observation.json"
        label_time = parse_time(stamp)
        row = {"filename": path.name, "label": label, "label_time": label_time, "status": "processing", "rules": model.by_id}
        started = time.monotonic()
        try:
            duration = read_metadata(path)["duration_sec"]
            if not duration:
                raise ValueError("Cannot read video duration")
            if label_time is not None:
                timestamps = sorted({round(max(.1, min(duration - .2, label_time + delta)), 3) for delta in (-4, -2, 0, 2, 4)})
            else:
                timestamps = [round((duration - .3) * fraction + .1, 3) for fraction in (0, .2, .4, .6, .8, 1)]
            if cache.exists():
                observation = json.loads(cache.read_text(encoding="utf-8"))
            else:
                observation = model.observe(path, timestamps, folder=output, timeout=180)
                cache.write_text(json.dumps(observation, ensure_ascii=False, indent=2), encoding="utf-8")
            row.update(status="done", observation=observation)
        except Exception as exc:
            row.update(status="failed", error=exc.code if isinstance(exc, AnalysisError) else type(exc).__name__)
        rows.append(row)
        (target / "review.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        write_reports(rows, target)
        print(json.dumps({"processed": index + 1, "total": len(labels), "status": row["status"], "seconds": round(time.monotonic() - started, 1)}), flush=True)
    print(json.dumps({"finished": True, "processed": len(rows), "completed": sum(row["status"] == "done" for row in rows), "report": "local_review/video_review.html"}), flush=True)


if __name__ == "__main__":
    main()
