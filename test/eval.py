"""
Measure image-classification accuracy against the hand labels.

    python test/eval.py [--set sample|holdout] [--workers N] [--show-wrong]

Create the sample + contact sheets first with test/make_sample.py, then
write one `<id> <label>` line per image into test/<set>_labels.txt.

Reads test/<set>_manifest.csv + test/<set>_labels.txt, classifies every image
with the real pipeline (media_organizer.classifiers.image.classify) using
the repo's media-organizer.conf, maps the chosen folder to a label letter,
and prints overall + per-class accuracy, a confusion matrix and the list
of mistakes.  Results are also written to test/<set>_last_eval.csv.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

import os                                                  # noqa: E402

from media_organizer import config as config_mod          # noqa: E402
from media_organizer import workers as workers_mod        # noqa: E402
from media_organizer.classifiers import clip               # noqa: E402
from media_organizer.classifiers import image as image_cls  # noqa: E402

# Folder name -> label letter.  Old layout names are mapped too so the
# script can measure the pre-change baseline.
FOLDER_TO_LABEL = {
    "personal": "P", "documents": "D", "docs": "D", "objects": "O",
    "nonsense": "O", "screenshots": "S", "other": "X", "unknown": "X",
    "painting": "X",
}
NAMES = {"P": "personal", "D": "documents", "O": "objects",
         "S": "screenshots", "X": "other"}

def _run(item):
    idx, path = item
    t = time.perf_counter()
    r = image_cls.classify(Path(path), workers_mod._WORKER_CFG)
    return idx, r.folder, r.confidence, "|".join(r.tags), time.perf_counter() - t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="sample", help="sample | holdout")
    ap.add_argument("--workers", type=int, default=0, help="0 = auto (by free RAM)")
    ap.add_argument("--config", default=str(ROOT / "media-organizer.conf"))
    ap.add_argument("--show-wrong", action="store_true")
    args = ap.parse_args()

    paths = {}
    with (HERE / f"{args.set}_manifest.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            paths[int(row["id"])] = row["path"]
    labels = {}
    for line in (HERE / f"{args.set}_labels.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            i, lab = line.split()
            labels[int(i)] = lab.split("/")

    items = [(i, paths[i]) for i in sorted(labels)]
    cfg = config_mod.load(args.config)
    n_workers = args.workers or workers_mod._auto_workers()
    threads = max(1, (os.cpu_count() or 4) // n_workers)
    print(f"{len(items)} images, {n_workers} workers x {threads} threads")
    t0 = time.perf_counter()
    text = clip.compute_text_embeddings(cfg)
    with ProcessPoolExecutor(n_workers, initializer=workers_mod._init_worker,
                             initargs=(cfg, text, threads)) as ex:
        results = sorted(ex.map(_run, items, chunksize=4))
    wall = time.perf_counter() - t0

    ok = 0
    per_total, per_ok = Counter(), Counter()
    confusion = defaultdict(Counter)
    wrong = []
    with (HERE / f"{args.set}_last_eval.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "pred", "folder", "conf", "tags", "path"])
        for idx, folder, conf, tags, _dt in results:
            pred = FOLDER_TO_LABEL.get(folder, "X")
            want = labels[idx]
            hit = pred in want
            ok += hit
            per_total[want[0]] += 1
            per_ok[want[0]] += hit
            confusion[want[0]][pred] += 1
            if not hit:
                wrong.append((idx, "/".join(want), pred, conf, tags))
            w.writerow([idx, "/".join(want), pred, folder, f"{conf:.2f}", tags, paths[idx]])

    n = len(results)
    print(f"\nAccuracy: {ok}/{n} = {ok / n:.1%}   ({wall:.1f}s wall, {wall / n * 1000:.0f} ms/img)")
    print("\nPer class (by first label):")
    for k in "PDOSX":
        if per_total[k]:
            print(f"  {NAMES[k]:<12} {per_ok[k]:>3}/{per_total[k]:<3} {per_ok[k] / per_total[k]:.0%}")
    print("\nConfusion (rows = truth, cols = predicted):")
    print("        " + "  ".join(f"{k:>3}" for k in "PDOSX"))
    for k in "PDOSX":
        print(f"  {k}    " + "  ".join(f"{confusion[k][c]:>3}" for c in "PDOSX"))
    if args.show_wrong:
        print("\nMistakes:")
        for idx, want, pred, conf, tags in wrong:
            print(f"  #{idx:<4} want={want:<4} got={pred} conf={conf:.2f}  {tags}")


if __name__ == "__main__":
    main()
