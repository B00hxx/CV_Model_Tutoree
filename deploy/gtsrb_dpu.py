"""Deploy step 4: run the compiled model on the Ultra96-V2 DPU (VART, Python).

Copy this file, src/classes.py and gtsrb_cnn.xmodel to the board, then:

    python3 gtsrb_dpu.py gtsrb_cnn.xmodel sign1.png sign2.png      # tightly cropped sign images
    python3 gtsrb_dpu.py gtsrb_cnn.xmodel --csv Test.csv --limit 2000   # accuracy + timing on GTSRB

Needs only what the Vitis AI board image already has: numpy, OpenCV, vart, xir.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import vart
import xir

from classes import CLASS_NAMES

# Must match src/data.py
MEAN = np.array([0.3403, 0.3121, 0.3214], np.float32)
STD = np.array([0.2724, 0.2608, 0.2669], np.float32)


def dpu_runner(xmodel_path):
    graph = xir.Graph.deserialize(str(xmodel_path))
    subgraphs = graph.get_root_subgraph().toposort_child_subgraph()
    dpu = [s for s in subgraphs if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
    assert len(dpu) == 1, f"expected 1 DPU subgraph, got {len(dpu)}; check the compiler log"
    return vart.Runner.create_runner(dpu[0], "run")


def preprocess(bgr, roi, size, in_scale):
    """BGR uint8 image -> int8 NHWC tensor, same maths as the eval transform in training."""
    if roi is not None:
        x1, y1, x2, y2 = roi
        bgr = bgr[y1:y2 + 1, x1:x2 + 1]          # ROI corners are inclusive
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    interp = cv2.INTER_AREA if rgb.shape[0] > size else cv2.INTER_LINEAR
    rgb = cv2.resize(rgb, (size, size), interpolation=interp)
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return np.clip(np.round(x * in_scale), -128, 127).astype(np.int8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("xmodel", type=Path)
    p.add_argument("images", nargs="*", type=Path)
    p.add_argument("--csv", type=Path, help="GTSRB-style CSV (Test.csv); paths relative to its folder")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    runner = dpu_runner(args.xmodel)
    in_t, out_t = runner.get_input_tensors()[0], runner.get_output_tensors()[0]
    in_scale = 2 ** in_t.get_attr("fix_point")      # float -> int8
    out_scale = 2 ** -out_t.get_attr("fix_point")   # int8 -> float logits
    _, h, _, _ = in_t.dims                           # NHWC, batch 1 on DPUCZDX8G
    inp = np.empty(tuple(in_t.dims), np.int8)
    out = np.empty(tuple(out_t.dims), np.int8)

    def infer(x):
        inp[0] = x
        job = runner.execute_async([inp], [out])
        runner.wait(job)
        logits = out.reshape(-1).astype(np.float32) * out_scale
        e = np.exp(logits - logits.max())            # softmax on the CPU
        return e / e.sum()

    if args.csv:
        rows = list(csv.DictReader(open(args.csv, newline="")))
        rows = rows[:args.limit] if args.limit else rows
        correct, t_pre, t_dpu = 0, 0.0, 0.0
        for r in rows:
            t0 = time.perf_counter()
            img = cv2.imread(str(args.csv.parent / r["Path"]))
            roi = tuple(int(r[k]) for k in ("Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"))
            x = preprocess(img, roi, h, in_scale)
            t1 = time.perf_counter()
            probs = infer(x)
            t2 = time.perf_counter()
            correct += int(probs.argmax() == int(r["ClassId"]))
            t_pre, t_dpu = t_pre + t1 - t0, t_dpu + t2 - t1
        n = len(rows)
        print(f"accuracy {correct}/{n} = {correct / n * 100:.2f}%")
        print(f"per image: load+preprocess {t_pre / n * 1e3:.2f} ms, DPU+softmax {t_dpu / n * 1e3:.2f} ms "
              f"(~{n / t_dpu:.0f} FPS single-thread, DPU only)")

    for path in args.images:
        probs = infer(preprocess(cv2.imread(str(path)), None, h, in_scale))
        print(f"\n{path}")
        for i in probs.argsort()[::-1][:args.top_k]:
            print(f"  {probs[i] * 100:6.2f}%  [{i:2d}] {CLASS_NAMES[i]}")


if __name__ == "__main__":
    main()
