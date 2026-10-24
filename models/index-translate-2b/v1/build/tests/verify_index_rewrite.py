#!/usr/bin/env python3
"""Validate an algebraic ONNX rewrite against HF-validated original partitions.
Uses three representative recurrent/KV states per partition, then copies the
original calibration dataset with corrected container paths for the new graph.
"""

import argparse, gc, json, pathlib, shutil, numpy as np, onnxruntime as ort

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("original", type=pathlib.Path)
p.add_argument("rewritten", type=pathlib.Path)
a = p.parse_args()
assert len(json.loads((a.original / "float-parity.json").read_text())) == 7
meta = json.loads((a.rewritten / "decoder.json").read_text())
reports = []
opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
for part in meta["partitions"]:
    name = pathlib.Path(part["modelFile"]).stem
    old = a.original / name
    new = a.rewritten / name
    print("Compare rewrite", name, flush=True)
    original = ort.InferenceSession(
        str(old / "decoder.onnx"), opts, providers=["CPUExecutionProvider"]
    )
    rewritten = ort.InferenceSession(
        str(new / "decoder.onnx"), opts, providers=["CPUExecutionProvider"]
    )
    locations = [
        sorted(
            (old / "calibration").glob(f"{i}-*"),
            key=lambda p: int(p.name.split("-")[1]),
        )[-1]
        for i in range(3)
    ]
    maximum = 0
    for location in locations:
        feed = {
            v.name: np.fromfile(location / (v.name + ".raw"), np.float32).reshape(
                v.shape
            )
            for v in original.get_inputs()
        }
        ref = original.run(None, feed)
        actual = rewritten.run(None, feed)
        for x, y in zip(ref, actual):
            maximum = max(maximum, float(np.max(np.abs(x - y))))
            assert np.allclose(x, y, rtol=0.002, atol=0.001), (name, maximum)
    reports.append(
        {
            "partition": name,
            "statesCompared": len(locations),
            "maxAbsError": maximum,
            "reference": "original ONNX, independently validated against Hugging Face weights",
        }
    )
    print(reports[-1], flush=True)
    # Retain authoritative measured calibration values; this rewrite is numerically equivalent.
    shutil.copytree(old / "calibration", new / "calibration", dirs_exist_ok=True)
    inputs = (
        (new / "calibration/inputs-onnx.txt")
        .read_text()
        .replace(str(a.original), str(a.rewritten))
    )
    (new / "calibration/inputs-onnx.txt").write_text(inputs)
    del original, rewritten
    gc.collect()
(a.rewritten / "float-parity.json").write_text(json.dumps(reports, indent=2) + "\n")
