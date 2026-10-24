#!/usr/bin/env python3
"""Calibrate FP32 Index partitions as W8A16 DLCs. Run after float parity and conversion."""

import argparse, json, os, pathlib, subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("folder", type=pathlib.Path)
p.add_argument("--sdk", type=pathlib.Path, default=pathlib.Path("/sdk"))
p.add_argument("--parts", nargs="+", help="Only named partitions (for resuming a successful earlier stage)")
a = p.parse_args()
meta = json.loads((a.folder / "decoder.json").read_text())
assert len(json.loads((a.folder / "float-parity.json").read_text())) == len(
    meta["partitions"]
)
env = dict(
    os.environ,
    LD_LIBRARY_PATH=str(a.sdk / "lib/x86_64-linux-clang"),
    PATH=str(a.sdk / "bin/x86_64-linux-clang") + ":" + os.environ["PATH"],
)
for part in meta["partitions"]:
    name = pathlib.Path(part["modelFile"]).stem
    if a.parts and name not in a.parts:
        continue
    source = a.folder / (name + ".preserved.float.dlc")
    output = a.folder / part["modelFile"]
    pending = output.with_name(output.stem + ".partial.dlc")
    print("Quantizing", name, flush=True)
    subprocess.run(
        [
            "bash",
            str(a.sdk / "bin/x86_64-linux-clang/snpe-dlc-quantize"),
            "--input_dlc",
            str(source),
            "--input_list",
            str(a.folder / name / "calibration/inputs-onnx.txt"),
            "--output_dlc",
            str(pending),
            "--act_bitwidth",
            "16",
            "--weights_bitwidth",
            "8",
            "--bias_bitwidth",
            "32",
        ],
        env=env,
        check=True,
    )
    pending.replace(output)
