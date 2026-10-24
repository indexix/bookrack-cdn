#!/usr/bin/env python3
"""Run in QAIRT 2.47 Linux container, after export_index_snpe.py and float parity.
Converts each static partition. Does not mark a model ready or publish it.
"""

import argparse, json, pathlib, subprocess, os

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("folder", type=pathlib.Path)
p.add_argument("--sdk", type=pathlib.Path, default=pathlib.Path("/sdk"))
a = p.parse_args()
env = dict(
    os.environ,
    PYTHONPATH=str(a.sdk / "lib/python"),
    LD_LIBRARY_PATH=str(a.sdk / "lib/x86_64-linux-clang"),
)
meta = json.loads((a.folder / "decoder.json").read_text())
for part in meta["partitions"]:
    name = pathlib.Path(part["modelFile"]).stem
    target = a.folder / (name + ".preserved.float.dlc")
    if target.exists() and target.stat().st_size > 1000:
        continue
    pending = target.with_name(target.stem + ".partial.dlc")
    print("Converting", name, flush=True)
    subprocess.run(
        [
            str(a.sdk / "bin/x86_64-linux-clang/snpe-onnx-to-dlc"),
            "-i",
            str(a.folder / name / "decoder.onnx"),
            "-o",
            str(pending),
            "--preserve_io",
            "layout",
            # The final norm/projection is already static; folding its 2 GB
            # matrix duplicates memory without simplifying any useful work.
            *(["--no_simplification"] if name == "head" else []),
        ],
        env=env,
        check=True,
    )
    pending.replace(target)
