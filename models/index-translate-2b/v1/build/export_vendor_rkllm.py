#!/usr/bin/env python3
"""Run on Linux x86_64 with the pinned RKLLM Toolkit 1.3.1. Uses original HF weights."""

import argparse, json, pathlib, shutil
from package_vendor_model import package


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("output", type=pathlib.Path)
    p.add_argument("--model", choices=["qwen3", "qwen35", "index2"], required=True)
    p.add_argument(
        "--target", choices=["rk3588", "rk3576", "rk3562", "rv1126b"], required=True
    )
    p.add_argument(
        "--calibration",
        type=pathlib.Path,
        required=True,
        help="Representative translation prompts/targets in the SDK JSON format",
    )
    p.add_argument("--context", type=int, default=2048)
    p.add_argument("--license-file", type=pathlib.Path)
    p.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32"
    )
    a = p.parse_args()
    if not 256 <= a.context <= 4096:
        p.error("context must be between 256 and 4096")
    from rkllm.api import RKLLM

    llm = RKLLM()
    a.output.mkdir(parents=True, exist_ok=True)
    source = {}
    if a.model == "index2":
        source = json.loads((pathlib.Path(a.checkpoint) / "source-pin.json").read_text())
        assert source["source"] == "IndexTeam/Index-Translate-2B"
        assert source["revision"] == "9bc08445315723eb9b77d1b93505915eb03c3f01"
    if a.license_file:
        shutil.copyfile(a.license_file, a.output / "LICENSE.txt")

    def checked(status, stage):
        if status != 0:
            raise RuntimeError(f"RKLLM {stage} failed ({status}); no bundle published")

    try:
        checked(
            llm.load_huggingface(
                model=a.checkpoint, device="cpu", dtype=a.dtype, load_weight=True
            ),
            "HF import",
        )
        checked(
            llm.build(
                do_quantization=True,
                optimization_level=1,
                quantized_dtype="W8A8",
                quantized_algorithm="normal",
                target_platform=a.target.upper(),
                num_npu_core={"rk3588": 3, "rk3576": 2, "rk3562": 1, "rv1126b": 1}[
                    a.target
                ],
                dataset=str(a.calibration.resolve()),
                hybrid_rate=0,
                max_context=a.context,
            ),
            "quantization",
        )
        checked(llm.export_rkllm(str((a.output / "model.rkllm").resolve())), "export")
    finally:
        del llm  # Toolkit 1.3.1 has no release() API.
    (a.output / "conversion.json").write_text(
        json.dumps(
            {
                "toolkit": "1.3.1",
                "checkpoint": a.checkpoint,
                "target": a.target,
                "context": a.context,
                "sourceDtype": a.dtype,
                "onDeviceVerified": False,
                "conversionVerified": True,
                "referenceVerified": False,
                "validationScope": "RKLLM Toolkit conversion and W8A8 calibration completed; generated model execution requires target Rockchip hardware and has not been verified here.",
                **source,
            },
            indent=2,
        )
        + "\n"
    )
    package(
        a.output,
        a.output.with_suffix(".zip"),
        {
            "runtime": "rkllm",
            "model": a.model,
            "modelFile": "model.rkllm",
            "target": a.target,
            "quantization": "W8A8",
            "contextTokens": a.context,
            "maxOutputTokens": min(512, a.context // 2),
        },
    )


if __name__ == "__main__":
    main()
