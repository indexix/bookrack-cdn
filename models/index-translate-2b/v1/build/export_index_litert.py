#!/usr/bin/env python3
"""Index-only LiteRT NPU pipeline using pinned litert-torch source and QAIRT.
Run stages independently; CPU/GPU exports are never published as NPU artifacts.
"""

import argparse, json, pathlib

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("stage", choices=["export", "calibrate", "quantize", "compile"])
p.add_argument("checkpoint")
p.add_argument("output", type=pathlib.Path)
p.add_argument("--soc", default="SM8750")
p.add_argument("--context", type=int, default=512)
p.add_argument("--metadata", required=True)
a = p.parse_args()
from litert_torch.generative.export_hf.experimental.npu_export.config_manager import (
    NpuPipelineConfig,
)
from litert_torch.generative.export_hf.experimental.npu_export.stages import (
    npu_export,
    npu_calibrate,
    npu_quantize,
    npu_compile,
)

c = NpuPipelineConfig(
    model_id=a.checkpoint,
    backend="qualcomm",
    soc_model=a.soc,
    cache_length=a.context,
    prefill_lengths=[32],
)
a.output.mkdir(parents=True, exist_ok=True)
exported = a.output / "export/model.litertlm"
quantized = a.output / "quantized.litertlm"
if a.stage == "export":
    npu_export(
        c,
        str(a.output / "export"),
        keep_temporary_files=True,
        model_dtype="float32",
        litert_lm_llm_metadata_override=a.metadata,
        export_vision_encoder=False,
    )
elif a.stage == "calibrate":
    dataset = a.output / "calibration-dataset"
    dataset.mkdir(exist_ok=True)
    samples = json.loads(
        (
            pathlib.Path(__file__).parent / "calibration/index-translation.json"
        ).read_text()
    )
    (dataset / "translation.jsonl").write_text(
        "\n".join(json.dumps({"text": s["input"]}, ensure_ascii=False) for s in samples)
        + "\n"
    )
    npu_calibrate(
        c,
        str(exported),
        str(dataset),
        str(a.output / "calibration"),
        enable_formatting=False,
    )
elif a.stage == "quantize":
    npu_quantize(c, str(exported), str(quantized), str(a.output / "calibration"))
else:
    npu_compile(
        c,
        quantized,
        a.output / f"index2-{a.soc.lower()}.litertlm",
        keep_temporary_files=True,
        intermediate_dir=a.output / f"compile-{a.soc.lower()}",
    )
