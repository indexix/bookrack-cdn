# Hybrid model build sources

These are the model export/packaging sources used with OCR Transer 0.9.2.
QAIRT/SNPE 2.47.0.260601 must be installed separately; its proprietary SDK is
not redistributed here. The native streaming harness additionally requires the
OCR Transer application source (`app/src/main/cpp` and the llama.cpp JSON header);
it is not a standalone test of this model-data repository.

Start with the pinned checkpoint and the four-layer export from
`export_index_snpe.py`. Verify the float export against the source checkpoint
with `tests/verify_index_snpe.py`, collecting fresh calibration via
`--calibration-output` and the official non-thinking chat template. Split the
vocabulary projection with `split_index_snpe_head.py`, then apply
`rewrite_index_snpe_norm.py` to each head. Split the six decoder body graphs
with `split_index_snpe_body.py --head-parts <head-parts.json>`. Its one-row body
calibration is only intended for the explicit CPU FLOAT32 execution plan.

Convert all 12 body and four rewritten head ONNX files using:

```
snpe-onnx-to-dlc -i decoder.onnx -o float.dlc --preserve_io layout --no_simplification
```

For each body, run the SDK binary `snpe-dlc-quant` with `--act_bitwidth 16
--weights_bitwidth 8 --bias_bitwidth 32`, its generated `inputs.txt`, and the
converted DLC. Body activations are evaluated by CPU FLOAT32 in the app; this
does not make the body an A16 DSP graph. For each head, use `--act_bitwidth 8
--weights_bitwidth 8 --bias_bitwidth 32` and the fresh full head calibration
input list. Do not use the old `quantize_index_snpe.py` defaults for these heads:
that script is the original all-A16 conversion pipeline, retained for context.

Package the 12 body and four head DLCs with the split decoder metadata,
`--execution-mode cpu_body_dsp_head --activation-bits 8`, context 512, output
limit 256 and attention mask -10000. Preserve the embedding, tokenizer and
Apache-2.0 license from the pinned source. See `conversion.json` and
`validation.json` for the exact tested scope. A successful conversion or
preparation alone is not evidence of usable translation quality.
