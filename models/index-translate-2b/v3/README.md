# Index A8 hybrid v3 (OCR Transer 0.9.3+)

This profile keeps the pinned Q4_K_M decoder body on CPU and runs 16 small W8A8 final-normalization/vocabulary shards on Qualcomm DSP. It is not full-model NPU execution. SHA-256-verified CPU weights are fetched separately from the pinned official GGUF URL in conversion.json, or reused from the app's existing CPU model store.

The 520 MB data bundle replaces only the explicitly named v2 A8 bundle. A16 and all four Rockchip bundles retain their original files and validation scope. Runtime code remains packaged with the app.

On BOOX Go103_2 / SM6350 / V66 / Android 15, Japanese generation fell from 164.3 seconds to 41.5 seconds; CPU alone was 7.1 seconds (5.7 on repeat). English was 27.3 seconds vs CPU 4.8. Cached generation/repetition, cancellation followed by a longer 22-token translation, and float/body parity checks passed. The longer translation took 106.2 seconds; peak sampled app PSS was 1.35 GiB, excluding separate DSP/driver allocations. This remains an experimental compatibility path and is slower than CPU on this device. Other devices, long documents and broad translation quality are not certified.

Reproduce the original head ONNX and representative calibration from the pinned checkpoint with the v2 build scripts, split using `split_index_snpe_head.py --shards 16`, rewrite each head using `rewrite_index_snpe_norm.py`, compare float outputs, then run `convert_index_snpe_small_heads.py` inside the QAIRT 2.47 Linux container. Device generation is a separate validation gate. The opt-in llama.cpp patch applies after the app's existing compatibility patches; it exposes pre-normalization residual states to the separately normalized SNPE heads.
