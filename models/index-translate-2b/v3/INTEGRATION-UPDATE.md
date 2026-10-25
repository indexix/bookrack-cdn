# Validated runtime integration update (2026-10-25)

The initial 41.5-second graph-cache candidate documented at the v3 model tag later reproduced Android LOW_MEMORY termination on the same BOOX device. The 96 MiB candidate and early batching candidates also failed later checks. Those implementations are rejected; their initial passing fixtures do not establish stability. Original weights, model assets, catalog hashes and the v3 tag are unchanged.

The final OCR Transer 0.9.3 integration uses the same 16 small W8A8 heads with up to eight CPU Q4 draft rows per loaded DSP graph. All emitted tokens are selected from complete DSP vocabulary logits. Mismatches discard future drafts and replay accepted history through the CPU. The model/context remains reusable; only one large DSP graph is live at a time. This is explicit CPU prediction plus NPU verification, not all-NPU execution or automatic CPU fallback.

Before DSP allocation, read-only GGUF mappings are identified by file device/inode (Android paths can have aliases), then relinquished with MADV_DONTNEED. POSIX_FADV_DONTNEED also requests release of the model file cache. Unloaded DLC cache pages receive the same hint. After each batch, all graph instances and the small session anchor are destroyed before SDK cleanupRuntime releases backend resources. A global mutex and live-graph counter prevent cleanup while another SNPE instance is active. This trades extra I/O/session startup for less overlapping memory pressure. Missing cleanup is an error, not permission to keep allocating.

BOOX Go103_2 / SM6350 / Hexagon V66 / Android 15, approximately 3.37 GiB reported system RAM:

| Fixture | CPU alone (seconds) | Final hybrid (seconds) |
| --- | ---: | ---: |
| Japanese weather sentence | 5.764 | 16.324 |
| English thank-you sentence | 4.849 | 15.286 |
| Repeated Japanese sentence | 5.732 | 16.422 |

The original FP32-body hybrid took 164.306 seconds for the same Japanese input. CPU remains faster. Same source/prompt/greedy sampling, but head quantization, context and memory policies differ; these are application-path timings, not equal-precision accelerator throughput tests. Model preparation/download is excluded.

Cancellation/resume, repeated output, a two-sentence Japanese fixture (79.235 seconds, 22 generated tokens), and an intentionally wrong draft followed by exact expected-output recovery (32.796 seconds) passed. The final series had no app LOW_MEMORY termination. Max sampled app PSS was 1.47 GiB; separate DSP/driver allocations and sub-sample peaks are excluded. Device host samples showed roughly 1.1-1.2 GB free RAM during DSP processing after file-cache reclamation, compared with roughly 70-90 MB in earlier failed runs. This does not certify every device or concurrent workload.

Raw state and CPU draft logits matched the original CPU graph across 16 positions and two resets per mode. Host CPU SNPE tests verified sequential/batched projection equivalence, handle reuse counts, recurrent state, cancellation and validation boundaries. Kotlin unit tests (41), Android Lint, debug app/test builds, APK signature and ZIP alignment passed. The final APK's installed/local checksums match; APKs are not distributed in this model release.

The actual Wi-Fi download/migration from the published v2 A8 profile succeeded; final tests use that same downloaded v3 bundle. Earlier network-test runtime measurements are not final-runtime stability results. Other Qualcomm devices, A16 and Rockchip execution, long documents, broad translation quality and battery behavior remain outside this physical test scope.

See [validation-final.json](validation-final.json) and the replacement app-local llama.cpp patch [index-batched-verification.patch](build/index-batched-verification.patch). The earlier raw-state-only patch and validation.json remain as historical records. App native code also implements batching, rollback, memory hints and SDK cleanup; the llama.cpp patch alone is not the complete runtime.
