# BookRack model distribution

Versioned model files for [BookRack](https://github.com/indexix/bookrack).
The app downloads only the upscaling or TTS variants selected by the user.
Native execution libraries are distributed with the app, not this repository.

## Real-ESRGAN general-x4v3 v1

[Download assets](https://github.com/indexix/bookrack-cdn/releases/tag/realesr-general-x4v3-v1)

- Original checkpoint: [`realesr-general-x4v3`](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0), by Xintao Wang.
- License: [BSD 3-Clause](LICENSE-Real-ESRGAN.txt).
- [Manifest](models/realesr-general-x4v3/v1/manifest.json): exact file lengths and SHA-256 digests.
- ncnn: `.param` and `.bin`, CPU/Vulkan, original 4x network reduced to 2x in the app.
- LiteRT: `.tflite`, static tiles 64/96/128, NHWC, valid-page mask and 2x output.
- SNPE: `.dlc`, static tiles 64/96/128, INT8 (numerically different from float models).
- RKNN: `.rknn`, tiles 64/96/128 for RK3566/RK3568, RK3562, RK3576 and RK3588.

These are the existing BookRack 0.27.1 model bytes, moved out of the APK without retraining.
Runtime compatibility still depends on the device's drivers and firmware. The app checks actual
execution; publishing a variant does not certify every device with that chipset.

Models are release assets, not Git source blobs. Do not replace files under an existing version:
publish a new version and update its manifest. The app pins the expected version and digests.
No credentials are required to download these public assets.

## Supertonic 3 TTS v1

[Download voice models](https://github.com/indexix/bookrack-cdn/releases/tag/supertonic3-v1).
Choose one bundle: INT8 (126 MB download) or FP32 (372 MB download).
Both include four ONNX graphs, ten voice styles, the character index, configuration, and original license/provenance.
Models and voice data are not packaged in the BookRack application. The native ONNX Runtime library ships in the application.

- [Pinned manifest](models/supertonic3/v1/tts-model-catalog.json) contains archive and individual-file SHA-256 checksums.
- [Model license: OpenRAIL-M](models/supertonic3/v1/Supertonic-model-OpenRAIL-M.txt). Its use-based restrictions apply to both bundles and downstream redistribution.
- FP32: `supertone-oss-archive/supertonic-3` revision `aafc6e32416a594460b32413efc49d7fe4ce6d46` on Hugging Face.
- INT8: `csukuangfj2/sherpa-onnx-supertonic-3-tts-int8-2026-05-11` revision `cca5a0e6c96e1d2c720986bf7e75fcc81dee3ae4` on Hugging Face.
- The INT8 bundle retains the FP32 duration predictor and text encoder; only vector estimator/vocoder differ. No retraining or weight changes were made by BookRack.
- Shared voice and text resources come from the pinned official FP32 repository.

BookRack performs local CPU inference. These downloads do not certify GPU/NPU compatibility or performance on a particular e-reader.

## Index-Translate-2B NPU conversion pipeline

The manual `index-npu-rkllm` workflow converts only `IndexTeam/Index-Translate-2B`
revision `9bc08445315723eb9b77d1b93505915eb03c3f01` using RKLLM Toolkit 1.3.1.
RK3588, RK3576, RK3562 and RV1126B have independent conversions and checksums.
The workflow uploads candidates to an existing **draft** release; it does not
publish a release or claim on-device validation. Conversion metadata distinguishes
SDK conversion from runtime execution, which requires the target hardware.

Model weights are Apache 2.0; see [the model license](models/index-translate-2b/v1/LICENSE.txt).
Runtime libraries are not included in model assets. Download catalogs pin every
asset's exact length and SHA-256 digest, including files split into 1 GB pieces.
