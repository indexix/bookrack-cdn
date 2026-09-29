# BookRack model distribution

Versioned model files for [BookRack](https://github.com/indexix/bookrack).
The app downloads only the Real-ESRGAN variants needed for the selected execution path.
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
