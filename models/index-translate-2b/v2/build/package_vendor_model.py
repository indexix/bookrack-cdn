#!/usr/bin/env python3
"""Package converted model DATA for the app. Never bundles native libraries."""

import argparse, hashlib, json, pathlib, zipfile


def package(folder, output, metadata):
    names = [metadata["modelFile"]]
    if metadata["runtime"] == "snpe":
        names += ["embedding.f32", "tokenizer.gguf"]
        metadata["decoder"] = json.loads((folder / "decoder.json").read_text())
        for part in metadata["decoder"].get("partitions", []):
            if part["modelFile"] not in names:
                names.append(part["modelFile"])
    for optional in ["LICENSE.txt", "NOTICE.txt", "conversion.json"]:
        if (folder / optional).is_file():
            names.append(optional)
    metadata["schema"] = 1
    metadata["files"] = {}
    for name in names:
        if pathlib.Path(name).name != name or ".." in name:
            raise ValueError("Only flat model filenames are allowed")
        p = folder / name
        with p.open("rb") as f:
            sha = hashlib.file_digest(f, "sha256").hexdigest()
        metadata["files"][name] = {"sha256": sha, "size": p.stat().st_size}
    manifest_text = json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    (folder / "manifest.json").write_text(manifest_text)
    if output is None:
        return metadata  # Automatic-download bundles need a manifest, not a duplicate ZIP.
    temporary = output.with_suffix(output.suffix + ".part")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as z:
            z.writestr(
                "manifest.json",
                manifest_text,
            )
            for name in names:
                z.write(folder / name, name)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(output)
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=pathlib.Path)
    p.add_argument("output", type=pathlib.Path, nargs="?", help="Omit to create only manifest.json for automatic downloads")
    p.add_argument("--runtime", choices=["litert", "snpe", "rkllm"], required=True)
    p.add_argument("--model", choices=["qwen3", "qwen35", "index2"], required=True)
    p.add_argument("--model-file", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--quantization", required=True)
    p.add_argument("--context", type=int, default=2048)
    p.add_argument("--max-output", type=int, default=512)
    p.add_argument("--execution-mode", choices=["npu", "cpu_body_dsp_head"], default="npu")
    p.add_argument("--activation-bits", type=int, choices=[8, 16])
    a = p.parse_args()
    if (
        not 256 <= a.context <= 4096
        or not 16 <= a.max_output <= 1024
        or a.max_output >= a.context
    ):
        p.error("Invalid context/output limit")
    if a.execution_mode == "cpu_body_dsp_head" and (a.runtime != "snpe" or a.model != "index2" or a.activation_bits != 8):
        p.error("The CPU-body/DSP-head plan requires Index SNPE with 8-bit DSP activations")
    package(
        a.folder,
        a.output,
        {
            "runtime": a.runtime,
            "model": a.model,
            "modelFile": a.model_file,
            "target": a.target.lower(),
            "quantization": a.quantization,
            "contextTokens": a.context,
            "maxOutputTokens": a.max_output,
            **({"executionMode": a.execution_mode} if a.runtime == "snpe" else {}),
            **({"activationBits": a.activation_bits} if a.activation_bits else {}),
        },
    )


if __name__ == "__main__":
    main()
