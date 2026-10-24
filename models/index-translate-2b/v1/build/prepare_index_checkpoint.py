#!/usr/bin/env python3
"""Fetch the pinned Index checkpoint and optionally shard its text-only weights.
No remote Python code is loaded. Vision/MTP tensors are excluded from the text export.
"""

import argparse, json, pathlib, shutil

REVISION = "9bc08445315723eb9b77d1b93505915eb03c3f01"
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("original", type=pathlib.Path)
p.add_argument("--text-output", type=pathlib.Path)
a = p.parse_args()
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

snapshot_download(
    "IndexTeam/Index-Translate-2B",
    revision=REVISION,
    local_dir=str(a.original),
    allow_patterns=[
        "*.json",
        "*.safetensors",
        "*.txt",
        "*.jinja",
        "LICENSE*",
        "README.md",
    ],
)
source = {
    "source": "IndexTeam/Index-Translate-2B",
    "revision": REVISION,
    "license": "Apache-2.0",
}
(a.original / "source-pin.json").write_text(json.dumps(source, indent=2) + "\n")
if a.text_output:
    out = a.text_output
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((a.original / "config.json").read_text())["text_config"]
    config["architectures"] = ["Qwen3_5ForCausalLM"]
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ]:
        if (a.original / name).exists():
            shutil.copyfile(a.original / name, out / name)
    index = {}
    chunk = {}
    size = 0
    total = 0
    number = 0

    def save():
        global chunk, size, number
        name = f"model-{number:03d}.safetensors"
        save_file(chunk, str(out / name), metadata={"format": "pt"})
        for key in chunk:
            index[key] = name
        print(name, size, flush=True)
        chunk = {}
        size = 0
        number += 1

    with safe_open(str(a.original / "model.safetensors"), framework="pt") as reader:
        for key in reader.keys():
            if not key.startswith("model.language_model."):
                continue
            tensor = reader.get_tensor(key)
            length = tensor.numel() * tensor.element_size()
            if chunk and size + length > 1_100_000_000:
                save()
            chunk[key.replace("model.language_model.", "model.")] = tensor
            size += length
            total += length
        if chunk:
            save()
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": index}, indent=2)
        + "\n"
    )
    (out / "source-pin.json").write_text(json.dumps(source, indent=2) + "\n")
