#!/usr/bin/env python3
"""Compare every exported Index partition with original HF decoder layers at every token.
Also emits representative recurrent/KV-state calibration inputs (ONNX layouts).
"""

import argparse, gc, json, pathlib
import numpy as np
import onnxruntime as ort
import torch
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3_5TextConfig, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("checkpoint", type=pathlib.Path)
p.add_argument("export", type=pathlib.Path)
a = p.parse_args()
c = json.loads((a.checkpoint / "config.json").read_text())["text_config"]
config = Qwen3_5TextConfig(**c)
config._attn_implementation = "eager"
meta = json.loads((a.export / "decoder.json").read_text())
capacity = 512
options = ort.SessionOptions()
options.intra_op_num_threads = 2
options.inter_op_num_threads = 1
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
# SNPE float reference is deterministic; disable quantized/prepacked ORT alternatives.
torch.set_num_threads(2)
tok = AutoTokenizer.from_pretrained(a.checkpoint)
samples = [
    ("日语", "韩语", "今日は良い天気です。", "오늘은 날씨가 좋습니다."),
    (
        "英语",
        "韩语",
        "Please close the door and wait here.",
        "문을 닫고 여기서 기다려 주세요.",
    ),
    ("韩语", "日语", "어디로 가고 있나요?", "どこへ行っていますか？"),
]
sequences = []
for source, target, text, answer in samples:
    prompt = f"请将以下{source}文本翻译为{target}，直接输出翻译结果，不要进行任何解释。\n\n{text}"
    sequences.append(
        tok.encode(
            "<|im_start|>user\n"
            + prompt
            + "<|im_end|>\n<|im_start|>assistant\n"
            + answer
            + "<|im_end|>",
            add_special_tokens=False,
        )
    )
embedding = np.memmap(
    a.export / "embedding.f32",
    dtype="<f4",
    mode="r",
    shape=(c["vocab_size"], c["hidden_size"]),
)
actual = [np.array(embedding[ids]).reshape(len(ids), 1, 1, -1) for ids in sequences]
references = [x.copy() for x in actual]
report = []
reader = safe_open(str(a.checkpoint / "model.safetensors"), framework="pt")
for part in meta["partitions"]:
    name = pathlib.Path(part["modelFile"]).stem
    folder = a.export / name
    print("Checking", name, flush=True)
    session = ort.InferenceSession(
        str(folder / "decoder.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    is_head = part["output"] == "logits"
    layers = []
    if not is_head:
        start = int(name.split("-")[1]) * 4
        for i in range(start, start + 4):
            layer = Qwen3_5DecoderLayer(config, i).eval()
            prefix = f"model.language_model.layers.{i}."
            layer.load_state_dict(
                {
                    k[len(prefix) :]: reader.get_tensor(k).float()
                    for k in reader.keys()
                    if k.startswith(prefix)
                }
            )
            layers.append(layer)
    else:
        norm = Qwen3_5RMSNorm(c["hidden_size"], eps=c["rms_norm_eps"])
        norm.load_state_dict(
            {"weight": reader.get_tensor("model.language_model.norm.weight").float()}
        )
        head = reader.get_tensor("model.language_model.embed_tokens.weight").float()
    cal = folder / "calibration"
    cal.mkdir(exist_ok=True)
    lines = []
    max_error = 0
    top_equal = 0
    count = 0
    next_actual = []
    next_ref = []
    for s, ids in enumerate(sequences):
        cache = DynamicCache(config=config)
        state = {
            x.name: np.zeros(x.shape, np.float32)
            for x in session.get_inputs()
            if x.name not in ["embedding", "cos", "sin", "mask", "slot"]
        }
        values = []
        refs = []
        chosen = set(np.linspace(0, len(ids) - 1, 8, dtype=int))
        for pos, token in enumerate(ids):
            d = meta["headDim"]
            angles = pos / (
                meta["ropeTheta"] ** (np.arange(d // 2, dtype=np.float32) * 2 / d)
            )
            angles = np.concatenate([angles, angles])
            cos = np.cos(angles).reshape(1, 1, 1, -1)
            sin = np.sin(angles).reshape(1, 1, 1, -1)
            mask = np.full((1, 1, 1, capacity), -10000, np.float32)
            mask[..., : pos + 1] = 0
            slot = np.zeros((1, 1, capacity, 1), np.float32)
            slot[:, :, pos, :] = 1
            allfeed = dict(
                state, embedding=actual[s][pos], cos=cos, sin=sin, mask=mask, slot=slot
            )
            feed = {v.name: allfeed[v.name] for v in session.get_inputs()}
            if pos in chosen:
                location = cal / f"{s}-{pos}"
                location.mkdir(exist_ok=True)
                entries = []
                for key, value in feed.items():
                    value.astype("<f4").tofile(location / (key + ".raw"))
                    entries.append(
                        f"{key}:=/work/.local/index-npu/snpe/{name}/calibration/{s}-{pos}/{key}.raw"
                    )
                lines.append(" ".join(entries))
            out = session.run(None, feed)
            y = torch.from_numpy(references[s][pos])
            with torch.no_grad():
                if is_head:
                    y = torch.nn.functional.linear(norm(y), head)
                else:
                    for layer in layers:
                        y = layer(
                            y,
                            position_embeddings=(
                                torch.from_numpy(cos[:, 0]),
                                torch.from_numpy(sin[:, 0]),
                            ),
                            attention_mask=torch.zeros(1, 1, 1, pos + 1),
                            past_key_values=cache,
                        )
            ref = y.numpy()
            err = float(np.max(np.abs(ref - out[0])))
            max_error = max(max_error, err)
            if not np.allclose(ref, out[0], atol=0.008, rtol=0.005):
                raise AssertionError(
                    f"{name} sample{s} token{pos}: max absolute error {err}"
                )
            if is_head:
                top_equal += int(ref.argmax() == out[0].argmax())
            count += 1
            if not is_head:
                values.append(out[0])
                refs.append(ref)
            state = {
                item["input"]: out[idx + 1] for idx, item in enumerate(part["states"])
            }
        if not is_head:
            next_actual.append(np.stack(values))
            next_ref.append(np.stack(refs))
    (cal / "inputs-onnx.txt").write_text("\n".join(lines) + "\n")
    item = {
        "partition": name,
        "tokens": count,
        "maxAbsError": max_error,
        "top1Matches": top_equal if is_head else None,
    }
    print(item, flush=True)
    report.append(item)
    if not is_head:
        actual, references = next_actual, next_ref
    del session, layers, cache, out
    gc.collect()
(a.export / "float-parity.json").write_text(json.dumps(report, indent=2) + "\n")
print("All partitions match the original Hugging Face weights.", flush=True)
