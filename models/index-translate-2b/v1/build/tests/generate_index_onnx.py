#!/usr/bin/env python3
"""Greedy Index reference generation with the same static partitions and chat template as Android."""

import argparse, json, pathlib, numpy as np, onnxruntime as ort
from transformers import AutoTokenizer

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("checkpoint", type=pathlib.Path)
p.add_argument("export", type=pathlib.Path)
p.add_argument("--limit", type=int, default=96)
a = p.parse_args()
meta = json.loads((a.export / "decoder.json").read_text())
capacity = json.loads((a.export / "conversion.json").read_text())["contextTokens"]
t = AutoTokenizer.from_pretrained(a.checkpoint)
eos = json.loads((a.checkpoint / "generation_config.json").read_text())["eos_token_id"]
if isinstance(eos, int):
    eos = [eos]
opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sessions = []
for part in meta["partitions"]:
    name = pathlib.Path(part["modelFile"]).stem
    print("Loading", name, flush=True)
    sessions.append(
        ort.InferenceSession(
            str(a.export / name / "decoder.onnx"),
            opts,
            providers=["CPUExecutionProvider"],
        )
    )
emb = np.memmap(
    a.export / "embedding.f32",
    dtype="<f4",
    mode="r",
    shape=(meta["vocabSize"], meta["hiddenSize"]),
)
report = []
for sample in [
    {"text": "今日は良い天気です。", "target": "韩语"},
    {"text": "ここから駅までは歩いて十分です。", "target": "韩语"},
]:
    prompt = f"请将以下日语文本翻译为{sample['target']}，直接输出翻译结果，不要进行任何解释。\n\n{sample['text']}"
    ids = t.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    states = [
        {
            v.name: np.zeros(v.shape, np.float32)
            for v in s.get_inputs()
            if v.name not in ["embedding", "cos", "sin", "mask", "slot"]
        }
        for s in sessions
    ]

    def step(token, position):
        d = meta["headDim"]
        angle = position / (
            meta["ropeTheta"] ** (np.arange(d // 2, dtype=np.float32) * 2 / d)
        )
        angle = np.concatenate([angle, angle])
        mask = np.full((1, 1, 1, capacity), -10000, np.float32)
        mask[..., : position + 1] = 0
        slot = np.zeros((1, 1, capacity, 1), np.float32)
        slot[:, :, position, :] = 1
        hidden = emb[token].reshape(1, 1, -1).copy()
        for i, s in enumerate(sessions):
            feed = dict(
                states[i],
                embedding=hidden,
                cos=np.cos(angle).reshape(1, 1, 1, -1),
                sin=np.sin(angle).reshape(1, 1, 1, -1),
                mask=mask,
                slot=slot,
            )
            result = s.run(None, {v.name: feed[v.name] for v in s.get_inputs()})
            hidden = result[0]
            states[i] = {
                v["input"]: result[j + 1]
                for j, v in enumerate(meta["partitions"][i]["states"])
            }
        return int(hidden[0, 0].argmax())

    position = 0
    for token in ids:
        token = step(token, position)
        position += 1
    generated = []
    for _ in range(a.limit):
        if token in eos:
            break
        generated.append(token)
        token = step(token, position)
        position += 1
    else:
        raise RuntimeError("Output limit reached before EOS: " + t.decode(generated))
    result = {
        **sample,
        "prompt": ids,
        "eos": eos,
        "generated": generated,
        "translation": t.decode(generated, skip_special_tokens=True),
    }
    report.append(result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
(a.export / "greedy-reference.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n"
)
