#!/usr/bin/env python3
"""Index-Translate-2B text decoder: static single-token ONNX partitions for SNPE.

Uses original BF16 HF weights, not GGUF-dequantized weights. Float reference parity,
DLC conversion, calibration, quantization and target execution are separate gates.
"""

import argparse
import json
import math
import pathlib
import sys
import numpy as np
import onnx
from onnx import helper as H, numpy_helper as N, TensorProto as T
from safetensors import safe_open


class Graph:
    def __init__(self, folder, reader, config, capacity):
        self.folder, self.reader, self.c, self.capacity = (
            folder,
            reader,
            config,
            capacity,
        )
        folder.mkdir(parents=True, exist_ok=True)
        self.external = (folder / "weights.bin").open("wb")
        self.nodes, self.constants, self.inputs, self.outputs, self.states = (
            [],
            [],
            [],
            [],
            [],
        )
        self.serial = 0
        self.names = set()

    def const(self, a, name=None):
        name = name or f"c{len(self.constants)}"
        self.constants.append(N.from_array(np.ascontiguousarray(a), name))
        return name

    def weight(self, suffix, transpose=False, plus_one=False):
        name = "model.language_model." + suffix
        a = self.reader.get_tensor(name).float().numpy()
        if transpose:
            a = a.T.copy()
        if plus_one:
            a = a + 1
        if name in self.names:
            raise ValueError("Duplicate weight " + name)
        self.names.add(name)
        t = T(name=name, data_type=T.FLOAT, dims=a.shape, data_location=T.EXTERNAL)
        offset = self.external.tell()
        self.external.write(a.astype("<f4").tobytes())
        for k, v in [
            ("location", "weights.bin"),
            ("offset", str(offset)),
            ("length", str(a.nbytes)),
        ]:
            e = t.external_data.add()
            e.key, e.value = k, v
        self.constants.append(t)
        return name

    def op(self, kind, *args, name=None, **attrs):
        self.serial += 1
        name = name or f"n{self.serial}"
        self.nodes.append(H.make_node(kind, list(args), [name], name=name, **attrs))
        return name

    def shape(self, x, *shape):
        return self.op("Reshape", x, self.const(np.array(shape, np.int64)))

    def slice(self, x, start, end, axis=-1):
        return self.op(
            "Slice",
            x,
            self.const(np.array([start], np.int64)),
            self.const(np.array([end], np.int64)),
            self.const(np.array([axis], np.int64)),
        )

    def trans(self, x, perm):
        return self.op("Transpose", x, perm=perm)

    def linear(self, x, w):
        return self.op("MatMul", x, self.weight(w, transpose=True))

    def silu(self, x):
        return self.op("Mul", x, self.op("Sigmoid", x))

    def sum(self, x, axis):
        return self.op(
            "ReduceSum", x, self.const(np.array([axis], np.int64)), keepdims=1
        )

    def norm(self, x, w=None, l2=False, plus_one=True):
        square = self.op("Mul", x, x)
        # Keep epsilon inside the reduction to avoid QAIRT's fused RMSNorm,
        # whose CPU reference path differs measurably for small input norms.
        # mean(x*x + eps) is algebraically identical to mean(x*x) + eps.
        eps = self.const(np.array(1e-6, np.float32))
        var = (
            self.op("Add", self.sum(square, -1), eps)
            if l2
            else self.op(
                "ReduceMean", self.op("Add", square, eps), axes=[-1], keepdims=1
            )
        )
        y = self.op(
            "Div",
            x,
            self.op("Sqrt", var),
        )
        return self.op("Mul", y, self.weight(w, plus_one=plus_one)) if w else y

    def input(self, name, shape):
        self.inputs.append(H.make_tensor_value_info(name, T.FLOAT, shape))
        return name

    def state(self, name, shape, value):
        self.input(name, shape)
        output = name + "_out"
        self.op("Identity", value, name=output)
        self.outputs.append(H.make_tensor_value_info(output, T.FLOAT, shape))
        self.states.append({"input": name, "output": output})
        return output

    def rotary(self, x):
        d = int(self.c["head_dim"] * self.c["rope_parameters"]["partial_rotary_factor"])
        first = self.slice(x, 0, d // 2)
        second = self.slice(x, d // 2, d)
        rotated = self.op("Concat", self.op("Neg", second), first, axis=-1)
        prefix = self.op(
            "Add",
            self.op("Mul", self.slice(x, 0, d), "cos"),
            self.op("Mul", rotated, "sin"),
        )
        return self.op("Concat", prefix, self.slice(x, d, self.c["head_dim"]), axis=-1)

    def attention(self, x, i):
        c, cap = self.c, self.capacity
        heads, kv, d = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        p = f"layers.{i}.self_attn."
        qg = self.shape(self.linear(x, p + "q_proj.weight"), 1, 1, heads, 2 * d)
        gate = self.shape(self.slice(qg, d, 2 * d), 1, 1, heads * d)
        q = self.rotary(
            self.trans(
                self.norm(self.slice(qg, 0, d), p + "q_norm.weight"), [0, 2, 1, 3]
            )
        )
        k = self.rotary(
            self.trans(
                self.norm(
                    self.shape(self.linear(x, p + "k_proj.weight"), 1, 1, kv, d),
                    p + "k_norm.weight",
                ),
                [0, 2, 1, 3],
            )
        )
        v = self.trans(
            self.shape(self.linear(x, p + "v_proj.weight"), 1, 1, kv, d), [0, 2, 1, 3]
        )
        vals = []
        inv = self.op("Sub", self.const(np.array(1, np.float32)), "slot")
        for key, val in [("key", k), ("value", v)]:
            name = f"{key}_{i}"
            y = self.state(
                name,
                [1, kv, cap, d],
                self.op("Add", self.op("Mul", name, inv), self.op("Mul", val, "slot")),
            )
            # Replicate heads without Expand/5D tensors, keeping DSP-supported ranks.
            y = self.op(
                "Concat",
                *[
                    self.op(
                        "Concat",
                        *([self.slice(y, j, j + 1, 1)] * (heads // kv)),
                        axis=1,
                    )
                    for j in range(kv)
                ],
                axis=1,
            )
            vals.append(y)
        scores = self.op(
            "Add",
            self.op(
                "Mul",
                self.op("MatMul", q, self.trans(vals[0], [0, 1, 3, 2])),
                self.const(np.array(1 / math.sqrt(d), np.float32)),
            ),
            "mask",
        )
        y = self.op("MatMul", self.op("Softmax", scores, axis=-1), vals[1])
        y = self.shape(self.trans(y, [0, 2, 1, 3]), 1, 1, heads * d)
        return self.linear(
            self.op("Mul", y, self.op("Sigmoid", gate)), p + "o_proj.weight"
        )

    def delta(self, x, i):
        c = self.c
        h, k, v = (
            c["linear_num_value_heads"],
            c["linear_key_head_dim"],
            c["linear_value_head_dim"],
        )
        assert c["linear_num_key_heads"] == h
        width = h * (2 * k + v)
        kernel = c["linear_conv_kernel_dim"]
        p = f"layers.{i}.linear_attn."
        qkv = self.trans(self.linear(x, p + "in_proj_qkv.weight"), [0, 2, 1])
        conv = f"conv_{i}"
        updated = self.op("Concat", self.slice(conv, 1, kernel), qkv, axis=-1)
        updated = self.state(conv, [1, width, kernel], updated)
        w = self.weight(p + "conv1d.weight")
        w = self.shape(w, 1, width, kernel)
        qkv = self.silu(self.sum(self.op("Mul", updated, w), -1))
        qkv = self.trans(qkv, [0, 2, 1])
        q = self.norm(self.shape(self.slice(qkv, 0, h * k), 1, h, 1, k), l2=True)
        q = self.op("Mul", q, self.const(np.array(1 / math.sqrt(k), np.float32)))
        key = self.norm(
            self.shape(self.slice(qkv, h * k, 2 * h * k), 1, h, 1, k), l2=True
        )
        value = self.shape(self.slice(qkv, 2 * h * k, width), 1, h, 1, v)
        z = self.shape(self.linear(x, p + "in_proj_z.weight"), 1, h, 1, v)
        beta = self.shape(
            self.op("Sigmoid", self.linear(x, p + "in_proj_b.weight")), 1, h, 1, 1
        )
        a = self.op(
            "Add", self.linear(x, p + "in_proj_a.weight"), self.weight(p + "dt_bias")
        )
        # Stable softplus: max(x,0)+log(1+exp(-abs(x))).
        soft = self.op(
            "Add",
            self.op("Relu", a),
            self.op(
                "Log",
                self.op(
                    "Add",
                    self.const(np.array(1, np.float32)),
                    self.op("Exp", self.op("Neg", self.op("Abs", a))),
                ),
            ),
        )
        g = self.op(
            "Neg", self.op("Mul", self.op("Exp", self.weight(p + "A_log")), soft)
        )
        decay = self.shape(self.op("Exp", g), 1, h, 1, 1)
        state = f"recurrent_{i}"
        old = self.op("Mul", state, decay)
        kt = self.trans(key, [0, 1, 3, 2])
        memory = self.sum(self.op("Mul", old, kt), -2)
        delta = self.op("Mul", self.op("Sub", value, memory), beta)
        new = self.state(
            state, [1, h, k, v], self.op("Add", old, self.op("Mul", kt, delta))
        )
        y = self.sum(self.op("Mul", new, self.trans(q, [0, 1, 3, 2])), -2)
        y = self.op(
            "Mul", self.norm(y, p + "norm.weight", plus_one=False), self.silu(z)
        )
        return self.linear(self.shape(y, 1, 1, h * v), p + "out_proj.weight")

    def layer(self, x, i):
        p = f"layers.{i}."
        n = self.norm(x, p + "input_layernorm.weight")
        y = (
            self.delta(n, i)
            if self.c["layer_types"][i] == "linear_attention"
            else self.attention(n, i)
        )
        x = self.op("Add", x, y)
        n = self.norm(x, p + "post_attention_layernorm.weight")
        y = self.op(
            "Mul",
            self.silu(self.linear(n, p + "mlp.gate_proj.weight")),
            self.linear(n, p + "mlp.up_proj.weight"),
        )
        return self.op("Add", x, self.linear(y, p + "mlp.down_proj.weight"))

    def save(self, output, shape):
        self.outputs.insert(0, H.make_tensor_value_info(output, T.FLOAT, shape))
        self.external.close()
        model = H.make_model(
            H.make_graph(
                self.nodes,
                "index_text_decoder",
                self.inputs,
                self.outputs,
                self.constants,
            ),
            opset_imports=[H.make_opsetid("", 13)],
        )
        model.ir_version = 9
        onnx.save_model(model, self.folder / "decoder.onnx")
        onnx.checker.check_model(str(self.folder / "decoder.onnx"))


def export(checkpoint, gguf, out, capacity, group):
    pin = json.loads((checkpoint / "source-pin.json").read_text())
    assert pin["source"] == "IndexTeam/Index-Translate-2B"
    assert pin["revision"] == "9bc08445315723eb9b77d1b93505915eb03c3f01"
    config = json.loads((checkpoint / "config.json").read_text())
    assert config["model_type"] == "qwen3_5"
    c = config["text_config"]
    hidden, vocab = c["hidden_size"], c["vocab_size"]
    out.mkdir(parents=True, exist_ok=True)
    parts, states = [], []
    with safe_open(
        str(checkpoint / "model.safetensors"), framework="pt", device="cpu"
    ) as reader:
        for start in range(0, c["num_hidden_layers"], group):
            end = min(start + group, c["num_hidden_layers"])
            name = f"part-{start//group:02d}"
            graph = Graph(out / name, reader, c, capacity)
            x = graph.input("embedding", [1, 1, hidden])
            if "full_attention" in c["layer_types"][start:end]:
                d = int(c["head_dim"] * c["rope_parameters"]["partial_rotary_factor"])
                for n in ["cos", "sin"]:
                    graph.input(n, [1, 1, 1, d])
                graph.input("mask", [1, 1, 1, capacity])
                graph.input("slot", [1, 1, capacity, 1])
            for i in range(start, end):
                x = graph.layer(x, i)
            graph.op("Identity", x, name="hidden")
            graph.save("hidden", [1, 1, hidden])
            parts.append(
                {"modelFile": name + ".dlc", "output": "hidden", "states": graph.states}
            )
            states += graph.states
            print("Exported", name, flush=True)
        graph = Graph(out / "head", reader, c, capacity)
        x = graph.norm(graph.input("embedding", [1, 1, hidden]), "norm.weight")
        graph.op("Identity", graph.linear(x, "embed_tokens.weight"), name="logits")
        graph.save("logits", [1, 1, vocab])
        parts.append({"modelFile": "head.dlc", "output": "logits", "states": []})
        embedding = reader.get_tensor("model.language_model.embed_tokens.weight")
        # Bounded conversion buffer: the full FP32 embedding is over 2 GB.
        with (out / "embedding.f32").open("wb") as f:
            for row in embedding.split(1024):
                f.write(row.float().numpy().astype("<f4").tobytes())
    sys.path.insert(
        0,
        str(
            pathlib.Path(__file__).resolve().parents[1]
            / "third_party/llama.cpp/gguf-py"
        ),
    )
    from gguf import GGUFReader, GGUFWriter

    r = GGUFReader(gguf)
    w = GGUFWriter(out / "tokenizer.gguf", "qwen35")
    for key, field in r.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        w.add_key_value(
            key,
            field.contents(),
            field.types[0],
            field.types[1] if len(field.types) > 1 else None,
        )
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    decoder = {
        "hiddenSize": hidden,
        "headDim": int(c["head_dim"] * c["rope_parameters"]["partial_rotary_factor"]),
        "vocabSize": vocab,
        "ropeTheta": c["rope_parameters"]["rope_theta"],
        "states": states,
        "partitions": parts,
    }
    (out / "decoder.json").write_text(json.dumps(decoder, indent=2) + "\n")
    (out / "conversion.json").write_text(
        json.dumps(
            {
                "source": "IndexTeam/Index-Translate-2B",
                "revision": "9bc08445315723eb9b77d1b93505915eb03c3f01",
                "architecture": "qwen3_5",
                "contextTokens": capacity,
                "stage": "ONNX export; parity, DLC conversion, calibration and validation required",
                "onDeviceVerified": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=pathlib.Path)
    p.add_argument("tokenizer_gguf", type=pathlib.Path)
    p.add_argument("output", type=pathlib.Path)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--layers-per-partition", type=int, default=4)
    a = p.parse_args()
    if not 256 <= a.context <= 4096 or not 1 <= a.layers_per_partition <= 24:
        p.error("Invalid context/partition size")
    export(a.checkpoint, a.tokenizer_gguf, a.output, a.context, a.layers_per_partition)
