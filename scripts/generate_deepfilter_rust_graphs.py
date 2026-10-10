#!/usr/bin/env python3
"""Expose streaming state in the official Rust DeepFilterNet3 model bundle.

Only graph structure is embedded. Both initializer and Constant-node weights
refer to validated byte ranges in the original enc/erb_dec/df_dec.onnx files.
The three independent sessions preserve Rust's conditional decoder updates.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from generate_deepfilter_streaming_graph import fields, weight_offsets


MODELS = {
    "Enc": ("enc.onnx", "7c5399d3da8a50ebef1c1a0ae421b33376aa5e45d0e92df16da7e83c9c131916",
            {"/erb_conv0/0/Pad": ("erb_cache", [1, 1, 2, 32]),
             "/df_conv0/0/Pad": ("spec_cache", [1, 2, 2, 96])},
            ["e0", "e1", "e2", "e3", "emb", "c0", "lsnr"]),
    "Erb": ("erb_dec.onnx", "ab669a1d10afe20911728b33053a452071042317a90581092b325da7b2f9d895",
            {}, ["m"]),
    "Df": ("df_dec.onnx", "23114ce3b0f6464b763ee62f7bb8aab6b2a129a21eabd5bcfe59413db05f278a",
           {"/df_convp/df_convp.0/Pad": ("df_cache", [1, 64, 4, 96])}, ["coefs"]),
}


def constant_offsets(data):
    offsets = {}
    for field, first, last in fields(data):
        if field != 7:
            continue
        for field, first, last in fields(data, first, last):
            if field != 1:  # GraphProto.node
                continue
            parts = list(fields(data, first, last))
            outputs = [data[a:b].decode() for f, a, b in parts if f == 2]
            for field, first, last in parts:
                if field != 5:  # NodeProto.attribute
                    continue
                for field, first, last in fields(data, first, last):
                    if field != 5:  # AttributeProto.t
                        continue
                    for field, first, last in fields(data, first, last):
                        if field == 9 and last - first > 128:
                            if len(outputs) != 1:
                                raise ValueError("Expected one Constant output")
                            offsets[outputs[0]] = (first, last)
    return offsets


def make_graph(data, caches, output_names):
    model = onnx.load_model_from_string(data)
    graph = model.graph
    outputs = [value for value in graph.output if value.name in output_names]
    del graph.output[:]
    graph.output.extend(outputs)
    states, extra = [], []
    for node in graph.node:
        if node.op_type == "GRU":
            name, shape = f"h{len(states)}", [1, 1, 256]
            node.input[5] = name
            extra.append(helper.make_node("Identity", [node.output[1]], [name + "_out"]))
            states.append((name, shape))
    expected_grus = 1 if "lsnr" in output_names else 2
    if len(states) != expected_grus:
        raise ValueError("Unexpected recurrent state contract")
    for node in graph.node:
        if node.name not in caches:
            continue
        name, shape = caches[node.name]
        source, output = node.input[0], node.output[0]
        node.CopyFrom(helper.make_node("Concat", [name, source], [output], axis=2))
        prefix = "stream_" + name
        for suffix, values in (("start", [-shape[2]]), ("end", [2**63 - 1]), ("axis", [2])):
            graph.initializer.append(helper.make_tensor(prefix + suffix, TensorProto.INT64, [1], values))
        extra.append(helper.make_node("Slice", [output, prefix + "start", prefix + "end", prefix + "axis"],
                                      [name + "_out"]))
        states.append((name, shape))
    if len(states) != expected_grus + len(caches):
        raise ValueError("Missing convolution cache")
    graph.node.extend(extra)
    for name, shape in states:
        graph.input.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))
        graph.output.append(helper.make_tensor_value_info(name + "_out", TensorProto.FLOAT, shape))
    required = {value.name for value in graph.output}
    kept = []
    for node in reversed(graph.node):
        if any(output in required for output in node.output):
            required.update(node.input)
            kept.append(node)
    del graph.node[:]
    graph.node.extend(reversed(kept))
    initializers = [value for value in graph.initializer if value.name in required]
    del graph.initializer[:]
    graph.initializer.extend(initializers)
    del graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    return model, states


def verify(data, model, states, output_names):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    reference = ort.InferenceSession(data, options, providers=["CPUExecutionProvider"])
    streaming = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(147)
    worst = 0.0
    for frames in (1, 5, 20, 137):
        inputs = {value.name: rng.standard_normal(
            [frames if dimension == "S" else dimension for dimension in value.shape], dtype=np.float32)
            for value in reference.get_inputs()}
        expected = reference.run(output_names, inputs)
        feedback = {name: np.zeros(shape, np.float32) for name, shape in states}
        collected = [[] for _ in output_names]
        for frame in range(frames):
            packet = {}
            for value in reference.get_inputs():
                axis = value.shape.index("S")
                slices = [slice(None)] * len(value.shape)
                slices[axis] = slice(frame, frame + 1)
                packet[value.name] = inputs[value.name][tuple(slices)]
            results = streaming.run(None, {**packet, **feedback})
            for index, value in enumerate(results[:len(output_names)]):
                collected[index].append(value)
            feedback = {name: value for (name, _), value in zip(states, results[len(output_names):])}
        for name, want, chunks in zip(output_names, expected, collected):
            axis = 2 if want.ndim == 4 and name != "coefs" else 1
            actual = np.concatenate(chunks, axis)
            error = float(np.max(np.abs(actual - want)))
            worst = max(worst, error)
            if not np.isfinite(actual).all() or error > 2e-5:
                raise ValueError(f"{name} streaming parity failed: {error}")
    print(f"Batch/stateful graph maximum absolute error: {worst:.3g}")


def generate(directory, destination):
    lines = ["// Generated by scripts/generate_deepfilter_rust_graphs.py; do not edit.",
             "// Official DeepFilterNet3 Rust bundle; weights remain in separately downloaded ONNX files.",
             "// DeepFilterNet: Copyright (c) 2021 Hendrik Schröter (MIT).",
             "// See third_party/deepfilter/LICENSE-MIT and THIRD-PARTY-NOTICES.md."]
    for kind, (filename, digest, caches, output_names) in MODELS.items():
        data = (directory / filename).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Expected the pinned official {filename}")
        model, states = make_graph(data, caches, output_names)
        verify(data, model, states, output_names)
        offsets = {**weight_offsets(data), **constant_offsets(data)}
        nodes = []
        for node in model.graph.node:
            if node.op_type == "Constant" and node.output[0] in offsets:
                tensor = onnx.TensorProto()
                tensor.CopyFrom(node.attribute[0].t)
                tensor.name = node.output[0]
                model.graph.initializer.append(tensor)
            else:
                nodes.append(node)
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        for value in model.graph.initializer:
            if value.name not in offsets:
                if len(value.raw_data) > 128:
                    raise ValueError("Unexpected embedded weights")
                continue
            first, last = offsets[value.name]
            if data[first:last] != value.raw_data:
                raise ValueError("Initializer byte range mismatch")
            onnx.external_data_helper.set_external_data(value, filename, first, last - first)
            value.ClearField("raw_data")
        graph_bytes = model.SerializeToString()
        fingerprint = 14695981039346656037
        for byte in data:
            fingerprint = ((fingerprint ^ byte) * 1099511628211) & (2**64 - 1)
        prefix = "kRust" + kind
        lines.extend([f"// Source SHA-256: {digest}",
                      f"static constexpr size_t {prefix}SourceBytes = {len(data)};",
                      f"static constexpr uint64_t {prefix}Fingerprint = UINT64_C(0x{fingerprint:016x});",
                      f"static constexpr unsigned char {prefix}Graph[] = {{"])
        for start in range(0, len(graph_bytes), 16):
            lines.append("    " + ", ".join(f"0x{byte:02x}" for byte in graph_bytes[start:start + 16]) + ",")
        lines.extend(["};", f"static constexpr const char* {prefix}StateInputs[] = {{"])
        lines.extend(f'    "{name}",' for name, _ in states)
        lines.extend(["};", f"static constexpr const char* {prefix}StateOutputs[] = {{"])
        lines.extend(f'    "{name}_out",' for name, _ in states)
        lines.extend(["};", f"static constexpr size_t {prefix}StateSizes[] = {{"])
        lines.extend(f"    {int(np.prod(shape))}," for _, shape in states)
        lines.extend(["};", f"static constexpr std::array<int64_t, 4> {prefix}StateShapes[] = {{"])
        lines.extend("    {" + ", ".join(map(str, shape + [0] * (4 - len(shape)))) + "}," for _, shape in states)
        lines.extend(["};", f"static constexpr size_t {prefix}StateRanks[] = {{" +
                      ", ".join(str(len(shape)) for _, shape in states) + "};", ""])
        print(f"{filename}: {len(graph_bytes)} graph bytes, no weights")
    destination.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent /
                        "src/models/deepfilter/deepfilter_rust_graphs.inc")
    args = parser.parse_args()
    generate(args.directory, args.output)
