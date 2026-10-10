#!/usr/bin/env python3
"""Generate the weight-free streaming graph used by DeepFilterEnhancer.

The published batch model is left unchanged. Its five GRUs and three causal
convolutions gain explicit state tensors; feature lookahead is handled by the
C++ frontend. External initializers refer to byte ranges in the original ONNX
file, supplied to ORT in memory at session creation. No Python is needed at
runtime and no second copy of the weights is distributed.

Regenerate with Python's onnx, numpy, and onnxruntime packages installed:
  python scripts/generate_deepfilter_streaming_graph.py /models/deepfilter.onnx
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


SOURCE_SHA256 = "e1157049059434ae0d5857e32c812abea227b975e946b2eb64d001abbce156d3"
EXTERNAL_FILE = "deepfilter_source.onnx"


def fields(data: bytes, start: int = 0, end: int | None = None):
    """Locate protobuf payloads without reserializing weight tensors."""
    end = len(data) if end is None else end
    pos = start

    def varint():
        nonlocal pos
        value = 0
        for shift in range(0, 70, 7):
            if pos >= end:
                raise ValueError("Truncated protobuf")
            byte = data[pos]
            pos += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value
        raise ValueError("Invalid protobuf varint")

    while pos < end:
        tag = varint()
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            varint()
            continue
        if wire == 1:
            length = 8
        elif wire == 5:
            length = 4
        elif wire == 2:
            length = varint()
        else:
            raise ValueError("Unsupported protobuf wire type")
        if pos + length > end:
            raise ValueError("Truncated protobuf payload")
        yield field, pos, pos + length
        pos += length


def weight_offsets(data: bytes):
    offsets = {}
    for field, start, end in fields(data):
        if field != 7:  # ModelProto.graph
            continue
        for field, start, end in fields(data, start, end):
            if field != 5:  # GraphProto.initializer
                continue
            parts = {field: (start, end) for field, start, end in fields(data, start, end)}
            if 8 in parts and 9 in parts:  # TensorProto.name / raw_data
                first, last = parts[8]
                offsets[data[first:last].decode()] = parts[9]
    return offsets


def make_graph(data: bytes):
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("Expected the published DeepFilterNet3 v0.5.6 FP32 model")
    model = onnx.load_model_from_string(data)
    graph = model.graph
    states = []
    # Removing only the input's two-frame shift/pad lets the native frontend
    # choose when lookahead is available without advancing hidden state twice.
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name == "/Pad_output_0":
                node.input[index] = "feat_erb"
            elif name == "/Pad_1_output_0":
                node.input[index] = "feat_spec"

    extra = []
    for node in graph.node:
        if node.op_type != "GRU":
            continue
        name = f"h{len(states)}"
        shape = [1, 1, 256]
        node.input[5] = name
        extra.append(helper.make_node("Identity", [node.output[1]], [name + "_out"]))
        states.append((name, shape))
    if len(states) != 5:
        raise ValueError("Expected five GRU states")

    caches = {
        "/enc/erb_conv0/erb_conv0.0/Pad": ("erb_cache", [1, 1, 2, 32]),
        "/enc/df_conv0/df_conv0.0/Pad": ("spec_cache", [1, 2, 2, 96]),
        "/df_dec/df_convp/df_convp.0/Pad": ("df_cache", [1, 64, 4, 96]),
    }
    for node in graph.node:
        if node.name not in caches:
            continue
        name, shape = caches[node.name]
        source, output = node.input[0], node.output[0]
        node.CopyFrom(helper.make_node("Concat", [name, source], [output], axis=2))
        prefix = "stream_" + name
        for suffix, values in [("start", [-shape[2]]), ("end", [2**63 - 1]), ("axis", [2])]:
            graph.initializer.append(helper.make_tensor(prefix + suffix, TensorProto.INT64, [1], values))
        extra.append(helper.make_node(
            "Slice", [output, prefix + "start", prefix + "end", prefix + "axis"], [name + "_out"]
        ))
        states.append((name, shape))
    if len(states) != 8:
        raise ValueError("Expected three convolution caches")
    graph.node.extend(extra)
    for name, shape in states:
        graph.input.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))
        graph.output.append(helper.make_tensor_value_info(name + "_out", TensorProto.FLOAT, shape))

    # Strip unreachable batch lookahead/zero-state nodes. Preserve original
    # topological order and weight names so the transformation is reviewable.
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


def verify(data: bytes, model, states):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    reference = ort.InferenceSession(data, options, providers=["CPUExecutionProvider"])
    streaming = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(147)
    worst = 0.0
    for frames in (5, 20, 137):
        erb = rng.standard_normal((1, 1, frames, 32), dtype=np.float32)
        spec = rng.standard_normal((1, 2, frames, 96), dtype=np.float32)
        expected = reference.run(None, {"feat_erb": erb, "feat_spec": spec})
        shifted = [np.pad(value[:, :, 2:, :], ((0, 0), (0, 0), (0, 2), (0, 0))) for value in (erb, spec)]
        state = {name: np.zeros(shape, np.float32) for name, shape in states}
        masks, coefs = [], []
        for frame in range(frames):
            outputs = streaming.run(None, {"feat_erb": shifted[0][:, :, frame:frame + 1],
                                           "feat_spec": shifted[1][:, :, frame:frame + 1], **state})
            masks.append(outputs[0])
            coefs.append(outputs[1])
            state = {name: value for (name, _), value in zip(states, outputs[2:])}
        for expected_value, actual in zip(expected, (np.concatenate(masks, 2), np.concatenate(coefs, 2))):
            error = float(np.max(np.abs(expected_value - actual)))
            worst = max(worst, error)
            if error > 2e-5:
                raise ValueError(f"Streaming graph parity failed: {error}")
    print(f"Batch vs stateful graph maximum absolute error: {worst:.3g}")


def generate(source: Path, destination: Path):
    data = source.read_bytes()
    model, states = make_graph(data)
    verify(data, model, states)
    offsets = weight_offsets(data)
    for value in model.graph.initializer:
        if value.name not in offsets:
            continue
        first, last = offsets[value.name]
        if data[first:last] != value.raw_data:
            raise ValueError("Initializer byte range mismatch")
        onnx.external_data_helper.set_external_data(value, EXTERNAL_FILE, first, last - first)
        value.ClearField("raw_data")
    graph_bytes = model.SerializeToString()
    fingerprint = 14695981039346656037
    for byte in data:
        fingerprint = ((fingerprint ^ byte) * 1099511628211) & (2**64 - 1)
    lines = [
        "// Generated by scripts/generate_deepfilter_streaming_graph.py; do not edit.",
        f"// Source model SHA-256: {SOURCE_SHA256}",
        "// Contains graph structure only. All neural weights stay in the source file.",
        "// DeepFilterNet: Copyright (c) 2021 Hendrik Schröter (MIT).",
        "// See third_party/deepfilter/LICENSE-MIT and THIRD-PARTY-NOTICES.md.",
        f"static constexpr size_t kSourceModelBytes = {len(data)};",
        f"static constexpr uint64_t kSourceModelFingerprint = UINT64_C(0x{fingerprint:016x});",
        "static constexpr unsigned char kStreamingGraph[] = {",
    ]
    for start in range(0, len(graph_bytes), 16):
        lines.append("    " + ", ".join(f"0x{byte:02x}" for byte in graph_bytes[start:start + 16]) + ",")
    lines.extend(["};", "static constexpr const char* kStateInputs[] = {"])
    lines.extend(f'    "{name}",' for name, _ in states)
    lines.extend(["};", "static constexpr const char* kStateOutputs[] = {"])
    lines.extend(f'    "{name}_out",' for name, _ in states)
    lines.extend(["};", "static constexpr size_t kStateSizes[] = {"])
    lines.extend(f"    {int(np.prod(shape))}," for _, shape in states)
    lines.extend(["};", "static constexpr std::array<int64_t, 4> kStateShapes[] = {"])
    lines.extend("    {" + ", ".join(map(str, shape + [0] * (4 - len(shape)))) + "}," for _, shape in states)
    lines.extend(["};", "static constexpr size_t kStateRanks[] = {3, 3, 3, 3, 3, 4, 4, 4};", ""])
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated {destination}: {len(graph_bytes)} graph bytes, no weights")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent /
                        "src/models/deepfilter/deepfilter_streaming_graph.inc")
    args = parser.parse_args()
    generate(args.source, args.output)
