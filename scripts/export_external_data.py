#!/usr/bin/env python3
"""Repack an ONNX model into an aligned external-data bundle.

Inline tensor payloads above ``--threshold`` are moved to one sidecar. Tensors
that already use external data are copied a chunk at a time, so repacking a
multi-gigabyte bundle does not materialize all weights in memory.

The output model and ``<output>.data`` must not already exist. Both are written
to temporary files in the destination directory; the model appears only after
the complete sidecar has been installed.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import BinaryIO, Dict, Iterable, Iterator, Optional, Tuple

import numpy as np
import onnx
from google.protobuf.message import DecodeError
from onnx import AttributeProto, GraphProto, ModelProto, TensorProto


DEFAULT_ALIGNMENT = 64 * 1024
DEFAULT_THRESHOLD = DEFAULT_ALIGNMENT
COPY_CHUNK_SIZE = 8 * 1024 * 1024


class ExportError(RuntimeError):
    """Raised when an input bundle cannot be repacked safely."""


@dataclass(frozen=True)
class ExternalDataRef:
    path: Path
    offset: int
    length: int


@dataclass(frozen=True)
class ExportStats:
    inline_externalized: int
    external_repacked: int
    empty_skipped: int
    bytes_written: int
    padding_bytes: int

    @property
    def tensors_written(self) -> int:
        return self.inline_externalized + self.external_repacked


def _iter_sparse_tensors(sparse_tensor) -> Iterator[TensorProto]:
    yield sparse_tensor.values
    yield sparse_tensor.indices


def _iter_attribute_tensors(
    attributes: Iterable[AttributeProto],
) -> Iterator[TensorProto]:
    for attribute in attributes:
        if attribute.HasField("t"):
            yield attribute.t
        yield from attribute.tensors

        if attribute.HasField("sparse_tensor"):
            yield from _iter_sparse_tensors(attribute.sparse_tensor)
        for sparse_tensor in attribute.sparse_tensors:
            yield from _iter_sparse_tensors(sparse_tensor)

        if attribute.HasField("g"):
            yield from _iter_graph_tensors(attribute.g)
        for graph in attribute.graphs:
            yield from _iter_graph_tensors(graph)


def _iter_node_tensors(nodes) -> Iterator[TensorProto]:
    for node in nodes:
        yield from _iter_attribute_tensors(node.attribute)


def _iter_graph_tensors(graph: GraphProto) -> Iterator[TensorProto]:
    yield from graph.initializer
    for sparse_tensor in graph.sparse_initializer:
        yield from _iter_sparse_tensors(sparse_tensor)
    yield from _iter_node_tensors(graph.node)


def iter_model_tensors(model: ModelProto) -> Iterator[TensorProto]:
    """Yield every TensorProto using only public protobuf fields."""
    yield from _iter_graph_tensors(model.graph)

    for function in model.functions:
        yield from _iter_node_tensors(function.node)
        yield from _iter_attribute_tensors(function.attribute_proto)

    for training_info in model.training_info:
        yield from _iter_graph_tensors(training_info.initialization)
        yield from _iter_graph_tensors(training_info.algorithm)


def _validate_alignment(align: int) -> None:
    if align < DEFAULT_ALIGNMENT:
        raise ValueError(
            f"alignment must be at least {DEFAULT_ALIGNMENT} bytes for portable mmap"
        )
    if align & (align - 1):
        raise ValueError("alignment must be a power of two")


def _validate_threshold(size_threshold: int) -> None:
    if size_threshold < 0:
        raise ValueError("threshold must be non-negative")


def _is_external(tensor: TensorProto) -> bool:
    return tensor.data_location == TensorProto.EXTERNAL or bool(tensor.external_data)


def _external_metadata(tensor: TensorProto) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    for entry in tensor.external_data:
        if entry.key in metadata:
            raise ExportError(
                f"{tensor.name or '<unnamed>'}: duplicate external_data key {entry.key!r}"
            )
        metadata[entry.key] = entry.value
    return metadata


def _parse_non_negative_int(value: str, field: str, tensor_name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ExportError(
            f"{tensor_name}: external_data {field} is not an integer: {value!r}"
        ) from exc
    if parsed < 0:
        raise ExportError(f"{tensor_name}: external_data {field} must be non-negative")
    return parsed


def _tensor_storage_size(tensor: TensorProto) -> Optional[int]:
    """Return the raw byte size implied by a tensor's type and shape."""
    element_count = 1
    for dimension in tensor.dims:
        if dimension < 0:
            raise ExportError(f"{tensor.name or '<unnamed>'}: negative tensor dimension")
        element_count *= dimension

    if element_count == 0:
        return 0
    if tensor.data_type in (TensorProto.UNDEFINED, TensorProto.STRING):
        return None

    packed_bits = {}
    for type_name, bits in (
        ("INT2", 2),
        ("UINT2", 2),
        ("INT4", 4),
        ("UINT4", 4),
        ("FLOAT4E2M1", 4),
    ):
        type_value = getattr(TensorProto, type_name, None)
        if type_value is not None:
            packed_bits[type_value] = bits

    if tensor.data_type in packed_bits:
        return math.ceil(element_count * packed_bits[tensor.data_type] / 8)

    try:
        dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type))
    except (KeyError, TypeError, ValueError) as exc:
        raise ExportError(
            f"{tensor.name or '<unnamed>'}: unsupported tensor data type "
            f"{tensor.data_type}"
        ) from exc
    return element_count * dtype.itemsize


def _resolve_external_path(model_dir: Path, location: str, tensor_name: str) -> Path:
    if not location:
        raise ExportError(f"{tensor_name}: external_data location is missing")

    location_path = Path(location)
    if location_path.is_absolute() or PureWindowsPath(location).is_absolute():
        raise ExportError(f"{tensor_name}: external_data location must be relative")

    # Check lexical containment without resolving the final path: Hugging Face
    # cache entries are intentionally symlinks to blobs outside the snapshot.
    model_dir = model_dir.absolute()
    resolved = Path(os.path.abspath(model_dir / location_path))
    try:
        inside_model_dir = os.path.commonpath((str(model_dir), str(resolved))) == str(
            model_dir
        )
    except ValueError:
        inside_model_dir = False
    if not inside_model_dir:
        raise ExportError(
            f"{tensor_name}: external_data location escapes the model directory"
        )
    return resolved


def _external_ref(tensor: TensorProto, model_dir: Path) -> Optional[ExternalDataRef]:
    tensor_name = tensor.name or "<unnamed>"
    expected_length = _tensor_storage_size(tensor)
    if expected_length == 0:
        return None

    metadata = _external_metadata(tensor)
    source_path = _resolve_external_path(
        model_dir, metadata.get("location", ""), tensor_name
    )
    if not source_path.is_file():
        raise ExportError(f"{tensor_name}: external data file not found: {source_path}")

    offset = _parse_non_negative_int(metadata.get("offset", "0"), "offset", tensor_name)
    declared_length = _parse_non_negative_int(
        metadata.get("length", "0"), "length", tensor_name
    )
    if declared_length:
        length = declared_length
    elif expected_length is not None:
        length = expected_length
    else:
        length = source_path.stat().st_size - offset

    if expected_length is not None and length != expected_length:
        raise ExportError(
            f"{tensor_name}: external_data length {length} does not match "
            f"shape/type size {expected_length}"
        )
    if length <= 0:
        raise ExportError(f"{tensor_name}: external_data length must be positive")
    if offset + length > source_path.stat().st_size:
        raise ExportError(
            f"{tensor_name}: external_data range exceeds {source_path.name}"
        )
    return ExternalDataRef(source_path, offset, length)


def _clear_external_data(tensor: TensorProto) -> None:
    tensor.ClearField("external_data")
    tensor.data_location = TensorProto.DEFAULT


def _set_external_data(
    tensor: TensorProto, location: str, offset: int, length: int
) -> None:
    tensor.ClearField("external_data")
    tensor.data_location = TensorProto.EXTERNAL
    for key, value in (
        ("location", location),
        ("offset", str(offset)),
        ("length", str(length)),
    ):
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = value


def _write_padding(output: BinaryIO, align: int) -> int:
    padding = (-output.tell()) % align
    if padding:
        output.write(b"\0" * padding)
    return padding


def _copy_range(source_ref: ExternalDataRef, output: BinaryIO) -> None:
    remaining = source_ref.length
    with source_ref.path.open("rb") as source:
        source.seek(source_ref.offset)
        while remaining:
            chunk = source.read(min(COPY_CHUNK_SIZE, remaining))
            if not chunk:
                raise ExportError(
                    f"unexpected EOF while reading external data from {source_ref.path}"
                )
            output.write(chunk)
            remaining -= len(chunk)


def _reserve_temp_file(parent: Path, prefix: str, suffix: str) -> Tuple[int, Path]:
    descriptor, path = tempfile.mkstemp(dir=str(parent), prefix=prefix, suffix=suffix)
    return descriptor, Path(path)


def _install_no_replace(temporary: Path, destination: Path) -> None:
    """Atomically publish a completed file without a check/rename race."""
    os.link(temporary, destination)
    try:
        temporary.unlink()
    except OSError:
        # The destination is already a complete hard link. A leftover hidden
        # temporary file is safer than reporting failure after publication.
        pass


def _validate_output_bundle(model_path: Path, align: int, expected_count: int) -> None:
    """Validate the serialized external-data contract without judging graph ops."""
    saved_model = onnx.load(str(model_path), load_external_data=False)
    external_count = 0
    for tensor in iter_model_tensors(saved_model):
        if not _is_external(tensor):
            continue
        tensor_name = tensor.name or "<unnamed>"
        source_ref = _external_ref(tensor, model_path.parent)
        if source_ref is None:
            raise ExportError(f"{tensor_name}: empty tensor was externalized")
        if source_ref.offset % align:
            raise ExportError(
                f"{tensor_name}: external_data offset {source_ref.offset} "
                f"is not aligned to {align} bytes"
            )
        if tensor.HasField("raw_data") and tensor.raw_data:
            raise ExportError(
                f"{tensor_name}: tensor contains both raw and external data"
            )
        external_count += 1

    if external_count != expected_count:
        raise ExportError(
            f"serialized model contains {external_count} external tensors; "
            f"expected {expected_count}"
        )


def export_aligned(
    input_path,
    output_path,
    align: int = DEFAULT_ALIGNMENT,
    size_threshold: int = DEFAULT_THRESHOLD,
) -> ExportStats:
    """Write a new aligned ONNX bundle without overwriting either input or output."""
    _validate_alignment(align)
    _validate_threshold(size_threshold)

    # Keep the logical input directory intact. Hugging Face snapshots commonly
    # store the model and sidecar as separate symlinks into a blob directory,
    # while external_data locations remain relative to the snapshot.
    source_model = Path(input_path).expanduser().absolute()
    destination_model = Path(output_path).expanduser().absolute()
    if not source_model.is_file():
        raise FileNotFoundError(f"input model not found: {source_model}")
    if source_model.resolve() == destination_model.resolve():
        raise ExportError("input and output paths must be different")

    output_mode = source_model.stat().st_mode & 0o777
    destination_model.parent.mkdir(parents=True, exist_ok=True)
    data_name = destination_model.name + ".data"
    destination_data = destination_model.parent / data_name
    for destination in (destination_model, destination_data):
        if os.path.lexists(destination):
            raise FileExistsError(f"refusing to overwrite existing output: {destination}")

    model = onnx.load(str(source_model), load_external_data=False)
    data_descriptor = None
    model_descriptor = None
    temporary_data = None
    temporary_model = None
    inline_externalized = 0
    external_repacked = 0
    empty_skipped = 0
    bytes_written = 0
    padding_bytes = 0
    installed_data = False
    deduplicated_refs: Dict[ExternalDataRef, int] = {}

    try:
        data_descriptor, temporary_data = _reserve_temp_file(
            destination_model.parent, f".{data_name}.", ".tmp"
        )
        model_descriptor, temporary_model = _reserve_temp_file(
            destination_model.parent, f".{destination_model.name}.", ".tmp"
        )
        os.close(model_descriptor)
        model_descriptor = None

        output = os.fdopen(data_descriptor, "wb")
        data_descriptor = None
        with output:
            for tensor in iter_model_tensors(model):
                if _is_external(tensor):
                    source_ref = _external_ref(tensor, source_model.parent)
                    if source_ref is None:
                        _clear_external_data(tensor)
                        tensor.ClearField("raw_data")
                        empty_skipped += 1
                        continue

                    offset = deduplicated_refs.get(source_ref)
                    if offset is None:
                        padding_bytes += _write_padding(output, align)
                        offset = output.tell()
                        _copy_range(source_ref, output)
                        deduplicated_refs[source_ref] = offset
                        bytes_written += source_ref.length
                    _set_external_data(tensor, data_name, offset, source_ref.length)
                    tensor.ClearField("raw_data")
                    external_repacked += 1
                    continue

                if not tensor.HasField("raw_data"):
                    continue
                raw_length = len(tensor.raw_data)
                if raw_length == 0:
                    empty_skipped += 1
                    continue
                if raw_length < size_threshold:
                    continue

                padding_bytes += _write_padding(output, align)
                offset = output.tell()
                output.write(tensor.raw_data)
                _set_external_data(tensor, data_name, offset, raw_length)
                tensor.ClearField("raw_data")
                inline_externalized += 1
                bytes_written += raw_length

            output.flush()
            os.fsync(output.fileno())

        os.chmod(temporary_data, output_mode)
        onnx.save_model(model, str(temporary_model), format="protobuf")
        os.chmod(temporary_model, output_mode)
        with temporary_model.open("r+b") as saved_model:
            os.fsync(saved_model.fileno())

        tensors_written = inline_externalized + external_repacked
        if tensors_written:
            _install_no_replace(temporary_data, destination_data)
            installed_data = True
        else:
            temporary_data.unlink()
        # The bundle validator resolves external data relative to the model.
        # At this point the complete sidecar is installed, but the final model
        # name is still absent; a validation failure therefore remains atomic
        # from the consumer's perspective and the sidecar is removed below.
        _validate_output_bundle(temporary_model, align, tensors_written)
        _install_no_replace(temporary_model, destination_model)
    except BaseException:
        for descriptor in (data_descriptor, model_descriptor):
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        if temporary_data is not None:
            temporary_data.unlink(missing_ok=True)
        if temporary_model is not None:
            temporary_model.unlink(missing_ok=True)
        if installed_data:
            destination_data.unlink(missing_ok=True)
        raise

    return ExportStats(
        inline_externalized=inline_externalized,
        external_repacked=external_repacked,
        empty_skipped=empty_skipped,
        bytes_written=bytes_written,
        padding_bytes=padding_bytes,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Repack an ONNX model with 64 KiB-aligned external tensor data. "
            "Existing sidecars are streamed instead of loaded into memory."
        )
    )
    parser.add_argument("input", help="Input .onnx model")
    parser.add_argument("output", help="New output .onnx model")
    parser.add_argument(
        "--align",
        type=int,
        default=DEFAULT_ALIGNMENT,
        help=f"Alignment in bytes (default: {DEFAULT_ALIGNMENT}; minimum: {DEFAULT_ALIGNMENT})",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        help=(
            "Minimum inline raw tensor size to externalize "
            f"(default: {DEFAULT_THRESHOLD})"
        ),
    )
    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        stats = export_aligned(args.input, args.output, args.align, args.threshold)
    except (
        DecodeError,
        ExportError,
        OSError,
        ValueError,
    ) as exc:
        parser.exit(1, f"error: {exc}\n")

    print(
        f"wrote {args.output}: {stats.tensors_written} external tensors, "
        f"{stats.bytes_written} payload bytes, {stats.padding_bytes} padding bytes"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
