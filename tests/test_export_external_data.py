import hashlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_external_data.py"
SPEC = importlib.util.spec_from_file_location("export_external_data", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
repacker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = repacker
SPEC.loader.exec_module(repacker)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(tensor):
    return {entry.key: entry.value for entry in tensor.external_data}


def _model_with_initializers(initializers):
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "repacker-test",
        [input_info],
        [output_info],
        initializer=list(initializers),
    )
    return helper.make_model(
        graph,
        producer_name="speech-core-repacker-test",
        opset_imports=[helper.make_opsetid("", 18)],
    )


def _save_inline_model(path, arrays):
    tensors = [numpy_helper.from_array(array, name=name) for name, array in arrays.items()]
    onnx.save_model(_model_with_initializers(tensors), str(path))


def _save_external_model(path, arrays):
    tensors = [numpy_helper.from_array(array, name=name) for name, array in arrays.items()]
    model = _model_with_initializers(tensors)
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=path.name + ".source-data",
        size_threshold=0,
    )
    return path.with_name(path.name + ".source-data")


def _initializer_arrays(path):
    model = onnx.load(str(path), load_external_data=True)
    return {
        tensor.name: numpy_helper.to_array(tensor)
        for tensor in model.graph.initializer
    }


class ExportExternalDataTests(unittest.TestCase):
    def test_default_threshold_does_not_pad_small_inline_tensors(self):
        arrays = {
            "large": np.arange(64 * 1024, dtype=np.uint8),
            "small": np.arange((64 * 1024) - 1, dtype=np.uint8),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.onnx"
            output = root / "source-aligned-v2.onnx"
            _save_inline_model(source, arrays)

            stats = repacker.export_aligned(source, output)

            self.assertEqual(stats.inline_externalized, 1)
            self.assertEqual(stats.padding_bytes, 0)
            skeleton = onnx.load(str(output), load_external_data=False)
            tensors = {tensor.name: tensor for tensor in skeleton.graph.initializer}
            self.assertEqual(tensors["large"].data_location, TensorProto.EXTERNAL)
            self.assertEqual(tensors["small"].data_location, TensorProto.DEFAULT)
            self.assertGreater(len(tensors["small"].raw_data), 0)
            actual = _initializer_arrays(output)
            for name, expected in arrays.items():
                np.testing.assert_array_equal(actual[name], expected)

    def test_inline_tensors_are_64k_aligned_and_empty_tensor_is_skipped(self):
        arrays = {
            "large_float": np.arange(300, dtype=np.float32),
            "large_int": np.arange(257, dtype=np.int64),
            "small": np.array([3.25], dtype=np.float32),
            "empty": np.empty((0,), dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.onnx"
            output = root / "source-aligned-v2.onnx"
            _save_inline_model(source, arrays)
            source_hash = _sha256(source)

            stats = repacker.export_aligned(source, output, size_threshold=0)

            self.assertEqual(source_hash, _sha256(source))
            self.assertEqual(stats.inline_externalized, 3)
            self.assertEqual(stats.external_repacked, 0)
            self.assertEqual(stats.empty_skipped, 1)
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_name(output.name + ".data").is_file())

            skeleton = onnx.load(str(output), load_external_data=False)
            tensors = {tensor.name: tensor for tensor in skeleton.graph.initializer}
            for name in ("large_float", "large_int", "small"):
                tensor = tensors[name]
                self.assertEqual(tensor.data_location, TensorProto.EXTERNAL)
                metadata = _metadata(tensor)
                self.assertEqual(metadata["location"], output.name + ".data")
                self.assertEqual(int(metadata["offset"]) % (64 * 1024), 0)
                self.assertGreater(int(metadata["length"]), 0)

            empty = tensors["empty"]
            self.assertEqual(empty.data_location, TensorProto.DEFAULT)
            self.assertEqual(list(empty.external_data), [])
            self.assertEqual(len(empty.raw_data), 0)

            onnx.checker.check_model(str(output))
            actual = _initializer_arrays(output)
            for name, expected in arrays.items():
                np.testing.assert_array_equal(actual[name], expected)

    def test_existing_sidecar_is_repacked_without_loading_it_whole(self):
        arrays = {
            "first": np.arange(41, dtype=np.float32),
            "second": np.arange(37, dtype=np.int64),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.onnx"
            output = root / "source-aligned-v2.onnx"
            source_data = _save_external_model(source, arrays)
            source_hashes = (_sha256(source), _sha256(source_data))

            with mock.patch.object(repacker, "COPY_CHUNK_SIZE", 7):
                stats = repacker.export_aligned(source, output)

            self.assertEqual(source_hashes, (_sha256(source), _sha256(source_data)))
            self.assertEqual(stats.inline_externalized, 0)
            self.assertEqual(stats.external_repacked, 2)
            skeleton = onnx.load(str(output), load_external_data=False)
            offsets = [
                int(_metadata(tensor)["offset"])
                for tensor in skeleton.graph.initializer
            ]
            self.assertEqual(offsets[0], 0)
            self.assertEqual(offsets[1] % (64 * 1024), 0)
            actual = _initializer_arrays(output)
            for name, expected in arrays.items():
                np.testing.assert_array_equal(actual[name], expected)

    @unittest.skipIf(os.name == "nt", "symlink creation is not portable on Windows")
    def test_hugging_face_style_symlink_bundle_uses_logical_model_directory(self):
        arrays = {"weights": np.arange(41, dtype=np.float32)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blobs = root / "blobs"
            snapshot = root / "snapshots" / "revision"
            output_dir = root / "output"
            blobs.mkdir()
            snapshot.mkdir(parents=True)
            output_dir.mkdir()

            source_blob = blobs / "model-blob"
            data_blob = _save_external_model(source_blob, arrays)
            model_link = snapshot / "model.onnx"
            data_link = snapshot / (source_blob.name + ".source-data")
            model_link.symlink_to(source_blob)
            data_link.symlink_to(data_blob)
            data_blob.rename(blobs / "weight-blob")
            data_link.unlink()
            data_link.symlink_to(blobs / "weight-blob")

            output = output_dir / "model-external-v2.onnx"
            repacker.export_aligned(model_link, output)

            actual = _initializer_arrays(output)
            np.testing.assert_array_equal(actual["weights"], arrays["weights"])

    def test_copy_range_limits_each_read_to_the_chunk_size(self):
        read_sizes = []

        class TrackingReader(io.BytesIO):
            def read(self, size=-1):
                read_sizes.append(size)
                return super().read(size)

        class InMemoryPath:
            def open(self, mode):
                self.assert_mode = mode
                return TrackingReader(b"0123456789abcdef")

            def __str__(self):
                return "in-memory-sidecar"

        source = repacker.ExternalDataRef(InMemoryPath(), 1, 14)
        output = io.BytesIO()
        with mock.patch.object(repacker, "COPY_CHUNK_SIZE", 5):
            repacker._copy_range(source, output)

        self.assertEqual(output.getvalue(), b"123456789abcde")
        self.assertGreater(len(read_sizes), 1)
        self.assertTrue(all(0 < size <= 5 for size in read_sizes))

    def test_tensor_inside_node_attribute_is_externalized(self):
        expected = np.arange(512, dtype=np.float32)
        constant = numpy_helper.from_array(expected, name="constant-value")
        node = helper.make_node("Constant", [], ["output"], value=constant)
        graph = helper.make_graph(
            [node],
            "attribute-test",
            [],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [512])],
        )
        model = helper.make_model(
            graph,
            producer_name="speech-core-repacker-test",
            opset_imports=[helper.make_opsetid("", 18)],
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "constant.onnx"
            output = root / "constant-aligned-v2.onnx"
            onnx.save_model(model, str(source))

            repacker.export_aligned(source, output, size_threshold=1)

            skeleton = onnx.load(str(output), load_external_data=False)
            tensor = skeleton.graph.node[0].attribute[0].t
            self.assertEqual(tensor.data_location, TensorProto.EXTERNAL)
            self.assertEqual(int(_metadata(tensor)["offset"]), 0)
            loaded = onnx.load(str(output), load_external_data=True)
            actual = numpy_helper.to_array(loaded.graph.node[0].attribute[0].t)
            np.testing.assert_array_equal(actual, expected)

    def test_empty_external_tensor_does_not_require_a_sidecar(self):
        empty = helper.make_tensor("empty", TensorProto.FLOAT, [0], [])
        empty.data_location = TensorProto.EXTERNAL
        entry = empty.external_data.add()
        entry.key = "location"
        entry.value = "missing.data"
        model = _model_with_initializers([empty])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "empty.onnx"
            output = root / "empty-aligned-v2.onnx"
            onnx.save_model(model, str(source))

            stats = repacker.export_aligned(source, output, size_threshold=0)

            self.assertEqual(stats.tensors_written, 0)
            self.assertEqual(stats.empty_skipped, 1)
            self.assertFalse(output.with_name(output.name + ".data").exists())
            skeleton = onnx.load(str(output), load_external_data=False)
            result = skeleton.graph.initializer[0]
            self.assertEqual(result.data_location, TensorProto.DEFAULT)
            self.assertEqual(list(result.external_data), [])
            onnx.checker.check_model(str(output))

    def test_failure_removes_partial_bundle_and_temporary_files(self):
        arrays = {"weights": np.arange(64, dtype=np.float32)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.onnx"
            output = root / "source-aligned-v2.onnx"
            _save_external_model(source, arrays)

            def fail_after_partial_write(source_ref, destination):
                destination.write(b"partial")
                raise repacker.ExportError("injected copy failure")

            with mock.patch.object(repacker, "_copy_range", fail_after_partial_write):
                with self.assertRaisesRegex(repacker.ExportError, "injected"):
                    repacker.export_aligned(source, output)

            self.assertFalse(output.exists())
            self.assertFalse(output.with_name(output.name + ".data").exists())
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_bundle_validation_failure_removes_installed_sidecar(self):
        arrays = {"weights": np.arange(64, dtype=np.float32)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.onnx"
            output = root / "source-aligned-v2.onnx"
            _save_inline_model(source, arrays)

            error = repacker.ExportError("injected validation failure")
            with mock.patch.object(
                repacker, "_validate_output_bundle", side_effect=error
            ):
                with self.assertRaisesRegex(repacker.ExportError, "injected"):
                    repacker.export_aligned(source, output, size_threshold=0)

            self.assertFalse(output.exists())
            self.assertFalse(output.with_name(output.name + ".data").exists())
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_existing_output_is_never_overwritten(self):
        arrays = {"weights": np.arange(64, dtype=np.float32)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.onnx"
            output = root / "source-aligned-v2.onnx"
            _save_inline_model(source, arrays)
            output.write_bytes(b"keep-me")

            with self.assertRaises(FileExistsError):
                repacker.export_aligned(source, output)

            self.assertEqual(output.read_bytes(), b"keep-me")
            self.assertFalse(output.with_name(output.name + ".data").exists())

    def test_output_created_during_export_is_not_overwritten(self):
        arrays = {"weights": np.arange(64, dtype=np.float32)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.onnx"
            output = root / "source-aligned-v2.onnx"
            _save_inline_model(source, arrays)
            validate = repacker._validate_output_bundle

            def validate_then_race(*args):
                validate(*args)
                output.write_bytes(b"racing-writer")

            with mock.patch.object(
                repacker, "_validate_output_bundle", side_effect=validate_then_race
            ):
                with self.assertRaises(FileExistsError):
                    repacker.export_aligned(source, output, size_threshold=0)

            self.assertEqual(output.read_bytes(), b"racing-writer")
            self.assertFalse(output.with_name(output.name + ".data").exists())
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_invalid_options_fail_before_creating_output(self):
        arrays = {"weights": np.arange(64, dtype=np.float32)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.onnx"
            _save_inline_model(source, arrays)

            cases = ((4096, 0), (65537, 0), (65536, -1))
            for index, (alignment, threshold) in enumerate(cases):
                output = root / f"invalid-{index}.onnx"
                with self.assertRaises(ValueError):
                    repacker.export_aligned(
                        source,
                        output,
                        align=alignment,
                        size_threshold=threshold,
                    )
                self.assertFalse(output.exists())
                self.assertFalse(output.with_name(output.name + ".data").exists())

    def test_external_location_cannot_escape_model_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "bundle"
            model_dir.mkdir()
            (root / "outside.data").write_bytes(np.array([1.0], np.float32).tobytes())
            tensor = TensorProto()
            tensor.name = "weights"
            tensor.data_type = TensorProto.FLOAT
            tensor.dims.extend([1])
            tensor.data_location = TensorProto.EXTERNAL
            for key, value in (
                ("location", "../outside.data"),
                ("offset", "0"),
                ("length", "4"),
            ):
                entry = tensor.external_data.add()
                entry.key = key
                entry.value = value
            source = model_dir / "source.onnx"
            output = model_dir / "source-aligned-v2.onnx"
            onnx.save_model(_model_with_initializers([tensor]), str(source))

            with self.assertRaisesRegex(repacker.ExportError, "escapes"):
                repacker.export_aligned(source, output)

            self.assertFalse(output.exists())
            self.assertFalse(output.with_name(output.name + ".data").exists())


if __name__ == "__main__":
    unittest.main()
