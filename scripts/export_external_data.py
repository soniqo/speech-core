#!/usr/bin/env python3
import os
import argparse
import onnx
from onnx.external_data_helper import set_external_data, _get_all_tensors

def export_aligned(input_path, output_path, align=4096, size_threshold=1024):
    print(f"Loading {input_path}...")
    model = onnx.load(input_path)
    
    data_file = os.path.basename(output_path) + ".data"
    data_path = os.path.join(os.path.dirname(output_path) or ".", data_file)
    
    print(f"Writing external data to {data_path} (alignment={align})...")
    with open(data_path, "wb") as f:
        for tensor in _get_all_tensors(model):
            if tensor.HasField("raw_data") and len(tensor.raw_data) >= size_threshold:
                pad = (align - (f.tell() % align)) % align
                if pad > 0:
                    f.write(b'\0' * pad)
                
                offset = f.tell()
                length = len(tensor.raw_data)
                f.write(tensor.raw_data)
                
                set_external_data(tensor, location=data_file, offset=offset, length=length)
                tensor.ClearField("raw_data")
                tensor.data_location = onnx.TensorProto.EXTERNAL
                
    print(f"Saving model skeleton to {output_path}...")
    onnx.save_model(model, output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ONNX model with page-aligned external data for zero-copy mmap.")
    parser.add_argument("input", help="Input .onnx model")
    parser.add_argument("output", help="Output .onnx model")
    parser.add_argument("--align", type=int, default=4096, help="Alignment size (default: 4096)")
    parser.add_argument("--threshold", type=int, default=1024, help="Min bytes to externalize (default: 1024)")
    args = parser.parse_args()
    
    export_aligned(args.input, args.output, args.align, args.threshold)
