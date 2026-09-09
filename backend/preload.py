"""
Pre-exports and quantizes the QA model to ONNX + dynamic INT8 during the build step.

This runs once during deployment (called by build.sh), not at request time.
The quantized model is saved to onnx_model/ and loaded by main.py at runtime.
"""
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForQuestionAnswering, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig


MODEL_NAME = "deepset/minilm-uncased-squad2"
ONNX_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_model")


if __name__ == "__main__":
    print(f"Exporting {MODEL_NAME} to ONNX and quantizing to dynamic INT8...")

    # Step 1: Export the PyTorch model to ONNX format in a temporary directory.
    with TemporaryDirectory() as tmp_dir:
        print("  [1/3] Exporting to ONNX...")
        model = ORTModelForQuestionAnswering.from_pretrained(MODEL_NAME, export=True)
        model.save_pretrained(tmp_dir)

        # Step 2: Apply dynamic INT8 quantization (no calibration data needed).
        print("  [2/3] Applying dynamic INT8 quantization...")
        quantizer = ORTQuantizer.from_pretrained(tmp_dir)
        qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
        quantizer.quantize(save_dir=ONNX_MODEL_DIR, quantization_config=qconfig)

    # Report the exact filename produced so the runtime loader can be pointed
    # at the right file (optimum names it model_quantized.onnx by convention,
    # but this confirms it explicitly after each build).
    onnx_files = sorted(Path(ONNX_MODEL_DIR).glob("*.onnx"))
    if onnx_files:
        print(f"  Quantized ONNX file(s) in output dir:")
        for f in onnx_files:
            print(f"    {f.name}  ({f.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print("  WARNING: no .onnx file found in output directory!")

    # Step 3: Save the tokenizer alongside the quantized model so the entire
    # onnx_model/ directory is self-contained for loading at runtime.
    # AutoTokenizer.save_pretrained() writes tokenizer.json (among other files);
    # main.py loads only tokenizer.json via tokenizers.Tokenizer.from_file().
    print("  [3/3] Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(ONNX_MODEL_DIR)
    tok_json = Path(ONNX_MODEL_DIR) / "tokenizer.json"
    if tok_json.exists():
        print(f"  tokenizer.json saved: {tok_json}  ({tok_json.stat().st_size / 1024:.1f} KB)")
    else:
        print("  WARNING: tokenizer.json not found after save_pretrained!")

    print(f"Quantized ONNX model saved to {ONNX_MODEL_DIR}/")

    # Report final on-disk size.
    total_size = sum(f.stat().st_size for f in Path(ONNX_MODEL_DIR).rglob("*") if f.is_file())
    print(f"Total model size: {total_size / 1024 / 1024:.1f} MB")
