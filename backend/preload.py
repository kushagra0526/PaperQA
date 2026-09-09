"""
Pre-exports and quantizes models to ONNX + dynamic INT8 during the build step.

Models exported here:
  onnx_model/         — deepset/minilm-uncased-squad2 (extractive QA)
  onnx_reranker/      — cross-encoder/ms-marco-MiniLM-L-6-v2 (reranker)

This runs once during deployment (called by build.sh), not at request time.
"""
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from transformers import AutoTokenizer
from optimum.onnxruntime import (
    ORTModelForQuestionAnswering,
    ORTModelForSequenceClassification,
    ORTQuantizer,
)
from optimum.onnxruntime.configuration import AutoQuantizationConfig


MODEL_NAME    = "deepset/minilm-uncased-squad2"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ONNX_MODEL_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_model")
ONNX_RERANKER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_reranker")


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # NLTK punkt data — download once here so the runtime container never
    # needs network access to NLTK's servers.  Data is saved to backend/nltk_data/
    # which is baked into the Docker image by the COPY . . layer that runs
    # before this script.  extraction.py sets NLTK_DATA at import time to
    # ensure nltk finds it at runtime.
    # -------------------------------------------------------------------------
    import nltk

    _NLTK_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data")
    os.makedirs(_NLTK_DATA_DIR, exist_ok=True)
    print(f"Downloading NLTK punkt data to {_NLTK_DATA_DIR} ...")
    for _resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{_resource}", paths=[_NLTK_DATA_DIR])
            print(f"  {_resource}: already present")
        except LookupError:
            nltk.download(_resource, download_dir=_NLTK_DATA_DIR, quiet=False)
            print(f"  {_resource}: downloaded")

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

    # -------------------------------------------------------------------------
    # Cross-encoder reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
    # -------------------------------------------------------------------------
    print()
    print(f"Exporting {RERANKER_NAME} to ONNX and quantizing to dynamic INT8...")

    with TemporaryDirectory() as tmp_dir:
        print("  [1/3] Exporting reranker to ONNX...")
        reranker = ORTModelForSequenceClassification.from_pretrained(
            RERANKER_NAME, export=True
        )
        reranker.save_pretrained(tmp_dir)

        print("  [2/3] Applying dynamic INT8 quantization to reranker...")
        quantizer = ORTQuantizer.from_pretrained(tmp_dir)
        qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
        quantizer.quantize(save_dir=ONNX_RERANKER_DIR, quantization_config=qconfig)

    # Confirm the output filename.
    onnx_files = sorted(Path(ONNX_RERANKER_DIR).glob("*.onnx"))
    if onnx_files:
        print("  Quantized ONNX file(s) in reranker output dir:")
        for f in onnx_files:
            print(f"    {f.name}  ({f.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print("  WARNING: no .onnx file found in reranker output directory!")

    # Step 3: Save the reranker tokenizer (for runtime use via AutoTokenizer or
    # tokenizers.Tokenizer.from_file — main.py uses AutoTokenizer here because
    # cross-encoder tokenizers may not expose a plain tokenizer.json).
    print("  [3/3] Saving reranker tokenizer...")
    reranker_tok = AutoTokenizer.from_pretrained(RERANKER_NAME)
    reranker_tok.save_pretrained(ONNX_RERANKER_DIR)
    tok_json = Path(ONNX_RERANKER_DIR) / "tokenizer.json"
    if tok_json.exists():
        print(f"  tokenizer.json saved: {tok_json}  ({tok_json.stat().st_size / 1024:.1f} KB)")
    else:
        print("  WARNING: tokenizer.json not found after save_pretrained!")

    print(f"Quantized reranker ONNX model saved to {ONNX_RERANKER_DIR}/")
    total_size = sum(f.stat().st_size for f in Path(ONNX_RERANKER_DIR).rglob("*") if f.is_file())
    print(f"Total reranker size: {total_size / 1024 / 1024:.1f} MB")
