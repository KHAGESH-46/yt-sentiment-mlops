"""PHASE 2d — Export the winning model to ONNX (for the serving image).

Baseline (tfidf_logreg):  pip install skl2onnx, then:

    from skl2onnx import to_onnx
    # convert fitted pipeline; initial_types=[("text", StringTensorType([None, 1]))]

DistilBERT (later): use optimum:

    pip install optimum[onnxruntime]
    optimum-cli export onnx --model <hf-model> --task text-classification out_dir/

TODO: implement conversion + verify parity (same predictions as joblib/HF
within 1e-4) before swapping the API over. Until then the API serves the
joblib baseline directly.
"""
import sys


def main() -> None:
    sys.exit(
        "export_onnx: not implemented yet (see docstring). "
        "The API currently serves models/model.joblib for the baseline."
    )


if __name__ == "__main__":
    main()
