from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as rt
from skl2onnx import to_onnx


def save_sklearn_onnx(model: Any, x: np.ndarray, path: str) -> None:
    onx = to_onnx(model, x[:1])
    with Path(path).open("wb") as file:
        file.write(onx.SerializeToString())


def predict_sklearn_onnx(path: str, x: np.ndarray) -> np.ndarray:
    sess = rt.InferenceSession(path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run([label_name], {input_name: x.astype(np.float32)})[0]
    return pred_onx
