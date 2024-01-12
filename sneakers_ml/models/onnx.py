from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as rt
from skl2onnx import to_onnx


def save_sklearn_onnx(model: Any, x: np.ndarray, path: str) -> None:
    onx = to_onnx(model, x[:1].astype(np.float32))
    with Path(path).open("wb") as file:
        file.write(onx.SerializeToString())


def load_sklearn_onnx(path: str) -> rt.InferenceSession:
    return rt.InferenceSession(path, providers=["CPUExecutionProvider"])


def load_catboost_onnx(path: str) -> rt.InferenceSession:
    return rt.InferenceSession(path)


def predict_sklearn_onnx(sess: rt.InferenceSession, x: np.ndarray) -> np.ndarray:
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run([label_name], {input_name: x.astype(np.float32)})
    return pred_onx[0]


def predict_catboost_onnx(sess: rt.InferenceSession, x: np.ndarray) -> np.ndarray:
    return sess.run(["label"], {"features": x.astype(np.float32)})
