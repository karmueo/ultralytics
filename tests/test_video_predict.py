# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import sys
import types

import video_predict


def test_resolve_batch_size_auto_detects_onnx_static_batch(monkeypatch):
    """验证 ONNX 静态 batch 可以被自动推断。"""
    fake_session = types.SimpleNamespace(
        get_inputs=lambda: [types.SimpleNamespace(shape=[4, 3, 352, 640])]
    )  # 模拟 ONNX 会话
    fake_ort = types.SimpleNamespace(InferenceSession=lambda *args, **kwargs: fake_session)  # 模拟 ort 模块
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    batch = video_predict.resolve_batch_size("model.onnx", None)  # 自动推断 batch

    assert batch == 4


def test_resolve_batch_size_honors_user_override(monkeypatch):
    """验证用户显式指定 batch 时不走自动推断。"""
    monkeypatch.setitem(sys.modules, "onnxruntime", None)

    batch = video_predict.resolve_batch_size("model.onnx", 2)  # 用户指定 batch

    assert batch == 2
