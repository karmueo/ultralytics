# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import numpy as np
import torch

from ultralytics.nn.backends.onnx import ONNXBackend


class _FakeInput:
    """模拟 ONNX 输入节点对象。"""

    def __init__(self, name: str):
        self.name = name


class _FakeSession:
    """模拟 ONNX Runtime Session。"""

    def __init__(self, input_name: str):
        self._input_name = input_name
        self.run_calls = []
        self.iobinding_calls = 0

    def get_inputs(self):
        """返回模拟输入节点列表。"""
        return [_FakeInput(self._input_name)]

    def run(self, output_names, inputs):
        """记录普通推理调用参数。"""
        self.run_calls.append((output_names, inputs))
        return ["fake-output"]

    def run_with_iobinding(self, io):
        """模拟 IO binding 推理调用。"""
        self.io = io
        self.iobinding_calls += 1


class _FakeIOBinding:
    """模拟 ONNX Runtime IO binding 对象。"""

    def __init__(self):
        self.bind_calls = []

    def bind_input(self, **kwargs):
        """记录绑定输入时的参数。"""
        self.bind_calls.append(kwargs)


def test_onnx_backend_forward_uses_model_input_name_for_iobinding():
    """验证 ONNX IO binding 绑定的是模型真实输入名，而不是硬编码名称。"""
    backend = object.__new__(ONNXBackend)
    backend.format = "onnx"
    backend.use_io_binding = True
    backend.device = torch.device("cpu")
    backend.fp16 = False
    backend.input_name = "input"
    backend.dynamic = False
    backend.static_batch = None
    backend.io = _FakeIOBinding()
    backend.session = _FakeSession("input")
    backend.bindings = [torch.zeros(1)]

    im = torch.zeros(1, 3, 8, 8)  # 输入张量
    output = ONNXBackend.forward(backend, im)

    assert backend.io.bind_calls[0]["name"] == "input"
    assert output == backend.bindings


def test_onnx_backend_forward_pads_tail_batch_for_iobinding():
    """验证静态 batch ONNX 在尾批不足时会补齐并裁剪输出。"""
    backend = object.__new__(ONNXBackend)
    backend.format = "onnx"
    backend.use_io_binding = True
    backend.device = torch.device("cpu")
    backend.fp16 = False
    backend.input_name = "input"
    backend.dynamic = False
    backend.static_batch = 4
    backend.io = _FakeIOBinding()
    backend.session = _FakeSession("input")
    backend.bindings = [torch.arange(8, dtype=torch.float32).reshape(4, 2)]

    im = torch.zeros(3, 3, 8, 8)  # 尾批只有 3 帧
    output = ONNXBackend.forward(backend, im)

    assert backend.io.bind_calls[0]["shape"] == (4, 3, 8, 8)
    assert output[0].shape == (3, 2)
    np.testing.assert_array_equal(output[0].cpu().numpy(), backend.bindings[0][:3].cpu().numpy())


def test_onnx_backend_forward_uses_model_input_name_for_session_run():
    """验证 ONNX 普通推理分支使用模型真实输入名。"""
    backend = object.__new__(ONNXBackend)
    backend.format = "onnx"
    backend.use_io_binding = False
    backend.device = torch.device("cpu")
    backend.fp16 = False
    backend.input_name = "input"
    backend.dynamic = False
    backend.static_batch = None
    backend.session = _FakeSession("input")
    backend.output_names = ["output"]

    im = torch.zeros(1, 3, 8, 8)  # 输入张量
    output = ONNXBackend.forward(backend, im)

    run_inputs = backend.session.run_calls[0][1]  # 普通推理输入映射
    assert list(run_inputs) == ["input"]
    np.testing.assert_array_equal(run_inputs["input"], im.numpy())
    assert output == ["fake-output"]


def test_onnx_backend_forward_pads_tail_batch_for_session_run():
    """验证普通 ONNX Runtime 分支在尾批不足时也会补齐并裁剪输出。"""
    backend = object.__new__(ONNXBackend)
    backend.format = "onnx"
    backend.use_io_binding = False
    backend.device = torch.device("cpu")
    backend.fp16 = False
    backend.input_name = "input"
    backend.dynamic = False
    backend.static_batch = 4
    backend.session = _FakeSession("input")
    backend.output_names = ["output"]
    backend.session.run = lambda output_names, inputs: [np.arange(8, dtype=np.float32).reshape(4, 2)]

    im = torch.zeros(3, 3, 8, 8)  # 尾批只有 3 帧
    output = ONNXBackend.forward(backend, im)

    assert output[0].shape == (3, 2)
    np.testing.assert_array_equal(output[0], np.arange(6, dtype=np.float32).reshape(3, 2))
