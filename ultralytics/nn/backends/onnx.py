# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class ONNXBackend(BaseBackend):
    """Microsoft ONNX Runtime inference backend with optional OpenCV DNN support.

    Loads and runs inference with ONNX models (.onnx files) using either Microsoft ONNX Runtime with CUDA/CoreML
    execution providers, or OpenCV DNN for lightweight CPU inference. Supports IO binding for optimized GPU inference
    with static input shapes.
    """

    def __init__(self, weight: str | Path, device: torch.device, fp16: bool = False, format: str = "onnx"):
        """Initialize the ONNX backend.

        Args:
            weight (str | Path): Path to the .onnx model file.
            device (torch.device): Device to run inference on.
            fp16 (bool): Whether to use FP16 half-precision inference.
            format (str): Inference engine, either "onnx" for ONNX Runtime or "dnn" for OpenCV DNN.
        """
        assert format in {"onnx", "dnn"}, f"Unsupported ONNX format: {format}."
        self.format = format
        super().__init__(weight, device, fp16)

    def load_model(self, weight: str | Path) -> None:
        """Load an ONNX model using ONNX Runtime or OpenCV DNN.

        Args:
            weight (str | Path): Path to the .onnx model file.
        """
        cuda = isinstance(self.device, torch.device) and torch.cuda.is_available() and self.device.type != "cpu"

        if self.format == "dnn":
            # OpenCV DNN
            LOGGER.info(f"Loading {weight} for ONNX OpenCV DNN inference...")
            check_requirements("opencv-python>=4.5.4")
            import cv2

            self.net = cv2.dnn.readNetFromONNX(weight)
        else:
            # ONNX Runtime
            LOGGER.info(f"Loading {weight} for ONNX Runtime inference...")
            check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))
            import onnxruntime

            # Select execution provider
            available = onnxruntime.get_available_providers()
            if cuda and "CUDAExecutionProvider" in available:
                providers = [("CUDAExecutionProvider", {"device_id": self.device.index}), "CPUExecutionProvider"]
            elif self.device.type == "mps" and "CoreMLExecutionProvider" in available:
                providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
                if cuda:
                    LOGGER.warning("CUDA requested but CUDAExecutionProvider not available. Using CPU...")
                    self.device = torch.device("cpu")
                    cuda = False

            LOGGER.info(
                f"Using ONNX Runtime {onnxruntime.__version__} with "
                f"{providers[0] if isinstance(providers[0], str) else providers[0][0]}"
            )

            self.session = onnxruntime.InferenceSession(weight, providers=providers)
            self.input_info = self.session.get_inputs()[0]
            self.input_name = self.input_info.name
            self.output_names = [x.name for x in self.session.get_outputs()]

            # Get metadata
            metadata_map = self.session.get_modelmeta().custom_metadata_map
            if metadata_map:
                self.apply_metadata(dict(metadata_map))

            # Check if dynamic shapes
            self.dynamic = isinstance(self.session.get_outputs()[0].shape[0], str)
            self.fp16 = "float16" in self.input_info.type
            self.static_batch = self.input_info.shape[0] if self.input_info.shape else None
            if not isinstance(self.static_batch, int) or self.static_batch <= 0:
                self.static_batch = None

            # Setup IO binding for CUDA
            self.use_io_binding = not self.dynamic and cuda
            if self.use_io_binding:
                self.io = self.session.io_binding()
                self.bindings = []
                for output in self.session.get_outputs():
                    out_fp16 = "float16" in output.type
                    y_tensor = torch.empty(output.shape, dtype=torch.float16 if out_fp16 else torch.float32).to(
                        self.device
                    )
                    self.io.bind_output(
                        name=output.name,
                        device_type=self.device.type,
                        device_id=self.device.index if cuda else 0,
                        element_type=np.float16 if out_fp16 else np.float32,
                        shape=tuple(y_tensor.shape),
                        buffer_ptr=y_tensor.data_ptr(),
                    )
                    self.bindings.append(y_tensor)

    def _pad_input(self, im: torch.Tensor) -> tuple[torch.Tensor, int]:
        """对静态 batch ONNX 输入做补齐。

        Args:
            im (torch.Tensor): 输入张量。

        Returns:
            (tuple[torch.Tensor, int]): 补齐后的输入张量与原始 batch 大小。
        """
        batch_size = int(im.shape[0])  # 原始 batch 大小
        dynamic = getattr(self, "dynamic", False)  # 是否动态 batch
        static_batch = getattr(self, "static_batch", None)  # 静态 batch 大小
        if dynamic or not static_batch or batch_size == static_batch:
            return im, batch_size
        if batch_size > static_batch:
            raise RuntimeError(
                f"ONNX static batch expects {static_batch}, but got {batch_size}. "
                "Please reduce batch size or re-export the model with dynamic batch."
            )

        pad_count = static_batch - batch_size  # 需要补齐的帧数
        pad_tensor = im[-1:].repeat(pad_count, 1, 1, 1)  # 重复最后一帧补齐尾批
        return torch.cat((im, pad_tensor), dim=0), batch_size

    @staticmethod
    def _slice_output(y: torch.Tensor | np.ndarray, batch_size: int) -> torch.Tensor | np.ndarray:
        """按原始 batch 大小裁剪输出。

        Args:
            y (torch.Tensor | np.ndarray): 模型输出。
            batch_size (int): 原始 batch 大小。

        Returns:
            (torch.Tensor | np.ndarray): 裁剪后的输出。
        """
        return y[:batch_size] if getattr(y, "shape", None) and len(y.shape) > 0 else y

    def forward(self, im: torch.Tensor) -> torch.Tensor | list[torch.Tensor] | np.ndarray:
        """Run ONNX inference using IO binding (CUDA) or standard session execution.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

        Returns:
            (torch.Tensor | list[torch.Tensor] | np.ndarray): Model predictions as tensor(s) or numpy array(s).
        """
        if self.format == "dnn":
            # OpenCV DNN
            self.net.setInput(im.cpu().numpy())
            return self.net.forward()

        # ONNX Runtime
        im, batch_size = self._pad_input(im)
        if self.use_io_binding:
            if self.device.type == "cpu":
                im = im.cpu()
            self.io.bind_input(
                name=self.input_name,
                device_type=im.device.type,
                device_id=im.device.index if im.device.type == "cuda" else 0,
                element_type=np.float16 if self.fp16 else np.float32,
                shape=tuple(im.shape),
                buffer_ptr=im.data_ptr(),
            )
            self.session.run_with_iobinding(self.io)
            return [self._slice_output(binding, batch_size) for binding in self.bindings]
        else:
            outputs = self.session.run(self.output_names, {self.input_name: im.cpu().numpy()})
            return [self._slice_output(output, batch_size) for output in outputs]


class ONNXIMXBackend(ONNXBackend):
    """ONNX IMX inference backend for NXP i.MX processors.

    Extends `ONNXBackend` with support for quantized models targeting NXP i.MX edge devices. Uses MCT (Model Compression
    Toolkit) quantizers and custom NMS operations for optimized inference.
    """

    def load_model(self, weight: str | Path) -> None:
        """Load a quantized ONNX model from an IMX model directory.

        Args:
            weight (str | Path): Path to the IMX model directory containing the .onnx file.
        """
        check_requirements(("model-compression-toolkit>=2.4.1", "edge-mdt-cl<1.1.0", "onnxruntime-extensions"))
        check_requirements(("onnx", "onnxruntime"))
        import mct_quantizers as mctq
        import onnxruntime
        from edgemdt_cl.pytorch.nms import nms_ort  # noqa - register custom NMS ops

        w = Path(weight)
        onnx_file = next(w.glob("*.onnx"))
        LOGGER.info(f"Loading {onnx_file} for ONNX IMX inference...")

        session_options = mctq.get_ort_session_options()
        session_options.enable_mem_reuse = False

        self.session = onnxruntime.InferenceSession(onnx_file, session_options, providers=["CPUExecutionProvider"])
        self.input_info = self.session.get_inputs()[0]
        self.input_name = self.input_info.name
        self.output_names = [x.name for x in self.session.get_outputs()]
        self.dynamic = isinstance(self.session.get_outputs()[0].shape[0], str)
        self.fp16 = "float16" in self.input_info.type
        self.static_batch = self.input_info.shape[0] if self.input_info.shape else None
        if not isinstance(self.static_batch, int) or self.static_batch <= 0:
            self.static_batch = None
        metadata_map = self.session.get_modelmeta().custom_metadata_map
        if metadata_map:
            self.apply_metadata(dict(metadata_map))

    def forward(self, im: torch.Tensor) -> np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]:
        """Run IMX inference with task-specific output concatenation for detect, pose, and segment tasks.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

        Returns:
            (np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]): Task-formatted model predictions.
        """
        im, batch_size = self._pad_input(im)
        y = self.session.run(self.output_names, {self.input_name: im.cpu().numpy()})
        y = [self._slice_output(output, batch_size) for output in y]

        if self.task == "detect":
            # boxes, conf, cls
            return np.concatenate([y[0], y[1][:, :, None], y[2][:, :, None]], axis=-1)
        elif self.task == "pose":
            # boxes, conf, kpts
            return np.concatenate([y[0], y[1][:, :, None], y[2][:, :, None], y[3]], axis=-1, dtype=y[0].dtype)
        elif self.task == "segment":
            return (
                np.concatenate([y[0], y[1][:, :, None], y[2][:, :, None], y[3]], axis=-1, dtype=y[0].dtype),
                y[4],
            )
        return y
