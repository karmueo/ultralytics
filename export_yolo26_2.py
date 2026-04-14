import os
import sys
from copy import deepcopy

import onnx
import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Detect, v10Detect
import ultralytics.models.yolo
import ultralytics.utils
import ultralytics.utils.tal as _m

sys.modules["ultralytics.yolo"] = ultralytics.models.yolo
sys.modules["ultralytics.yolo.utils"] = ultralytics.utils


def _dist2bbox(distance, anchor_points, xywh=False, dim=-1):
    """将距离编码转换为边界框坐标（固定输出 xyxy）。

    Args:
        distance (torch.Tensor): 距离预测张量。
        anchor_points (torch.Tensor): 锚点坐标张量。
        xywh (bool): 占位参数，导出时忽略以固定输出格式。
        dim (int): 切分维度。

    Returns:
        (torch.Tensor): xyxy 格式的边界框张量。
    """
    lt, rb = distance.chunk(2, dim)  # 左上/右下距离
    x1y1 = anchor_points - lt  # 左上角坐标
    x2y2 = anchor_points + rb  # 右下角坐标
    return torch.cat((x1y1, x2y2), dim)


_m.dist2bbox.__code__ = _dist2bbox.__code__


class DeepStreamOutput(nn.Module):
    """DeepStream 期望输出封装。

    Args:
        end2end (bool): 是否为端到端 NMS-free 输出。
    """

    def __init__(self, end2end: bool):
        super().__init__()
        self.end2end = end2end  # 是否端到端输出

    def forward(self, x):
        """根据输出结构整理为 DeepStream 兼容格式。

        Args:
            x (torch.Tensor): 模型输出张量。

        Returns:
            (torch.Tensor): 形状为 [B, N, 6] 的输出。
        """
        if self.end2end:
            return x
        x = x.transpose(1, 2)  # 转置为 [B, N, C]
        boxes = x[:, :, :4]  # 边界框坐标
        scores, labels = torch.max(x[:, :, 4:], dim=-1, keepdim=True)  # 分数与类别
        return torch.cat([boxes, scores, labels.to(boxes.dtype)], dim=-1)


def _is_end2end(model: nn.Module) -> bool:
    """判断模型是否为端到端输出。

    Args:
        model (nn.Module): YOLO 模型实例。

    Returns:
        (bool): 是否启用 end2end。
    """
    for module in model.modules():  # 遍历模块
        if isinstance(module, (Detect, v10Detect)):
            return bool(getattr(module, "end2end", False))
    return False


def yolo26_export(weights: str, device: torch.device, fuse: bool = True) -> nn.Module:
    """构建用于导出的 YOLO26 模型。

    Args:
        weights (str): 权重文件路径。
        device (torch.device): 运行设备。
        fuse (bool): 是否融合卷积与 BN。

    Returns:
        (nn.Module): 导出用模型。
    """
    model = YOLO(weights)  # YOLO 包装器
    model = deepcopy(model.model).to(device)  # 推理模型
    for p in model.parameters():  # 参数
        p.requires_grad = False
    model.eval()
    model.float()
    if fuse:
        model = model.fuse()
    for _, m in model.named_modules():  # 模块遍历
        if isinstance(m, (Detect, v10Detect)):
            m.dynamic = False
            m.export = True
            m.format = "onnx"
        elif isinstance(m, C2f):
            m.forward = m.forward_split
    return model


def suppress_warnings() -> None:
    """屏蔽导出过程中的常见告警。"""
    import warnings
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=ResourceWarning)


def main(args) -> None:
    """导出 YOLO26 ONNX 并生成 labels.txt。

    Args:
        args (argparse.Namespace): 命令行参数。
    """
    suppress_warnings()

    print(f"\nStarting: {args.weights}")
    print("Opening YOLO26 model")

    device = torch.device("cpu")  # 运行设备
    model = yolo26_export(args.weights, device)  # 导出模型
    end2end = _is_end2end(model)  # 是否端到端输出

    if len(model.names.keys()) > 0:
        print("Creating labels.txt file")
        with open("labels.txt", "w", encoding="utf-8") as f:  # 标签文件
            for name in model.names.values():
                f.write(f"{name}\n")

    model = nn.Sequential(model, DeepStreamOutput(end2end))  # 输出封装模型

    img_size = args.size * 2 if len(args.size) == 1 else args.size  # 输入尺寸
    onnx_input_im = torch.zeros(args.batch, 3, *img_size).to(device)  # ONNX 输入
    onnx_output_file = args.weights.rsplit(".", 1)[0] + ".onnx"  # 输出路径

    dynamic_axes = {  # 动态维度配置
        "input": {
            0: "batch"
        },
        "output": {
            0: "batch"
        }
    }

    print("Exporting the model to ONNX")
    torch.onnx.export(
        model,
        onnx_input_im,
        onnx_output_file,
        verbose=False,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes if args.dynamic else None
    )

    if args.simplify:
        print("Simplifying the ONNX model")
        import onnxslim
        model_onnx = onnx.load(onnx_output_file)  # ONNX 模型
        model_onnx = onnxslim.slim(model_onnx)
        onnx.save(model_onnx, onnx_output_file)

    print(f"Done: {onnx_output_file}\n")


def parse_args():
    """解析命令行参数。

    Returns:
        (argparse.Namespace): 参数对象。
    """
    import argparse
    parser = argparse.ArgumentParser(description="DeepStream YOLO26 conversion")  # 参数解析器
    parser.add_argument("-w", "--weights", required=True, type=str, help="Input weights (.pt) file path (required)")
    parser.add_argument("-s", "--size", nargs="+", type=int, default=[640], help="Inference size [H,W] (default [640])")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--simplify", action="store_true", help="ONNX simplify model")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic batch-size")
    parser.add_argument("--batch", type=int, default=1, help="Static batch-size")
    args = parser.parse_args()  # 解析结果
    if not os.path.isfile(args.weights):
        raise RuntimeError("Invalid weights file")
    if args.dynamic and args.batch > 1:
        raise RuntimeError("Cannot set dynamic batch-size and static batch-size at same time")
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
