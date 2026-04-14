#!/usr/bin/env python3
"""使用 ONNX Runtime 对导出的 YOLO ONNX 模型执行单图推理并保存结果图。"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Args:
        无。

    Returns:
        (argparse.Namespace) 解析后的命令行参数对象。
    """
    parser = argparse.ArgumentParser(description="ONNX 单图推理脚本")
    parser.add_argument("--model", type=str, required=True, help="ONNX 模型路径")
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--output", type=str, required=True, help="输出图片路径")
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=None,
        help="模型输入尺寸，单值表示方形输入，双值表示 [height width]",
    )
    parser.add_argument("--conf-thres", type=float, default=0.25, help="置信度阈值")
    return parser.parse_args()


def normalize_imgsz(imgsz: list[int] | None, input_shape: list) -> tuple[int, int]:
    """标准化模型输入尺寸。

    Args:
        imgsz: 命令行传入的输入尺寸列表。
        input_shape: ONNX 模型输入张量形状。

    Returns:
        (tuple[int, int]) 标准化后的 (height, width)。
    """
    if imgsz:
        if len(imgsz) == 1:
            target_h = imgsz[0]  # 目标高度。
            target_w = imgsz[0]  # 目标宽度。
        else:
            target_h = imgsz[0]  # 目标高度。
            target_w = imgsz[1]  # 目标宽度。
        return target_h, target_w

    model_h = input_shape[2] if len(input_shape) > 2 else 640  # 模型输入高度。
    model_w = input_shape[3] if len(input_shape) > 3 else 640  # 模型输入宽度。

    if not isinstance(model_h, int):
        model_h = 640  # 动态输入时的默认高度。
    if not isinstance(model_w, int):
        model_w = 640  # 动态输入时的默认宽度。

    return model_h, model_w


def preprocess_image(image_path: str, target_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """读取并预处理图片。

    Args:
        image_path: 输入图片路径。
        target_size: 模型输入尺寸 (height, width)。

    Returns:
        (tuple[np.ndarray, np.ndarray, tuple[int, int]])
            预处理后的输入张量、原始 BGR 图像、原始尺寸 (width, height)。
    """
    image_bgr = cv2.imread(image_path)  # 原始 BGR 图像。
    if image_bgr is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    original_size = (image_bgr.shape[1], image_bgr.shape[0])  # 原始尺寸 (width, height)。
    resized_bgr = cv2.resize(image_bgr, (target_size[1], target_size[0]))  # 按模型尺寸缩放后的图像。
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)  # RGB 图像。

    input_tensor = resized_rgb.astype(np.float32) / 255.0  # 归一化后的输入数据。
    input_tensor = np.transpose(input_tensor, (2, 0, 1))  # 转为 CHW。
    input_tensor = np.expand_dims(input_tensor, axis=0)  # 增加 batch 维度。

    return input_tensor, image_bgr, original_size


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """将边界框从 xywh 转为 xyxy。

    Args:
        boxes: [N, 4] 格式边界框。

    Returns:
        (np.ndarray) [N, 4] 的 xyxy 边界框。
    """
    converted = boxes.copy()  # 转换后的边界框数组。
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def postprocess(output: np.ndarray, conf_thres: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对 ONNX 输出做基础后处理。

    Args:
        output: ONNX 输出张量。
        conf_thres: 置信度阈值。

    Returns:
        (tuple[np.ndarray, np.ndarray, np.ndarray])
            boxes(xyxy)、scores、class_ids。
    """
    predictions = output[0]  # 去掉 batch 维度后的预测结果。

    if len(output.shape) == 3 and output.shape[1] < output.shape[2]:
        predictions = output.transpose(0, 2, 1)[0]  # [1, C, N] 转 [N, C]。

    if predictions.shape[1] == 6 and predictions.shape[0] <= 300:
        boxes = predictions[:, :4]  # 检测框。
        scores = predictions[:, 4]  # 置信度。
        class_ids = predictions[:, 5].astype(np.int32)  # 类别索引。
    else:
        boxes = predictions[:, :4]  # 原始 xywh 检测框。
        class_probs = predictions[:, 4:]  # 类别概率。
        scores = np.max(class_probs, axis=1)  # 最大类别置信度。
        class_ids = np.argmax(class_probs, axis=1).astype(np.int32)  # 最大概率对应类别。
        boxes = xywh2xyxy(boxes)

    valid_mask = scores > conf_thres  # 置信度筛选掩码。
    boxes = boxes[valid_mask]  # 筛选后的检测框。
    scores = scores[valid_mask]  # 筛选后的置信度。
    class_ids = class_ids[valid_mask]  # 筛选后的类别索引。
    return boxes, scores, class_ids


def scale_boxes(boxes: np.ndarray, original_size: tuple[int, int], model_size: tuple[int, int]) -> np.ndarray:
    """将模型坐标缩放回原图尺寸。

    Args:
        boxes: 模型输入尺度下的 xyxy 边界框。
        original_size: 原图尺寸 (width, height)。
        model_size: 模型输入尺寸 (height, width)。

    Returns:
        (np.ndarray) 原图尺度下的边界框。
    """
    if boxes.size == 0:
        return boxes

    scale_x = original_size[0] / model_size[1]  # x 方向缩放系数。
    scale_y = original_size[1] / model_size[0]  # y 方向缩放系数。
    scaled = boxes.copy()  # 缩放后的边界框。
    scaled[:, [0, 2]] *= scale_x
    scaled[:, [1, 3]] *= scale_y
    return scaled


def draw_detections(image: np.ndarray, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
    """在图像上绘制检测框。

    Args:
        image: 原始 BGR 图像。
        boxes: 原图尺度下的 xyxy 边界框。
        scores: 置信度。
        class_ids: 类别索引。

    Returns:
        (np.ndarray) 绘制后的 BGR 图像。
    """
    canvas = image.copy()  # 待绘制图像。
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.astype(np.int32)  # 框坐标整数值。
        label = f"cls {int(class_id)} {score:.2f}"  # 标签文本。
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_top = max(y1 - label_h - baseline - 4, 0)  # 文本背景上边界。
        cv2.rectangle(canvas, (x1, text_top), (x1 + label_w + 2, text_top + label_h + baseline + 4), (0, 255, 0), -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 1, text_top + label_h + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


def run_inference(
    model_path: str,
    image_path: str,
    output_path: str,
    imgsz: list[int] | None,
    conf_thres: float,
) -> None:
    """执行 ONNX 推理并保存结果。

    Args:
        model_path: ONNX 模型路径。
        image_path: 输入图片路径。
        output_path: 输出图片路径。
        imgsz: 输入尺寸参数。
        conf_thres: 置信度阈值。

    Returns:
        无。
    """
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])  # ONNX Runtime 会话。
    input_info = session.get_inputs()[0]  # 输入节点信息。
    output_info = session.get_outputs()[0]  # 输出节点信息。
    model_size = normalize_imgsz(imgsz, input_info.shape)  # 实际推理输入尺寸。

    input_tensor, image_bgr, original_size = preprocess_image(image_path, model_size)  # 输入张量和原图数据。
    outputs = session.run([output_info.name], {input_info.name: input_tensor})  # 推理输出列表。
    raw_output = outputs[0]  # 主输出张量。

    boxes, scores, class_ids = postprocess(raw_output, conf_thres)  # 初步后处理结果。

    boxes = scale_boxes(boxes, original_size, model_size)  # 缩放到原图坐标系。
    result = draw_detections(image_bgr, boxes, scores, class_ids)  # 绘制后的图像。

    output_file = Path(output_path)  # 输出文件路径对象。
    output_file.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在。
    success = cv2.imwrite(str(output_file), result)  # 写入结果图。
    if not success:
        raise RuntimeError(f"保存结果图片失败: {output_path}")

    print(f"模型输入节点: {input_info.name}, shape={input_info.shape}")
    print(f"模型输出节点: {output_info.name}, shape={output_info.shape}")
    print(f"推理输入尺寸: {model_size[0]}x{model_size[1]}")
    print(f"检测数量: {len(boxes)}")
    print(f"结果已保存: {output_file}")


def main() -> None:
    """脚本入口函数。

    Args:
        无。

    Returns:
        无。
    """
    args = parse_args()  # 命令行参数。
    run_inference(
        model_path=args.model,
        image_path=args.image,
        output_path=args.output,
        imgsz=args.imgsz,
        conf_thres=args.conf_thres,
    )


if __name__ == "__main__":
    main()
