#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用自训练模型对视频推理并保存结果视频（支持等比缩放与直接缩放）。"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Iterable, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 解析后的参数对象。
    """
    parser = argparse.ArgumentParser(description="YOLO 视频推理（自定义预处理）")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型权重路径，例如 runs/detect/train/weights/best.pt",
    )
    parser.add_argument("--video", type=str, required=True, help="输入视频路径")
    parser.add_argument(
        "--output",
        type=str,
        default="runs/predict/video_result.mp4",
        help="输出视频路径",
    )
    parser.add_argument(
        "--imgsz",
        type=str,
        default="640",
        help="推理尺寸，示例：640 或 352,640",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        choices=["letterbox", "resize"],
        default="letterbox",
        help="预处理方式：letterbox 等比缩放填充；resize 直接缩放",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    parser.add_argument("--device", type=str, default="", help="设备，例如 0 或 cpu")
    return parser.parse_args()


def parse_imgsz(imgsz: str) -> Tuple[int, int]:
    """解析推理尺寸字符串。

    Args:
        imgsz (str): 推理尺寸字符串，格式为 640 或 352,640 或 352x640。

    Returns:
        Tuple[int, int]: (高, 宽)
    """
    imgsz = imgsz.lower().replace("x", ",")
    parts = [p for p in imgsz.split(",") if p.strip()]
    if len(parts) == 1:
        size = int(parts[0])  # 尺寸值
        return size, size
    if len(parts) == 2:
        height = int(parts[0])  # 目标高度
        width = int(parts[1])  # 目标宽度
        return height, width
    raise ValueError(f"无效的 --imgsz 参数：{imgsz}")


def letterbox_image(
    image: np.ndarray,
    new_shape: Tuple[int, int],
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """等比缩放并填充到目标尺寸。

    Args:
        image (np.ndarray): 输入图像（BGR）。
        new_shape (Tuple[int, int]): 目标尺寸 (高, 宽)。
        color (Tuple[int, int, int]): 填充颜色。

    Returns:
        Tuple[np.ndarray, float, Tuple[int, int]]: 预处理后的图像、缩放比例、左上角填充值 (pad_x, pad_y)。
    """
    height = image.shape[0]  # 原图高度
    width = image.shape[1]  # 原图宽度
    target_h, target_w = new_shape  # 目标尺寸

    ratio = min(target_w / width, target_h / height)  # 缩放比例
    resized_w = int(round(width * ratio))  # 缩放后宽度
    resized_h = int(round(height * ratio))  # 缩放后高度
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)  # 缩放图像

    pad_w = target_w - resized_w  # 需要填充的宽度
    pad_h = target_h - resized_h  # 需要填充的高度
    left = pad_w // 2  # 左侧填充
    right = pad_w - left  # 右侧填充
    top = pad_h // 2  # 上侧填充
    bottom = pad_h - top  # 下侧填充

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 填充图像
    return padded, ratio, (left, top)


def scale_boxes_from_letterbox(
    boxes_xyxy: np.ndarray,
    ratio: float,
    pad: Tuple[int, int],
    original_shape: Tuple[int, int],
) -> np.ndarray:
    """将 letterbox 坐标映射回原图。

    Args:
        boxes_xyxy (np.ndarray): letterbox 图像中的 xyxy 坐标。
        ratio (float): 缩放比例。
        pad (Tuple[int, int]): (pad_x, pad_y)。
        original_shape (Tuple[int, int]): 原图尺寸 (高, 宽)。

    Returns:
        np.ndarray: 映射回原图的 xyxy 坐标。
    """
    pad_x, pad_y = pad  # 填充值
    boxes = boxes_xyxy.copy()  # 坐标副本
    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes /= ratio

    orig_h, orig_w = original_shape  # 原图尺寸
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h - 1)
    return boxes


def scale_boxes_from_resize(
    boxes_xyxy: np.ndarray,
    original_shape: Tuple[int, int],
    resized_shape: Tuple[int, int],
) -> np.ndarray:
    """将直接缩放后的坐标映射回原图。

    Args:
        boxes_xyxy (np.ndarray): 直接缩放图像中的 xyxy 坐标。
        original_shape (Tuple[int, int]): 原图尺寸 (高, 宽)。
        resized_shape (Tuple[int, int]): 缩放后尺寸 (高, 宽)。

    Returns:
        np.ndarray: 映射回原图的 xyxy 坐标。
    """
    orig_h, orig_w = original_shape  # 原图尺寸
    resized_h, resized_w = resized_shape  # 缩放后尺寸
    gain_w = orig_w / resized_w  # 宽度缩放回原图的比例
    gain_h = orig_h / resized_h  # 高度缩放回原图的比例

    boxes = boxes_xyxy.copy()  # 坐标副本
    boxes[:, [0, 2]] *= gain_w
    boxes[:, [1, 3]] *= gain_h
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h - 1)
    return boxes


def iter_detections(result) -> Iterable[Tuple[np.ndarray, float, int]]:
    """从推理结果中迭代检测框。

    Args:
        result: Ultralytics 单帧结果对象。

    Yields:
        Tuple[np.ndarray, float, int]: (xyxy, conf, cls_id)
    """
    if result.boxes is None:
        return []
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # xyxy 坐标
    confs = result.boxes.conf.cpu().numpy()  # 置信度
    clss = result.boxes.cls.cpu().numpy().astype(int)  # 类别
    for xyxy, conf, cls_id in zip(boxes_xyxy, confs, clss):
        yield xyxy, float(conf), int(cls_id)


def get_color(cls_id: int) -> Tuple[int, int, int]:
    """为类别生成稳定颜色。

    Args:
        cls_id (int): 类别 id。

    Returns:
        Tuple[int, int, int]: BGR 颜色。
    """
    palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)  # 颜色调色板
    color = [int((p * (cls_id**2 - cls_id + 1)) % 255) for p in palette]  # 颜色计算
    return color[2], color[1], color[0]


def draw_boxes(
    frame: np.ndarray,
    boxes_xyxy: np.ndarray,
    confs: np.ndarray,
    clss: np.ndarray,
    names,
) -> np.ndarray:
    """在帧上绘制检测框与标签。

    Args:
        frame (np.ndarray): 原始帧。
        boxes_xyxy (np.ndarray): xyxy 坐标。
        confs (np.ndarray): 置信度。
        clss (np.ndarray): 类别。
        names: 类别名称映射。

    Returns:
        np.ndarray: 绘制后的帧。
    """
    for xyxy, conf, cls_id in zip(boxes_xyxy, confs, clss):
        x1, y1, x2, y2 = [int(v) for v in xyxy]  # 坐标
        color = get_color(int(cls_id))  # 颜色
        label_name = names[int(cls_id)] if isinstance(names, dict) else names[int(cls_id)]  # 标签
        label = f"{label_name} {conf:.2f}"  # 文本标签

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(
            frame,
            (x1, y1 - text_h - baseline - 4),
            (x1 + text_w, y1),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return frame


def main() -> None:
    """执行视频推理并保存结果。"""
    args = parse_args()  # 参数
    model_path = args.model  # 模型路径
    video_path = args.video  # 视频路径
    output_path = args.output  # 输出路径
    preprocess = args.preprocess  # 预处理方式
    target_h, target_w = parse_imgsz(args.imgsz)  # 推理尺寸

    model = YOLO(model_path)  # 模型对象
    print(f"torch.cuda.is_available()={torch.cuda.is_available()}")  # CUDA 可用性
    print(f"model.device={model.device}")  # 模型设备
    cap = cv2.VideoCapture(video_path)  # 视频读取器
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # 帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 原视频宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 原视频高度

    output_file = Path(output_path)  # 输出文件路径
    output_file.parent.mkdir(parents=True, exist_ok=True)  # 创建输出目录
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 编码器
    writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))  # 视频写入器

    names = model.names  # 类别名称
    total_frames = 0  # 总帧数
    log_interval = 30  # 打印间隔帧数
    window_e2e_times = []  # 端到端耗时窗口
    window_infer_times = []  # 仅推理耗时窗口
    latest_speed = None  # 最近一次分段耗时
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1  # 帧计数
        e2e_start = time.perf_counter()  # 端到端起始时间

        if preprocess == "letterbox":
            pre_start = time.perf_counter()  # 前处理起始时间
            input_img, ratio, pad = letterbox_image(frame, (target_h, target_w))  # 等比缩放
            _ = pre_start  # 保留变量防止误删（便于扩展）
            infer_start = time.perf_counter()  # 推理起始时间
            results = model.predict(
                source=input_img,
                imgsz=(target_h, target_w),
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )
            infer_end = time.perf_counter()  # 推理结束时间
            result = results[0]  # 单帧结果
            latest_speed = getattr(result, "speed", None)  # 分段耗时
            dets = list(iter_detections(result))  # 检测列表
            if dets:
                boxes = np.array([d[0] for d in dets], dtype=np.float32)  # 检测框
                confs = np.array([d[1] for d in dets], dtype=np.float32)  # 置信度
                clss = np.array([d[2] for d in dets], dtype=np.int32)  # 类别
                boxes = scale_boxes_from_letterbox(boxes, ratio, pad, (height, width))  # 坐标映射
        else:
            pre_start = time.perf_counter()  # 前处理起始时间
            resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)  # 直接缩放
            _ = pre_start  # 保留变量防止误删（便于扩展）
            infer_start = time.perf_counter()  # 推理起始时间
            results = model.predict(
                source=resized,
                imgsz=(target_h, target_w),
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )
            infer_end = time.perf_counter()  # 推理结束时间
            result = results[0]  # 单帧结果
            latest_speed = getattr(result, "speed", None)  # 分段耗时
            dets = list(iter_detections(result))  # 检测列表
            if dets:
                boxes = np.array([d[0] for d in dets], dtype=np.float32)  # 检测框
                confs = np.array([d[1] for d in dets], dtype=np.float32)  # 置信度
                clss = np.array([d[2] for d in dets], dtype=np.int32)  # 类别
                boxes = scale_boxes_from_resize(boxes, (height, width), (target_h, target_w))  # 坐标映射
        e2e_end = time.perf_counter()  # 端到端结束时间（不含绘制与写视频）
        if dets:
            frame = draw_boxes(frame, boxes, confs, clss, names)  # 绘制
        writer.write(frame)
        e2e_time = e2e_end - e2e_start  # 端到端耗时
        infer_time = infer_end - infer_start  # 仅推理耗时
        window_e2e_times.append(e2e_time)
        window_infer_times.append(infer_time)
        if len(window_e2e_times) > log_interval:
            window_e2e_times.pop(0)
        if len(window_infer_times) > log_interval:
            window_infer_times.pop(0)
        if total_frames % log_interval == 0:
            avg_e2e = sum(window_e2e_times) / max(len(window_e2e_times), 1)  # 平均端到端耗时
            avg_infer = sum(window_infer_times) / max(len(window_infer_times), 1)  # 平均推理耗时
            fps_e2e = 1.0 / avg_e2e if avg_e2e > 0 else 0.0  # 端到端 FPS
            fps_infer = 1.0 / avg_infer if avg_infer > 0 else 0.0  # 推理 FPS
            speed_text = ""
            if isinstance(latest_speed, dict):
                preprocess_ms = latest_speed.get("preprocess", 0.0)
                inference_ms = latest_speed.get("inference", 0.0)
                postprocess_ms = latest_speed.get("postprocess", 0.0)
                speed_text = (
                    f" | speed(ms) preprocess: {preprocess_ms:.2f}, "
                    f"inference: {inference_ms:.2f}, postprocess: {postprocess_ms:.2f}"
                )
            print(f"[Frame {total_frames}] 端到端 FPS: {fps_e2e:.2f} | 仅推理 FPS: {fps_infer:.2f}{speed_text}")

    cap.release()
    writer.release()


if __name__ == "__main__":
    main()
