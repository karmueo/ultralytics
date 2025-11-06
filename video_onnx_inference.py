import argparse
from pathlib import Path
import time
import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2


def preprocess_image(image, imgsz=640):
    """
    预处理图像用于 ONNX 推理

    Args:
        image: numpy array 或 PIL Image
        imgsz: 输入图像尺寸

    Returns:
        处理后的图像数组 [1, 3, H, W], 原始图像尺寸
    """
    if isinstance(image, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        img = image

    original_size = img.size  # (width, height)
    img = img.convert("RGB")
    img_resized = img.resize((imgsz, imgsz))

    # 转换为 numpy 数组并归一化
    img_array = np.array(img_resized).astype(np.float32) / 255.0

    # 转换维度从 [H, W, C] 到 [C, H, W]
    img_array = img_array.transpose(2, 0, 1)

    # 添加 batch 维度 [1, C, H, W]
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, original_size


def xywh2xyxy(boxes):
    """
    将边界框从 [x_center, y_center, width, height] 转换为 [x1, y1, x2, y2]

    Args:
        boxes: numpy array of shape [N, 4] in xywh format

    Returns:
        boxes in xyxy format
    """
    xyxy = boxes.copy()
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 = x_center - w/2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 = y_center - h/2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2 = x_center + w/2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2 = y_center + h/2
    return xyxy


def postprocess_detections(output, conf_threshold=0.25, verbose=False):
    """
    后处理 ONNX 模型输出

    Args:
        output: ONNX 模型输出
        conf_threshold: 置信度阈值
        verbose: 是否打印调试信息

    Returns:
        检测结果列表
    """
    # 检查输出形状
    if verbose:
        print(f"Raw output shape: {output.shape}")

    # 如果是 YOLOv5 格式 [1, 6, 8400]，转换为 [1, 8400, 6]
    if len(output.shape) == 3 and output.shape[1] < output.shape[2]:
        output = output.transpose(0, 2, 1)
        if verbose:
            print(f"Transposed output shape: {output.shape}")

    predictions = output[0]  # 移除 batch 维度

    # 如果输出已经包含 NMS (shape like [300, 6])
    if predictions.shape[1] == 6 and predictions.shape[0] <= 300:
        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        class_ids = predictions[:, 5].astype(int)

        # 过滤无效检测
        valid_mask = scores > conf_threshold
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        class_ids = class_ids[valid_mask]
    else:
        # 原始输出，需要手动处理
        boxes = predictions[:, :4]  # [x_center, y_center, w, h]
        class_probs = predictions[:, 4:]

        # 获取最大类别概率和类别ID
        scores = np.max(class_probs, axis=1)
        class_ids = np.argmax(class_probs, axis=1)

        # 置信度过滤
        valid_mask = scores > conf_threshold
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        class_ids = class_ids[valid_mask]

        # 转换坐标格式从 xywh 到 xyxy
        if len(boxes) > 0:
            boxes = xywh2xyxy(boxes)
            if verbose:
                print("Converted boxes from xywh to xyxy format")

    return boxes, scores, class_ids


def nms_boxes(boxes, scores, iou_threshold=0.45):
    """
    执行非极大值抑制 (NMS)

    Args:
        boxes: numpy array of shape [N, 4], format [x1, y1, x2, y2]
        scores: numpy array of shape [N]
        iou_threshold: IOU 阈值

    Returns:
        保留的框的索引
    """
    if len(boxes) == 0:
        return np.array([])

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep)


def draw_detections(image, boxes, scores, class_ids, class_names=None):
    """
    在图像上绘制检测结果

    Args:
        image: numpy array (BGR format)
        boxes: numpy array of shape [N, 4]
        scores: numpy array of shape [N]
        class_ids: numpy array of shape [N]
        class_names: 类别名称列表

    Returns:
        绘制了检测框的图像
    """
    img = image.copy()

    # 定义颜色
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        x1, y1, x2, y2 = box.astype(int)
        color = colors[int(class_id) % len(colors)]

        # 绘制边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 准备标签文本
        if class_names and class_id < len(class_names):
            label = f"{class_names[int(class_id)]}: {score:.2f}"
        else:
            label = f"Class {int(class_id)}: {score:.2f}"

        # 绘制标签背景
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            img,
            (x1, y1 - label_h - baseline - 5),
            (x1 + label_w, y1),
            color,
            -1,
        )

        # 绘制标签文本
        cv2.putText(
            img,
            label,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return img


def scale_boxes(boxes, original_size, model_size):
    """
    将模型输出的坐标缩放回原始图像尺寸

    Args:
        boxes: numpy array of shape [N, 4], 坐标在模型尺寸下
        original_size: 原始图像尺寸 (width, height)
        model_size: 模型输入尺寸

    Returns:
        缩放后的坐标
    """
    if len(boxes) == 0:
        return boxes

    scale_x = original_size[0] / model_size
    scale_y = original_size[1] / model_size

    scaled_boxes = boxes.copy()
    scaled_boxes[:, [0, 2]] *= scale_x  # x1, x2
    scaled_boxes[:, [1, 3]] *= scale_y  # y1, y2

    return scaled_boxes


def process_video(
    onnx_path,
    video_path,
    output_path,
    imgsz=640,
    conf_threshold=0.25,
    iou_threshold=0.45,
    apply_nms=False,
    class_names=None,
):
    """
    处理视频文件，对每一帧进行推理并生成结果视频

    Args:
        onnx_path: ONNX 模型路径
        video_path: 输入视频路径
        output_path: 输出视频路径
        imgsz: 输入图像尺寸
        conf_threshold: 置信度阈值
        iou_threshold: NMS IOU 阈值
        apply_nms: 是否应用 NMS
        class_names: 类别名称列表
    """
    print(f"\n{'='*60}")
    print(f"Processing video: {video_path}")
    print(f"Using ONNX model: {onnx_path}")
    print(f"{'='*60}\n")

    # 创建 ONNX Runtime 会话
    session = ort.InferenceSession(str(onnx_path))

    # 获取输入输出信息
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(f"Input name: {input_name}")
    print(f"Output name: {output_name}\n")

    # 打开视频文件
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Video properties:")
    print(f"  FPS: {fps}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Total frames: {total_frames}\n")

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    processed_frames = 0
    first_frame = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 预处理帧
        img_array, original_size = preprocess_image(frame, imgsz)

        # 运行推理
        start_time = time.time()
        outputs = session.run([output_name], {input_name: img_array})
        inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
        output = outputs[0]

        # 打印推理时间
        print(f"Frame {frame_count}: Inference time: {inference_time:.2f} ms")

        # 后处理（第一帧显示详细信息）
        boxes, scores, class_ids = postprocess_detections(
            output, conf_threshold=conf_threshold, verbose=first_frame
        )

        if first_frame:
            first_frame = False

        # 应用 NMS（如果需要）
        if apply_nms and len(boxes) > 0:
            # 按类别分别进行 NMS
            unique_classes = np.unique(class_ids)
            keep_indices = []

            for cls in unique_classes:
                cls_mask = class_ids == cls
                cls_boxes = boxes[cls_mask]
                cls_scores = scores[cls_mask]
                cls_indices = np.where(cls_mask)[0]

                cls_keep = nms_boxes(cls_boxes, cls_scores, iou_threshold)
                keep_indices.extend(cls_indices[cls_keep])

            keep_indices = np.array(keep_indices)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            class_ids = class_ids[keep_indices]

        # 缩放坐标回原始图像尺寸
        boxes_scaled = scale_boxes(boxes, original_size, imgsz)

        # 绘制检测结果
        result_frame = draw_detections(frame, boxes_scaled, scores, class_ids, class_names)

        # 写入输出视频
        out.write(result_frame)

        processed_frames += 1

        # 打印进度
        if frame_count % 100 == 0 or frame_count == total_frames:
            print(f"Processed {frame_count}/{total_frames} frames")

    # 释放资源
    cap.release()
    out.release()

    print(f"\n{'='*60}")
    print("Video processing completed!")
    print(f"Output video saved to: {output_path}")
    print(f"Total frames processed: {processed_frames}")
    print(f"{'='*60}\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference on video using ONNX model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the ONNX model",
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the input video file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output video file (default: input_video_result.mp4)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for inference (default: 640)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for detection (default: 0.25)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
        help="IOU threshold for NMS (default: 0.45)",
    )
    parser.add_argument(
        "--apply-nms",
        action="store_true",
        default=True,
        help="Apply NMS during inference (for models without NMS)",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        nargs="+",
        default=None,
        help="Class names list (optional)",
    )
    return parser.parse_args()


def main():
    """Main function to process video with ONNX model."""
    args = parse_args()

    # 设置默认输出路径
    if args.output is None:
        video_path = Path(args.video)
        args.output = video_path.parent / f"{video_path.stem}_result.mp4"

    # 处理视频
    process_video(
        onnx_path=args.model,
        video_path=args.video,
        output_path=args.output,
        imgsz=args.imgsz,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        apply_nms=args.apply_nms,
        class_names=args.class_names,
    )


if __name__ == "__main__":
    main()
