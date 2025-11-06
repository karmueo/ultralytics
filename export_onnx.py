import argparse
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
import cv2
from ultralytics import YOLO


def preprocess_image(image_path, imgsz=640):
    """
    预处理图像用于 ONNX 推理

    Args:
        image_path: 图像路径
        imgsz: 输入图像尺寸

    Returns:
        处理后的图像数组 [1, 3, H, W], 处理后的图像, 原始图像尺寸
    """
    img = Image.open(image_path)
    original_size = img.size  # (width, height)
    img = img.convert("RGB")
    img_resized = img.resize((imgsz, imgsz))

    # 转换为 numpy 数组并归一化
    img_array = np.array(img_resized).astype(np.float32) / 255.0

    # 转换维度从 [H, W, C] 到 [C, H, W]
    img_array = img_array.transpose(2, 0, 1)

    # 添加 batch 维度 [1, C, H, W]
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, img_resized, original_size


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


def postprocess_detections(output, conf_threshold=0.25, img_shape=(640, 640)):
    """
    后处理 ONNX 模型输出

    Args:
        output: ONNX 模型输出
        conf_threshold: 置信度阈值
        img_shape: 图像尺寸

    Returns:
        检测结果列表
    """
    # 检查输出形状
    print(f"Raw output shape: {output.shape}")

    # 如果是 YOLOv5 格式 [1, 6, 8400]，转换为 [1, 8400, 6]
    if len(output.shape) == 3 and output.shape[1] < output.shape[2]:
        output = output.transpose(0, 2, 1)
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
        image: PIL Image 或 numpy array
        boxes: numpy array of shape [N, 4]
        scores: numpy array of shape [N]
        class_ids: numpy array of shape [N]
        class_names: 类别名称列表

    Returns:
        绘制了检测框的图像
    """
    # 转换为 numpy array
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image.copy()

    # 转换为 BGR (OpenCV 格式)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

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

    # 转换回 RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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


def test_onnx_model(
    onnx_path,
    image_path,
    imgsz=640,
    conf_threshold=0.25,
    iou_threshold=0.45,
    apply_nms=False,
    save_img=True,
    class_names=None,
):
    """
    测试导出的 ONNX 模型

    Args:
        onnx_path: ONNX 模型路径
        image_path: 测试图像路径
        imgsz: 输入图像尺寸
        conf_threshold: 置信度阈值
        iou_threshold: NMS IOU 阈值
        apply_nms: 是否应用 NMS
        save_img: 是否保存结果图像
        class_names: 类别名称列表
    """
    print(f"\n{'='*60}")
    print(f"Testing ONNX model: {onnx_path}")
    print(f"{'='*60}\n")

    # 创建 ONNX Runtime 会话
    session = ort.InferenceSession(str(onnx_path))

    # 获取输入输出信息
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_shape = session.get_outputs()[0].shape

    print(f"Input name: {input_name}")
    print(f"Input shape: {input_shape}")
    print(f"Output name: {output_name}")
    print(f"Output shape: {output_shape}\n")

    # 预处理图像
    img_array, resized_img, original_size = preprocess_image(image_path, imgsz)
    print(f"Original image size: {original_size} (W x H)")
    print(f"Preprocessed image shape: {img_array.shape}")

    # 检查模型的批次大小并调整输入
    model_batch_size = input_shape[0]
    if isinstance(model_batch_size, int) and model_batch_size > 1:
        print(f"\nModel expects batch size: {model_batch_size}")
        print(f"Repeating input image to match batch size...")
        # 重复图像以匹配批次大小
        img_array = np.repeat(img_array, model_batch_size, axis=0)
        print(f"Adjusted input shape: {img_array.shape}")

    # 运行推理
    print("Running inference...")
    outputs = session.run([output_name], {input_name: img_array})
    output = outputs[0]

    # 如果是批次输出，只处理第一个图像的结果
    if len(output.shape) > 2 and output.shape[0] > 1:
        print(f"Model output batch size: {output.shape[0]}, using first image result")
        output = output[0:1]  # 保持第一个维度为 1

    # 后处理
    boxes, scores, class_ids = postprocess_detections(
        output, conf_threshold=conf_threshold
    )

    # 应用 NMS（如果需要且模型输出不包含 NMS）
    if apply_nms and len(boxes) > 0:
        print(f"\nApplying NMS with IOU threshold: {iou_threshold}")
        print(f"Detections before NMS: {len(boxes)}")

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

        print(f"Detections after NMS: {len(boxes)}")

    # 缩放坐标回原始图像尺寸
    boxes_scaled = scale_boxes(boxes, original_size, imgsz)

    # 打印结果
    print(f"\n{'='*60}")
    print("Detection Results (Original Image Coordinates):")
    print(f"{'='*60}")
    print(f"Total detections (confidence > {conf_threshold}): {len(boxes)}\n")

    if len(boxes) > 0:
        for i, (box, score, class_id) in enumerate(
            zip(boxes_scaled, scores, class_ids)
        ):
            print(f"Detection {i+1}:")
            print(f"  Class ID: {class_id}")
            print(f"  Confidence: {score:.4f}")
            print(
                f"  Bounding Box (original size): "
                f"[{box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}, {box[3]:.2f}]"
            )
            print()
    else:
        print("No detections found.\n")

    # 保存结果图像（使用缩放后的坐标）
    if save_img and len(boxes) > 0:
        # 加载原始图像用于绘制
        original_img = Image.open(image_path).convert("RGB")
        result_img = draw_detections(
            original_img, boxes_scaled, scores, class_ids, class_names
        )

        # 生成输出文件名
        img_path = Path(image_path)
        onnx_name = Path(onnx_path).stem
        output_path = img_path.parent / f"{img_path.stem}_{onnx_name}_result.jpg"

        # 保存图像
        result_img_pil = Image.fromarray(result_img)
        result_img_pil.save(output_path)
        print(f"Result image saved to: {output_path}\n")

    return boxes_scaled, scores, class_ids


def rename_onnx_input(onnx_path, old_name="images", new_name="input"):
    """
    重命名 ONNX 模型的输入名称

    Args:
        onnx_path: ONNX 模型路径
        old_name: 原始输入名称
        new_name: 新的输入名称

    Returns:
        修改后的模型路径
    """
    import onnx

    print(f"\nRenaming ONNX input from '{old_name}' to '{new_name}'...")

    # 加载模型
    model = onnx.load(str(onnx_path))

    # 修改输入名称
    for input_tensor in model.graph.input:
        if input_tensor.name == old_name:
            input_tensor.name = new_name
            print(f"✅ Input renamed: {old_name} -> {new_name}")
            break

    # 修改图中所有引用该输入的节点
    for node in model.graph.node:
        for i, input_name in enumerate(node.input):
            if input_name == old_name:
                node.input[i] = new_name

    # 保存修改后的模型
    onnx.save(model, str(onnx_path))
    print("✅ Model saved successfully!")

    return onnx_path


def transpose_onnx_output(onnx_path):
    """
    修改 ONNX 模型输出格式从 [1, 6, 8400] 到 [1, 8400, 6]

    Args:
        onnx_path: ONNX 模型路径

    Returns:
        修改后的模型路径
    """
    import onnx
    from onnx import helper

    print("\nTransposing ONNX output from [1, 6, 8400] to [1, 8400, 6]...")

    # 加载模型
    model = onnx.load(str(onnx_path))

    # 获取原始输出信息
    original_output = model.graph.output[0]
    output_name = original_output.name

    # 创建新的输出名称
    transposed_output_name = output_name + "_transposed"

    # 添加转置节点
    transpose_node = helper.make_node(
        "Transpose",
        inputs=[output_name],
        outputs=[transposed_output_name],
        perm=[0, 2, 1],  # [1, 6, 8400] -> [1, 8400, 6]
    )

    # 添加节点到图中
    model.graph.node.append(transpose_node)

    # 更新输出
    # 移除原始输出
    model.graph.output.remove(original_output)

    # 创建新的输出（维度为 [1, 8400, 6]）
    new_output = helper.make_tensor_value_info(
        transposed_output_name, original_output.type.tensor_type.elem_type, [1, 8400, 6]
    )
    model.graph.output.append(new_output)

    # 保存修改后的模型
    onnx.save(model, str(onnx_path))

    print("✅ Model output transposed successfully!")
    print("   New output shape: [1, 8400, 6]")

    return onnx_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX format")
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Path to the YOLO model (default: yolo11n.pt)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Optimize the ONNX model for inference",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=[640],
        help="Image size for export (default: 640). "
        "Can be single value or two values [height, width]",
    )
    parser.add_argument(
        "--dynamic", action="store_true", help="Enable dynamic input shapes"
    )
    parser.add_argument(
        "--simplify", action="store_true", help="Simplify the ONNX model"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: None, uses model default)",
    )
    parser.add_argument(
        "--nms", action="store_true", help="Add NMS module to ONNX model"
    )
    parser.add_argument(
        "--apply-nms",
        action="store_true",
        help="Apply NMS during ONNX inference testing " "(for models without NMS)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
        help="IOU threshold for NMS (default: 0.45)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for detection (default: 0.25)",
    )
    parser.add_argument(
        "--save-img",
        action="store_true",
        default=True,
        help="Save detection result image (default: True)",
    )
    parser.add_argument(
        "--transpose-output",
        action="store_true",
        help="Transpose output from [1, 6, 8400] to [1, 8400, 6]",
    )
    parser.add_argument(
        "--rename-input",
        type=str,
        default=None,
        help="Rename ONNX input (e.g., 'input' to rename from 'images' to 'input')",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size for export (default: 1)",
    )
    return parser.parse_args()


def main():
    """Main function to export YOLO model to ONNX."""
    args = parse_args()

    # Load the YOLO model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Prepare export parameters
    export_kwargs = {
        "format": "onnx",
        "optimize": args.optimize,
        "dynamic": args.dynamic,
        "simplify": args.simplify,
        "nms": args.nms,
        "batch": args.batch,
    }

    # Handle imgsz parameter
    if len(args.imgsz) == 1:
        export_kwargs["imgsz"] = args.imgsz[0]
    else:
        export_kwargs["imgsz"] = args.imgsz

    # Add opset if specified
    if args.opset is not None:
        export_kwargs["opset"] = args.opset

    # Export the model to ONNX format
    print(f"Exporting model to ONNX format with parameters: {export_kwargs}")
    model.export(**export_kwargs)

    # Get the exported ONNX model path
    model_path = Path(args.model)
    onnx_path = model_path.with_suffix(".onnx")

    print(f"Model exported successfully to: {onnx_path}")

    # Rename input if requested
    if args.rename_input:
        onnx_path = rename_onnx_input(onnx_path, old_name="images", new_name=args.rename_input)

    # Transpose output if requested
    if args.transpose_output:
        onnx_path = transpose_onnx_output(onnx_path)

    # 如果没有重命名输入，则使用 Ultralytics YOLO 包装器测试
    if not args.rename_input:
        # Load the exported ONNX model
        print(f"Loading exported ONNX model: {onnx_path}")
        onnx_model = YOLO(str(onnx_path))

        # Run inference with Ultralytics YOLO
        print("\n" + "=" * 60)
        print("Testing with Ultralytics YOLO wrapper:")
        print("=" * 60)
        results = onnx_model("uav.jpg")
        print(f"Inference completed. Detected {len(results[0].boxes)} objects.")

        # 打印详细结果
        if len(results[0].boxes) > 0:
            for i, box in enumerate(results[0].boxes):
                print(f"\nDetection {i+1}:")
                print(f"  Class ID: {int(box.cls[0])}")
                print(f"  Confidence: {float(box.conf[0]):.4f}")
                print(f"  Bounding Box: {box.xyxy[0].tolist()}")
    else:
        print("\n" + "=" * 60)
        print("⚠️  Input renamed: Skipping Ultralytics YOLO wrapper test")
        print("   (Ultralytics expects input name 'images')")
        print("=" * 60)

    # 使用原生 ONNX Runtime 进行测试
    # 检查是否需要自动启用 NMS（对于没有内置 NMS 的模型）
    auto_apply_nms = args.apply_nms
    if not args.nms and not args.apply_nms:
        # 如果导出时没有使用 --nms，则自动启用 NMS
        auto_apply_nms = True
        print("\n⚠️  Model exported without NMS. " "Auto-enabling NMS for testing.")

    test_onnx_model(
        onnx_path=onnx_path,
        image_path="uav.jpg",
        imgsz=args.imgsz[0] if len(args.imgsz) == 1 else args.imgsz[0],
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        apply_nms=auto_apply_nms,
        save_img=args.save_img,
        class_names=None,  # 可以传入类别名称列表，如 ['class0', 'class1']
    )


if __name__ == "__main__":
    # 用法：python3 export_onnx.py --model runs/detect/yolov11m_110_rgb_640_v7.1/weights/best.pt --dynamic --opset 11 --save-img
    main()
