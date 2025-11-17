import argparse
from pathlib import Path
import time
import cv2
import numpy as np
from ultralytics import YOLO
import sys
try:
    from tqdm import tqdm
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tqdm'])
    from tqdm import tqdm


def get_video_files(input_path):
    """
    获取视频文件列表
    
    Args:
        input_path: 输入路径，可以是文件或文件夹
    
    Returns:
        list: 视频文件路径列表
    """
    input_path = Path(input_path)
    
    # 支持的视频格式
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm'}
    
    if input_path.is_file():
        # 如果是文件，检查是否是视频文件
        if input_path.suffix.lower() in video_extensions:
            return [input_path]
        else:
            raise ValueError(f"输入文件不是支持的视频格式: {input_path}")
    
    elif input_path.is_dir():
        # 如果是文件夹，查找所有视频文件
        video_files = []
        for ext in video_extensions:
            video_files.extend(input_path.glob(f'*{ext}'))
            video_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        if not video_files:
            raise ValueError(f"在文件夹中未找到视频文件: {input_path}")
        
        # 按文件名排序
        video_files.sort()
        return video_files
    
    else:
        raise ValueError(f"输入路径不存在: {input_path}")


def calculate_crop_params(frame_height, frame_width, overlap_ratio=0.1):
    """
    计算子图的裁剪参数
    
    Args:
        frame_height: 视频帧高度
        frame_width: 视频帧宽度
        overlap_ratio: 重叠率 (默认0.1)
    
    Returns:
        dict: 包含5个子图的裁剪坐标和中心点
    """
    # 计算子图尺寸 (考虑重叠)
    # 对于上下两个子图: h1 + h2 - overlap*h = H
    # 如果h1 = h2 = h, 则 2h - overlap*h = H, h = H / (2 - overlap)
    sub_height = int(frame_height / (2 - overlap_ratio))
    sub_width = int(frame_width / (2 - overlap_ratio))
    
    crops = {}
    
    # 左上 (top-left)
    crops['top_left'] = {
        'x1': 0,
        'y1': 0,
        'x2': sub_width,
        'y2': sub_height,
        'center_x': sub_width // 2,
        'center_y': sub_height // 2
    }
    
    # 左下 (bottom-left)
    y1_bottom = frame_height - sub_height
    crops['bottom_left'] = {
        'x1': 0,
        'y1': y1_bottom,
        'x2': sub_width,
        'y2': frame_height,
        'center_x': sub_width // 2,
        'center_y': y1_bottom + sub_height // 2
    }
    
    # 右上 (top-right)
    x1_right = frame_width - sub_width
    crops['top_right'] = {
        'x1': x1_right,
        'y1': 0,
        'x2': frame_width,
        'y2': sub_height,
        'center_x': x1_right + sub_width // 2,
        'center_y': sub_height // 2
    }
    
    # 右下 (bottom-right)
    crops['bottom_right'] = {
        'x1': x1_right,
        'y1': y1_bottom,
        'x2': frame_width,
        'y2': frame_height,
        'center_x': x1_right + sub_width // 2,
        'center_y': y1_bottom + sub_height // 2
    }
    
    # 中心 (center) - 与左上子图尺寸相同，中心点对齐视频帧中心
    center_x = frame_width // 2
    center_y = frame_height // 2
    half_w = sub_width // 2
    half_h = sub_height // 2
    
    crops['center'] = {
        'x1': max(0, center_x - half_w),
        'y1': max(0, center_y - half_h),
        'x2': min(frame_width, center_x + half_w),
        'y2': min(frame_height, center_y + half_h),
        'center_x': center_x,
        'center_y': center_y
    }
    
    return crops, sub_width, sub_height


def crop_frame(frame, crops, num_crops=5):
    """
    将帧裁剪成子图
    
    Args:
        frame: 输入帧
        crops: 裁剪参数字典
        num_crops: 子图数量 (4 或 5)
    
    Returns:
        list: 裁剪后的子图列表
        list: 对应的crop信息列表
    """
    crop_order = ['top_left', 'bottom_left', 'top_right', 'bottom_right', 'center']
    if num_crops == 4:
        crop_order = crop_order[:4]  # 只使用前4个
    
    cropped_frames = []
    crop_infos = []
    
    for name in crop_order:
        crop = crops[name]
        cropped = frame[crop['y1']:crop['y2'], crop['x1']:crop['x2']]
        cropped_frames.append(cropped)
        crop_infos.append(crop)
    
    return cropped_frames, crop_infos


def global_nms_with_center_priority(detections, iou_threshold=0.5):
    """
    对所有检测结果进行全局NMS，优先保留距离子图中心更近的检测框
    
    Args:
        detections: list of dict, 每个dict包含:
            - boxes: xyxy坐标 (N, 4)
            - scores: 置信度 (N,)
            - classes: 类别 (N,)
            - crop_info: 子图信息
        iou_threshold: IoU阈值
    
    Returns:
        dict: 合并后的检测结果
    """
    if not detections:
        return {'boxes': np.array([]), 'scores': np.array([]), 'classes': np.array([])}
    
    # 收集所有检测框
    all_boxes = []
    all_scores = []
    all_classes = []
    all_distances = []  # 到子图中心的距离
    
    for det in detections:
        if len(det['boxes']) == 0:
            continue
        
        boxes = det['boxes']  # (N, 4) xyxy格式
        scores = det['scores']  # (N,)
        classes = det['classes']  # (N,)
        crop_info = det['crop_info']
        
        # 计算每个框的中心到子图中心的距离
        box_centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
        box_centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
        distances = np.sqrt(
            (box_centers_x - crop_info['center_x']) ** 2 +
            (box_centers_y - crop_info['center_y']) ** 2
        )
        
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_classes.append(classes)
        all_distances.append(distances)
    
    if not all_boxes:
        return {'boxes': np.array([]), 'scores': np.array([]), 'classes': np.array([])}
    
    # 合并所有检测结果
    all_boxes = np.vstack(all_boxes)
    all_scores = np.concatenate(all_scores)
    all_classes = np.concatenate(all_classes)
    all_distances = np.concatenate(all_distances)
    
    # 执行全局NMS (不分类别)
    keep_indices = []
    
    # 按置信度排序
    order = all_scores.argsort()[::-1]
    
    while len(order) > 0:
        # 保留当前最高置信度的框
        i = order[0]
        keep_indices.append(i)
        
        if len(order) == 1:
            break
        
        # 计算IoU
        ious = calculate_iou(all_boxes[i:i+1], all_boxes[order[1:]])
        
        # 找出IoU大于阈值的框
        overlap_mask = ious[0] > iou_threshold
        overlap_indices = order[1:][overlap_mask]
        
        # 对于重叠的框，比较距离中心的远近
        # 如果当前框距离中心更远，则替换
        if len(overlap_indices) > 0:
            current_distance = all_distances[i]
            overlap_distances = all_distances[overlap_indices]
            
            # 找到距离更近的框
            closer_mask = overlap_distances < current_distance
            if np.any(closer_mask):
                # 有更近的框，移除当前框，保留更近的框
                keep_indices.pop()
                # 找到最近的那个框
                closest_idx = overlap_indices[np.argmin(overlap_distances)]
                keep_indices.append(closest_idx)
                
                # 更新order，移除当前框和所有与最近框重叠的框
                ious_with_closest = calculate_iou(
                    all_boxes[closest_idx:closest_idx+1],
                    all_boxes[order[1:]]
                )[0]
                remove_mask = np.zeros(len(order) - 1, dtype=bool)
                remove_mask[0] = True  # 移除原来的i
                remove_mask |= (ious_with_closest > iou_threshold)
                order = order[1:][~remove_mask]
            else:
                # 当前框更近，移除重叠的框
                remove_mask = np.zeros(len(order) - 1, dtype=bool)
                remove_mask[overlap_mask] = True
                order = order[1:][~remove_mask]
        else:
            # 没有重叠，继续下一个
            order = order[1:]
    
    keep_indices = np.array(keep_indices)
    
    return {
        'boxes': all_boxes[keep_indices],
        'scores': all_scores[keep_indices],
        'classes': all_classes[keep_indices]
    }


def calculate_iou(boxes1, boxes2):
    """
    计算两组框的IoU
    
    Args:
        boxes1: (N, 4) xyxy格式
        boxes2: (M, 4) xyxy格式
    
    Returns:
        iou: (N, M) IoU矩阵
    """
    # 扩展维度以便广播
    boxes1 = boxes1[:, None, :]  # (N, 1, 4)
    boxes2 = boxes2[None, :, :]  # (1, M, 4)
    
    # 计算交集
    x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
    y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
    x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
    y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # 计算各自面积
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    
    # 计算并集
    union = area1 + area2 - intersection
    
    # 计算IoU
    iou = intersection / (union + 1e-6)
    
    return iou


def process_video_split(
    model_path,
    video_path,
    output_path,
    overlap_ratio=0.1,
    conf_threshold=0.25,
    nms_iou_threshold=0.5,
    show_fps=True,
    num_crops=5,
    save_mismatch=False,
    mismatch_classes=None,
    mismatch_output_dir=None,
    video_prefix="",
    show_detail_log=False,
):
    """
    使用子图切分方式处理视频文件
    
    Args:
        model_path: YOLO模型路径
        video_path: 输入视频路径
        output_path: 输出视频路径
        overlap_ratio: 子图重叠率
        conf_threshold: 检测置信度阈值
        nms_iou_threshold: NMS的IoU阈值
        show_fps: 是否显示FPS
        num_crops: 子图数量 (4 或 5)
        save_mismatch: 是否保存检测类别与指定类别不一致的帧
        mismatch_classes: 指定类别列表（类别名称），用于比对检测结果
        mismatch_output_dir: 保存不一致帧的输出目录
        video_prefix: 视频文件名前缀，用于生成唯一的保存文件名
        show_detail_log: 是否显示每帧的详细日志 (推理时间、检测结果)
    """
    print(f"\n{'='*60}")
    print(f"Processing video with split inference: {video_path}")
    print(f"Using model: {model_path}")
    print(f"Overlap ratio: {overlap_ratio}")
    print(f"{'='*60}\n")
    
    # 加载模型
    print("Loading YOLO model...")
    model = YOLO(model_path)
    print("Model loaded successfully!\n")
    
    # 验证num_crops参数
    if num_crops not in [4, 5]:
        raise ValueError(f"错误: num_crops 必须是 4 或 5, 当前值: {num_crops}")
    
    print(f"Using {num_crops} crops per frame\n")
    
    # 打开视频
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
    
    # 计算裁剪参数
    crops, sub_width, sub_height = calculate_crop_params(height, width, overlap_ratio)
    print(f"Sub-image size: {sub_width}x{sub_height}\n")
    print("Crop regions:")
    for name, crop in crops.items():
        print(f"  {name}: ({crop['x1']}, {crop['y1']}) -> ({crop['x2']}, {crop['y2']})")
    print()
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # 创建保存不一致帧的文件夹
    mismatch_raw_frames_dir = None
    mismatch_raw_crops_dir = None
    mismatch_annotated_crops_dir = None
    mismatch_count = 0
    
    if save_mismatch:
        if not mismatch_classes:
            raise ValueError("启用 save_mismatch 时必须提供 mismatch_classes 参数")
        
        # 如果未指定输出目录，使用视频文件所在目录
        if mismatch_output_dir is None:
            video_dir = Path(video_path).parent
            mismatch_output_dir = video_dir / "mismatch_frames"
        else:
            mismatch_output_dir = Path(mismatch_output_dir)
        
        # 创建三个子目录
        mismatch_raw_frames_dir = mismatch_output_dir / "raw_frames"
        mismatch_raw_crops_dir = mismatch_output_dir / "raw_crops"
        mismatch_annotated_crops_dir = mismatch_output_dir / "annotated_crops"
        
        mismatch_raw_frames_dir.mkdir(parents=True, exist_ok=True)
        mismatch_raw_crops_dir.mkdir(parents=True, exist_ok=True)
        mismatch_annotated_crops_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Mismatch detection enabled. Expected classes: {mismatch_classes}")
        print(f"Raw frames will be saved to: {mismatch_raw_frames_dir}")
        print(f"Raw crops will be saved to: {mismatch_raw_crops_dir}")
        print(f"Annotated crops will be saved to: {mismatch_annotated_crops_dir}\n")
    
    frame_count = 0
    processed_frames = 0
    total_inference_time = 0.0

    # tqdm帧进度条
    pbar = tqdm(total=total_frames, desc=f"{Path(video_path).name}", unit="frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 裁剪帧
        cropped_frames, crop_infos = crop_frame(frame, crops, num_crops)
        
        # 批量推理
        start_time = time.time()
        
        # 收集所有检测结果
        detections = []
        
        # 使用batch推理
        results = model(cropped_frames, conf=conf_threshold, verbose=False)
        
        for i, result in enumerate(results):
            if len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4)
                scores = result.boxes.conf.cpu().numpy()  # (N,)
                classes = result.boxes.cls.cpu().numpy()  # (N,)
                
                # 将坐标转换回原始帧坐标系
                crop_info = crop_infos[i]
                boxes[:, 0] += crop_info['x1']  # x1
                boxes[:, 1] += crop_info['y1']  # y1
                boxes[:, 2] += crop_info['x1']  # x2
                boxes[:, 3] += crop_info['y1']  # y2
                
                detections.append({
                    'boxes': boxes,
                    'scores': scores,
                    'classes': classes,
                    'crop_info': crop_info
                })
        
        # 全局NMS
        merged_result = global_nms_with_center_priority(detections, nms_iou_threshold)
        
        inference_time = (time.time() - start_time) * 1000
        total_inference_time += inference_time
        
        # 打印每帧的详细日志
        if show_detail_log:
            print(f"Frame {frame_count}: Inference time: {inference_time:.2f} ms, "
                  f"Detections: {len(merged_result['boxes'])}")
        
        # 绘制结果
        annotated_frame = frame.copy()
        
        for i in range(len(merged_result['boxes'])):
            box = merged_result['boxes'][i]
            score = merged_result['scores'][i]
            cls = int(merged_result['classes'][i])
            
            # 获取类别名称
            class_name = results[0].names[cls] if hasattr(results[0], 'names') else str(cls)
            
            # 绘制边界框
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"{class_name} {score:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            label_y = max(y1 - 10, label_size[1] + 10)
            
            cv2.rectangle(
                annotated_frame,
                (x1, label_y - label_size[1] - 5),
                (x1 + label_size[0], label_y + 5),
                (0, 255, 0),
                -1
            )
            cv2.putText(
                annotated_frame,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )
        
        # 显示FPS
        if show_fps:
            avg_fps = 1000.0 / (total_inference_time / processed_frames) if processed_frames > 0 else 0
            fps_text = f"FPS: {avg_fps:.1f}"
            cv2.putText(
                annotated_frame,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
        
        # 检查是否需要保存不一致的帧
        if (save_mismatch and mismatch_classes and mismatch_raw_frames_dir and 
            mismatch_raw_crops_dir and mismatch_annotated_crops_dir and 
            len(merged_result['boxes']) > 0):
            # 检查检测到的类别是否在指定类别列表中
            detected_classes = []
            for i in range(len(merged_result['boxes'])):
                cls = int(merged_result['classes'][i])
                class_name = results[0].names[cls] if hasattr(results[0], 'names') else str(cls)
                detected_classes.append(class_name)
            
            detected_classes = list(set(detected_classes))  # 去重
            has_mismatch = False
            
            for det_class in detected_classes:
                if det_class not in mismatch_classes:
                    has_mismatch = True
                    break
            
            if has_mismatch:
                # 保存原始帧
                frame_filename = f"frame_{video_prefix}_{frame_count:06d}.jpg"
                raw_frame_path = mismatch_raw_frames_dir / frame_filename
                cv2.imwrite(str(raw_frame_path), frame)
                
                # 保存有目标的原始子图和标注子图
                crop_order = ['top_left', 'bottom_left', 'top_right', 'bottom_right', 'center']
                if num_crops == 4:
                    crop_order = crop_order[:4]
                
                for idx, result in enumerate(results):
                    if len(result.boxes) > 0:
                        crop_name = crop_order[idx]
                        
                        # 保存原始子图
                        crop_filename = f"frame_{video_prefix}_{frame_count:06d}_{crop_name}.jpg"
                        raw_crop_path = mismatch_raw_crops_dir / crop_filename
                        cv2.imwrite(str(raw_crop_path), cropped_frames[idx])
                        
                        # 绘制并保存标注子图
                        annotated_crop = result.plot()
                        annotated_crop_path = mismatch_annotated_crops_dir / crop_filename
                        cv2.imwrite(str(annotated_crop_path), annotated_crop)
                
                mismatch_count += 1
                print(f"  -> Mismatch detected! Detected classes: {detected_classes}. Saved frame {frame_count}")
        
        # 写入输出视频
        out.write(annotated_frame)
        processed_frames += 1
        pbar.update(1)
    
    pbar.close()
    
    # 释放资源
    cap.release()
    out.release()
    
    # 统计信息
    avg_inference_time = total_inference_time / processed_frames if processed_frames > 0 else 0
    avg_fps = 1000.0 / avg_inference_time if avg_inference_time > 0 else 0
    
    print(f"\n{'='*60}")
    print("Video processing completed!")
    print(f"Output video saved to: {output_path}")
    print(f"Total frames processed: {processed_frames}")
    print(f"Average inference time: {avg_inference_time:.2f} ms per frame")
    print(f"Average FPS: {avg_fps:.1f}")
    if save_mismatch:
        print(f"Mismatch frames detected: {mismatch_count}")
        if mismatch_count > 0:
            print(f"Raw frames saved to: {mismatch_raw_frames_dir}")
            print(f"Raw crops saved to: {mismatch_raw_crops_dir}")
            print(f"Annotated crops saved to: {mismatch_annotated_crops_dir}")
    print(f"{'='*60}\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run YOLO inference on video with split crops"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the YOLO model (.pt file)",
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the input video file or directory containing video files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output video file (default: input_video_split_result.mp4)",
    )
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=0.01,
        help="Overlap ratio for crops (default: 0.1)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for detection (default: 0.25)",
    )
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=0.01,
        help="IoU threshold for global NMS (default: 0.5)",
    )
    parser.add_argument(
        "--show-fps",
        action="store_true",
        default=True,
        help="Show FPS on the output video (default: True)",
    )
    parser.add_argument(
        "--num-crops",
        type=int,
        choices=[4, 5],
        default=5,
        help="Number of crops (4 or 5, default: 5)",
    )
    parser.add_argument(
        "--save-mismatch",
        action="store_true",
        default=False,
        help="Save frames where detected classes don't match expected classes",
    )
    parser.add_argument(
        "--mismatch-classes",
        type=str,
        nargs="+",
        default=None,
        help="Expected class names for mismatch detection (e.g., 'person' 'car')",
    )
    parser.add_argument(
        "--mismatch-output-dir",
        type=str,
        default=None,
        help="Output directory for saving mismatch frames (default: video_dir/mismatch_frames)",
    )
    parser.add_argument(
        "--show-detail-log",
        action="store_true",
        default=False,
        help="Show detailed log for each frame (inference time, detections)",
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # 检查num_crops参数
    if args.num_crops not in [4, 5]:
        print(f"错误: num_crops 必须是 4 或 5, 当前值: {args.num_crops}")
        return
    
    # 获取视频文件列表
    try:
        video_files = get_video_files(args.video)
    except ValueError as e:
        print(f"错误: {e}")
        return
    
    print(f"\n找到 {len(video_files)} 个视频文件\n")
    
    # 如果启用mismatch保存且输入是文件夹,使用统一的输出目录
    if args.save_mismatch and len(video_files) > 1:
        if args.mismatch_output_dir is None:
            # 使用输入文件夹作为基础目录
            input_path = Path(args.video)
            if input_path.is_dir():
                args.mismatch_output_dir = str(input_path / "mismatch_frames")
            else:
                args.mismatch_output_dir = str(input_path.parent / "mismatch_frames")
    
    # 创建输出视频文件夹，与输入同级
    if len(video_files) > 0:
        first_video = video_files[0]
        out_dir = Path(first_video).parent / "output_videos"
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path("output_videos")
        out_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理每个视频文件
    for idx, video_file in enumerate(tqdm(video_files, desc='All Videos', unit='video'), 1):
        print(f"\n{'='*80}")
        print(f"处理视频 {idx}/{len(video_files)}: {video_file.name}")
        print(f"{'='*80}\n")
        
        # 设置输出路径
        if args.output is None or len(video_files) > 1:
            output_path = out_dir / f"{video_file.stem}_split_result.mp4"
        else:
            output_path = args.output
        
        # 为每个视频生成唯一的文件名前缀(用于mismatch保存)
        video_prefix = video_file.stem
        
        try:
            # 处理视频
            process_video_split(
                model_path=args.model,
                video_path=str(video_file),
                output_path=str(output_path),
                overlap_ratio=args.overlap_ratio,
                conf_threshold=args.conf_threshold,
                nms_iou_threshold=args.nms_iou_threshold,
                show_fps=args.show_fps,
                num_crops=args.num_crops,
                save_mismatch=args.save_mismatch,
                mismatch_classes=args.mismatch_classes,
                mismatch_output_dir=args.mismatch_output_dir,
                video_prefix=video_prefix,
                show_detail_log=args.show_detail_log,
            )
        except Exception as e:
            print(f"\n错误: 处理视频 {video_file} 时发生错误: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"所有视频处理完成! 共处理 {len(video_files)} 个视频")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

"""
使用示例:

# 处理单个视频文件
python video_pt_inference_split.py \
    --model yolo11n.pt \
    --video input.mp4

# 处理文件夹中的所有视频
python video_pt_inference_split.py \
    --model yolo11n.pt \
    --video ./video_folder/

# 处理文件夹并保存不一致的帧
python video_pt_inference_split.py \
    --model yolo11n.pt \
    --video ./video_folder/ \
    --save-mismatch \
    --mismatch-classes uav

# 完整参数
python video_pt_inference_split.py \
    --model yolo11n.pt \
    --video ./video_folder/ \
    --overlap-ratio 0.1 \
    --conf-threshold 0.3 \
    --nms-iou-threshold 0.5 \
    --num-crops 5 \
    --save-mismatch \
    --mismatch-classes uav
"""
