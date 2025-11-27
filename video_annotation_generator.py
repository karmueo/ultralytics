import argparse
from pathlib import Path
import cv2
import json
from datetime import datetime
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


def process_video_to_annotations(
    model_path,
    video_path,
    images_output_dir,
    labels_output_dir,
    frame_interval=1,
    conf_threshold=0.25,
    coco_data=None,
    global_image_id=1,
    global_annotation_id=1,
):
    """
    处理视频文件，生成YOLO格式标注
    
    Args:
        model_path: YOLO模型路径
        video_path: 输入视频路径
        images_output_dir: 图像帧输出目录
        labels_output_dir: 标注文件输出目录
        frame_interval: 帧采样间隔（1表示每帧都处理，5表示每5帧处理一次）
        conf_threshold: 置信度阈值
        coco_data: COCO格式数据结构（可选，如果提供则添加到此结构中）
        global_image_id: 全局图像ID起始值
        global_annotation_id: 全局标注ID起始值
    
    Returns:
        dict: 包含处理统计信息
    """
    video_path = Path(video_path)
    images_output_dir = Path(images_output_dir)
    labels_output_dir = Path(labels_output_dir)
    images_output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    model = YOLO(model_path)
    
    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_count = 0
    processed_frames = 0
    total_detections = 0
    current_image_id = global_image_id
    current_annotation_id = global_annotation_id
    
    pbar = tqdm(total=total_frames, desc=f"Processing {video_path.name}", unit="frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        pbar.update(1)
        
        # 根据采样间隔决定是否处理当前帧
        if (frame_count - 1) % frame_interval != 0:
            continue
        
        # 运行推理，agnostic_nms=True 对所有类使用NMS
        results = model(frame, conf=conf_threshold, agnostic_nms=True, verbose=False)
        result = results[0]
        
        # 生成标注文件名
        annotation_filename = f"{video_path.stem}_frame_{frame_count:06d}"
        yolo_label_path = labels_output_dir / f"{annotation_filename}.txt"
        image_path = images_output_dir / f"{annotation_filename}.jpg"
        
        # 保存视频帧到images文件夹
        cv2.imwrite(str(image_path), frame)
        
        # 写入YOLO格式标注
        detections = []
        if len(result.boxes) > 0:
            boxes = result.boxes
            for box in boxes:
                # 获取边界框信息
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                
                # 转换为YOLO格式 (归一化的中心坐标和宽高)
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                
                detections.append({
                    'cls_id': cls_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': w,
                    'height': h,
                    'conf': conf,
                    'bbox_pixels': [x1, y1, x2 - x1, y2 - y1]  # COCO格式 [x, y, width, height]
                })
        
        # 写入YOLO标注文件
        with open(yolo_label_path, 'w') as f:
            for det in detections:
                f.write(f"{det['cls_id']} {det['x_center']:.6f} {det['y_center']:.6f} "
                       f"{det['width']:.6f} {det['height']:.6f}\n")
        
        # 添加到COCO格式数据
        if coco_data is not None:
            # 添加图像信息
            coco_data["images"].append({
                "id": current_image_id,
                "file_name": f"{annotation_filename}.jpg",
                "width": width,
                "height": height
            })
            
            # 添加标注信息
            for det in detections:
                x, y, w, h = det['bbox_pixels']
                coco_data["annotations"].append({
                    "id": current_annotation_id,
                    "image_id": current_image_id,
                    "category_id": int(det['cls_id']),
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "area": float(w * h),
                    "iscrowd": 0,
                    "score": float(det['conf'])
                })
                current_annotation_id += 1
            
            current_image_id += 1
        
        processed_frames += 1
        total_detections += len(detections)
    
    pbar.close()
    cap.release()
    
    return {
        'total_frames': total_frames,
        'processed_frames': processed_frames,
        'total_detections': total_detections,
        'images_dir': images_output_dir,
        'labels_dir': labels_output_dir,
        'next_image_id': current_image_id,
        'next_annotation_id': current_annotation_id
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Generate YOLO format annotations from video files"
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
        "--frame-interval",
        type=int,
        default=1,
        help="Frame sampling interval (1=every frame, 5=every 5th frame, etc. Default: 1)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for detection (default: 0.25)",
    )
    parser.add_argument(
        "--save-coco",
        action="store_true",
        default=False,
        help="Save annotations in COCO format (JSON)",
    )
    parser.add_argument(
        "--labels-output-dir",
        type=str,
        default=None,
        help="Output directory for YOLO label files (default: auto-generated based on input)",
    )
    parser.add_argument(
        "--coco-output-path",
        type=str,
        default=None,
        help="Output path for COCO JSON file (default: auto-generated based on input)",
    )
    
    args = parser.parse_args()
    
    # 获取视频文件列表
    try:
        video_files = get_video_files(args.video)
    except ValueError as e:
        print(f"错误: {e}")
        return
    
    print(f"\n{'='*80}")
    print(f"找到 {len(video_files)} 个视频文件")
    print(f"使用模型: {args.model}")
    print(f"帧采样间隔: {args.frame_interval}")
    print(f"置信度阈值: {args.conf_threshold}")
    print(f"{'='*80}\n")
    
    # 确定输出目录
    input_path = Path(args.video)
    
    # 根据输入类型自动生成输出目录
    if input_path.is_file():
        # 单个文件：在视频文件同级目录下创建images和labels文件夹
        base_dir = input_path.parent
    else:
        # 文件夹：在文件夹所在位置创建images和labels文件夹
        base_dir = input_path
    
    images_base_dir = base_dir / "images"
    labels_base_dir = base_dir / "labels"
    
    # COCO输出路径
    if args.save_coco:
        if args.coco_output_path:
            coco_output_path = Path(args.coco_output_path)
        else:
            # 默认在images和labels同级目录下创建coco.json
            coco_output_path = base_dir / "coco.json"
    else:
        coco_output_path = None
    
    print(f"图像帧输出目录: {images_base_dir}")
    print(f"YOLO标注输出目录: {labels_base_dir}")
    if args.save_coco:
        print(f"COCO标注输出路径: {coco_output_path}")
    print()
    
    # 初始化COCO数据结构
    coco_data = None
    if args.save_coco:
        # 加载模型以获取类别信息
        temp_model = YOLO(args.model)
        class_names = temp_model.names
        
        coco_data = {
            "info": {
                "description": "Video annotations",
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # 添加类别信息
        for cls_id, cls_name in class_names.items():
            coco_data["categories"].append({
                "id": int(cls_id),
                "name": cls_name,
                "supercategory": "object"
            })
    
    # 统计信息
    total_processed_frames = 0
    total_detections = 0
    global_image_id = 1
    global_annotation_id = 1
    
    # 处理每个视频
    for idx, video_file in enumerate(video_files, 1):
        print(f"\n{'='*80}")
        print(f"处理视频 {idx}/{len(video_files)}: {video_file.name}")
        print(f"{'='*80}\n")
        
        try:
            # 所有文件直接保存到images和labels文件夹，不创建子文件夹
            # 处理视频
            stats = process_video_to_annotations(
                model_path=args.model,
                video_path=video_file,
                images_output_dir=images_base_dir,
                labels_output_dir=labels_base_dir,
                frame_interval=args.frame_interval,
                conf_threshold=args.conf_threshold,
                coco_data=coco_data,
                global_image_id=global_image_id,
                global_annotation_id=global_annotation_id,
            )
            
            # 更新全局ID
            global_image_id = stats['next_image_id']
            global_annotation_id = stats['next_annotation_id']
            
            # 打印统计信息
            print("\n视频处理完成:")
            print(f"  总帧数: {stats['total_frames']}")
            print(f"  处理帧数: {stats['processed_frames']}")
            print(f"  检测目标总数: {stats['total_detections']}")
            print(f"  图像帧保存至: {stats['images_dir']}")
            print(f"  YOLO标注保存至: {stats['labels_dir']}")
            
            total_processed_frames += stats['processed_frames']
            total_detections += stats['total_detections']
            
        except Exception as e:
            print(f"\n错误: 处理视频 {video_file} 时发生错误: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存COCO格式标注
    if args.save_coco and coco_data is not None and coco_output_path:
        coco_output_path = Path(coco_output_path)
        coco_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(coco_output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)
        print(f"\nCOCO标注已保存至: {coco_output_path}")
        print(f"  总图像数: {len(coco_data['images'])}")
        print(f"  总标注数: {len(coco_data['annotations'])}")
        print(f"  类别数: {len(coco_data['categories'])}")
    
    # 打印总体统计
    print(f"\n{'='*80}")
    print("所有视频处理完成!")
    print(f"  处理视频数: {len(video_files)}")
    print(f"  总处理帧数: {total_processed_frames}")
    print(f"  总检测目标数: {total_detections}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

"""
使用示例:

# 处理单个视频文件，每帧都处理
python video_annotation_generator.py \
    --model yolo11n.pt \
    --video input.mp4

# 处理单个视频文件，每5帧处理一次
python video_annotation_generator.py \
    --model yolo11n.pt \
    --video input.mp4 \
    --frame-interval 5

# 处理文件夹中的所有视频
python video_annotation_generator.py \
    --model yolo11n.pt \
    --video ./video_folder/ \
    --frame-interval 10

# 处理视频并生成COCO格式标注
python video_annotation_generator.py \
    --model yolo11n.pt \
    --video input.mp4 \
    --save-coco

# 完整参数示例
python video_annotation_generator.py \
    --model yolo11n.pt \
    --video ./video_folder/ \
    --frame-interval 5 \
    --conf-threshold 0.3 \
    --save-coco \
    --coco-output-path ./custom_coco.json

# 输出结构:
# - 单个视频文件: 在视频同级目录创建 images/ 和 labels/ 文件夹
# - 视频文件夹: 在文件夹内创建 images/ 和 labels/ 文件夹
# - 所有视频帧和标注直接保存在 images/ 和 labels/ 中，不再创建子文件夹
# - COCO格式: 在 images/ 和 labels/ 同级目录创建 coco.json
"""
