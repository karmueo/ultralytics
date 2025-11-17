import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO
import shutil
from tqdm import tqdm


def get_image_files(input_dir):
    """
    获取文件夹中所有图片文件
    
    Args:
        input_dir: 输入文件夹路径
    
    Returns:
        list: 图片文件路径列表
    """
    input_path = Path(input_dir)
    
    if not input_path.is_dir():
        raise ValueError(f"输入路径不是文件夹: {input_dir}")
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    # 查找所有图片文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    if not image_files:
        raise ValueError(f"在文件夹中未找到图片文件: {input_dir}")
    
    # 按文件名排序
    image_files.sort()
    return image_files


def save_yolo_annotation(image_path, boxes, classes, scores, output_dir, img_width, img_height):
    """
    保存YOLO格式的标注
    
    Args:
        image_path: 图片路径
        boxes: 检测框 (N, 4) xyxy格式
        classes: 类别 (N,)
        scores: 置信度 (N,)
        output_dir: 输出目录
        img_width: 图片宽度
        img_height: 图片高度
    """
    # 创建标注文件路径
    label_file = output_dir / f"{image_path.stem}.txt"
    
    with open(label_file, 'w') as f:
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            cls = int(classes[i])
            # conf = scores[i]  # 可选：如果需要保存置信度
            
            # 转换为YOLO格式 (归一化的中心点坐标和宽高)
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            # YOLO格式: class_id x_center y_center width height
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def process_image_folder(
    model_path,
    image_folder,
    conf_threshold=0.25,
    show_detail_log=False,
):
    """
    处理图片文件夹，进行YOLO推理并生成标注
    
    Args:
        model_path: YOLO模型路径
        image_folder: 输入图片文件夹
        conf_threshold: 置信度阈值
        show_detail_log: 是否显示详细日志
    """
    print(f"\n{'='*60}")
    print(f"Processing image folder: {image_folder}")
    print(f"Using model: {model_path}")
    print(f"{'='*60}\n")
    
    # 加载YOLO模型
    print("Loading YOLO model...")
    model = YOLO(model_path)
    print("Model loaded successfully!\n")
    
    # 获取图片文件列表
    try:
        image_files = get_image_files(image_folder)
    except ValueError as e:
        print(f"错误: {e}")
        return
    
    print(f"找到 {len(image_files)} 个图片文件\n")
    
    # 创建输出文件夹
    input_path = Path(image_folder)
    labels_dir = input_path / "labels"
    annotated_dir = input_path / "annotated"
    missannotated_dir = input_path / "missannotated"
    
    labels_dir.mkdir(exist_ok=True)
    annotated_dir.mkdir(exist_ok=True)
    missannotated_dir.mkdir(exist_ok=True)
    
    print("输出目录:")
    print(f"  YOLO标注: {labels_dir}")
    print(f"  标注图片: {annotated_dir}")
    print(f"  无目标图片: {missannotated_dir}\n")
    
    # 统计信息
    total_detections = 0
    images_with_detections = 0
    images_without_detections = 0
    
    # 处理每个图片
    for image_file in tqdm(image_files, desc="Processing images", unit="image"):
        # 读取图片
        img = cv2.imread(str(image_file))
        if img is None:
            print(f"警告: 无法读取图片 {image_file}")
            continue
        
        img_height, img_width = img.shape[:2]
        
        # 运行推理
        results = model(img, conf=conf_threshold, verbose=False)
        result = results[0]
        
        # 获取检测结果
        num_detections = len(result.boxes)
        
        if num_detections > 0:
            # 有检测目标
            images_with_detections += 1
            total_detections += num_detections
            
            boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4) xyxy格式
            scores = result.boxes.conf.cpu().numpy()  # (N,)
            classes = result.boxes.cls.cpu().numpy()  # (N,)
            
            # 保存YOLO格式标注
            save_yolo_annotation(
                image_file, boxes, classes, scores, 
                labels_dir, img_width, img_height
            )
            
            # 绘制并保存标注图片
            annotated_img = result.plot()
            annotated_path = annotated_dir / image_file.name
            cv2.imwrite(str(annotated_path), annotated_img)
            
            if show_detail_log:
                print(f"  {image_file.name}: {num_detections} detections")
        
        else:
            # 没有检测到目标
            images_without_detections += 1
            
            # 复制到missannotated文件夹
            miss_path = missannotated_dir / image_file.name
            shutil.copy2(image_file, miss_path)
            
            # 创建空的标注文件
            label_file = labels_dir / f"{image_file.stem}.txt"
            label_file.touch()
            
            if show_detail_log:
                print(f"  {image_file.name}: No detections")
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print("Processing completed!")
    print(f"Total images processed: {len(image_files)}")
    print(f"Images with detections: {images_with_detections}")
    print(f"Images without detections: {images_without_detections}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {total_detections / len(image_files):.2f}")
    print("\nOutput directories:")
    print(f"  YOLO labels: {labels_dir}")
    print(f"  Annotated images: {annotated_dir}")
    print(f"  Images without detections: {missannotated_dir}")
    print(f"{'='*60}\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run YOLO inference on image folder and generate YOLO format annotations"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the YOLO model (.pt file)",
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        required=True,
        help="Path to the folder containing images",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for detection (default: 0.25)",
    )
    parser.add_argument(
        "--show-detail-log",
        action="store_true",
        default=False,
        help="Show detailed log for each image",
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # 处理图片文件夹
    process_image_folder(
        model_path=args.model,
        image_folder=args.image_folder,
        conf_threshold=args.conf_threshold,
        show_detail_log=args.show_detail_log,
    )


if __name__ == "__main__":
    main()

"""
使用示例:

# 基本使用
python image_folder_inference.py \
    --model yolo11n.pt \
    --image-folder ./images/

# 指定置信度阈值
python image_folder_inference.py \
    --model yolo11n.pt \
    --image-folder ./images/ \
    --conf-threshold 0.5

# 显示详细日志
python image_folder_inference.py \
    --model yolo11n.pt \
    --image-folder ./images/ \
    --show-detail-log

输出结构:
images/
├── image1.jpg
├── image2.jpg
├── labels/              # YOLO格式标注
│   ├── image1.txt
│   └── image2.txt
├── annotated/           # 绘制了检测框的图片
│   ├── image1.jpg
│   └── image2.jpg
└── missannotated/       # 没有检测到目标的图片
    └── image3.jpg

YOLO标注格式 (每行):
class_id x_center y_center width height
其中坐标都是归一化的 (0-1之间)
"""
