import argparse
from pathlib import Path
import time
import cv2
from ultralytics import YOLO


def process_video_yolo(
    model_path,
    video_path,
    output_path,
    conf_threshold=0.25,
    show_fps=True,
    class_names=None,
    use_classifier=False,
    classifier_path=None,
    classifier_conf=0.5,
    save_mismatch=False,
    mismatch_classes=None,
    mismatch_output_dir="mismatch_frames",
):
    """
    使用Ultralytics YOLO处理视频文件,对每一帧进行推理并生成结果视频

    Args:
        model_path: YOLO模型路径 (.pt文件)
        video_path: 输入视频路径
        output_path: 输出视频路径
        conf_threshold: 置信度阈值
        show_fps: 是否在视频上显示FPS
        class_names: 类别名称列表（可选）
        use_classifier: 是否使用分类模型二次推理
        classifier_path: 分类模型路径 (.pt文件)
        classifier_conf: 分类模型置信度阈值
        save_mismatch: 是否保存检测类别与指定类别不一致的帧
        mismatch_classes: 指定类别列表（类别名称），用于比对检测结果
        mismatch_output_dir: 保存不一致帧的输出目录
    """
    print(f"\n{'='*60}")
    print(f"Processing video with YOLO: {video_path}")
    print(f"Using model: {model_path}")
    print(f"{'='*60}\n")

    # 加载YOLO模型
    print("Loading YOLO model...")
    model = YOLO(model_path)
    print("Model loaded successfully!\n")

    # 加载分类模型（如果需要）
    classifier = None
    if use_classifier:
        if not classifier_path:
            raise ValueError("使用分类模型时必须提供 classifier_path 参数")
        print(f"Loading classification model: {classifier_path}")
        classifier = YOLO(classifier_path)
        print("Classification model loaded successfully!\n")

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

    # 创建保存不一致帧的文件夹
    mismatch_raw_dir = None
    mismatch_annotated_dir = None
    mismatch_count = 0
    if save_mismatch:
        if not mismatch_classes:
            raise ValueError("启用 save_mismatch 时必须提供 mismatch_classes 参数")
        mismatch_output_path = Path(mismatch_output_dir)
        mismatch_raw_dir = mismatch_output_path / "raw_frames"
        mismatch_annotated_dir = mismatch_output_path / "annotated_frames"
        mismatch_raw_dir.mkdir(parents=True, exist_ok=True)
        mismatch_annotated_dir.mkdir(parents=True, exist_ok=True)
        print(f"Mismatch detection enabled. Expected classes: {mismatch_classes}")
        print(f"Raw frames will be saved to: {mismatch_raw_dir}")
        print(f"Annotated frames will be saved to: {mismatch_annotated_dir}\n")

    frame_count = 0
    processed_frames = 0
    total_inference_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 运行YOLO推理
        start_time = time.time()
        results = model(frame, conf=conf_threshold, verbose=False)
        inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
        total_inference_time += inference_time

        # 打印推理时间
        print(f"Frame {frame_count}: Inference time: {inference_time:.2f} ms")

        # 获取检测结果
        result = results[0]

        # 绘制检测结果
        annotated_frame = result.plot()

        # 如果启用分类模型，对检测到的目标进行二次分类
        if use_classifier and classifier is not None and len(result.boxes) > 0:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                # 获取边界框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # 裁剪目标区域
                crop = frame[y1:y2, x1:x2]
                
                if crop.size > 0:
                    # 使用分类模型推理
                    cls_results = classifier(crop, verbose=False)
                    cls_result = cls_results[0]
                    
                    # 获取分类结果
                    if hasattr(cls_result, 'probs') and cls_result.probs is not None:
                        # 获取top1分类结果
                        top1_idx = cls_result.probs.top1
                        top1_conf = cls_result.probs.top1conf.item()
                        
                        # 只有当分类置信度大于阈值时才显示
                        if top1_conf >= classifier_conf:
                            cls_name = cls_result.names[top1_idx]
                            det_cls_id = int(box.cls[0].item())
                            det_name = result.names[det_cls_id]
                            
                            # 在边界框上方绘制检测类别-分类类别
                            label = f"{det_name}-{cls_name} {top1_conf:.2f}"
                            
                            # 计算文本位置
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            label_y = max(y1 - 10, label_size[1] + 10)
                            
                            # 绘制背景矩形
                            cv2.rectangle(
                                annotated_frame,
                                (x1, label_y - label_size[1] - 5),
                                (x1 + label_size[0], label_y + 5),
                                (0, 255, 0),
                                -1,
                            )
                            
                            # 绘制文本
                            cv2.putText(
                                annotated_frame,
                                label,
                                (x1, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 0, 0),
                                2,
                            )

        # 如果需要显示FPS，在帧上绘制
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
                2,
            )

        # 检查是否需要保存不一致的帧
        if save_mismatch and mismatch_classes and mismatch_raw_dir and mismatch_annotated_dir and len(result.boxes) > 0:
            # 检查检测到的类别是否在指定类别列表中
            detected_classes = [result.names[int(box.cls[0].item())] for box in result.boxes]
            has_mismatch = False
            
            for det_class in detected_classes:
                if det_class not in mismatch_classes:
                    has_mismatch = True
                    break
            
            if has_mismatch:
                # 保存原始帧和标注帧
                frame_filename = f"frame_{frame_count:06d}.jpg"
                raw_frame_path = mismatch_raw_dir / frame_filename
                annotated_frame_path = mismatch_annotated_dir / frame_filename
                
                cv2.imwrite(str(raw_frame_path), frame)
                cv2.imwrite(str(annotated_frame_path), annotated_frame)
                
                mismatch_count += 1
                print(f"  -> Mismatch detected! Detected classes: {detected_classes}. Saved frame {frame_count}")

        # 写入输出视频
        out.write(annotated_frame)

        processed_frames += 1

        # 打印进度
        if frame_count % 100 == 0 or frame_count == total_frames:
            print(f"Processed {frame_count}/{total_frames} frames")

    # 释放资源
    cap.release()
    out.release()

    # 计算平均性能
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
            print(f"Raw frames saved to: {mismatch_raw_dir}")
            print(f"Annotated frames saved to: {mismatch_annotated_dir}")
    print(f"{'='*60}\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run YOLO inference on video")
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
        help="Path to the input video file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output video file (default: input_video_yolo_result.mp4)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for detection (default: 0.25)",
    )
    parser.add_argument(
        "--show-fps",
        action="store_true",
        default=True,
        help="Show FPS on the output video (default: True)",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        nargs="+",
        default=None,
        help="Class names list (optional, will use model defaults if not provided)",
    )
    parser.add_argument(
        "--use-classifier",
        action="store_true",
        default=False,
        help="Enable secondary classification for detected objects",
    )
    parser.add_argument(
        "--classifier-path",
        type=str,
        default=None,
        help="Path to the classification model (.pt file) for secondary inference",
    )
    parser.add_argument(
        "--classifier-conf",
        type=float,
        default=0.5,
        help="Confidence threshold for classification model (default: 0.5)",
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
        default="mismatch_frames",
        help="Output directory for saving mismatch frames (default: mismatch_frames)",
    )
    return parser.parse_args()


def main():
    """Main function to process video with YOLO model."""
    args = parse_args()

    # 设置默认输出路径
    if args.output is None:
        video_path = Path(args.video)
        args.output = video_path.parent / f"{video_path.stem}_yolo_result.mp4"

    # 处理视频
    process_video_yolo(
        model_path=args.model,
        video_path=args.video,
        output_path=args.output,
        conf_threshold=args.conf_threshold,
        show_fps=args.show_fps,
        class_names=args.class_names,
        use_classifier=args.use_classifier,
        classifier_path=args.classifier_path,
        classifier_conf=args.classifier_conf,
        save_mismatch=args.save_mismatch,
        mismatch_classes=args.mismatch_classes,
        mismatch_output_dir=args.mismatch_output_dir,
    )


if __name__ == "__main__":
    main()

"""
# 不使用分类模型(原有功能)
python video_pt_inference.py --model yolo11n.pt --video input.mp4

# 使用分类模型二次推理
python video_pt_inference.py \
    --model yolo11n.pt \
    --video input.mp4 \
    --use-classifier \
    --classifier-path yolo11n-cls.pt \
    --classifier-conf 0.5

# 保存检测类别与指定类别不一致的帧
python video_pt_inference.py \
    --model yolo11n.pt \
    --video input.mp4 \
    --save-mismatch \
    --mismatch-classes person car \
    --mismatch-output-dir ./mismatch_output

# 结合分类模型和不一致检测
python video_pt_inference.py \
    --model yolo11n.pt \
    --video input.mp4 \
    --use-classifier \
    --classifier-path yolo11n-cls.pt \
    --save-mismatch \
    --mismatch-classes person bicycle
"""
