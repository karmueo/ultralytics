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
):
    """
    使用Ultralytics YOLO处理视频文件，对每一帧进行推理并生成结果视频

    Args:
        model_path: YOLO模型路径 (.pt文件)
        video_path: 输入视频路径
        output_path: 输出视频路径
        conf_threshold: 置信度阈值
        show_fps: 是否在视频上显示FPS
        class_names: 类别名称列表（可选）
    """
    print(f"\n{'='*60}")
    print(f"Processing video with YOLO: {video_path}")
    print(f"Using model: {model_path}")
    print(f"{'='*60}\n")

    # 加载YOLO模型
    print("Loading YOLO model...")
    model = YOLO(model_path)
    print("Model loaded successfully!\n")

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
    )


if __name__ == "__main__":
    main()