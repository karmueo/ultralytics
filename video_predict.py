import argparse
from pathlib import Path

from ultralytics import YOLO


def resolve_batch_size(model_path: str, user_batch: int | None) -> int:
    """解析视频推理使用的 batch 大小。

    Args:
        model_path (str): 模型权重路径。
        user_batch (int | None): 用户手工指定的 batch 大小。

    Returns:
        (int): 最终使用的 batch 大小。
    """
    if user_batch is not None:
        return user_batch

    if Path(model_path).suffix.lower() != ".onnx":
        return 1

    try:
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(Path(model_path).expanduser()),
            providers=["CPUExecutionProvider"],
        )  # ONNX 会话
        input_shape = session.get_inputs()[0].shape  # 模型输入形状
        batch_dim = input_shape[0] if input_shape else None  # batch 维度
        if isinstance(batch_dim, int) and batch_dim > 0:
            return batch_dim
    except Exception:
        pass

    return 1


def parse_args() -> argparse.Namespace:
    """解析视频推理的命令行参数。

    Returns:
        (argparse.Namespace): 解析后的命令行参数对象。
    """
    parser = argparse.ArgumentParser(description="Ultralytics YOLO 视频推理脚本")
    parser.add_argument("--model", type=str, required=True, help="模型权重路径，例如 yolov11s.pt")
    parser.add_argument("--source", type=str, required=True, help="视频路径，例如 /path/to/video.mp4")
    parser.add_argument("--imgsz", type=str, default="352,640", help="推理输入尺寸，格式 H,W")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--device", type=str, default="0", help="推理设备，例如 0 或 cpu")
    parser.add_argument("--batch", type=int, default=None, help="推理 batch；留空时对 ONNX 自动推断")
    parser.add_argument("--vid-stride", type=int, default=1, help="视频抽帧间隔，1 表示逐帧")
    parser.add_argument("--save", action="store_true", help="是否保存可视化结果视频")
    parser.add_argument("--project", type=str, default="runs/predict", help="输出根目录")
    parser.add_argument("--name", type=str, default="exp", help="输出子目录名称")
    parser.add_argument("--exist-ok", action="store_true", help="若输出目录已存在则允许覆盖")
    return parser.parse_args()


def main() -> None:
    """执行视频推理并按需保存结果。

    Returns:
        (None): 无返回值。
    """
    args = parse_args()  # 命令行参数
    model_path = args.model  # 模型权重路径
    source_path = args.source  # 视频路径
    imgsz_tokens = args.imgsz.split(",")  # 尺寸字符串切分
    imgsz = (int(imgsz_tokens[0]), int(imgsz_tokens[1]))  # 推理输入尺寸
    conf = args.conf  # 置信度阈值
    device = args.device  # 推理设备
    batch = resolve_batch_size(model_path, args.batch)  # 推理 batch 大小
    vid_stride = args.vid_stride  # 抽帧间隔
    save = args.save  # 是否保存可视化结果
    project = args.project  # 输出根目录
    name = args.name  # 输出子目录名称
    exist_ok = args.exist_ok  # 是否允许覆盖

    model = YOLO(model_path)  # 加载模型
    model.predict(
        source=source_path,
        imgsz=imgsz,
        conf=conf,
        device=device,
        batch=batch,
        vid_stride=vid_stride,
        save=save,
        project=project,
        name=name,
        exist_ok=exist_ok,
    )


if __name__ == "__main__":
    main()
