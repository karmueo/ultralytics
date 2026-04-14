# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import argparse
import sys
from pathlib import Path

FILE = Path(__file__).resolve()  # 当前脚本文件路径
ROOT = FILE.parents[1]  # 仓库根目录路径
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS


def parse_args() -> argparse.Namespace:
    """解析目录图片推理导出YOLO标注所需的命令行参数.

    Returns:
        (argparse.Namespace): 解析后的命令行参数对象.
    """
    parser = argparse.ArgumentParser(description="对指定图片目录执行YOLO推理，并生成YOLO检测格式txt标注。")  # 参数解析器
    parser.add_argument("--model", required=True, type=Path, help="YOLO模型权重路径，例如 runs/detect/train/weights/best.pt")
    parser.add_argument("--source-dir", required=True, type=Path, help="待推理图片目录")
    parser.add_argument("--source-label-dir", default=None, type=Path, help="已有YOLO标注目录，若指定则会先读取原标注再追加person预测框")
    parser.add_argument("--output-dir", required=True, type=Path, help="标注输出目录，txt 会保存到该目录下的 labels 子目录")
    parser.add_argument("--imgsz", default=640, type=int, help="推理输入尺寸")
    parser.add_argument("--conf", default=0.5, type=float, help="检测置信度阈值")
    parser.add_argument("--device", default="cuda", help="推理设备，例如 cuda、cpu、0、0,1")
    parser.add_argument("--batch", default=1, type=int, help="推理batch大小")
    parser.add_argument("--person-source-class-id", default=0, type=int, help="person模型输出中的原始person类别ID")
    parser.add_argument("--person-target-class-id", default=1, type=int, help="写入输出labels时person要重映射成的目标类别ID")
    parser.add_argument("--recursive", action="store_true", help="递归扫描 source-dir 下的子目录图片")
    return parser.parse_args()


def collect_image_paths(source_dir: Path, recursive: bool = False) -> list[Path]:
    """收集输入目录中的图片路径并按路径排序.

    Args:
        source_dir (Path): 待扫描的图片目录.
        recursive (bool): 是否递归扫描子目录.

    Returns:
        (list[Path]): 已排序的图片路径列表.

    Raises:
        FileNotFoundError: 当 source_dir 不存在或不是目录时抛出.
        ValueError: 当目录下没有可支持格式图片，或存在同名stem图片导致标签覆盖风险时抛出.
    """
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source-dir 不存在或不是目录: {source_dir}")

    pattern = "**/*" if recursive else "*"  # 图片扫描通配符
    image_paths = sorted(
        path for path in source_dir.glob(pattern) if path.is_file() and path.suffix[1:].lower() in IMG_FORMATS
    )  # 待推理图片路径列表
    if not image_paths:
        supported_formats = ", ".join(sorted(IMG_FORMATS))  # 支持的图片格式说明
        raise ValueError(f"source-dir 中没有可推理图片: {source_dir}，支持格式: {supported_formats}")

    stem_counts: dict[str, int] = {}  # 图片stem重复计数
    for image_path in image_paths:
        stem_counts[image_path.stem] = stem_counts.get(image_path.stem, 0) + 1
    duplicate_stems = sorted(stem for stem, count in stem_counts.items() if count > 1)  # 重复stem列表
    if duplicate_stems:
        raise ValueError(
            "存在同名图片stem，按YOLO默认labels落盘会互相覆盖，请先重命名图片: " + ", ".join(duplicate_stems)
        )

    return image_paths


def ensure_empty_label_files(image_paths: list[Path], label_dir: Path) -> None:
    """为无检测结果的图片补充空txt标注文件，保证图片与标签一一对应.

    Args:
        image_paths (list[Path]): 已推理的图片路径列表.
        label_dir (Path): YOLO标签输出目录.
    """
    label_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"  # 单张图片对应的标注路径
        label_path.touch(exist_ok=True)


def read_source_label_lines(source_label_dir: Path | None, image_stem: str) -> list[str]:
    """读取指定图片对应的原始YOLO标注行.

    Args:
        source_label_dir (Path | None): 原始YOLO标注目录，None表示不读取原始标注.
        image_stem (str): 图片文件stem.

    Returns:
        (list[str]): 去除空行后的原始标注行列表.
    """
    if source_label_dir is None:
        return []

    source_label_path = source_label_dir / f"{image_stem}.txt"  # 原始标注文件路径
    if not source_label_path.is_file():
        return []

    return [line for line in source_label_path.read_text(encoding="utf-8").splitlines() if line]


def build_prediction_label_lines(result, source_label_dir: Path | None, person_source_class_id: int, person_target_class_id: int) -> list[str]:
    """把单张图片预测结果转换为YOLO标注行，并按需过滤和重映射person类别.

    Args:
        result: 单张图片的YOLO推理结果对象.
        source_label_dir (Path | None): 原始YOLO标注目录，None表示保留所有预测类别原始ID.
        person_source_class_id (int): person模型输出中的原始person类别ID.
        person_target_class_id (int): 写入输出labels时person要重映射成的目标类别ID.

    Returns:
        (list[str]): 转换后的预测标注行列表.
    """
    if result.boxes is None or len(result.boxes) == 0:
        return []

    label_lines = []  # 预测标注行列表
    for box in result.boxes:
        class_id = int(box.cls)  # 预测框原始类别ID
        if source_label_dir is not None:
            if class_id != person_source_class_id:
                continue
            class_id = person_target_class_id
        xywhn = box.xywhn.view(-1).tolist()  # 归一化xywh坐标
        line_values = (class_id, *xywhn)  # 单行YOLO标注字段
        label_lines.append(("%g " * len(line_values)).rstrip() % line_values)

    return label_lines


def predict_images_to_yolo_labels(
    model_path: Path,
    source_dir: Path,
    output_dir: Path,
    source_label_dir: Path | None = None,
    imgsz: int = 640,
    conf: float = 0.25,
    device: str | None = "cuda",
    batch: int = 1,
    person_source_class_id: int = 0,
    person_target_class_id: int = 1,
    recursive: bool = False,
) -> Path:
    """使用指定YOLO模型对目录图片执行推理，并输出YOLO检测格式txt标注.

    Args:
        model_path (Path): YOLO模型权重路径.
        source_dir (Path): 待推理图片目录.
        output_dir (Path): 推理输出目录.
        source_label_dir (Path | None): 原始YOLO标注目录，若指定则先保留原标注再追加person预测框.
        imgsz (int): 推理输入尺寸.
        conf (float): 检测置信度阈值.
        device (str | None): 推理设备.
        batch (int): 推理batch大小.
        person_source_class_id (int): person模型输出中的原始person类别ID.
        person_target_class_id (int): 写入输出labels时person要重映射成的目标类别ID.
        recursive (bool): 是否递归扫描source_dir子目录.

    Returns:
        (Path): 实际保存labels的目录路径.

    Raises:
        FileNotFoundError: 当输入图片目录或原始标注目录不存在时抛出.
        ValueError: 当输入目录无图片或图片stem重复时抛出.
    """
    image_paths = collect_image_paths(source_dir=source_dir, recursive=recursive)  # 待推理图片路径列表
    if source_label_dir is not None and not source_label_dir.is_dir():
        raise FileNotFoundError(f"source-label-dir 不存在或不是目录: {source_label_dir}")

    model = YOLO(model_path)  # YOLO模型实例
    prediction_source = str(source_dir / "**" / "*") if recursive else str(source_dir)  # 传给predict的源路径
    label_dir = output_dir / "labels"  # 最终labels输出目录
    label_dir.mkdir(parents=True, exist_ok=True)
    predict_classes = [person_source_class_id] if source_label_dir is not None else None  # 推理类别过滤列表

    results = model.predict(
        source=prediction_source,
        imgsz=imgsz,
        conf=conf,
        device=device,
        batch=batch,
        classes=predict_classes,
        save=False,
        save_txt=False,
        save_conf=False,
        project=output_dir.parent,
        name=output_dir.name,
        exist_ok=True,
        verbose=False,
        stream=True,
    )  # 流式预测结果生成器
    for result in results:
        image_stem = Path(result.path).stem  # 当前图片stem
        source_label_lines = read_source_label_lines(source_label_dir=source_label_dir, image_stem=image_stem)  # 原始标注行
        prediction_label_lines = build_prediction_label_lines(
            result=result,
            source_label_dir=source_label_dir,
            person_source_class_id=person_source_class_id,
            person_target_class_id=person_target_class_id,
        )  # 预测标注行
        merged_label_lines = [*source_label_lines, *prediction_label_lines]  # 合并后的标注行
        output_label_path = label_dir / f"{image_stem}.txt"  # 合并后标注文件路径
        output_label_path.write_text(
            "".join(f"{line}\n" for line in merged_label_lines),
            encoding="utf-8",
        )

    ensure_empty_label_files(image_paths=image_paths, label_dir=label_dir)
    return label_dir


def main() -> int:
    """脚本主入口，执行参数解析、批量推理和异常转可读错误输出.

    Returns:
        (int): 进程退出码，0 表示成功，1 表示失败.
    """
    args = parse_args()  # 命令行参数
    try:
        label_dir = predict_images_to_yolo_labels(
            model_path=args.model,
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            source_label_dir=args.source_label_dir,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            batch=args.batch,
            person_source_class_id=args.person_source_class_id,
            person_target_class_id=args.person_target_class_id,
            recursive=args.recursive,
        )  # 预测输出labels目录
    except (FileNotFoundError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    print(f"YOLO标注已保存到: {label_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
