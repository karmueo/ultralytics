# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from tests import MODEL, SOURCE
from tools.predict_images_to_yolo_labels import parse_args


SCRIPT_PATH = Path("tools/predict_images_to_yolo_labels.py")  # 待验证的推理脚本路径


def _run_label_export_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """运行图片目录推理导出脚本并返回进程结果.

    Args:
        *args (str): 传递给脚本的命令行参数.

    Returns:
        (subprocess.CompletedProcess[str]): 脚本执行结果对象.
    """
    cmd = [sys.executable, str(SCRIPT_PATH), *args]  # 脚本执行命令
    return subprocess.run(cmd, capture_output=True, check=False, text=True)


def test_predict_images_to_yolo_labels_exports_labels_and_empty_files(tmp_path: Path) -> None:
    """验证目录图片推理后会导出YOLO检测标签，并为无检测图片补空txt文件."""
    source_dir = tmp_path / "images"  # 输入图片目录
    output_dir = tmp_path / "labels_output"  # 输出标注目录
    source_dir.mkdir(parents=True, exist_ok=True)

    Image.open(SOURCE).save(source_dir / "bus.jpg")
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(source_dir / "blank.jpg")
    (source_dir / "ignore.txt").write_text("not an image", encoding="utf-8")

    result = _run_label_export_cli(
        "--model",
        str(MODEL),
        "--source-dir",
        str(source_dir),
        "--output-dir",
        str(output_dir),
        "--imgsz",
        "160",
        "--conf",
        "0.25",
        "--device",
        "cpu",
    )

    assert result.returncode == 0, result.stderr

    bus_label = output_dir / "labels" / "bus.txt"  # 有目标图片标注文件
    blank_label = output_dir / "labels" / "blank.txt"  # 无目标图片空标注文件
    assert bus_label.is_file()
    assert blank_label.is_file()
    assert blank_label.read_text(encoding="utf-8") == ""

    label_lines = [line for line in bus_label.read_text(encoding="utf-8").splitlines() if line]  # 有效标注行
    assert label_lines
    assert all(len(line.split()) == 5 for line in label_lines)


def test_predict_images_to_yolo_labels_merges_source_labels_and_remaps_person_class(tmp_path: Path) -> None:
    """验证会保留原始无人机标注，并把person模型预测框重映射为类别1追加到新labels目录."""
    source_dir = tmp_path / "images"  # 输入图片目录
    source_label_dir = tmp_path / "labels"  # 原始标注目录
    output_dir = tmp_path / "merged_output"  # 合并后标注输出目录
    source_dir.mkdir(parents=True, exist_ok=True)
    source_label_dir.mkdir(parents=True, exist_ok=True)

    Image.open(SOURCE).save(source_dir / "bus.jpg")

    source_label_path = source_label_dir / "bus.txt"  # 原始无人机标注路径
    source_label_text = "0 0.5 0.5 0.1 0.1\n"  # 原始无人机标注内容
    source_label_path.write_text(source_label_text, encoding="utf-8")

    result = _run_label_export_cli(
        "--model",
        str(MODEL),
        "--source-dir",
        str(source_dir),
        "--source-label-dir",
        str(source_label_dir),
        "--output-dir",
        str(output_dir),
        "--imgsz",
        "320",
        "--conf",
        "0.25",
        "--device",
        "cpu",
        "--person-source-class-id",
        "0",
        "--person-target-class-id",
        "1",
    )

    assert result.returncode == 0, result.stderr
    assert source_label_path.read_text(encoding="utf-8") == source_label_text

    merged_lines = [
        line for line in (output_dir / "labels" / "bus.txt").read_text(encoding="utf-8").splitlines() if line
    ]  # 合并后的有效标注行
    assert source_label_text.strip() in merged_lines
    assert any(line.startswith("1 ") for line in merged_lines)


def test_predict_images_to_yolo_labels_rejects_missing_source_dir(tmp_path: Path) -> None:
    """验证输入图片目录不存在时脚本会返回非0状态码并输出可读错误."""
    result = _run_label_export_cli(
        "--model",
        str(MODEL),
        "--source-dir",
        str(tmp_path / "missing"),
        "--output-dir",
        str(tmp_path / "labels_output"),
    )

    assert result.returncode != 0
    assert "source-dir" in result.stderr


def test_parse_args_uses_default_inference_settings(monkeypatch, tmp_path: Path) -> None:
    """验证仅传必填参数时会默认使用640尺寸、0.25置信度和cuda设备."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict_images_to_yolo_labels.py",
            "--model",
            str(MODEL),
            "--source-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "labels_output"),
        ],
    )

    args = parse_args()  # 解析后的默认参数

    assert args.imgsz == 640
    assert args.conf == 0.25
    assert args.device == "cuda"
