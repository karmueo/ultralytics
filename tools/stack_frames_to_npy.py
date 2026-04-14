"""将连续多帧堆叠为多通道 .npy 文件（用于时序上下文输入）。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

FRAME_PATTERN = re.compile(r"^(?P<prefix>.+)_(?P<index>\d{6})$")


def parse_frame_stem(stem: str) -> tuple[str, int] | None:
    """解析帧文件名 stem，提取公共前缀与帧序号。

    Args:
        stem (str): 不包含扩展名的文件名，例如 "xxx_000123"。

    Returns:
        (tuple[str, int] | None): 若匹配成功，返回 (prefix, index)；否则返回 None。
    """
    m = FRAME_PATTERN.match(stem)
    if not m:
        return None
    prefix = m.group("prefix")  # 公共前缀
    index = int(m.group("index"))  # 当前帧序号
    return prefix, index


def build_frame_paths(current_image: Path, n: int) -> list[Path] | None:
    """根据当前帧路径构建连续 n 帧的路径列表（含当前帧）。

    Args:
        current_image (Path): 当前帧图片路径。
        n (int): 堆叠帧数（例如 3）。

    Returns:
        (list[Path] | None): 从最早帧到当前帧的路径列表；若无法解析或存在缺帧则返回 None。
    """
    parsed = parse_frame_stem(current_image.stem)
    if parsed is None:
        return None

    prefix, index = parsed
    width = 6  # 序号位宽（与你的数据命名一致）
    frames: list[Path] = []  # 连续帧路径列表
    for i in range(index - n + 1, index + 1):
        safe_i = max(i, 0)  # 安全帧序号（对起始帧做0号填充）
        frame_name = f"{prefix}_{safe_i:0{width}d}{current_image.suffix}"  # 构造帧文件名
        frame_path = current_image.with_name(frame_name)  # 构造帧路径
        if not frame_path.exists():
            frame_path = current_image  # 若缺帧，则回退为当前帧（保证通道数一致）
        frames.append(frame_path)
    return frames


def load_and_stack(frames: Iterable[Path]) -> np.ndarray | None:
    """读取多帧并在通道维度进行堆叠。

    Args:
        frames (Iterable[Path]): 从最早到最新的帧路径序列。

    Returns:
        (np.ndarray | None): 形状为 (H, W, 3*n) 的 uint8 数组；若读取失败返回 None。
    """
    imgs: list[np.ndarray] = []  # 读取到的多帧图像
    base_hw: tuple[int, int] | None = None  # 基准高宽
    for fp in frames:
        im = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if im is None:
            return None
        if base_hw is None:
            base_hw = im.shape[:2]
        elif im.shape[:2] != base_hw:
            return None
        imgs.append(im)
    stacked = np.concatenate(imgs, axis=2)  # 在通道维度堆叠
    return stacked.astype(np.uint8, copy=False)


def iter_label_files(labels_dir: Path) -> Iterable[Path]:
    """遍历 labels 目录下的所有标签文件。

    Args:
        labels_dir (Path): 标签目录路径。

    Yields:
        Path: 每个 .txt 标签文件路径。
    """
    yield from sorted(labels_dir.rglob("*.txt"))


def iter_image_files(images_dir: Path) -> Iterable[Path]:
    """遍历 images 目录下的所有图片文件。

    Args:
        images_dir (Path): 图片目录路径。

    Yields:
        Path: 每个图片文件路径。
    """
    for suf in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        yield from sorted(images_dir.glob(suf))


def process_images_labels_dir(
    images_dir: Path,
    labels_dir: Path,
    n: int,
    overwrite: bool,
    all_images: bool,
) -> tuple[int, int, int]:
    """处理一对 images/labels 目录，为图片或标签生成对应 .npy。

    Args:
        images_dir (Path): 图片目录路径。
        labels_dir (Path): 标签目录路径。
        n (int): 堆叠帧数。
        overwrite (bool): 是否覆盖已存在的 .npy 文件。
        all_images (bool): 是否对 images 目录下的所有图片生成 .npy（默认 False）。

    Returns:
        (tuple[int, int, int]): (成功数, 跳过数, 失败数)。
    """
    success = 0  # 成功计数
    skipped = 0  # 跳过计数
    failed = 0  # 失败计数

    if not images_dir.exists() or not labels_dir.exists():
        return success, skipped, failed

    targets = list(iter_image_files(images_dir)) if all_images else list(iter_label_files(labels_dir))  # 处理目标列表
    total_targets = len(targets)  # 目标总数
    processed = 0  # 已处理计数

    for item in targets:
        processed += 1
        if all_images:
            img_path = item  # 直接使用图片路径
        else:
            lb = item  # 当前标签文件
            img_path = images_dir / f"{lb.stem}.jpg"  # 默认图片后缀为 .jpg
            if not img_path.exists():
                # 若不是 .jpg，再尝试常见后缀
                alt_found = False  # 是否找到替代后缀
                for suf in (".png", ".jpeg", ".JPG", ".PNG", ".JPEG"):
                    candidate = images_dir / f"{lb.stem}{suf}"  # 替代候选路径
                    if candidate.exists():
                        img_path = candidate
                        alt_found = True
                        break
                if not alt_found:
                    failed += 1
                    continue

        npy_path = img_path.with_suffix(".npy")  # 对应的 .npy 路径
        if npy_path.exists() and not overwrite:
            skipped += 1
            continue

        frame_paths = build_frame_paths(img_path, n)
        if frame_paths is None:
            failed += 1
            continue

        stacked = load_and_stack(frame_paths)
        if stacked is None:
            failed += 1
            print(
                f"\r[progress] {processed}/{total_targets} success={success} skipped={skipped} failed={failed}",
                end="",
                flush=True,
            )
            continue

        np.save(npy_path, stacked, allow_pickle=False)
        success += 1
        print(
            f"\r[progress] {processed}/{total_targets} success={success} skipped={skipped} failed={failed}",
            end="",
            flush=True,
        )

    if total_targets > 0:
        print()
    return success, skipped, failed


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        (argparse.Namespace): 解析后的参数对象。
    """
    parser = argparse.ArgumentParser(description="将连续n帧堆叠为多通道 .npy 文件")
    parser.add_argument("--root", type=Path, required=True, help="数据集根目录（包含 images/labels 子目录）")
    parser.add_argument("-n", "--num-frames", type=int, default=3, help="堆叠帧数 n（默认 3）")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的 .npy 文件（风险操作：会覆盖旧的堆叠结果）",
    )
    parser.add_argument(
        "--all-images",
        action="store_true",
        help="对 images 目录下的所有图片生成 .npy（建议9通道训练开启）",
    )
    return parser.parse_args()


def main() -> None:
    """脚本入口：为指定划分生成多帧堆叠的 .npy 文件。"""
    args = parse_args()
    root = args.root  # 数据集根目录
    n = args.num_frames  # 堆叠帧数
    overwrite = args.overwrite  # 是否覆盖
    all_images = args.all_images  # 是否处理所有图片

    if n < 2:
        raise ValueError("--num-frames 必须 >= 2")

    total_success = 0  # 总成功计数
    total_skipped = 0  # 总跳过计数
    total_failed = 0  # 总失败计数

    # 仅支持结构：root/images + root/labels
    root_images = root / "images"  # 根目录下的图片目录
    root_labels = root / "labels"  # 根目录下的标签目录
    if not root_images.exists() or not root_labels.exists():
        raise FileNotFoundError(f"root 下必须包含 images/labels 子目录: {root}")

    success, skipped, failed = process_images_labels_dir(
        root_images, root_labels, n=n, overwrite=overwrite, all_images=all_images
    )
    total_success += success
    total_skipped += skipped
    total_failed += failed
    print(f"[split=root] success={success} skipped={skipped} failed={failed} all_images={all_images}")

    print(
        "[total] "
        f"n={n} success={total_success} skipped={total_skipped} failed={total_failed} root={root}"
    )


if __name__ == "__main__":
    main()
