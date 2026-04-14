"""将预训练权重的首层卷积输入通道从3扩展到9（多帧堆叠场景）。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics.nn.tasks import DetectionModel


def find_first_conv_key(state_dict: dict, old_in_channels: int) -> str:
    """在 state_dict 中查找首个输入通道数匹配的卷积权重键名。

    Args:
        state_dict (dict): 模型的 state_dict。
        old_in_channels (int): 期望的旧输入通道数（例如 3）。

    Returns:
        (str): 首个匹配的卷积权重键名。

    Raises:
        KeyError: 当未找到符合条件的卷积权重时抛出。
    """
    for k, v in state_dict.items():
        if k.endswith("weight") and getattr(v, "ndim", 0) == 4 and v.shape[1] == old_in_channels:
            return k
    raise KeyError(f"未找到输入通道为 {old_in_channels} 的4维卷积权重键名")


def expand_conv_weight(weight: torch.Tensor, new_in_channels: int, mode: str) -> torch.Tensor:
    """将卷积权重的输入通道维度从 old 扩展到 new。

    Args:
        weight (torch.Tensor): 原始卷积权重，形状为 (out_c, in_c, k, k)。
        new_in_channels (int): 新的输入通道数（例如 9）。
        mode (str): 扩展模式，可选 "repeat" 或 "repeat_avg"。

    Returns:
        (torch.Tensor): 扩展后的卷积权重张量。

    Raises:
        ValueError: 当 new_in_channels 不是 old_in_channels 的整数倍时抛出。
    """
    old_in_channels = weight.shape[1]  # 旧输入通道数
    if new_in_channels % old_in_channels != 0:
        raise ValueError(f"new_in_channels={new_in_channels} 必须是 old_in_channels={old_in_channels} 的整数倍")

    repeat_factor = new_in_channels // old_in_channels  # 通道重复次数
    expanded = weight.repeat(1, repeat_factor, 1, 1)  # 通过重复在通道维度扩展
    if mode == "repeat_avg":
        expanded = expanded / repeat_factor  # 做均值缩放，避免激活幅度显著放大
    elif mode != "repeat":
        raise ValueError(f"不支持的 mode: {mode}")
    return expanded


def rebuild_model_with_new_channels(
    model: torch.nn.Module, old_in_channels: int, new_in_channels: int, mode: str
) -> tuple[torch.nn.Module, str]:
    """重建指定输入通道数的新模型，并加载修补后的权重。

    Args:
        model (torch.nn.Module): Ultralytics 的模型对象（DetectionModel 等）。
        old_in_channels (int): 旧输入通道数（例如 3）。
        new_in_channels (int): 新输入通道数（例如 9）。
        mode (str): 扩展模式，可选 "repeat" 或 "repeat_avg"。

    Returns:
        (tuple[torch.nn.Module, str]): (重建后的模型对象, 被修补的卷积权重键名)。
    """
    state_dict = model.state_dict()  # 原模型权重字典
    key = find_first_conv_key(state_dict, old_in_channels=old_in_channels)  # 首层卷积权重键名
    weight = state_dict[key]  # 原始首层卷积权重
    state_dict[key] = expand_conv_weight(weight, new_in_channels=new_in_channels, mode=mode)  # 扩展首层卷积权重

    yaml_cfg = getattr(model, "yaml", None)  # 模型yaml配置
    nc = getattr(model, "nc", None)  # 类别数
    if yaml_cfg is None or nc is None:
        raise AttributeError("模型缺少 yaml 或 nc 属性，无法自动重建为新通道数")

    new_model = DetectionModel(cfg=yaml_cfg, ch=new_in_channels, nc=nc, verbose=False)  # 新通道数模型
    new_model.load_state_dict(state_dict, strict=False)
    return new_model, key


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        (argparse.Namespace): 解析后的参数对象。
    """
    parser = argparse.ArgumentParser(description="扩展首层卷积输入通道数（例如 3 -> 9）")
    parser.add_argument("--weights", type=Path, required=True, help="输入权重路径（.pt）")
    parser.add_argument("--out", type=Path, required=True, help="输出权重路径（.pt）")
    parser.add_argument("--old-in", type=int, default=3, help="旧输入通道数（默认 3）")
    parser.add_argument("--new-in", type=int, default=9, help="新输入通道数（默认 9）")
    parser.add_argument(
        "--mode",
        type=str,
        default="repeat_avg",
        choices=("repeat", "repeat_avg"),
        help="扩展模式：repeat 或 repeat_avg（默认 repeat_avg 更稳）",
    )
    return parser.parse_args()


def main() -> None:
    """脚本入口：加载权重、扩展首层卷积、保存新权重。"""
    args = parse_args()
    weights_path = args.weights  # 输入权重路径
    out_path = args.out  # 输出权重路径
    old_in = args.old_in  # 旧输入通道数
    new_in = args.new_in  # 新输入通道数
    mode = args.mode  # 扩展模式

    # 注意：weights_only=False 可能执行pickle反序列化，仅对可信权重使用。
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)  # 加载完整checkpoint

    if "model" not in ckpt:
        raise KeyError("checkpoint 中未找到 'model' 字段，无法自动修补首层卷积")

    new_model, key = rebuild_model_with_new_channels(
        ckpt["model"], old_in_channels=old_in, new_in_channels=new_in, mode=mode
    )
    ckpt["model"] = new_model  # 替换为新通道数模型
    print(f"[model] patched key={key} old_in={old_in} new_in={new_in} mode={mode}")

    # 若存在 EMA 权重，也一并修补，保持训练初期一致性。
    ema = ckpt.get("ema", None)  # EMA 模型对象
    if ema is not None and hasattr(ema, "state_dict"):
        new_ema, ema_key = rebuild_model_with_new_channels(
            ema, old_in_channels=old_in, new_in_channels=new_in, mode=mode
        )
        ckpt["ema"] = new_ema  # 替换EMA模型
        print(f"[ema] patched key={ema_key} old_in={old_in} new_in={new_in} mode={mode}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    print(f"[done] saved to: {out_path}")


if __name__ == "__main__":
    main()
