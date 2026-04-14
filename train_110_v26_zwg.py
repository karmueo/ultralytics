'''
Author: karmueo karmueo@163.com
Date: 2026-02-11 09:38:29
LastEditors: karmueo karmueo@163.com
LastEditTime: 2026-02-11 09:39:31
FilePath: /ultralytics/train_110_v26_zwg.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from ultralytics import YOLO

# Load a model
model = YOLO(
    # "ultralytics/cfg/models/26/yolo26-p2-p4-3nc.yaml"
    "ultralytics/cfg/models/26/yolo26n-p2-2nc.yaml"
).load("/home/tl/data/weights/yolo/yolo26n.pt")


# Train
results = model.train(
    data="./ultralytics/cfg/datasets/110_rgb.yaml",
    epochs=100,
    imgsz=640,
    # device=[0, 1, 2, 3],
    device=[2],
    batch=-1,
    multi_scale=False,
    cache=True,
    save_period=-1,
    name="zwg_yolo26n_110_rgb_352_640_cls0.6_v3",
    plots=True,
    optimizer="auto",
    amp=True,  # 自动混合精度训练
    exist_ok=True,  # 如果为true，则允许覆盖现有项目/名称目录。对于迭代实验有用，无需手动清除先前的输出
    cls=0.6,
    mixup=0.1,
    mosaic=1,
    rect=True,
    # resume=True
    patience=20,
    hsv_h=0.015,  # 色调增强 (原默认 0.015)，增加对不同颜色无人机的适应
    hsv_s=0.7,  # 饱和度增强 (原默认 0.7)
    hsv_v=0.4,  # 亮度增强 (原默认 0.4)，这对"夜间无人机"泛化至关重要
    degrees=10,  # 稍微旋转，防止模型只记住水平飞行的姿态
    copy_paste=0.1,  # 开启 CopyPaste，将鸟贴到无人机背景，反之亦然，极强地提升抗干扰能力
)
