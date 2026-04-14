from ultralytics import YOLO

# Load a model
model = YOLO(
    # "ultralytics/cfg/models/26/yolo26-p2-p4-3nc.yaml"
    # "ultralytics/cfg/models/26/yolo26n-p2-2nc.yaml"
    "ultralytics/cfg/models/26/yolo26n-2nc.yaml"
).load("/home/tl/data/weights/yolo/yolo26n.pt")


# Train
results = model.train(
    data="./ultralytics/cfg/datasets/simulation_rgb.yaml",
    epochs=100,
    imgsz=640,
    device=[0, 1, 2, 3],
    batch=128,
    multi_scale=False,
    cache=True,
    save_period=-1,
    name="yolo26n_uav_352_640_no-p2_simulation",
    plots=True,
    optimizer="auto",
    amp=True,  # 自动混合精度训练
    exist_ok=True,  # 如果为true，则允许覆盖现有项目/名称目录。对于迭代实验有用，无需手动清除先前的输出
    mixup=0.1,
    mosaic=1,
    rect=True,
    fraction=0.8,  # 训练集占比，剩余0.2作为验证集，simulation数据集没有单独的验证集
    # resume=True
    patience=20,
    hsv_h=0.015,  # 色调增强 (原默认 0.015)，增加对不同颜色无人机的适应
    hsv_s=0.7,  # 饱和度增强 (原默认 0.7)
    hsv_v=0.4,  # 亮度增强 (原默认 0.4)，这对"夜间无人机"泛化至关重要
    degrees=10,  # 稍微旋转，防止模型只记住水平飞行的姿态
    copy_paste=0.1,  # 开启 CopyPaste，将鸟贴到无人机背景，反之亦然，极强地提升抗干扰能力
)
