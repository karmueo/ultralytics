from ultralytics import YOLO

# 加载模型（与原脚本保持一致）
model = YOLO(
    "ultralytics/cfg/models/26/yolo26s-p2-2nc.yaml"  # 2类配置（与数据集类别数一致）
).load("/home/tl/data/weights/yolo/yolo26s_9ch.pt")

# 3帧堆叠训练配置（9通道输入）
results = model.train(
    data="./ultralytics/cfg/datasets/110_rgb_3frames.yaml",  # 数据集配置（9通道）
    epochs=100,
    imgsz=640,
    # device=[0, 1, 2, 3],
    device=[1],
    batch=-1,
    multi_scale=False,
    cache=True,
    save_period=-1,
    name="yolo26s_110_rgb_3frames_9ch_v1",  # 实验名称（3帧9通道）
    # name="yolov11s_110_rgb_640_v11",
    plots=True,
    # lr0=0.001,
    # optimizer="SGD",
    optimizer="auto",
    amp=True,  # 自动混合精度训练
    exist_ok=True,  # 允许覆盖同名实验目录
    # profile=True,
    cls=0.75,
    mixup=0.1,
    mosaic=1,
    rect=True,
    # resume=True
    patience=20,
    hsv_h=0.0,  # 关闭 HSV 增强：9通道输入不适配 HSV 颜色空间变换
    hsv_s=0.0,  # 关闭 HSV 增强：避免在多通道输入上触发 cv2.cvtColor 错误
    hsv_v=0.0,  # 关闭 HSV 增强：多帧堆叠时建议先禁用再逐步恢复
    degrees=10,  # 稍微旋转，防止模型只记住水平飞行的姿态
    copy_paste=0.2,  # 开启 CopyPaste，提升抗干扰能力
)
