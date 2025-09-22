from ultralytics import YOLO

# Load a model
model = YOLO(
    "./ultralytics/cfg/models/11/yolo11m_p2p.yaml"
    # "./ultralytics/cfg/models/11/yolo11m_ir_p2.yaml"
    # "./ultralytics/cfg/models/v8/yolov8m-p2p.yaml"
).load(
    # "runs/detect/yolov11m_110_ir_640_nanchang_bak/weights/last.pt"
    # "runs/detect/yolov11m_110_rgb_640_nanchang_v3/weights/last.pt"
    # "runs/detect/yolov8m_110_ir_640_nanchang/weights/last.pt"
    # "/home/tl/data/weights/yolov8m.pt"
    "/home/tl/data/weights/yolov11m.pt"
    # "/home/tl/data/weights/yolov11l.pt"
)  # load a pretrained model (recommended for training)


# Train the model with 2 GPUs
results = model.train(
    # data="./ultralytics/cfg/datasets/110_infrared.yaml",
    data="./ultralytics/cfg/datasets/110_rgb.yaml",
    epochs=100,
    imgsz=640,
    device=[0, 1, 2, 3],
    batch=32,
    multi_scale=True,
    cache=True,
    save_period=10,
    # single_cls=True,
    name="yolov11m_110_rgb_640_nanchang_v5",
    # name="yolov11m_110_ir_640_nanchang",
    plots=True,
    # lr0=0.001,
    # optimizer="SGD",
    optimizer="auto",
    amp=True,  # 自动混合精度训练
    exist_ok=True,  # 如果为true，则允许覆盖现有项目/名称目录。对于迭代实验有用，无需手动清除先前的输出
    profile=True,  # 在训练过程中启用onnx和TensorRT速度的分析，可用于优化模型部署。
    mixup=0,
    mosaic=0,
    # resume=True
    patience=10,
    hsv_h=0.001,
    hsv_v=0.001
)
