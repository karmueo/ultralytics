# UAV 固定尺寸训练

仅用于可见光 UAV 检测：标准预训练 YOLO26n，类别 `uav`，训练、验证和 PyTorch 推理都直接拉伸到宽 640、高 352。入口通过 Dataset、Trainer、Validator、Predictor 子类复用框架，不修改框架核心。

## 环境

在仓库根目录运行。需要 Python 3.12，单卡训练默认 GPU 0，双卡需要两张可用 NVIDIA GPU。

```bash
uv venv --python 3.12 .venv-uav
uv pip install --python .venv-uav/bin/python torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-uav/bin/python -e . 'cloudpickle>=3.1.1'
source .venv-uav/bin/activate
```

CUDA wheel 配套版本来自 [PyTorch 官方安装表](https://pytorch.org/get-started/previous-versions/)。CUDA 驱动必须支持所选 wheel；无需安装独立 CUDA toolkit。

## 准备数据

`--source` 指向包含 `source_mapping.csv`、`images`、`labels` 的 `normal` 目录。先设置本机路径，输出使用独立目录。输出目录非空时直接拒绝。

```bash
UAV_SOURCE=/path/to/dataset/data/uav/normal
UAV_PREPARED=/path/to/dataset/prepared/uav_normal_640x352_seed0
python -m examples.uav.prepare --source "$UAV_SOURCE" --output "$UAV_PREPARED"
```

固定 seed 0，对排序后的 33 个视频组生成 10,000 个约 80/10/10 分配候选，按图片、正图和负图比例总误差选择；同分按组归属序列排序。每个集合包含正负样本，视频组和原图 SHA256 均不能跨集合。重复图片和负样本全部保留。

每张原图仅用 `cv2.INTER_LINEAR` 直接拉伸一次，保存无损 PNG；归一化 YOLO 标签原样复制。输出包括 `images/{train,val,test}`、`labels/{train,val,test}`、`data.yaml`、`split_manifest.json` 和 `stats.json`。训练直接使用准备结果，不重新划分。

## 训练

```bash
python -m examples.uav.run train --data "$UAV_PREPARED/data.yaml" --name uav_640x352
```

默认 100 轮、batch 32、早停 20 轮、GPU 0、workers 8、AMP、`optimizer=auto`、seed 0、确定性模式和关闭缓存。只使用 HSV 扰动与水平翻转；没有 Mosaic、MixUp、随机仿射或微小框过滤。已准备图片不再重采样。新实验使用 `exist_ok=False`，复用名称时由框架生成新目录。

可以覆盖数据、预训练权重、设备、batch、epochs、workers、实验名、输出项目及保存周期。以下为短程验收与双卡命令；不需要运行默认完整 100 轮。

```bash
python -m examples.uav.run train --data "$UAV_PREPARED/data.yaml" --epochs 2 --batch 32 --workers 8 --save-period 1 --project runs/uav --name single
python -m examples.uav.run train --data "$UAV_PREPARED/data.yaml" --device 0,1 --epochs 1 --batch 32 --workers 8 --project runs/uav --name dual
```

续训必须显式提供仍含优化器状态的有效检查点。`save-period=1` 的第一个周期文件为 `epoch0.pt`。下面从第 1 轮恢复至原来的第 2 轮终点；已完成训练的 `last.pt` / `best.pt` 已被框架移除优化器状态，不能作为续训起点。恢复时沿用检查点保存的 epochs 等训练参数，仅框架允许的设备、batch、workers 等参数可覆盖。

```bash
python -m examples.uav.run train --resume runs/uav/single/weights/epoch0.pt
```

## 验证与原图推理

```bash
python -m examples.uav.run val --data "$UAV_PREPARED/data.yaml" --model runs/uav/single/weights/best.pt --project runs/uav --name validation
python -m examples.uav.run predict --model runs/uav/single/weights/best.pt --source "$UAV_SOURCE/images/000001.jpeg" --project runs/uav --name prediction --save --save-txt
```

验证使用预先固定的 `val` 集合。原图推理直接拉伸至 640×352，再分别按横向和纵向比例把预测框还原到原图坐标。保存的类别名为 `uav`。完整参数见对应模式的 `--help`。

640×352 的像素量为 640×640 的 55%。实际速度收益需要在目标部署设备测量；小目标召回仍需完整对照实验确认。后续对比固定本次划分，记录 mAP50–95、召回、背景误报率和目标设备延迟，不使用独立测试集调参。
