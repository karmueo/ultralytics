import torch
import torchvision.transforms as T

from ultralytics import YOLO
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator


class CustomizedDataset(ClassificationDataset):
    """A customized dataset class for image classification with enhanced data augmentation transforms."""

    def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
        """Initialize a customized classification dataset with enhanced data augmentation transforms."""
        super().__init__(root, args, augment, prefix)

        # Add your custom training transforms here
        train_transforms = T.Compose(
            [
                T.Resize(size=(args.imgsz, args.imgsz), interpolation=T.InterpolationMode.BILINEAR),
                T.RandomHorizontalFlip(p=args.fliplr),
                T.RandomVerticalFlip(p=args.flipud),
                T.RandAugment(interpolation=T.InterpolationMode.BILINEAR),
                T.ColorJitter(brightness=args.hsv_v, contrast=args.hsv_v,
                              saturation=args.hsv_s, hue=args.hsv_h),
                T.ToTensor(),
                T.Normalize(mean=torch.tensor(0), std=torch.tensor(1)),
                T.RandomErasing(p=args.erasing, inplace=True),
            ]
        )

        # Add your custom validation transforms here
        val_transforms = T.Compose(
            [
                T.Resize(size=(args.imgsz, args.imgsz), interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=torch.tensor(0), std=torch.tensor(1)),
            ]
        )
        self.torch_transforms = train_transforms if augment else val_transforms


class CustomizedTrainer(ClassificationTrainer):
    """A customized trainer class for YOLO classification models with enhanced dataset handling."""

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        """Build a customized dataset for classification training and the validation during training."""
        return CustomizedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)


class CustomizedValidator(ClassificationValidator):
    """A customized validator class for YOLO classification models with enhanced dataset handling."""

    def build_dataset(self, img_path: str, mode: str = "train"):
        """Build a customized dataset for classification standalone validation."""
        return CustomizedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=self.args.split)


model = YOLO("ultralytics/cfg/models/11/yolo11m-cls-110.yaml").load(
    "/home/tl/data/weights/yolo11m-cls.pt")

results = model.train(
    data="/home/tl/data/datasets/classify/RGB",
    trainer=CustomizedTrainer,
    imgsz=32,
    epochs=100,
    batch=8192,
    device=[3],
    cache=True,
    save_period=3,
    # lr0=0.001,
    # optimizer="SGD",
    optimizer="auto",
    amp=True,  # 自动混合精度训练
    exist_ok=True,
    name="110_rgb_32*32_v0",
    plots=True,
    patience=10,
    erasing=0.1,
    dropout=0.02,
    hsv_h=0.001,
    hsv_v=0.001
)
print("[DONE] 训练完成。")

model.val(data="/home/tl/data/datasets/classify/RGB",
          validator=CustomizedValidator, imgsz=32, batch=32)
