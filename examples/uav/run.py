# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Train, validate, and predict with fixed 640x352 UAV images."""

from __future__ import annotations

import argparse
from copy import copy
from pathlib import Path

from ultralytics import YOLO
from ultralytics.data.augment import Compose, LetterBox, RandomFlip, RandomHSV
from ultralytics.data.dataset import YOLODataset
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect import DetectionPredictor, DetectionTrainer, DetectionValidator
from ultralytics.utils import colorstr, ops
from ultralytics.utils.torch_utils import unwrap_model

IMAGE_SHAPE = (352, 640)
DEFAULT_DATA = str(Path(__file__).resolve().parents[3] / "dataset/prepared/uav_normal_640x352_seed0/data.yaml")
DEFAULT_MODEL = "yolo26n.pt"
DEFAULT_PROJECT = "runs/uav"


class UAVDataset(YOLODataset):
    """Detection dataset with fixed-size stretching and only color and horizontal-flip augmentation."""

    def build_transforms(self, hyp):
        """Build transforms that preserve every box and emit the fixed UAV image shape."""
        transforms = Compose([LetterBox(new_shape=IMAGE_SHAPE, scale_fill=True)])
        if self.augment:
            transforms.append(RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v))
            transforms.append(RandomFlip(p=hyp.fliplr, direction="horizontal"))
        transforms.append(
            self.format_class(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,
            )
        )
        return transforms


def build_uav_dataset(cfg, data, img_path, batch, stride, mode):
    """Build the shared fixed-shape dataset for training and validation."""
    return UAVDataset(
        img_path=img_path,
        imgsz=IMAGE_SHAPE[1],
        batch_size=batch,
        augment=mode == "train",
        hyp=cfg,
        prefix=colorstr(f"{mode}: "),
        rect=False,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        classes=cfg.classes,
        data=data,
        task="detect",
    )


class UAVTrainer(DetectionTrainer):
    """Detection trainer that uses the fixed-shape UAV dataset and validator."""

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build the UAV dataset without rectangular or multi-scale resizing."""
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_uav_dataset(self.args, self.data, img_path, batch, stride, mode)

    def get_validator(self):
        """Return the validator configured for fixed-shape UAV samples."""
        return UAVValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)


class UAVValidator(DetectionValidator):
    """Detection validator using the same fixed-shape dataset as training."""

    def build_dataset(self, img_path, mode="val", batch=None):
        """Build a non-rectangular UAV validation dataset."""
        return build_uav_dataset(self.args, self.data, img_path, batch, self.stride, mode)


class UAVPredictor(DetectionPredictor):
    """Predictor that stretches each source image and restores boxes with independent axis scales."""

    scale_fill = True

    def setup_source(self, source, stride=None):
        """Set up inference and retain the fixed 352x640 model input shape."""
        super().setup_source(source, stride)
        self.args.imgsz = IMAGE_SHAPE
        self.imgsz = list(IMAGE_SHAPE)

    def construct_result(self, pred, img, orig_img, img_path):
        """Scale detections independently in x and y to the original image dimensions."""
        ops.scale_boxes(
            img.shape[2:],
            pred[:, :4],
            orig_img.shape[:2],
            ratio_pad=((img.shape[2] / orig_img.shape[0], img.shape[3] / orig_img.shape[1]), (0, 0)),
            padding=False,
        )
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])


def add_common_args(parser, *, data=True, workers=True, batch=32, name="train", explicit=False):
    """Add the small set of options shared by the three modes."""
    if data:
        parser.add_argument("--data", default=None if explicit else DEFAULT_DATA, help="Dataset YAML")
    parser.add_argument(
        "--model",
        default=None if explicit else DEFAULT_MODEL,
        help="YOLO26n pretrained weights or a trained checkpoint",
    )
    parser.add_argument("--device", default=None if explicit else "0", help="Device, for example 0 or 0,1")
    parser.add_argument("--batch", type=int, default=None if explicit else batch, help="Batch size")
    if workers:
        parser.add_argument("--workers", type=int, default=None if explicit else 8, help="Data loader workers")
    parser.add_argument("--name", default=None if explicit else name, help="Run name")
    parser.add_argument("--project", default=None if explicit else DEFAULT_PROJECT, help="Output project directory")


def parse_args():
    """Parse the train, val, and predict command line modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)

    train = modes.add_parser("train", help="Train a UAV detector")
    add_common_args(train, explicit=True)
    train.add_argument("--epochs", type=int, default=None, help="Training epochs")
    train.add_argument("--save-period", type=int, default=None, help="Save a checkpoint every N epochs")
    train.add_argument("--resume", type=Path, metavar="CHECKPOINT", help="Resume from an explicit checkpoint")

    val = modes.add_parser("val", help="Validate a detector")
    add_common_args(val, name="val")

    predict = modes.add_parser("predict", help="Run prediction")
    add_common_args(predict, data=False, workers=False, batch=1, name="predict")
    predict.add_argument("--source", required=True, help="Image, video, directory, glob, or stream source")
    predict.add_argument("--save", action="store_true", help="Save annotated results")
    predict.add_argument("--save-txt", action="store_true", help="Save detections as text files")
    predict.add_argument("--show", action="store_true", help="Display prediction results")
    predict.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    predict.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")

    return parser.parse_args()


def main():
    """Run the selected model mode."""
    args = parse_args()
    if args.mode == "train":
        resume = args.resume.expanduser().resolve() if args.resume else None
        if resume is not None and not resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume}")
        if resume:
            if args.model is not None:
                raise ValueError("--model cannot be combined with --resume; the resume checkpoint supplies the model")
            model = YOLO(str(resume), task="detect")
            train_args = {"resume": str(resume)}
            for key in ("data", "device", "batch", "epochs", "workers", "save_period", "project", "name"):
                value = getattr(args, key)
                if value is not None:
                    train_args[key] = value
        else:
            model = YOLO(args.model or DEFAULT_MODEL, task="detect")
            train_args = {
                "data": args.data if args.data is not None else DEFAULT_DATA,
                "device": args.device if args.device is not None else "0",
                "batch": args.batch if args.batch is not None else 32,
                "epochs": args.epochs if args.epochs is not None else 100,
                "workers": args.workers if args.workers is not None else 8,
                "patience": 20,
                "imgsz": IMAGE_SHAPE[1],
                "multi_scale": 0.0,
                "rect": False,
                "mosaic": 0.0,
                "mixup": 0.0,
                "cutmix": 0.0,
                "copy_paste": 0.0,
                "degrees": 0.0,
                "translate": 0.0,
                "scale": 0.0,
                "shear": 0.0,
                "perspective": 0.0,
                "flipud": 0.0,
                "close_mosaic": 0,
                "amp": True,
                "optimizer": "auto",
                "seed": 0,
                "deterministic": True,
                "cache": False,
                "save_period": args.save_period if args.save_period is not None else -1,
                "project": args.project if args.project is not None else DEFAULT_PROJECT,
                "name": args.name if args.name is not None else "train",
                "exist_ok": False,
                "resume": False,
            }
        model.train(trainer=UAVTrainer, **train_args)
    elif args.mode == "val":
        model = YOLO(args.model, task="detect")
        model.val(
            validator=UAVValidator,
            data=args.data,
            device=args.device,
            batch=args.batch,
            workers=args.workers,
            imgsz=IMAGE_SHAPE[1],
            rect=False,
            project=args.project,
            name=args.name,
            exist_ok=False,
        )
    else:
        model = YOLO(args.model, task="detect")
        model.predict(
            predictor=UAVPredictor,
            source=args.source,
            device=args.device,
            batch=args.batch,
            imgsz=IMAGE_SHAPE,
            conf=args.conf,
            iou=args.iou,
            save=args.save,
            save_txt=args.save_txt,
            show=args.show,
            project=args.project,
            name=args.name,
            exist_ok=False,
        )


if __name__ == "__main__":
    main()
