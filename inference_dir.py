import argparse
from pathlib import Path
import shutil
from ultralytics import YOLO
from PIL import Image
from collections import Counter, defaultdict
import csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference on a directory")
    parser.add_argument(
        "--model",
        type=str,
        default=(
            "/home/tl/data_80/triton/server/ultralytics/runs/detect/"
            "yolov11m_110_rgb_640_nanchang_v5/weights/best.pt"
        ),
        help="Path to the YOLO model file",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="/home/tl/data_80/data/video/110/RGB/bird/big_bird_9.18/",
        help="Path to directory containing images and videos for inference",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results",
        help="Directory to save result images",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help=(
            "If set, clear previous output folder and reuse it. "
            "Otherwise create an indexed subfolder run_<n> inside output."
        ),
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="If set, do not save result images (counts CSV will still be written).",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use half precision for inference when supported (fp16)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size to use during inference",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.45,
        help="Confidence threshold for detections",
    )
    return parser.parse_args()


def prepare_output_dir(output_dir: str, clear: bool) -> Path:
    output_path = Path(output_dir)
    if clear:
        if output_path.exists() and output_path.is_dir():
            resolved = str(output_path.resolve())
            if resolved in ("/", ""):
                raise SystemExit("Refusing to delete root or empty directory")
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    # incremental run_N
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    runs = [p for p in output_path.iterdir() if p.is_dir() and p.name.startswith("run_")]
    max_n = 0
    for p in runs:
        try:
            n = int(p.name.split("_", 1)[1])
            if n > max_n:
                max_n = n
        except Exception:
            continue
    next_n = max_n + 1
    final_output = output_path / f"run_{next_n}"
    final_output.mkdir(parents=True, exist_ok=True)
    return final_output


def main(args: argparse.Namespace) -> None:
    # Load model
    model = YOLO(args.model)

    source = args.source
    final_output = prepare_output_dir(args.output, args.clear_output)

    # Run inference
    results = model(source, stream=True, half=args.half, batch=args.batch, conf=args.conf)

    counts = Counter()              # global counts
    detected_total = 0              # global detections
    save_images = not args.no_save
    source_counters = defaultdict(int)   # per-video frame counters
    video_counts: dict[str, Counter] = {}  # per-video class counts
    video_dirs: dict[str, Path] = {}       # map source path -> output dir

    for i, r in enumerate(results):
        # plot
        im_bgr = r.plot()
        im_rgb = Image.fromarray(im_bgr[..., ::-1])

        # determine source-based subfolder name (try various attributes)
        src_path = None
        for attr in ("path", "orig_img_path", "orig_path", "img_file"):
            try:
                val = getattr(r, attr, None)
            except Exception:
                val = None
            if val:
                src_path = val
                break

        if src_path:
            try:
                src_name = Path(str(src_path)).stem
            except Exception:
                src_name = f"item_{i}"
        else:
            src_name = f"item_{i}"

        # decide if source is video
        src_p = Path(str(src_path)) if src_path else None
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".mpg", ".mpeg"}
        is_video = bool(src_p and src_p.suffix.lower() in video_exts)

        # prepare per-video directory only once
        per_output: Path | None = None
        if is_video and src_path:
            key = str(src_p)
            if key not in video_dirs:
                safe_name = str(src_p.stem).replace(" ", "_")
                out_dir = final_output / safe_name
                # ensure unique if collision
                if out_dir.exists():
                    idx2 = 1
                    while (final_output / f"{safe_name}_{idx2}").exists():
                        idx2 += 1
                    out_dir = final_output / f"{safe_name}_{idx2}"
                out_dir.mkdir(parents=True, exist_ok=True)
                video_dirs[key] = out_dir
                video_counts[key] = Counter()
            per_output = video_dirs[key]

        # save image
        if save_images:
            if is_video and per_output is not None:
                key = str(src_p)
                frame_idx = source_counters[key] + 1
                source_counters[key] = frame_idx
                filename = f"{src_p.stem}_frame{frame_idx:06d}.jpg"
                out_path = per_output / filename
            else:
                # images or unknown source -> save directly under final_output
                base_name = src_p.stem if src_p else f"results_{i}"
                filename = f"{base_name}.jpg"
                out_path = final_output / filename
            try:
                r.save(filename=str(out_path))
            except Exception:
                im_rgb.save(out_path)

        # counting
        try:
            boxes = getattr(r, "boxes", None)
            if boxes is not None:
                cls_tensor = getattr(boxes, "cls", None)
                if cls_tensor is not None:
                    try:
                        cls_list = cls_tensor.detach().cpu().numpy().astype(int).tolist()
                    except Exception:
                        cls_list = []
                    if cls_list:
                        counts.update(cls_list)
                        detected_total += len(cls_list)
                        if is_video and src_path:
                            video_counts[str(src_p)].update(cls_list)
        except Exception:
            pass

    # write global CSV
    csv_path = final_output / "counts.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "count"])
        names = None
        try:
            names = getattr(results, "names", None)
        except Exception:
            names = None
        if names is None:
            try:
                names = getattr(model, "names", None)
            except Exception:
                names = None
        for class_id, cnt in sorted(counts.items()):
            class_name = None
            if names is not None:
                try:
                    class_name = names[int(class_id)]
                except Exception:
                    class_name = ""
            writer.writerow([int(class_id), class_name, int(cnt)])
        writer.writerow(["total", "", int(detected_total)])

    # write per-video CSVs
    for key, vc in video_counts.items():
        out_dir = video_dirs[key]
        per_csv = out_dir / "counts.csv"
        with open(per_csv, "w", newline="") as pf:
            pw = csv.writer(pf)
            pw.writerow(["class_id", "class_name", "count"])
            names = None
            try:
                names = getattr(model, "names", None)
            except Exception:
                names = None
            for class_id, cnt in sorted(vc.items()):
                class_name = ""
                if names is not None:
                    try:
                        class_name = names[int(class_id)]
                    except Exception:
                        class_name = ""
                pw.writerow([int(class_id), class_name, int(cnt)])
            pw.writerow(["total", "", int(sum(vc.values()))])


if __name__ == "__main__":
    main(parse_args())
