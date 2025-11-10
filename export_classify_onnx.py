import argparse
from ultralytics import YOLO

# Parse command line arguments
parser = argparse.ArgumentParser(description='Export YOLO model to ONNX format')
parser.add_argument('--model', required=True, help='Path to the YOLO model file')
parser.add_argument('--imgsz', type=int, nargs='+', default=[32, 32], help='Image size for export (default: 32 32)')
parser.add_argument('--batch', type=int, default=4, help='Batch size for export (default: 4)')
parser.add_argument('--simplify', action='store_true', help='Simplify the ONNX model')
parser.add_argument('--dynamic', action='store_true', help='Use dynamic axes')

args = parser.parse_args()

# Load the model
model = YOLO(args.model)

# Export the model
model.export(format="onnx",
             imgsz=tuple(args.imgsz),
             batch=args.batch,
             simplify=args.simplify,
             dynamic=args.dynamic)

# python export_classify_onnx.py --model runs/classify/110_rgb_32*32_v0/weights/best.pt --imgsz 32 32 --simplify