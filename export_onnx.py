# export_onnx.py
import torch
from stgcn_model import STGCN
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="stgcn_best_model.pth")
parser.add_argument("--onnx", default="stgcn.onnx")
parser.add_argument("--seq_len", type=int, default=30)
parser.add_argument("--num_joints", type=int, default=33)
parser.add_argument("--in_channels", type=int, default=3)
parser.add_argument("--opset", type=int, default=13)
args = parser.parse_args()

def export_to_onnx():
    ckpt = torch.load(args.ckpt, map_location="cpu")
    classes = ckpt.get("classes", None)
    if classes is None:
        print("[WARN] classes not found in checkpoint, using num_class from weights shape")
        # attempt infer num_classes from fc weight
        num_classes = next(iter(ckpt.values())).shape[0] if isinstance(ckpt, dict) else 1
    else:
        num_classes = len(classes)
    print("[INFO] inferred num_classes =", num_classes)

    # instantiate model and load weights
    model = STGCN(num_class=num_classes, num_point=args.num_joints, in_channels=args.in_channels)
    # If checkpoint contains dict with 'model_state' key:
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    dummy = torch.randn(1, args.in_channels, args.seq_len, args.num_joints, 1, dtype=torch.float32)

    print(f"[INFO] Exporting ONNX -> {args.onnx}  (opset={args.opset})")
    torch.onnx.export(
        model,
        dummy,
        args.onnx,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['logits'],
        # dynamic_axes can be given, but avoid frames dynamic if problematic
        dynamic_axes={
            'input': {0: 'batch'},   # keep frames static to avoid listconstruct issues
            'logits': {0: 'batch'}
        }
    )
    print("[INFO] ONNX export finished.")

if __name__ == "__main__":
    export_to_onnx()
