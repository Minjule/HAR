# export_onnx.py
import torch
from model import TemporalMobileNet

device = "cpu"
num_classes = 3  # set to your number
model = TemporalMobileNet(num_classes=num_classes, pretrained=False)
model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
model.eval()

# Dummy input shape: (1, T, C, H, W)
dummy = torch.randn(1, 8, 3, 160, 160)
torch.onnx.export(model, dummy, "har_mobilenet.onnx",
                  input_names=["input"], output_names=["logits"],
                  opset_version=12, dynamic_axes={"input": {0:"batch"}})
