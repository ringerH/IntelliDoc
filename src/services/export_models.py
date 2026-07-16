import os
import shutil
from pathlib import Path
import torch
import torchvision.models as models
from ultralytics import YOLO

def export_classifier(output_dir: Path):
    """Downloads MobileNetV3, modifies classifier head, and exports to PyTorch state dict and ONNX format."""
    print("Preparing Document Classifier model (MobileNetV3)...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pt_path = output_dir / "model.pt"
    onnx_path = output_dir / "model.onnx"
    
    # 1. Create model modified for 4 output classes (invoice, receipt, ID, form)
    # Using weights=models.MobileNet_V3_Small_Weights.DEFAULT in modern torchvision
    try:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    except AttributeError:
        # Fallback for older torchvision versions
        model = models.mobilenet_v3_small(pretrained=True)
        
    num_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(num_features, 4)
    model.eval()
    
    # 2. Save PyTorch state dict
    torch.save(model.state_dict(), pt_path)
    print(f"Saved PyTorch classifier state dict to {pt_path}")
    
    # 3. Export to ONNX
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=14
    )
    print(f"Exported ONNX classifier to {onnx_path}")

def export_detector(output_dir: Path):
    """Downloads YOLOv8 nano model and copies it and its ONNX export to the target directory."""
    print("Preparing Document Region Detector model (YOLOv8)...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # YOLOv8n download and ONNX export via ultralytics
    # This downloads yolov8n.pt in the current working directory
    model = YOLO("yolov8n.pt")
    exported_onnx_path = model.export(format="onnx", opset=12) # outputs yolov8n.onnx
    
    pt_dest = output_dir / "model.pt"
    onnx_dest = output_dir / "model.onnx"
    
    # Move/copy files to destination
    shutil.copy("yolov8n.pt", pt_dest)
    shutil.copy(exported_onnx_path, onnx_dest)
    
    # Cleanup local temp files
    if os.path.exists("yolov8n.pt"):
        os.remove("yolov8n.pt")
    if os.path.exists(exported_onnx_path):
        os.remove(exported_onnx_path)
        
    print(f"Saved PyTorch YOLOv8 detector to {pt_dest}")
    print(f"Saved ONNX YOLOv8 detector to {onnx_dest}")

if __name__ == "__main__":
    base_model_dir = Path(__file__).resolve().parent.parent.parent / "models"
    
    classifier_v1_dir = base_model_dir / "classifier" / "v1"
    detector_v1_dir = base_model_dir / "detector" / "v1"
    
    export_classifier(classifier_v1_dir)
    export_detector(detector_v1_dir)
    print("Model generation and registration completed successfully.")
