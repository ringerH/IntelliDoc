import os
import numpy as np
from PIL import Image
from pathlib import Path

class DocumentClassifier:
    def __init__(self, model_dir: Path, backend: str = "onnx"):
        self.backend = backend.lower()
        self.classes = ["invoice", "receipt", "id", "form"]
        
        if self.backend == "onnx":
            import onnxruntime as ort
            onnx_path = model_dir / "model.onnx"
            if not onnx_path.exists():
                raise FileNotFoundError(f"ONNX model file not found at {onnx_path}")
            # Load ONNX session (CPU execution provider)
            self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
        elif self.backend == "pytorch":
            # Lazy import PyTorch libraries only when active backend is PyTorch
            global torch, models
            import torch
            import torchvision.models as models
            
            pt_path = model_dir / "model.pt"
            if not pt_path.exists():
                raise FileNotFoundError(f"PyTorch model file not found at {pt_path}")
            
            # Recreate model architecture
            try:
                self.model = models.mobilenet_v3_small()
            except AttributeError:
                self.model = models.mobilenet_v3_small(pretrained=False)
                
            num_features = self.model.classifier[3].in_features
            self.model.classifier[3] = torch.nn.Linear(num_features, len(self.classes))
            
            # Load weights
            self.model.load_state_dict(torch.load(pt_path, map_location=torch.device("cpu")))
            self.model.eval()
        else:
            raise ValueError(f"Unsupported classification backend: {backend}")

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """Standardizes image shape and channel values for MobileNetV3."""
        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Resize to 224x224
        image = image.resize((224, 224), Image.Resampling.BILINEAR)
        
        # Convert to numpy array and scale to [0, 1]
        img_data = np.array(image).astype(np.float32) / 255.0
        
        # Normalize with ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_data = (img_data - mean) / std
        
        # Transpose from HWC to CHW and add batch dimension (BCHW)
        img_data = np.transpose(img_data, (2, 0, 1))
        img_data = np.expand_dims(img_data, axis=0)
        return img_data

    def predict(self, image: Image.Image) -> dict:
        """Predicts the document class.
        Returns:
            dict: {"predicted_class": str, "confidence": float, "probabilities": list}
        """
        processed_img = self._preprocess(image)
        
        if self.backend == "onnx":
            outputs = self.session.run(None, {self.input_name: processed_img})
            logits = outputs[0][0]
            # Softmax calculation using NumPy
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
        else:  # pytorch
            with torch.no_grad():
                tensor_img = torch.tensor(processed_img)
                outputs = self.model(tensor_img)
                probs = torch.softmax(outputs[0], dim=0).numpy()
                
        pred_idx = int(np.argmax(probs))
        return {
            "predicted_class": self.classes[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
        }
