import os
import numpy as np
from PIL import Image
from pathlib import Path

class DocumentDetector:
    def __init__(self, model_dir: Path, backend: str = "onnx"):
        self.backend = backend.lower()
        
        # Standard COCO classes for our pretrained YOLOv8 model.
        # We will map specific COCO classes to representation layout regions:
        # - "dining table" (class 60) -> "table"
        # - "book" (class 73) -> "text_block"
        # - "tie" (class 27) -> "signature"
        self.class_mapping = {
            60: "table",
            73: "text_block",
            27: "signature"
        }
        
        # Load all 80 COCO classes names for fallback
        self.coco_classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        ]

        if self.backend == "onnx":
            import onnxruntime as ort
            onnx_path = model_dir / "model.onnx"
            if not onnx_path.exists():
                raise FileNotFoundError(f"ONNX model file not found at {onnx_path}")
            self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
        elif self.backend == "pytorch":
            from ultralytics import YOLO
            pt_path = model_dir / "model.pt"
            if not pt_path.exists():
                raise FileNotFoundError(f"PyTorch model file not found at {pt_path}")
            self.model = YOLO(str(pt_path))
        else:
            raise ValueError(f"Unsupported detection backend: {backend}")

    def _preprocess_onnx(self, image: Image.Image):
        """Preprocesses image to shape (1, 3, 640, 640) for YOLOv8 ONNX."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        orig_width, orig_height = image.size
        
        # Resize to 640x640
        img_resized = image.resize((640, 640), Image.Resampling.BILINEAR)
        img_data = np.array(img_resized).astype(np.float32) / 255.0
        
        # HWC to CHW and expand to BCHW
        img_data = np.transpose(img_data, (2, 0, 1))
        img_data = np.expand_dims(img_data, axis=0)
        return img_data, orig_width, orig_height

    def _nms(self, boxes, scores, iou_threshold=0.45):
        """Non-Maximum Suppression (NMS) implementation in NumPy."""
        if len(boxes) == 0:
            return []
            
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        order = scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
            
        return keep

    def _postprocess_onnx(self, outputs, orig_w, orig_h, conf_threshold=0.25):
        """Parses raw YOLOv8 outputs (1, 84, 8400) into boxes, scores, and classes."""
        output = outputs[0][0]  # shape: (84, 8400)
        
        # Transpose output to (8400, 84)
        output = np.transpose(output, (1, 0))
        
        boxes = []
        scores = []
        class_ids = []
        
        for row in output:
            classes_scores = row[4:]
            class_id = np.argmax(classes_scores)
            score = classes_scores[class_id]
            
            if score > conf_threshold:
                # Box coordinates are cx, cy, w, h
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                
                # Scale coordinates back to original size
                cx = (cx / 640.0) * orig_w
                cy = (cy / 640.0) * orig_h
                w = (w / 640.0) * orig_w
                h = (h / 640.0) * orig_h
                
                # Convert to x1, y1, x2, y2
                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)
                x2 = int(cx + w / 2)
                y2 = int(cy + h / 2)
                
                boxes.append([x1, y1, x2, y2])
                scores.append(float(score))
                class_ids.append(int(class_id))
                
        if len(boxes) == 0:
            return []
            
        boxes = np.array(boxes)
        scores = np.array(scores)
        class_ids = np.array(class_ids)
        
        keep_indices = self._nms(boxes, scores)
        
        results = []
        for idx in keep_indices:
            cid = class_ids[idx]
            label = self.class_mapping.get(cid, self.coco_classes[cid])
            results.append({
                "box": boxes[idx].tolist(),
                "class": label,
                "confidence": float(scores[idx]),
                "class_id": int(cid)
            })
            
        return results

    def detect(self, image: Image.Image, conf_threshold: float = 0.25) -> list:
        """Detects regions of interest in the document.
        Returns:
            list of dicts: [{"box": [x1, y1, x2, y2], "class": str, "confidence": float}]
        """
        if self.backend == "onnx":
            img_data, orig_w, orig_h = self._preprocess_onnx(image)
            outputs = self.session.run(None, {self.input_name: img_data})
            return self._postprocess_onnx(outputs, orig_w, orig_h, conf_threshold)
        else:  # pytorch
            # Run inference through ultralytics
            results = self.model.predict(image, conf=conf_threshold, verbose=False)
            boxes = results[0].boxes
            
            detected = []
            for box in boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int).tolist() # [x1, y1, x2, y2]
                score = float(box.conf[0].cpu().numpy())
                cid = int(box.cls[0].cpu().numpy())
                label = self.class_mapping.get(cid, self.coco_classes[cid])
                detected.append({
                    "box": coords,
                    "class": label,
                    "confidence": score,
                    "class_id": cid
                })
            return detected
