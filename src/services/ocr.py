import numpy as np
from PIL import Image

class DocumentOCR:
    def __init__(self):
        # Defer EasyOCR initialization until first OCR operation to reduce startup footprint
        self.reader = None

    def _ensure_reader(self):
        """Lazy loads the EasyOCR library and reader engine on CPU."""
        if self.reader is None:
            import easyocr
            print("Initializing EasyOCR Engine...")
            self.reader = easyocr.Reader(['en'], gpu=False)

    def process_ocr(self, image: Image.Image, regions: list) -> tuple[list, str]:
        """Runs OCR once on the full page image and maps text blocks to the layout regions.
        
        Args:
            image (Image.Image): The source document image.
            regions (list): List of dicts representing layout regions, e.g.
                            [{"box": [x1, y1, x2, y2], "class": "table", "confidence": 0.90}]
        Returns:
            tuple: (ocr_regions, full_text)
                   ocr_regions is a list of regions with "text" and OCR "confidence" fields.
                   full_text is a single string of all extracted document text.
        """
        self._ensure_reader()
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        
        # 1. Run EasyOCR exactly once on the full-page image
        ocr_results = self.reader.readtext(img_np)
        
        # 2. Reconstruct full page text
        full_text = " ".join([res[1] for res in ocr_results]).strip()
        
        # 3. Parse EasyOCR coordinate format into standardized bounding boxes [x1, y1, x2, y2]
        parsed_texts = []
        for bbox, text, conf in ocr_results:
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            tx1 = int(max(0, min(xs)))
            ty1 = int(max(0, min(ys)))
            tx2 = int(min(w, max(xs)))
            ty2 = int(min(h, max(ys)))
            parsed_texts.append({
                "box": [tx1, ty1, tx2, ty2],
                "text": text,
                "confidence": float(conf)
            })
            
        # Helper to compute intersection ratio (intersection area / area of text box)
        def get_intersection_ratio(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            
            if boxAArea == 0:
                return 0.0
            return interArea / float(boxAArea)

        # 4. Map OCR detections to YOLOv8 layout regions using geometric overlap
        ocr_regions = []
        for region in regions:
            rx1, ry1, rx2, ry2 = region["box"]
            region_box = [rx1, ry1, rx2, ry2]
            
            overlapping_texts = []
            overlapping_confs = []
            for item in parsed_texts:
                ratio = get_intersection_ratio(item["box"], region_box)
                if ratio >= 0.5:
                    overlapping_texts.append(item["text"])
                    overlapping_confs.append(item["confidence"])
                    
            region_text = " ".join(overlapping_texts).strip()
            # Mean confidence of all text items inside the region, default to 0.0 if empty
            region_conf = float(np.mean(overlapping_confs)) if overlapping_confs else 0.0
            
            ocr_regions.append({
                "box": region_box,
                "class": region["class"],
                "text": region_text,
                "confidence": region_conf
            })
            
        return ocr_regions, full_text

    def extract_text(self, image: Image.Image) -> str:
        """Extracts all text from the given Pillow Image. (Maintained for backwards compatibility)"""
        _, full_text = self.process_ocr(image, [])
        return full_text

    def extract_regions(self, image: Image.Image, regions: list) -> list:
        """Extracts text from individual layout regions of the image. (Maintained for backwards compatibility)"""
        ocr_regions, _ = self.process_ocr(image, regions)
        return ocr_regions
