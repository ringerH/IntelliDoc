import os
import shutil
import json
from PIL import Image
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.services.pipeline import get_pipeline_manager

def main():
    demo_dir = Path(__file__).resolve().parent
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Copy generated images from artifact folder to workspace demo folder
    artifact_invoice = Path(r"C:\Users\delah\.gemini\antigravity-ide\brain\1cc7322a-7bdd-4896-ad18-15bc50091712\invoice_sample_1784224892630.png")
    artifact_receipt = Path(r"C:\Users\delah\.gemini\antigravity-ide\brain\1cc7322a-7bdd-4896-ad18-15bc50091712\receipt_sample_1784224909216.png")
    
    dest_invoice = demo_dir / "invoice.png"
    dest_receipt = demo_dir / "receipt.png"
    
    if artifact_invoice.exists():
        shutil.copy(artifact_invoice, dest_invoice)
        print(f"Copied invoice to {dest_invoice}")
    else:
        print(f"Artifact invoice not found at {artifact_invoice}")
        
    if artifact_receipt.exists():
        shutil.copy(artifact_receipt, dest_receipt)
        print(f"Copied receipt to {dest_receipt}")
    else:
        print(f"Artifact receipt not found at {artifact_receipt}")

    # 2. Load PipelineManager and process these images
    print("Loading ML pipeline manager...")
    manager = get_pipeline_manager()
    
    # Process invoice
    if dest_invoice.exists():
        print("Processing invoice document...")
        img = Image.open(dest_invoice)
        res = manager.process_document(img, backend="onnx")
        
        # Save output response JSON
        out_path = demo_dir / "invoice_response.json"
        with open(out_path, "w") as f:
            json.dump(res, f, indent=4)
        print(f"Saved invoice processing response to {out_path}")
        
    # Process receipt
    if dest_receipt.exists():
        print("Processing receipt document...")
        img = Image.open(dest_receipt)
        res = manager.process_document(img, backend="onnx")
        
        # Save output response JSON
        out_path = demo_dir / "receipt_response.json"
        with open(out_path, "w") as f:
            json.dump(res, f, indent=4)
        print(f"Saved receipt processing response to {out_path}")

if __name__ == "__main__":
    main()
