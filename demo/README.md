# Document Intelligence Demo Gallery

This directory contains static, pre-processed input and output artifacts. Reviewers and recruiters can instantly inspect the document inputs and corresponding API output payloads without needing to install packages or download model checkpoints.

---

## Showcase Artifacts

### 1. Business Invoice Document
* **Input Image**: **[invoice.png](file:///d:/AntiG/IntelliDoc/demo/invoice.png)** (A mock corporate invoice containing tables, line items, and totals).
* **API Response JSON**: **[invoice_response.json](file:///d:/AntiG/IntelliDoc/demo/invoice_response.json)** (The actual parsed JSON returned by the service).
  - *Classification Result*: `invoice` (High confidence).
  - *Extracted Regions*: Layout coordinates mapped to tables, text blocks, and signatures.
  - *Full text*: Full-page parsed string content.

### 2. Retail Cash Receipt
* **Input Image**: **[receipt.png](file:///d:/AntiG/IntelliDoc/demo/receipt.png)** (A mock thermal checkout receipt with itemized totals and tax details).
* **API Response JSON**: **[receipt_response.json](file:///d:/AntiG/IntelliDoc/demo/receipt_response.json)** (The actual parsed JSON returned by the service).
  - *Classification Result*: `receipt` (High confidence).
  - *Extracted Regions*: Layout coordinates mapped to prices, store headers, and item lines.
  - *Full text*: Continuous full-page cash receipt text list.

---

## How it was Generated
These outputs were generated directly by executing the optimized document processing pipeline:
1. Document was categorized as `invoice` or `receipt` via MobileNetV3.
2. Structure layout boxes were predicted via YOLOv8.
3. EasyOCR scanned the full page exactly **once**, and word coordinates were mapped back to the YOLOv8 regions using Python/NumPy geometric intersection ratios.
