---
language: vi
pipeline_tag: text-classification
base_model: vinai/phobert-base-v2
tags:
  - intent-classification
  - phobert
  - vietnamese
metrics:
  - accuracy
  - f1
---

# PhoBERT Vietnamese Intent Classification

Fine-tune `vinai/phobert-base-v2` để phân loại yêu cầu tiếng Việt thành 5 ý định,
phục vụ hệ thống hỗ trợ đọc chữ và tìm hiểu môi trường qua hình ảnh.
Model chỉ phân loại yêu cầu; đọc ảnh, mô tả ảnh và trả lời câu hỏi về ảnh do thành phần khác đảm nhận.

Mã nguồn: https://github.com/NguyenThHien/NLU

## Nhãn

| ID | Nhãn | Ví dụ |
|---:|---|---|
| 0 | `READ_TEXT` | Đọc dòng chữ trên cửa. |
| 1 | `DESCRIBE_SCENE` | Mô tả toàn cảnh nhà bếp. |
| 2 | `ANSWER_VISUAL_QUESTION` | Trong hình có bao nhiêu người? |
| 3 | `END_SESSION` | Đóng phiên này giúp tôi. |
| 4 | `UNKNOWN` | Bày tôi cách nấu cháo. |

## Cách dùng

**Đầu vào phải được tách từ bằng [VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP)**
(ví dụ `mô_tả toàn_cảnh nhà bếp`), giống lúc huấn luyện. Văn bản thô vẫn chạy được nhưng độ chính xác giảm.

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vncorenlp import VnCoreNLP

repo = "Hienne/phobert-vi-intent"
tokenizer = AutoTokenizer.from_pretrained(repo, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(repo).eval()

with VnCoreNLP("VnCoreNLP-1.2.jar", annotators="wseg") as segmenter:
    words = [w for s in segmenter.tokenize("Mô tả toàn cảnh nhà bếp.") for w in s]
    inputs = tokenizer(" ".join(words), truncation=True, max_length=128, return_tensors="pt")
    with torch.inference_mode():
        probs = model(**inputs).logits.softmax(-1)[0]
    print(model.config.id2label[int(probs.argmax())], float(probs.max()))
```

## Dữ liệu và kết quả

8.000 câu tiếng Việt (train/val/test = 6.000/1.000/1.000), cân bằng giữa 5 nhãn.
Checkpoint được chọn theo validation macro-F1.

| Tập | Accuracy | Macro-F1 |
|---|---:|---:|
| Validation | 96,00% | 95,94% |
| Test | 94,50% | 94,36% |

| Nhãn (test) | Precision | Recall | F1 |
|---|---:|---:|---:|
| `READ_TEXT` | 98,97% | 96,00% | 97,46% |
| `DESCRIBE_SCENE` | 94,34% | 100,00% | 97,09% |
| `ANSWER_VISUAL_QUESTION` | 88,29% | 98,00% | 92,89% |
| `END_SESSION` | 93,02% | 100,00% | 96,39% |
| `UNKNOWN` | 100,00% | 78,50% | 87,96% |

## Ngưỡng tin cậy

`thresholds.json` chứa điểm softmax tối thiểu theo nhãn, chọn trên validation (macro-F1).
Nhãn top-1 có điểm dưới ngưỡng được chuyển thành `UNKNOWN`; `END_SESSION` dùng ngưỡng cao hơn
vì kết thúc phiên nhầm gây hại nhiều nhất.

| Test | Macro-F1 | Recall `UNKNOWN` | Precision `END_SESSION` |
|---|---:|---:|---:|
| Không ngưỡng | 94,36% | 78,50% | 93,02% |
| Có ngưỡng | 94,58% | 81,50% | 94,31% |

## Hạn chế

- `UNKNOWN` có recall thấp: câu ngoài phạm vi chứa từ khóa như *dừng*, *tạm biệt*,
  *có màu gì*, *xung quanh* dễ bị xếp vào nhãn nghiệp vụ (ví dụ "dừng bài hát lại" → `END_SESSION`).
  Ngưỡng chỉ cải thiện nhẹ vì model quá tự tin; cần bổ sung dữ liệu đối chứng.
- Dữ liệu huấn luyện không có dấu câu cuối câu; câu có `?`/`.` có thể bị phân loại kém hơn
  (ví dụ "Trong hình có bao nhiêu người?" → `DESCRIBE_SCENE`).
- Dữ liệu sinh theo mẫu câu; độ chính xác trên câu thực tế ngoài phân phối có thể thấp hơn.
