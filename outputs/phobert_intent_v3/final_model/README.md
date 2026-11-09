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

## Phiên bản

| Tag | Mô tả |
|---|---|
| `v1.0` | 8.000 câu gốc. Bản đang deploy. |
| `v3.0` (main) | Thêm 296 câu đối lập nhắm vào lỗi `UNKNOWN`. **Nhãn bổ sung mới chỉ có AI duyệt, chưa có người duyệt.** |

## Dữ liệu và kết quả (v3.0)

8.000 câu gốc (train/val/test = 6.000/1.000/1.000) cộng 296 câu đối lập
(246 vào train, 50 vào val), dạy model phân biệt từ khóa giống nhau nhưng ý định khác nhau:
*dừng / tạm biệt* không nhắm vào phiên, *xung quanh / tả cảnh* nhưng hỏi kiến thức hoặc nhờ viết văn,
câu hỏi không cần ảnh. **Test giữ nguyên bản gốc** để so sánh được với v1.0.
Checkpoint được chọn theo validation macro-F1. Train trên GPU (fp16, batch 16, 3 epoch).

| Test gốc (1.000 câu) | v1.0 | **v3.0** |
|---|---:|---:|
| Accuracy | 94,50% | **96,20%** |
| Macro-F1 | 94,36% | **96,17%** |
| Recall `UNKNOWN` | 78,50% | **88,50%** |

McNemar v3.0 so với v1.0 trên test gốc: v3 đúng mà v1 sai 34 câu, ngược lại 17 câu, p = 0,024.

| Nhãn (test, v3.0) | Precision | Recall |
|---|---:|---:|
| `READ_TEXT` | 100,00% | 95,50% |
| `DESCRIBE_SCENE` | 94,34% | 100,00% |
| `ANSWER_VISUAL_QUESTION` | 95,61% | 98,00% |
| `END_SESSION` | 94,74% | 99,00% |
| `UNKNOWN` | 96,72% | 88,50% |

Trên bộ thử thách 115 câu (3 kiểu lỗi trên): accuracy v1.0 81,7% → v3.0 99,1%.
Bộ này do AI viết cùng văn phong với câu bổ sung, nên con số này lạc quan.

## Ngưỡng tin cậy

`thresholds.json` (v3.0) chỉ đặt ngưỡng cho `END_SESSION` = 0,95, chọn trên validation,
vì kết thúc phiên nhầm gây hại nhiều nhất. Nhãn top-1 có điểm dưới ngưỡng được chuyển thành `UNKNOWN`.
Ngưỡng cho mọi nhãn (lưới tìm trên validation) làm test giảm, nên không dùng.

| Test (v3.0) | Macro-F1 | Recall `UNKNOWN` | Precision `END_SESSION` |
|---|---:|---:|---:|
| Không ngưỡng | 96,17% | 88,50% | 94,74% |
| Có ngưỡng | 96,17% | 89,50% | 95,61% |

## Hạn chế

- Nhãn của 296 câu bổ sung chưa có người duyệt.
- Test và bộ thử thách đều là câu viết tay; chưa đo trên transcript thật từ ASR.
- Dữ liệu huấn luyện không có dấu câu cuối câu; câu có `?`/`.` có thể bị phân loại kém hơn.
- Dữ liệu sinh theo mẫu câu; độ chính xác trên câu thực tế ngoài phân phối có thể thấp hơn.
