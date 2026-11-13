---
language: vi
pipeline_tag: token-classification
base_model: vinai/phobert-base-v2
tags:
  - slot-filling
  - phobert
  - vietnamese
---

# PhoBERT Vietnamese Slot Tagger (target_obj)

Gán nhãn BIO cho yêu cầu tiếng Việt gửi trợ lý Hina (hỗ trợ đọc chữ, tìm hiểu môi trường qua ảnh).
Dùng cùng model intent `Hienne/phobert-vi-intent`; API trả `target_obj` = OBJECT đầu tiên (cụm lõi).

Mã nguồn: https://github.com/NguyenThHien/NLU (`finetune/train_slots.py`, `finetune/api.py`).

## Nhãn
`O`, `B/I-OBJECT`, `B/I-LOCATION`, `B/I-ATTRIBUTE`, `B/I-PLACE`. Quy tắc gán nhãn: `dataset/LABELING_GUIDE.md`
(OBJECT là **cụm lõi**: "con mèo nhà mình" → `con mèo`; vị trí tách thành LOCATION).

## Cách dùng
Đầu vào phải tách từ bằng VnCoreNLP (giống lúc train). Nhãn dự đoán ở subword đầu mỗi từ;
xem `extract_slots()` trong `finetune/predict.py`.

## Dữ liệu
- 8.000 câu gốc (6.000/1.000/1.000), đã sửa 203 nhãn theo guideline.
- 374 câu bổ sung (người, con vật, xe, đồ ăn, cảnh vật, câu hỏi tồn tại…), AI sinh, người duyệt.
- Tăng cường tự động 50% câu train: từ đệm, lặp từ, viết hoa, dấu câu (mô phỏng lời nói qua ASR).

## Kết quả `target_obj` (độ chính xác OBJECT đầu tiên)
| Bộ đo | Điểm | Ghi chú |
|---|---:|---|
| Test gốc (600 câu có intent cần slot) | 99,0% | Lạc quan: 80% vật thể test đã có trong train |
| Bộ thử thách khác văn phong (145 câu, AI viết) | 86,9% | Gồm câu ngập ngừng, phương ngữ, trộn tiếng Anh |

Chưa có số đo trên câu người dùng thật.

## Hạn chế
- Câu "trời có mưa không" và câu so sánh 2 vật còn yếu.
- VnCoreNLP đôi khi tách dính sang từ ngoài slot (`đường_không`) → bỏ sót vật.
- ATTRIBUTE kém tin cậy (F1 ~65%), không trả ra API.
