# Nháp dữ liệu đối lập — CẦN NGƯỜI DUYỆT trước khi dùng

Câu trong thư mục này do AI (Claude) sinh, **chưa được kiểm tra nhãn**. Không gộp vào
train/val/test khi cột `review` còn trống.

| File | Số câu | Dùng cho |
|---|---:|---|
| `contrast_train_draft.csv` | 299 | Gộp vào **train/val** sau khi duyệt |
| `challenge_draft.csv` | 116 | **Chỉ để đánh giá**, không bao giờ đưa vào train |

Cột `error_type`:
- `A_dung_tambiet`: có "dừng / tạm biệt / tắt / thoát / chào" nhưng không nhắm vào phiên hỗ trợ.
- `B_xung_quanh`: có "xung quanh / quanh đây / tả cảnh" nhưng hỏi kiến thức, bản đồ, hoặc nhờ viết văn.
- `C_hoi_khong_can_anh`: câu hỏi "có… không / màu gì / bao nhiêu" trả lời được mà không cần camera.

## Quy tắc gán nhãn

Câu hỏi đặt ra: **Hina có cần ảnh camera lúc này để trả lời không?**

- Nhắm vào **phiên hỗ trợ / ứng dụng / camera của Hina** ("tắt camera", "tạm biệt hina") → `END_SESSION`.
- Tắt hoặc dừng **thứ khác** (nhạc, báo thức, đèn, xe, gói cước), chào hỏi người khác → `UNKNOWN`.
- Mô tả **cảnh đang ở trước camera** → `DESCRIBE_SCENE`. Tả cảnh để viết văn, hỏi phong cảnh một địa danh → `UNKNOWN`.
- Hỏi về **vật thể đang ở trước camera** ("quả chuối tôi đang cầm") → `ANSWER_VISUAL_QUESTION`.
  Hỏi kiến thức chung ("chuối chín màu gì"), hỏi địa điểm cần bản đồ, giá cả, thời tiết → `UNKNOWN`.

## Cách duyệt

Điền cột `review`: `ok` / tên nhãn đúng (nếu nhãn nháp sai) / `xoa` (câu mơ hồ, câu không tự nhiên).
Câu mơ hồ thì **xóa**, đừng cố gán nhãn.

Bộ thử thách nên được **người trong nhóm viết thêm** khoảng 30–50 câu. Lý do: AI viết cả câu train lẫn câu
test thì hai bộ cùng một văn phong, nên điểm trên bộ thử thách sẽ cao hơn thực tế.

## Kết quả nền trên `challenge_draft.csv` (trước khi duyệt)

| | Model cũ | v2 |
|---|---:|---:|
| Accuracy | 81,9% | 87,1% |
| Recall `UNKNOWN` | 70,8% | 79,2% |
| `B_xung_quanh` / `UNKNOWN` | 48% | 60% |

So sánh lại bằng:
`python finetune/compare_models.py --data dataset/drafts/challenge_draft.csv --models <model1> <model2>`
