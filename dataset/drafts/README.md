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

## Nháp slot vật thể: `slot_objects_draft.csv` (213 câu, AI sinh)

Bổ sung nhóm vật thể thiếu trong data gốc (gốc chỉ có 250 vật, chủ yếu giấy tờ và biển bảng):
`nguoi`, `con_vat`, `xe`, `do_an`, `quan_ao`, `do_gia_dung`. Đã bỏ câu có vật thể trùng test.

Cột slot là cụm cắt nguyên văn từ `text`; nhiều OBJECT ngăn bằng `|`. Quy ước theo data gốc:
- OBJECT **không** kèm từ chỉ định: "cái áo này" → `cái áo`.
- ATTRIBUTE: câu "có X không" → `X`; câu có từ hỏi → cụm hỏi (`màu gì`, `ở đâu`).

`review` hiện là `ok_ai` (chỉ để train thử `phobert_slots_v2`). **Người cần duyệt lại** trước khi dùng v2
thật: sửa cột slot nếu cắt sai, hoặc `xoa`. Sau đó gộp lại:
`python finetune/merge_slot_drafts.py --draft dataset/drafts/slot_objects_draft.csv --name slots_v2`

## Nháp slot đợt 2: `slot_objects_draft2.csv` (116 câu, AI sinh)
Nhóm: `canh_vat` (vạch kẻ đường, ổ gà…), `thien_nhien`, `do_an`, `nguoi_vi_tri` (quy tắc O5), `hai_vat` (O6–O8).
Gán theo `dataset/LABELING_GUIDE.md`. Đã bỏ vật thể trùng test gốc và val/test của `dataset/slots_split`.

## Sửa nhãn data gốc: `label_fixes.csv`
Sinh bởi `python finetune/check_labels.py`: 195 chỗ vi phạm guideline (O4 = OBJECT gộp cụm vị trí,
L1 = LOCATION thiếu đại từ, O2 = OBJECT kèm từ chỉ định). Cột `suggestion` là gợi ý tự động;
người duyệt điền `review`: `ok` (theo gợi ý) / cách gán đúng / `giu` (nhãn cũ đúng).

## Bộ test do người viết: `human_test_template.csv`
**Người trong nhóm tự viết** 50–100 câu (không dùng AI), giống cách người dùng thật nói với Hina: có dấu câu, nói vòng,
vật thể đa dạng. `target_obj` để trống nếu câu không nhắc vật. Không được đưa bộ này vào train.
Chấm: `python finetune/eval_slots.py --data dataset/drafts/human_test.csv --models <model...>`

## Đợt 2: `contrast_train_draft2.csv` (126 câu, train) + `challenge_draft2.csv` (65 câu, chỉ đánh giá)

AI sinh, `review` **để trống** — cần người duyệt. Đã bỏ câu trùng với `dataset/v3`.
Nền v3 trên `challenge_draft2.csv`: accuracy 84,6%, macro-F1 78,7%.

| `error_type` | Lỗi của v3 | Quy ước nhãn đề xuất (cần nhóm chốt) |
|---|---|---|
| `E_phu_dinh` | "đừng tắt", "khoan kết thúc" → `END_SESSION` | Phủ định yêu cầu kết thúc → `UNKNOWN` (giữ phiên). Có kèm yêu cầu khác → nhãn của yêu cầu đó |
| `E_ghep` | "mô tả … rồi tắt" → `END_SESSION` | Làm X rồi kết thúc → nhãn của X (giống "khoan kết thúc, mô tả … đã") |
| `Q_ngan` | "bao nhiêu tiền", "cái nào là muối" → `UNKNOWN` | Câu hỏi ngắn có chỉ định ngầm vật trước camera → `ANSWER_VISUAL_QUESTION` |
| `Q_khong_dau` | "cai nay mau gi" → `DESCRIBE_SCENE` | Theo nghĩa câu (train cũ chỉ có 6 câu không dấu, đều `END_SESSION`) |
| `Q_canh_vs_hoi` | "phía trước có gì cản đường không" → `DESCRIBE_SCENE` | Hỏi có/không hoặc một chi tiết → `ANSWER_VISUAL_QUESTION`; tả tổng thể → `DESCRIBE_SCENE` |
| `U_tri_nho` | "cái bàn nhà tôi màu gì bạn nhớ không" → `ANSWER_VISUAL_QUESTION` | Hỏi vật không ở trước camera / hỏi trí nhớ → `UNKNOWN` |

## Đợt 3: `contrast_train_draft3.csv` (368 câu, train)

Cùng 6 `error_type` và quy ước nhãn như đợt 2; `review` **để trống** — cần người duyệt.
Đã bỏ câu trùng với `dataset/v3`, hai bộ challenge và hai bản nháp contrast trước.

- `source=claude_draft` (218 câu): AI viết tay theo từng nhóm lỗi.
- `source=strip_diacritics` (150 câu, nhóm `Q_khong_dau`): bỏ dấu 30 câu/nhãn lấy ngẫu nhiên
  (`seed=42`) từ `dataset/v3/train.csv`, giữ nguyên nhãn gốc. Nhãn gốc sai thì bản bỏ dấu cũng sai —
  khi duyệt nên soát cả nhãn, không chỉ cách viết.
