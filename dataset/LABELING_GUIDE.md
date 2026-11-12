# Hướng dẫn gán nhãn slot (nhóm cần chốt các quy tắc còn lại)

Áp dụng cho `tokens`/`tags` (BIO theo âm tiết) trong `dataset/*.jsonl` và các cột slot trong `dataset/drafts/*.csv`.
Chỉ gán slot cho câu `READ_TEXT`, `DESCRIBE_SCENE`, `ANSWER_VISUAL_QUESTION`. Câu `END_SESSION`/`UNKNOWN` không có slot.

Quy tắc dưới đây theo **cách gán chiếm đa số trong data hiện tại**, để phải sửa ít câu nhất.
**Đã chốt (2026-11-12): `target_obj` là cụm lõi (O3)**; downstream nhận kèm nguyên câu hỏi.
Số trong ngoặc là số lần quy ước đó xuất hiện trong 8.000 câu gốc.

## OBJECT: vật/người mà câu hỏi hoặc yêu cầu nhắm tới (API trả ra dưới tên `target_obj`)

| # | Quy tắc | Đúng | Sai |
|---|---|---|---|
| O1 | Gồm **loại từ + danh từ** | `con dao`, `cái áo`, `tờ báo` | `dao` |
| O2 | **Không** kèm từ chỉ định (này, kia, đó, ấy, nớ, ni, tê) (661 so với 0) | "cái áo **này**" → `cái áo` | `cái áo này` |
| O3 | **Cụm lõi**: bỏ phần mô tả (màu, chủ sở hữu, hành động, trang phục). Downstream nhận kèm nguyên câu hỏi nên vẫn đọc được mô tả. **Giữ danh từ ghép làm tên vật** và giống/loài | `con mèo` ("con mèo nhà mình"), `bé gái` ("bé gái mặc váy hồng"), `cái xe` ("cái xe màu trắng"); giữ: `sổ khám bệnh`, `bảng hiệu nhà thuốc`, `đèn tín hiệu`, `con gà trống`, `con chó săn` | `con mèo nhà mình`, `sổ khám bệnh của bà` |
| O4 | Cụm **vị trí** (trên/trong/dưới/ở/ngoài/cạnh/bên/trước/sau/gần/giữa + nơi chốn) tách ra thành **LOCATION** (854 so với 138) | "bảng tên **trên cửa**" → OBJECT `bảng tên`, LOCATION `trên cửa` | OBJECT `bảng tên trên cửa` |
| O5 | Người xác định bằng vị trí cũng theo O3 + O4: OBJECT là người, vị trí là LOCATION, động từ để O | "người **đứng sau lưng mình**" → OBJECT `người`, LOCATION `sau lưng mình`; "người trước mặt" → `người` + `trước mặt` | `người đứng sau lưng mình` |
| O6 | Hỏi về **một phần** của vật: OBJECT là vật chứa, phần được hỏi là ATTRIBUTE | "**giá** của hộp sữa" → OBJECT `hộp sữa`, ATTRIBUTE `giá`; "**bìa** sách hướng dẫn màu gì" → OBJECT `sách hướng dẫn` | OBJECT `bìa sách` |
| O7 | Nhiều vật ngang hàng thì gán OBJECT cho tất cả, vật nhắc trước đứng trước | `cái áo` + `cái quần` | |
| O8 | Món trên thực đơn hoặc giấy tờ: OBJECT là **thực đơn/giấy tờ**, món là ATTRIBUTE | "trà sữa trên **thực đơn**" → OBJECT `thực đơn` | OBJECT `trà sữa` |
| O9 | Cảnh vật và thiên nhiên được hỏi trực tiếp là OBJECT (15 lần) | "**trời** có mưa không" → `trời`; "**vạch qua đường**" | |
| O10 | **Hỏi tồn tại**: "[vị trí] có/còn X không", câu không có vật chủ → X là **OBJECT**, vị trí là LOCATION (35 lần) | "trong tủ lạnh còn **trứng** không" → OBJECT `trứng`, LOCATION `trong tủ lạnh`; "đường phía trước có **vũng nước** không" → OBJECT `vũng nước` | ATTRIBUTE `trứng`; OBJECT `đường` |
| O11 | **Hỏi thuộc tính**: "[vật] có X không" → vật là OBJECT, X là **ATTRIBUTE** (126 lần) | "con chó có đeo **rọ mõm** không" → OBJECT `con chó`, ATTRIBUTE `rọ mõm` | OBJECT `rọ mõm` |

## LOCATION: vị trí tương đối so với người dùng hoặc vật khác
- L1: **Kèm** đại từ chỉ người nếu có (262 so với 61): `bên phải mình`, `xung quanh tôi`, `trước mặt con`.
- L2: Kèm giới từ: `trên bàn`, `ở đầu hẻm`, `ngoài ban công`.

## PLACE: nơi người dùng đang ở ("mình đang ở **chùa**")

## ATTRIBUTE: cái được hỏi về OBJECT (API không trả ra; giữ cách gán hiện tại)
- Câu "có X không" → `X`; câu có từ hỏi → cụm hỏi (`màu gì`, `ở đâu`, `mấy giờ`).

## Câu khó: ghi lại ở đây khi nhóm chốt
| Câu | Quyết định |
|---|---|
| "biển trên đầu xe buýt" | OBJECT `biển`, LOCATION `trên đầu xe buýt` (O4) |
| "thông báo dán trên cửa" | OBJECT `thông báo`, LOCATION `trên cửa` (O4; "dán" là O) |

## Kiểm tra
`python finetune/check_labels.py` liệt kê câu vi phạm O2/O4/L1 trong data gốc ra CSV để người sửa.
