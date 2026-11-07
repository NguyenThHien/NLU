# Phân loại ý định tiếng Việt với PhoBERT-base-v2

Fine-tune PhoBERT-base-v2 để phân loại yêu cầu tiếng Việt thành 5 ý định
cho hệ thống hỗ trợ đọc chữ và nhận biết môi trường qua hình ảnh.
Model chỉ xác định ý định từ văn bản; không trực tiếp đọc ảnh, mô tả ảnh
hay trả lời câu hỏi về ảnh.

Môi trường: Python 3.11, Java 8+, PyTorch CPU.

| ID dataset | Nhãn |
|---|---|
| 1 | READ_TEXT |
| 2 | DESCRIBE_SCENE |
| 3 | ANSWER_VISUAL_QUESTION |
| 4 | END_SESSION |
| 5 | UNKNOWN |

CSV giữ ID 1–5. Khi tính loss, code đổi sang chỉ số nội bộ 0–4.
Kết quả dự đoán được trả về tên nhãn và ID gốc 1–5.

## Dữ liệu

Dataset hiện có **8.000 câu tiếng Việt**, chia sẵn theo tỷ lệ
**75% train / 12,5% validation / 12,5% test**. Mỗi câu có một nhãn ý định.

| Tập | File CSV | Số câu | Tỷ lệ | Số câu mỗi nhãn |
|---|---|---:|---:|---:|
| Train | `train.csv` | 6.000 | 75% | 1.200 |
| Validation | `val.csv` | 1.000 | 12,5% | 200 |
| Test | `test.csv` | 1.000 | 12,5% | 200 |

### Phân bố nhãn

| ID | Nhãn | Train | Validation | Test | Tổng |
|---|---|---:|---:|---:|---:|
| 1 | READ_TEXT | 1.200 | 200 | 200 | 1.600 |
| 2 | DESCRIBE_SCENE | 1.200 | 200 | 200 | 1.600 |
| 3 | ANSWER_VISUAL_QUESTION | 1.200 | 200 | 200 | 1.600 |
| 4 | END_SESSION | 1.200 | 200 | 200 | 1.600 |
| 5 | UNKNOWN | 1.200 | 200 | 200 | 1.600 |

Mỗi nhãn chiếm **20%** trong từng tập và toàn bộ dataset. Dữ liệu cân bằng
về số lượng nhãn; điều này không đồng nghĩa các cách diễn đạt hoặc tình
huống sử dụng đã được bao phủ đầy đủ.

### Định dạng và các trường dữ liệu

CSV gốc được đọc bằng UTF-8/UTF-8 có BOM, gồm:

| Cột | Kiểu dữ liệu | Ý nghĩa |
|---|---|---|
| `text` | Chuỗi | Câu tiếng Việt đầu vào |
| `label` | Chuỗi | Một trong 5 tên nhãn ý định |
| `label_id` | Số nguyên | ID nhãn gốc từ 1 đến 5 |

Tác vụ phân loại sử dụng các file CSV. File JSONL có thêm slot và tags
được giữ lại nhưng chưa sử dụng trong mô hình hiện tại.
Mỗi tập CSV có một file JSONL tương ứng với cùng số dòng; đã kiểm tra
`text`, `label`, `label_id` khớp theo từng dòng giữa hai định dạng.
Vì vậy, CSV và JSONL là hai biểu diễn của cùng dữ liệu, không cộng thành
16.000 mẫu độc lập.

JSONL gồm `text`, `label`, `label_id`, `slots`, `tokens`, `tags`.
Các loại slot quan sát được là `OBJECT`, `LOCATION`, `ATTRIBUTE`, `PLACE`.
Pipeline hiện tại tách từ lại từ `text` bằng VnCoreNLP, không sử dụng trực
tiếp trường `tokens` hoặc `tags` trong JSONL.

Sau tiền xử lý, CSV giữ các cột gốc và bổ sung:

| Cột | Ý nghĩa |
|---|---|
| `text_clean` | Văn bản đã chuẩn hóa Unicode NFC và khoảng trắng |
| `text_segmented` | Văn bản đã tách từ bằng VnCoreNLP, ví dụ `sinh_viên` |
| `labels` | Nhãn dùng trong pipeline, vẫn giữ ID gốc 1–5 trên file |

### Thống kê độ dài

Thống kê trên dữ liệu hiện có. “Đơn vị cách trắng” được đếm bằng
`text.split()` trước tách từ, không phải số từ tiếng Việt đã phân đoạn.
Độ dài ký tự tính cả khoảng trắng và dấu câu.

| Tập | Đơn vị cách trắng TB | Nhỏ nhất–lớn nhất | Ký tự TB | Ký tự nhỏ nhất–lớn nhất |
|---|---:|---|---:|---|
| Train | 9,50 | 1–27 | 38,94 | 1–113 |
| Validation | 9,79 | 1–22 | 40,53 | 3–96 |
| Test | 9,38 | 1–24 | 38,33 | 1–98 |

Trung vị độ dài là 9 đơn vị cách trắng trong cả ba tập; ít nhất 95% câu
trong mỗi tập có độ dài không quá 16 đơn vị cách trắng.

Độ dài đầu vào sau VnCoreNLP và tokenizer PhoBERT, **bao gồm token đặc biệt**:

| Tập | Token trung bình | Nhỏ nhất | P95 | Lớn nhất | Số câu vượt 128 token |
|---|---:|---:|---:|---:|---:|
| Train | 10,70 | 3 | 17 | 26 | 0 |
| Validation | 10,95 | 3 | 17 | 25 | 0 |
| Test | 10,61 | 3 | 17 | 25 | 0 |

Giới hạn `max_length=128` hiện không cắt câu nào trong dataset đã xử lý.
Thống kê này chủ yếu phản ánh câu ngắn, chưa chứng minh model xử lý tốt
đoạn văn dài hoặc đầu vào nhiều ý định.

### Kiểm tra chất lượng và giới hạn dữ liệu

- Không phát hiện giá trị thiếu trong `text`, `label`, `label_id` của CSV gốc.
- Không phát hiện câu trùng trong từng tập hoặc giữa các tập sau chuẩn hóa
  Unicode, khoảng trắng và chữ hoa/thường.
- Có 13/1.000 câu test có độ giống từ vựng với câu train gần nhất từ 0,9
  trở lên, đo bằng cosine similarity trên TF-IDF ký tự 3–5 gram. Đây là
  dấu hiệu cần xem xét, không phải kết luận trùng ngữ nghĩa hay rò rỉ dữ liệu.
- Tiền xử lý xáo trộn riêng từng tập với `seed=42`, giữ nguyên phân chia
  train/validation/test và không ghép các tập lại để chia lại.
- Nguồn thu thập, cách tạo câu, quy trình gán nhãn và tiêu chí chia tập
  chưa được xác minh từ các file hiện có. Không khẳng định dataset đã được
  chia độc lập theo người viết, nguồn hoặc mẫu câu.

## Kết quả đánh giá

Các chỉ số dưới đây được tính bằng model đã chọn theo validation macro-F1.
Precision, recall và F1 là trung bình macro trên 5 nhãn.

| Tập đánh giá | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Train | 99,90% | 99,90% | 99,90% | 99,90% |
| Validation | 96,00% | 95,99% | 96,00% | 95,94% |
| Test | 94,50% | 94,92% | 94,50% | 94,36% |
| 25 câu mới gán nhãn thủ công | 88,00% | 92,50% | 88,00% | 88,72% |

Bộ 25 câu là kiểm tra bổ sung có quy mô nhỏ, tách biệt với tập test 1.000 câu.
Các ví dụ kiểm thử tương tác dưới đây cũng là quan sát bổ sung, không được
gộp vào các chỉ số trong bảng.

### Khả năng xử lý yêu cầu cơ bản

Bốn nhãn nghiệp vụ `READ_TEXT`, `DESCRIBE_SCENE`, `ANSWER_VISUAL_QUESTION`
và `END_SESSION` hoạt động khá tốt với yêu cầu rõ ràng ở mức cơ bản.
Kiểm thử tương tác cho thấy model phân biệt được một số câu gần nhau về
từ vựng nhưng khác ý định, cũng như một số yêu cầu gián tiếp và phủ định.

| Đầu vào kiểm thử | Nhãn dự đoán | Đánh giá |
|---|---|---|
| Đọc các chữ trên chai nước. | READ_TEXT | Đúng |
| Các chữ trên chai nước có màu gì? | ANSWER_VISUAL_QUESTION | Đúng |
| Mô tả toàn cảnh nhà bếp. | DESCRIBE_SCENE | Đúng |
| Trong nhà bếp có bao nhiêu người? | ANSWER_VISUAL_QUESTION | Đúng |
| Khoan kết thúc phiên, mô tả căn phòng trước mặt tôi đã. | DESCRIBE_SCENE | Đúng |
| Tôi không muốn nghe mô tả nữa, đóng phiên hỗ trợ này giúp tôi. | END_SESSION | Đúng |
| Mắt tôi không nhìn rõ phiếu này, bạn nói thành tiếng những gì được in trên đó nhé. | READ_TEXT | Đúng |

### Điểm yếu: nhận diện UNKNOWN và đầu vào khó

`UNKNOWN` là điểm yếu nổi bật: các yêu cầu ngoài phạm vi nhưng chứa từ
khóa gần với nhãn nghiệp vụ có thể bị gán sang nhãn khác, kể cả khi điểm
softmax cao. Trên tập test, recall của `UNKNOWN` là **78,5%**: model nhận
đúng 157/200 câu và nhầm 43 câu sang nhãn khác. Vì vậy, chưa đủ căn cứ
để kết luận phần lớn mọi câu `UNKNOWN` đều bị nhầm; vấn đề tập trung ở
khả năng tổng quát hóa sang các cách diễn đạt ngoài phạm vi khó hơn.

| Đầu vào kiểm thử | Nhãn kỳ vọng | Nhãn dự đoán | Nhận xét |
|---|---|---|---|
| Dịch câu “kết thúc phiên hỗ trợ này” sang tiếng Anh. | UNKNOWN | READ_TEXT | Nhầm yêu cầu dịch với đọc chữ; softmax của nhãn sai là 98,35% |
| từ kết thúc trong tiếng Anh viết thế nào | UNKNOWN | READ_TEXT | Nhầm câu hỏi ngôn ngữ với đọc chữ trong bộ 25 câu mới |
| Bày tôi cách nấu một nồi cháo cho hai người ăn. | UNKNOWN | UNKNOWN | Nhận đúng một yêu cầu ngoài phạm vi rõ ràng |
| doc nguyen van nhung chu tren to giay nay giup toi | READ_TEXT | ANSWER_VISUAL_QUESTION | Chưa ổn định với tiếng Việt không dấu |

Các nhãn nghiệp vụ vẫn có thể nhầm ở câu khó. Trong bộ kiểm tra bổ sung,
“không cần đọc chữ, chỉ mô tả khung cảnh thôi” bị gán `READ_TEXT` thay vì
`DESCRIBE_SCENE`. Model hiện chưa xử lý phủ định ổn định trong mọi trường hợp.

### Nhận định và hướng cải thiện

Ở mức yêu cầu cơ bản, model đã phân biệt khá tốt các nhãn nghiệp vụ.
Ưu tiên cải thiện tiếp theo là giảm việc gán yêu cầu `UNKNOWN` sang các
nhãn khác, bổ sung câu chứa từ khóa gây nhiễu, câu phủ định và tiếng Việt
không dấu.

Không phát hiện câu trùng nguyên văn giữa các tập sau chuẩn hóa. Tuy nhiên,
chênh lệch accuracy train–test là 5,4 điểm phần trăm và kết quả câu khó
cho thấy dấu hiệu overfit nhẹ, bám từ khóa. Các kết quả hiện có chưa đủ
để khẳng định model không học vẹt hoặc đã hiểu đầy đủ ý định.

Nên xây dựng thêm tập kiểm tra độc lập theo nguồn hoặc mẫu câu. Điểm softmax
chưa được hiệu chỉnh; không dùng riêng điểm này để kết luận dự đoán đáng tin.
Chi tiết đánh giá: [MODEL_EVALUATION.md](MODEL_EVALUATION.md).

## Cấu trúc dự án

```text
dataset/                 # Dataset gốc CSV và JSONL
  processed/             # CSV sau tiền xử lý
finetune/
  common.py              # Đường dẫn, nhãn, chuẩn hóa và tách từ dùng chung
  preprocess.py          # Chuẩn hóa, tách từ, xáo trộn dữ liệu
  train.py               # Fine-tune, đánh giá và vẽ biểu đồ
  predict.py             # Dự đoán từ văn bản nhập qua terminal
  audit_model.py         # Kiểm tra tổng quát hóa và trùng lặp dữ liệu
model/
  download_model.py      # Tải PhoBERT-base-v2
  download_vncorenlp.py   # Tải thành phần tách từ VnCoreNLP
outputs/                 # Model đã train, checkpoint, báo cáo và biểu đồ
requirements.txt
```

## Cài đặt trên Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python .\model\download_model.py
python .\model\download_vncorenlp.py
```

VnCoreNLP cần Java trong PATH (`java -version`). Chỉ tải thành phần tách từ.
Model nền: [vinai/phobert-base-v2](https://huggingface.co/vinai/phobert-base-v2).
Bộ tách từ: [VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP).
Khi sử dụng hoặc phân phối các tài nguyên này, xem giấy phép của nguồn tương ứng.

## Tiền xử lý và huấn luyện

Đặt `train.csv`, `val.csv`, `test.csv` trong `dataset/`, với cột `text`,
`label` (và `label_id` nếu có). Chương trình tạo cột `labels` từ tên nhãn.

```powershell
python .\finetune\preprocess.py
python .\finetune\train.py --smoke
python .\finetune\train.py
```

Tiền xử lý chuẩn hóa Unicode/khoảng trắng, tách từ bằng VnCoreNLP và xáo trộn
riêng từng tập. Không sửa dataset gốc. Huấn luyện trên CPU; model tốt nhất
được chọn theo validation macro-F1. Test chỉ được đánh giá sau khi chọn model.

Kết quả nằm trong `outputs/phobert_intent/`: model, báo cáo từng lớp,
accuracy, macro precision/recall/F1, confusion matrix và biểu đồ huấn luyện.
`--smoke` chạy 2 bước và lưu riêng trong `outputs/smoke_test/`.

## Dự đoán và kiểm tra tổng quát hóa

```powershell
python .\finetune\predict.py
python .\finetune\audit_model.py
```

`predict.py` nhận câu qua terminal; nhập `/exit` để thoát. Mỗi đầu vào chỉ
được gán một nhãn; văn bản vượt 128 token sẽ bị cắt. Điểm softmax chưa được
hiệu chỉnh nên không phải mức đảm bảo dự đoán đúng.

`audit_model.py` không huấn luyện lại: đánh giá train/val/test, kiểm tra trùng
câu giữa các tập, đo độ giống từ vựng với train và thử 25 câu mới có nhãn kỳ
vọng gán thủ công. Báo cáo nằm trong `outputs/model_audit/`. Bộ câu nhỏ này
chỉ giúp tìm lỗi; không đủ chứng minh model không học vẹt. Những lần đánh giá
lại cùng model không tạo ra một tập kiểm tra độc lập mới.

## Đưa code lên Git

`.gitignore` bỏ qua môi trường, model tải về, checkpoint, kết quả và dữ liệu
đã tiền xử lý. Tất cả vẫn được giữ trên máy. Dataset gốc CSV/JSONL và các file
Python vẫn được giữ; JSONL có slot nhưng tác vụ hiện tại chỉ dùng CSV.
Các file thử nghiệm được giữ lại để tham khảo.
