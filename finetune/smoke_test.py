"""Smoke test cho API đang chạy. Dùng: python smoke_test.py [http://localhost:8000]"""

import json
import sys
import urllib.error
import urllib.request

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
# Câu model phân loại chắc chắn (kiểm tra service chạy đúng, không đo độ chính xác model).
CASES = {
    "đọc nội dung hóa đơn thành tiếng cho mình": "READ_TEXT",
    "mô tả toàn cảnh sân ga giúp tôi": "DESCRIBE_SCENE",
    "người đứng cạnh cửa đang mặc áo màu gì": "ANSWER_VISUAL_QUESTION",
    "đóng phiên trợ giúp hiện tại giúp tôi": "END_SESSION",
    "làm sao để luộc trứng lòng đào": "UNKNOWN",
}
# target_obj mong đợi (None = không có vật thể).
TARGETS = {
    "đọc nội dung hóa đơn thành tiếng cho mình": "hóa đơn",
    "làm sao để luộc trứng lòng đào": None,
}


def call(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        BASE_URL + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8")


def main():
    status, _ = call("/health")
    assert status == 200, f"/health trả {status}"

    for text, expected in CASES.items():
        status, body = call("/v1/predict", {"text": text})
        assert status == 200, f"{text!r}: HTTP {status} {body}"
        result = json.loads(body)
        assert result["label"] == expected, f"{text!r}: nhận {result['label']}, cần {expected}"
        if text in TARGETS:
            assert result["target_obj"] == TARGETS[text], f"{text!r}: target_obj {result['target_obj']!r}"

    status, _ = call("/v1/predict", {"text": ""})
    assert status == 422, f"câu rỗng phải trả 422, nhận {status}"
    status, _ = call("/v1/predict", {"text": "a" * 501})
    assert status == 422, f"câu quá dài phải trả 422, nhận {status}"

    status, body = call("/metrics")
    assert status == 200 and "nlu_predictions_total" in body, "/metrics thiếu counter"
    print(f"OK: {len(CASES)} câu đúng nhãn, target_obj đúng, validate đầu vào và /metrics hoạt động")


if __name__ == "__main__":
    main()
