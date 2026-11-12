"""Tìm câu vi phạm dataset/LABELING_GUIDE.md trong data slot, xuất CSV kèm gợi ý sửa để người duyệt.

  python check_labels.py                         # quét dataset/{train,val,test}.jsonl
  python check_labels.py --out ../dataset/drafts/label_fixes.csv

Chỉ báo cáo, KHÔNG tự sửa data.
"""

import argparse
import csv
import json
from collections import Counter

from common import ROOT_DIR

DEMONSTRATIVES = {"này", "kia", "đó", "ấy", "nớ", "ni", "tê", "đấy"}
LOCATION_WORDS = {"trên", "trong", "dưới", "ở", "ngoài", "cạnh", "bên", "trước", "sau", "gần", "giữa", "đằng", "phía"}
# "bạn" không có: cuối câu nó thường là lời gọi ("bên phải bạn" = "bên phải, bạn ơi").
PRONOUNS = {"tôi", "mình", "con", "em", "cháu", "bà", "ông", "má", "mẹ", "dì", "tui", "tớ", "anh", "chị",
            "o", "mạ", "bác", "cô", "chú"}
# Đại từ đứng trước các từ này là chủ ngữ vế sau ("trên quầy mình rất cần"), không thuộc LOCATION.
CLAUSE_STARTERS = {"rất", "cần", "muốn", "đang", "phải", "không"}
# Động từ nối OBJECT với vị trí ("thông báo dán trên cửa") → để O (quy tắc O3).
LINK_VERBS = {"dán", "treo", "gắn", "đặt", "để", "in", "ghi", "nằm", "đứng", "ngồi", "đi", "dựng"}
EXIST_VERBS = {"có", "còn"}
YES_NO = {"không", "hông", "hem", "ko", "chưa"}
PERSON_HEADS = {"người", "đứa", "cậu", "bé", "ông", "bà", "anh", "chị", "cô", "chú", "bác", "cụ"}


def violations(row):
    """→ [(quy tắc, loại slot, giá trị hiện tại, gợi ý)]."""
    found = []
    for slot in row["slots"]:
        words = slot["value"].split()
        after = row["text"][slot["end"]:].split()[:1]
        if slot["type"] == "OBJECT":
            if words[-1] in DEMONSTRATIVES:
                found.append(("O2", "OBJECT", slot["value"], " ".join(words[:-1])))
            cut = next((i for i, w in enumerate(words[1:], 1) if w in LOCATION_WORDS), None)
            if cut:
                # O5: người + (động từ) + vị trí → OBJECT chỉ giữ danh từ chỉ người.
                core = words[:1] if words[0] in PERSON_HEADS else words[:cut]
                while len(core) > 1 and core[-1] in LINK_VERBS:
                    core = core[:-1]
                found.append(("O4", "OBJECT", slot["value"],
                              f"OBJECT '{' '.join(core)}' + LOCATION '{' '.join(words[cut:])}'"))
        after2 = row["text"][slot["end"]:].split()[1:2]
        if (slot["type"] == "LOCATION" and after and after[0] in PRONOUNS and words[-1] not in PRONOUNS
                and not (after2 and after2[0] in CLAUSE_STARTERS)):
            found.append(("L1", "LOCATION", slot["value"], f"{slot['value']} {after[0]}"))
        before = row["text"][:slot["start"]].split()[-1:]
        has_object = any(s["type"] == "OBJECT" for s in row["slots"])
        if (slot["type"] == "ATTRIBUTE" and not has_object and before and before[0] in EXIST_VERBS
                and after and after[0] in YES_NO):
            found.append(("O10", "ATTRIBUTE", slot["value"], f"OBJECT '{slot['value']}' (hỏi tồn tại)"))
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT_DIR / "dataset" / "drafts" / "label_fixes.csv"))
    args = parser.parse_args()

    report = []
    for split in ("train", "val", "test"):
        for number, line in enumerate((ROOT_DIR / "dataset" / f"{split}.jsonl").open(encoding="utf-8"), 1):
            row = json.loads(line)
            for rule, slot_type, value, suggestion in violations(row):
                report.append({"split": split, "line": number, "text": row["text"], "rule": rule,
                               "slot": slot_type, "value": value, "suggestion": suggestion, "review": ""})

    with open(args.out, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "line", "text", "rule", "slot", "value", "suggestion", "review"])
        writer.writeheader()
        writer.writerows(report)
    print(f"{len(report)} vi phạm theo quy tắc {dict(Counter(r['rule'] for r in report))} → {args.out}")


if __name__ == "__main__":
    row = {"text": "đọc bảng tên trên cửa này cho xung quanh mình",
           "slots": [{"type": "OBJECT", "value": "bảng tên trên cửa", "start": 4, "end": 21},
                     {"type": "LOCATION", "value": "xung quanh", "start": 30, "end": 40}]}
    assert [v[0] for v in violations(row)] == ["O4", "L1"], violations(row)
    assert violations({"text": "người đi phía trước", "slots": [
        {"type": "OBJECT", "value": "người đi phía trước", "start": 0, "end": 19}]})[0][3] == (
        "OBJECT 'người' + LOCATION 'phía trước'")  # O5
    main()
