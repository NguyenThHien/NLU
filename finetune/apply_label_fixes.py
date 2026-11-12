"""Áp các sửa nhãn đã duyệt (dataset/drafts/label_fixes.csv) vào BẢN SAO data gốc → dataset/<tên>/.

review = ok → áp gợi ý (suggestion); giu / trống → bỏ qua. Data gốc không bị sửa.

  python apply_label_fixes.py --name slots_fixed
"""

import argparse
import json
import re

import pandas as pd

from common import ROOT_DIR, bio_spans


def find_span(row, value, slot_type):
    """Vị trí token [start, end) của slot (type, value) trong câu."""
    for slot in row["slots"]:
        if slot["type"] == slot_type and slot["value"] == value:
            start = len(row["text"][:slot["start"]].split())
            return start, start + len(value.split())
    raise ValueError(f"không thấy {slot_type} '{value}'")


def put(tags, start, words_count, slot_type):
    tags[start:start + words_count] = [f"B-{slot_type}"] + [f"I-{slot_type}"] * (words_count - 1)


def apply_fix(row, fix):
    """Sửa tags của một câu theo một dòng label_fixes; trả tags mới."""
    tags = list(row["tags"])
    start, end = find_span(row, fix["value"], fix["slot"])
    rule, suggestion = fix["rule"], fix["suggestion"]

    if rule == "O2":  # bỏ từ chỉ định cuối OBJECT
        tags[end - 1] = "O"
    elif rule == "O4":  # OBJECT 'x' + LOCATION 'y' (cả hai nằm trong span cũ)
        obj, loc = re.fullmatch(r"OBJECT '(.+)' \+ LOCATION '(.+)'", suggestion).groups()
        tags[start:end] = ["O"] * (end - start)
        put(tags, start, len(obj.split()), "OBJECT")
        put(tags, end - len(loc.split()), len(loc.split()), "LOCATION")
    elif rule == "L1":  # nối thêm đại từ ngay sau LOCATION
        if tags[end] != "O":
            raise ValueError("token sau LOCATION đã có nhãn")
        tags[end] = tags[end - 1].replace("B-", "I-")
    elif rule == "O10":  # ATTRIBUTE → OBJECT (hỏi tồn tại)
        put(tags, start, end - start, "OBJECT")
    else:
        raise ValueError(f"quy tắc lạ {rule}")
    return tags


def rebuild(row, tags):
    """Tạo lại slots (kèm vị trí ký tự) từ tags."""
    slots = []
    for slot_type, start, end in bio_spans(tags):
        char_start = len(" ".join(row["tokens"][:start])) + (1 if start else 0)
        value = " ".join(row["tokens"][start:end])
        slots.append({"type": slot_type, "value": value, "start": char_start, "end": char_start + len(value)})
    return {**row, "tags": tags, "slots": slots}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixes", default=str(ROOT_DIR / "dataset" / "drafts" / "label_fixes.csv"))
    parser.add_argument("--name", required=True, help="Thư mục con trong dataset/.")
    args = parser.parse_args()

    fixes = pd.read_csv(args.fixes, encoding="utf-8-sig", keep_default_na=False)
    fixes = fixes[fixes["review"].str.strip() == "ok"]
    out = ROOT_DIR / "dataset" / args.name
    out.mkdir(parents=True, exist_ok=True)

    applied, failed = 0, []
    for split in ("train", "val", "test"):
        rows = [json.loads(line) for line in (ROOT_DIR / "dataset" / f"{split}.jsonl").open(encoding="utf-8")]
        # Áp từng câu, có thể nhiều sửa trên một câu; slots được dựng lại sau mỗi sửa.
        for _, fix in fixes[fixes["split"] == split].iterrows():
            index = int(fix["line"]) - 1
            try:
                rows[index] = rebuild(rows[index], apply_fix(rows[index], fix))
                applied += 1
            except (ValueError, AttributeError, IndexError) as error:
                failed.append((split, fix["line"], fix["rule"], fix["text"], str(error)))
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as file:
            file.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)

    print(f"Áp {applied}/{len(fixes)} sửa → {out}")
    for item in failed:
        print("  KHÔNG ÁP:", item)


if __name__ == "__main__":
    row = {"text": "đọc bảng tên trên cửa cho mình", "tokens": "đọc bảng tên trên cửa cho mình".split(),
           "tags": ["O", "B-OBJECT", "I-OBJECT", "I-OBJECT", "I-OBJECT", "O", "O"],
           "slots": [{"type": "OBJECT", "value": "bảng tên trên cửa", "start": 4, "end": 21}]}
    fixed = rebuild(row, apply_fix(row, {"rule": "O4", "slot": "OBJECT", "value": "bảng tên trên cửa",
                                         "suggestion": "OBJECT 'bảng tên' + LOCATION 'trên cửa'"}))
    assert [(s["type"], s["value"]) for s in fixed["slots"]] == [("OBJECT", "bảng tên"), ("LOCATION", "trên cửa")]
    assert all(row["text"][s["start"]:s["end"]] == s["value"] for s in fixed["slots"])
    main()
