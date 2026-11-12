"""Gộp câu nháp slot đã duyệt vào train/val gốc → dataset/<tên>/*.jsonl (dataset gốc không bị sửa).

Nháp: CSV cột text, label, category, object (nhiều vật ngăn bằng |), location, place, attribute, review.
review: ok / ok_ai / tên nhãn → giữ; xoa hoặc trống → bỏ. Test giữ nguyên bản gốc để so với model cũ.
Ghi thêm new_val.jsonl = phần nháp rơi vào val, để đo riêng nhóm vật mới.

  python merge_slot_drafts.py --draft ../dataset/drafts/slot_objects_draft.csv --name slots_v2
  python train_slots.py --data-dir ../dataset/slots_v2 --run-name phobert_slots_v2
"""

import argparse
import json
import shutil

import pandas as pd

from common import ROOT_DIR, SEED
from merge_drafts import VAL_FRACTION, approved

SLOT_COLUMNS = {"object": "OBJECT", "location": "LOCATION", "place": "PLACE", "attribute": "ATTRIBUTE"}


def to_record(row):
    """Dòng nháp → bản ghi jsonl giống dataset gốc; trả (record, None) hoặc (None, lý do lỗi)."""
    tokens = row["text"].split()
    tags = ["O"] * len(tokens)
    slots = []
    pairs = [(slot_type, value) for column, slot_type in SLOT_COLUMNS.items()
             for value in str(row[column] or "").split("|") if value.split()]
    # Cụm dài gán trước để "thư" không chiếm chỗ trong "hộp thư".
    for slot_type, value in sorted(pairs, key=lambda pair: -len(pair[1].split())):
        words = value.split()
        starts = [i for i in range(len(tokens) - len(words) + 1) if tokens[i:i + len(words)] == words]
        if not starts:
            return None, f"không thấy '{value}'"
        start = next((i for i in starts if all(tag == "O" for tag in tags[i:i + len(words)])), None)
        if start is None:
            return None, f"'{value}' chồng lên slot khác"
        tags[start:start + len(words)] = [f"B-{slot_type}"] + [f"I-{slot_type}"] * (len(words) - 1)
        char_start = len(" ".join(tokens[:start])) + (1 if start else 0)
        slots.append({"type": slot_type, "value": value, "start": char_start, "end": char_start + len(value)})

    slots.sort(key=lambda slot: slot["start"])
    record = {"text": row["text"], "label": row["label"], "label_id": int(row["label_id"]),
              "slots": slots, "tokens": tokens, "tags": tags}
    return record, None


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", nargs="+", required=True, help="Một hoặc nhiều CSV nháp.")
    parser.add_argument("--name", required=True, help="Thư mục con trong dataset/.")
    parser.add_argument("--no-leak-guard", action="store_true",
                        help="Không bỏ câu nháp có vật thể trùng test (chỉ cho model chạy thật, không dùng để đo).")
    parser.add_argument("--base", default="", help="Thư mục con trong dataset/ làm nền (mặc định: bộ gốc).")
    args = parser.parse_args()

    src, out = ROOT_DIR / "dataset" / args.base, ROOT_DIR / "dataset" / args.name
    out.mkdir(parents=True, exist_ok=True)

    test_objects = {s["value"] for line in (src / "test.jsonl").open(encoding="utf-8")
                    for s in json.loads(line)["slots"] if s["type"] == "OBJECT"}
    frames = [pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False) for path in args.draft]
    draft = approved(pd.concat(frames, ignore_index=True))
    records = []
    for _, row in draft.iterrows():
        record, error = to_record(row)
        if not error and not args.no_leak_guard and {s["value"] for s in record["slots"] if s["type"] == "OBJECT"} & test_objects:
            error = "vật thể trùng test (leak)"
        if error:
            print(f"BỎ: {row['text']!r}: {error}")
            continue
        records.append((row["category"], {**record, "category": row["category"]}))
    print(f"Nháp: {len(draft)} câu đã duyệt, dùng được {len(records)}")

    # Chia val theo từng nhóm vật để nhóm nào cũng có câu đánh giá.
    frame = pd.DataFrame(records, columns=["category", "record"])
    val_index = frame.groupby("category").sample(frac=VAL_FRACTION, random_state=SEED).index
    extra = {"val": frame.loc[val_index, "record"].tolist(),
             "train": frame.drop(val_index)["record"].tolist()}

    for split in ["train", "val"]:
        base = [json.loads(line) for line in (src / f"{split}.jsonl").open(encoding="utf-8")]
        write_jsonl(out / f"{split}.jsonl", base + extra[split])
        print(f"{split}: {len(base)} gốc + {len(extra[split])} nháp")
    write_jsonl(out / "new_val.jsonl", extra["val"])
    shutil.copy(src / "test.jsonl", out / "test.jsonl")
    print(f"test: giữ nguyên bản gốc; new_val: {len(extra['val'])} câu → {out}")


if __name__ == "__main__":
    row = {"text": "à cái áo này với cái quần này có hợp không", "label": "ANSWER_VISUAL_QUESTION", "label_id": 3,
           "object": "cái áo|cái quần", "location": "", "place": "", "attribute": "hợp"}
    record, error = to_record(row)
    assert error is None and record["tags"][1:3] == ["B-OBJECT", "I-OBJECT"], record
    assert all(row["text"][s["start"]:s["end"]] == s["value"] for s in record["slots"]), record["slots"]
    assert to_record({**row, "attribute": "áo này"})[1] is not None  # chồng lên OBJECT
    main()
