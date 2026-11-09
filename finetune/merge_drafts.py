"""Gộp câu nháp đã duyệt vào train/val gốc → dataset/<tên>/ (dataset gốc không bị sửa).

review: ok / ok_ai → giữ nhãn; tên nhãn → đổi nhãn; xoa hoặc trống → bỏ.
Test giữ nguyên bản gốc để so sánh được với model cũ.

  python merge_drafts.py --draft ../dataset/drafts/contrast_train_draft.csv --name v3
  $env:NLU_DATA_DIR="dataset/v3"; python preprocess.py; python train.py --run-name phobert_intent_v3
"""

import argparse
import shutil

import pandas as pd

from common import LABEL2ID, ROOT_DIR, SEED

VAL_FRACTION = 0.15
# Biến thể có tiền tố phải nằm cùng tập với câu gốc, nếu không val sẽ "thấy" câu train.
PREFIXES = ("bạn ơi ", "hina ơi ", "ừm ", "à ", "này ")


def core(text):
    for prefix in PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def approved(draft):
    review = draft["review"].fillna("").str.strip()
    keep = draft[review.isin(["ok", "ok_ai"]) | review.isin(LABEL2ID)].copy()
    relabel = keep["review"].isin(LABEL2ID)
    keep.loc[relabel, "label"] = keep.loc[relabel, "review"]
    keep["label_id"] = keep["label"].map(LABEL2ID)
    return keep


def split_by_group(data):
    """Chia theo câu gốc, phân tầng theo (error_type, label)."""
    data = data.assign(group=data["text"].map(core))
    val_groups = set()
    for _, part in data.groupby(["error_type", "label"]):
        groups = part["group"].drop_duplicates().sample(frac=1, random_state=SEED)
        val_groups.update(groups.iloc[: round(len(groups) * VAL_FRACTION)])
    is_val = data["group"].isin(val_groups)
    return data[~is_val], data[is_val]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", required=True)
    parser.add_argument("--name", required=True, help="Thư mục con trong dataset/.")
    args = parser.parse_args()

    src, out = ROOT_DIR / "dataset", ROOT_DIR / "dataset" / args.name
    out.mkdir(parents=True, exist_ok=True)

    draft = approved(pd.read_csv(args.draft, encoding="utf-8-sig"))
    extra = dict(zip(["train", "val"], split_by_group(draft)))
    columns = ["text", "label", "label_id"]

    for split in ["train", "val"]:
        base = pd.read_csv(src / f"{split}.csv", encoding="utf-8-sig")[columns]
        merged = pd.concat([base, extra[split][columns]], ignore_index=True)
        merged.to_csv(out / f"{split}.csv", index=False, encoding="utf-8-sig")
        print(f"{split}: {len(base)} gốc + {len(extra[split])} nháp = {len(merged)}")
        print(merged["label"].value_counts().to_string())
    shutil.copy(src / "test.csv", out / "test.csv")
    print(f"test: giữ nguyên bản gốc → {out / 'test.csv'}")


if __name__ == "__main__":
    main()
