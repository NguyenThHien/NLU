"""Download only the VnCoreNLP components needed for word segmentation."""

from pathlib import Path
from urllib.request import urlretrieve


SAVE_DIR = Path(__file__).resolve().parent / "vncorenlp"
BASE_URL = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/"
FILES = [
    "VnCoreNLP-1.2.jar",
    "models/wordsegmenter/wordsegmenter.rdr",
    "models/wordsegmenter/vi-vocab",
]


def main():
    for relative_path in FILES:
        target = SAVE_DIR / relative_path
        if target.is_file() and target.stat().st_size > 0:
            print(f"Already downloaded: {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".part")
        print(f"Downloading: {relative_path}", flush=True)
        urlretrieve(BASE_URL + relative_path, temporary)
        temporary.replace(target)


if __name__ == "__main__":
    main()
