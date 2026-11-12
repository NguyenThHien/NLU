"""Tăng cường dữ liệu không cần gán nhãn: từ đệm, lặp từ, viết hoa đầu câu, dấu câu.

Mô phỏng lời nói qua ASR ("ừ thì cái cái hộp này") và văn bản có dấu câu; nhãn BIO dịch theo.
Chỉ dùng cho tập train.
"""

import random

FILLERS = [["ờ"], ["à"], ["ừ", "thì"], ["ờm"], ["ơ"], ["ừm"], ["ờ", "thì"]]
QUESTION_WORDS = {"gì", "không", "chưa", "nào", "đâu", "sao", "mấy", "bao", "hông", "chi", "mô", "à", "hả"}


def augment(tokens, tags, rng):
    """Trả (tokens, tags) mới với ít nhất một phép biến đổi; không sửa đầu vào."""
    tokens, tags = list(tokens), list(tags)
    applied = False
    while not applied:
        if rng.random() < 0.3:  # lặp một từ: ưu tiên từ đầu của OBJECT ("cái cái hộp")
            starts = [i for i, tag in enumerate(tags) if tag == "B-OBJECT"]
            i = rng.choice(starts) if starts and rng.random() < 0.6 else rng.randrange(len(tokens))
            tokens.insert(i, tokens[i])
            tags.insert(i, "O")  # bản lặp đứng trước, ngoài slot
            applied = True
        if rng.random() < 0.35:  # từ đệm đầu câu, có thể kèm dấu phẩy
            filler = rng.choice(FILLERS) + ([","] if rng.random() < 0.5 else [])
            tokens, tags = filler + tokens, ["O"] * len(filler) + tags
            applied = True
        if rng.random() < 0.5:  # dấu câu cuối
            is_question = any(token in QUESTION_WORDS for token in tokens[-3:])
            tokens, tags = tokens + ["?" if is_question else "."], tags + ["O"]
            applied = True
        if rng.random() < 0.5:  # viết hoa đầu câu
            tokens[0] = tokens[0][:1].upper() + tokens[0][1:]
            applied = True
    return tokens, tags


if __name__ == "__main__":
    rng = random.Random(0)
    for _ in range(200):
        new_tokens, new_tags = augment(["cái", "hộp", "này", "màu", "gì"], ["B-OBJECT", "I-OBJECT", "O", "B-ATTRIBUTE", "I-ATTRIBUTE"], rng)
        assert len(new_tokens) == len(new_tags)
        object_tokens = [t for t, g in zip(new_tokens, new_tags) if g.endswith("OBJECT")]
        assert [t.lower() for t in object_tokens] == ["cái", "hộp"], (new_tokens, new_tags)
    print(" ".join(augment("đọc cái hộp này".split(), ["O", "B-OBJECT", "I-OBJECT", "O"], random.Random(3))[0]))
    print("ok")
