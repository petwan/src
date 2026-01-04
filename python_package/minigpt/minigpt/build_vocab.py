import json
import argparse
from collections import Counter


def build_vocab(data_path: str, output_path: str):
    counter = Counter()

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
                # 收集 question 和 answer 中的所有字符
                counter.update(item["question"])
                counter.update(item["answer"])
            except Exception as e:
                print(f"Warning: skip invalid line: {line[:50]}... | Error: {e}")

    # 获取所有唯一字符
    chars = sorted(counter.keys())

    # 添加特殊 token
    special_tokens = ["<pad>", "<unk>", "<sep>"]

    # 构建 word2id：先放特殊 token，再放字符（顺序固定便于复现）
    word2id = {token: i for i, token in enumerate(special_tokens)}
    for char in chars:
        if char not in special_tokens:  # 防御：跳过特殊 token
            word2id[char] = len(word2id)  # 自动递增

    # 构建 id2word
    id2word = {i: token for token, i in word2id.items()}

    vocab = {"word2id": word2id, "id2word": id2word}

    # 保存 vocab
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)

    print(f"✅ Vocabulary built and saved to {output_path}")
    print(f"   Total tokens: {len(word2id)}")
    print(f"   Special tokens: {special_tokens}")
    print(f"   Sample chars: {chars[:10]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build vocabulary from QA dataset.")
    parser.add_argument(
        "--data", type=str, required=True, help="Path to training data (JSONL format)"
    )
    parser.add_argument(
        "--output", type=str, default="vocab.json", help="Output vocab file path"
    )
    args = parser.parse_args()

    build_vocab(args.data, args.output)
