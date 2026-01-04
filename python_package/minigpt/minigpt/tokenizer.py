import json
import token


class Tokenizer:
    def __init__(self, vocab_path: str):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        self.word2id = vocab["word2id"]
        self.id2word = {int(k): v for k, v in vocab["id2word"].items()}

        # 固定特殊 token ID
        self.pad_token_id = self.word2id["<pad>"]
        self.unk_token_id = self.word2id["<unk>"]
        self.sep_token_id = self.word2id["<sep>"]

    def encode(
        self,
        question: str,
        answer: str,
        max_length: int = 128,
        pad_to_max_length: bool = True,
    ):
        """将问答对编码为 token ID 序列。"""
        tokens = []

        # encode question
        for char in question:
            tokens.append(self.word2id.get(char, self.unk_token_id))
        tokens.append(self.sep_token_id)  # 添加分隔符

        # encode answer
        if answer is not None:
            for char in answer:
                tokens.append(self.word2id.get(char, self.unk_token_id))

            tokens.append(self.sep_token_id)

        # 构建 attention mask（1=真实 token，0=padding）
        attn_mask = [1] * len(tokens)

        # 截断或填充
        if pad_to_max_length:
            if len(tokens) > max_length:
                # 截断（保留开头）
                tokens = tokens[:max_length]
                attn_mask = attn_mask[:max_length]
            else:
                # 填充
                pad_len = max_length - len(tokens)
                tokens.extend([self.pad_token_id] * pad_len)
                attn_mask.extend([0] * pad_len)

        return tokens, attn_mask

    def decode(self, ids):
        """将 token ID 列表解码为原始文本（跳过 <pad>）。"""
        return "".join(
            self.id2word[i] for i in ids if i != self.pad_token_id  # 跳过填充符
        )

    def get_vocab_size(self):
        return len(self.id2word)


if __name__ == "__main__":
    question = "你好，最近怎么样？"
    answer = "我很好，谢谢！"

    tokenizer = Tokenizer("./data/vocab.json")

    input_ids, attn_mask = tokenizer.encode(question, answer, max_length=32)
    print(input_ids)
    print(tokenizer.decode(input_ids))
