import torch
from minigpt.tokenizer import Tokenizer
from minigpt.model import GPTLMHeadModel, GPTConfig


def generate(model, tokenizer, question, max_length, device):
    """
    外部问答生成函数（使用模型内部的 generate 方法）
    """
    # 1. 编码问题 + <sep>
    tokens, _ = tokenizer.encode(
        question, answer=None, max_length=max_length, pad_to_max_length=False
    )
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(
        0
    )  # [1, seq_len]

    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    # 2. 调用模型的标准 generate
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_length,  # 注意：这里实际是“最多新增”，不是总长
        stop_token_ids=[sep_id, pad_id],
        do_sample=False,  # greedy
    )

    # 3. 后处理：提取答案部分
    generated_ids = output_ids[0].tolist()

    try:
        first_sep = generated_ids.index(sep_id)
        answer_ids = generated_ids[first_sep + 1 :]
    except ValueError:
        answer_ids = generated_ids[len(tokens) :]  # fallback: skip input part

    # 移除答案中遇到的第一个 <sep> 或 <pad> 及之后内容
    answer_clean = []
    for tok in answer_ids:
        if tok != sep_id and tok != pad_id:
            answer_clean.append(tok)
        else:
            break

    return tokenizer.decode(answer_clean)


def main():
    vocab_path = "./data/vocab.json"
    max_length = 140  # 注意：现在这个值用于控制总长度（见上）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "output/best_model.pth"

    tokenizer = Tokenizer(vocab_path)
    config = GPTConfig(vocab_size=tokenizer.get_vocab_size())
    model = GPTLMHeadModel(config).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])  # ← 关键！
    model.eval()

    while True:
        question = input("Q: ")
        if question == "":
            break
        answer = generate(model, tokenizer, question, max_length, device)
        print("A:", answer)


if __name__ == "__main__":
    main()
