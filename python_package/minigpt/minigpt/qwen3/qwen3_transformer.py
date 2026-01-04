from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",  # 自动选择 float16 / bfloat16（若硬件支持）
    device_map="auto",  # 自动分配到 CPU/GPU（支持多卡）
)

prompt = "明天做点啥"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,  # ← 关键！启用 thinking 模式
)

# 当 enable_thinking=True 时，
# tokenizer 会在 assistant 回复前插入 <|im_start|>think\n 等特殊 token
# 模型会先生成一段 内部推理（thinking），再输出最终答案
# 特殊 token 示例：
# <|im_start|>think → 开始思考
# </think>（token ID 151668）→ 思考结束
# 然后才是 <|im_start|>assistant 的正式回答

print(f"text: {text}")
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
print(f"model_inputs: {model_inputs}")

# conduct text completion
generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

# parsing thinking content
# try:
#     # rindex finding 151668 (</think>)
#     index = len(output_ids) - output_ids[::-1].index(151668)
# except ValueError:
#     index = 0

# thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[0:], skip_special_tokens=True).strip("\n")

# print("thinking content:", thinking_content)
print("content:", content)
