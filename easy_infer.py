import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CHECKPOINT = "checkpoints/verl_examples/gsm8k/global_step_40/actor/model_world_size_1_rank_0.pt"
TOKENIZER_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# 1. 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)

# 2. 初始化模型架构（无权重）
model = AutoModelForCausalLM.from_pretrained(
    TOKENIZER_NAME,
    trust_remote_code=True,
    # 不自动下载权重，只建立模型结构
    from_tf=False,
    state_dict=None
)

# 3. 载入 Actor 权重
state_dict = torch.load(CHECKPOINT, map_location="cpu")
model.load_state_dict(state_dict)

# 4. 准备推理
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

# 5. 简单推理函数
def infer(prompt: str, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outs[0], skip_special_tokens=True)


Q = """

Given the following maze (0=passage,1=wall):
1 1 1 1 1
1 0 0 0 1
1 1 1 0 1
1 0 0 0 1
1 1 1 1 1
Start: (1, 1), End: (3, 3).
Let's think step by step.
At the end, please output **only** the final path steps in one line, prefixed by `#### `, as comma‑separated directions (up/down/left/right).
For example: `#### down, down, right, right, up`.

"""


# 测试
print(infer(Q))
