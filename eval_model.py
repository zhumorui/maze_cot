import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from verl.utils.reward_score.maze import extract_directions

model_dir = "checkpoints/maze_cot_sft/global_step_300"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")

df = pd.read_parquet("data/maze/maze_val.parquet")

total = 0
correct = 0

for i, (_, row) in enumerate(df.iterrows()):
    if i >= 10:
        break
    prompt = row["prompt"][0]["content"]

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    pred_dirs = extract_directions(pred_text)
    gt_dirs = extract_directions(row["reward_model"]["ground_truth"])

    print("pred_text: ", pred_text)
    print("pred_dirs: ", pred_dirs)
    print("gt_dirs: ", gt_dirs)

    is_match = (pred_dirs == gt_dirs)
    total += 1
    correct += int(is_match)

accuracy = correct / total * 100
print(f"Total samples: {total}")
print(f"Correct samples: {correct}")
print(f"Accuracy: {accuracy:.2f}%")
