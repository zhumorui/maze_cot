#!/usr/bin/env python3
"""
Preprocess maze dataset (train.jsonl / val.jsonl) into Parquet format for PPO/RLHF training.
"""
import argparse
import os
import json
from datasets import load_dataset
def extract_args():
    parser = argparse.ArgumentParser(
        description="Convert prompt/completion JSONL into Huggingface Dataset and output Parquet files"
    )
    parser.add_argument(
        "--train_jsonl", default="train.jsonl",
        help="Path to the train JSONL file (prompt/completion pairs)"
    )
    parser.add_argument(
        "--val_jsonl", default="val.jsonl",
        help="Path to the validation JSONL file"
    )
    parser.add_argument(
        "--out_dir", default="./",
        help="Directory to write the Parquet files"
    )
    parser.add_argument(
        "--trigger", default="Let's think step by step and output the final answer after we reach the end.",
        help="Trigger phrase to append to prompt if missing"
    )
    return parser.parse_args()

def preprocess_split(split_name, file_path, trigger):
    # Load JSONL as a Dataset split
    ds = load_dataset(
        'json', data_files={split_name: file_path}, split=split_name
    )
    def map_fn(example, idx):
        prompt = example['prompt']
        # Ensure trigger phrase present
        if trigger not in prompt:
            prompt = prompt.strip() + ' ' + trigger
        answer = example['completion'].strip()
        return {
            'data_source': 'maze_dataset',
            'prompt': [{'role': 'user', 'content': prompt}],
            'ability': 'maze_solving',
            'reward_model': {'style': 'rule', 'ground_truth': answer},
            'extra_info': {
                'split': split_name,
                'index': idx,
                'prompt_raw': example['prompt'],
                'completion_raw': example['completion'],
            }
        }
    # Apply mapping
    return ds.map(map_fn, with_indices=True)

def main():
    args = extract_args()
    os.makedirs(args.out_dir, exist_ok=True)
    # Process train and validation splits
    train_ds = preprocess_split('train', args.train_jsonl, args.trigger)
    val_ds = preprocess_split('validation', args.val_jsonl, args.trigger)
    # Write to Parquet
    train_path = os.path.join(args.out_dir, 'maze_train.parquet')
    val_path = os.path.join(args.out_dir, 'maze_val.parquet')
    train_ds.to_parquet(train_path)
    val_ds.to_parquet(val_path)
    print(f'Wrote Parquet files:\n  Train -> {train_path}\n  Validation -> {val_path}')

if __name__ == '__main__':
    main()
