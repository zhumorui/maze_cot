import argparse
import random
import numpy as np
import os
import re
from collections import deque
from datasets import Dataset
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def generate_maze(width, height, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    maze = np.ones((height, width), dtype=int)
    start = (1, 1)
    end = (width - 2, height - 2)
    maze[start[1], start[0]] = 0
    maze[end[1], end[0]] = 0
    dirs = [(0, 2), (2, 0), (0, -2), (-2, 0)]
    stack = [start]
    visited = {start}
    while stack:
        x, y = stack[-1]
        neighbors = []
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 < nx < width - 1 and 0 < ny < height - 1 and (nx, ny) not in visited:
                neighbors.append((dx, dy, nx, ny))
        if neighbors:
            dx, dy, nx, ny = random.choice(neighbors)
            maze[y + dy // 2, x + dx // 2] = 0
            maze[ny, nx] = 0
            visited.add((nx, ny))
            stack.append((nx, ny))
        else:
            stack.pop()
    path = find_path(maze, start, end)
    if not path:
        return generate_maze(width, height, seed)
    return maze, start, end, path

def find_path(maze, start, end):
    h, w = maze.shape
    q = deque([start])
    visited = {start}
    parent = {}
    moves = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    while q:
        x, y = q.popleft()
        if (x, y) == end:
            path = []
            while (x, y) != start:
                path.append((x, y))
                x, y = parent[(x, y)]
            path.append(start)
            return list(reversed(path))
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and maze[ny, nx] == 0 and (nx, ny) not in visited:
                visited.add((nx, ny))
                parent[(nx, ny)] = (x, y)
                q.append((nx, ny))
    return []

def path_to_directions(path):
    dir_map = {(1, 0): 'right', (-1, 0): 'left', (0, 1): 'down', (0, -1): 'up'}
    return [dir_map.get((x1 - x0, y1 - y0), '') for (x0, y0), (x1, y1) in zip(path, path[1:])]

def extract_answer_directions(answer_text):
    match = re.search(r"####\s*([a-z, ]+)", answer_text)
    if not match:
        return []
    return [d.strip() for d in match.group(1).split(',')]

def directions_to_path(start, directions):
    x, y = start
    path = [start]
    move_map = {'right': (1, 0), 'left': (-1, 0), 'down': (0, 1), 'up': (0, -1)}
    for d in directions:
        dx, dy = move_map.get(d, (0, 0))
        x, y = x + dx, y + dy
        path.append((x, y))
    return path

def visualize_and_save(maze, path, start, end, filename):
    solution = maze.copy()
    for x, y in path:
        solution[y, x] = 2
    cmap = ListedColormap(['white', 'black', 'blue'])
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(maze, cmap='binary')
    axes[0].plot(start[0], start[1], 'go')
    axes[0].plot(end[0], end[1], 'ro')
    axes[0].axis('off')
    axes[1].imshow(solution, cmap=cmap, vmin=0, vmax=2)
    axes[1].plot(start[0], start[1], 'go')
    axes[1].plot(end[0], end[1], 'ro')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)

def format_maze_matrix(maze):
    return '\n'.join(' '.join(str(cell) for cell in row) for row in maze)

def build_chain_of_thought(start, path):
    steps = [f"We start at {start}." ]
    dirs = path_to_directions(path)
    for (x0, y0), (x1, y1), d in zip(path, path[1:], dirs):
        steps.append(f"Move {d} to {(x1, y1)}.")
    steps.append(f"Thus we reach the end at {path[-1]}." )
    final_dirs = ', '.join(dirs)
    steps.append(f"#### {final_dirs}")
    return '\n'.join(steps)

def generate_samples(num_samples, width, height, val_ratio, output_dir):
    samples = []
    vis_dir = os.path.join(output_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    for seed in range(num_samples):
        maze, start, end, path = generate_maze(width, height, seed)
        maze_text = format_maze_matrix(maze)
        question = (
            f"Given the following maze (0=passage,1=wall):\n{maze_text}\n"
            f"Start: {start}, End: {end}."
        )
        prompt = (
            question +
            "\nLet's think step by step and output the final answer after \"####\"."
        )
        completion = build_chain_of_thought(start, path)
        gt_dirs = ', '.join(path_to_directions(path))
        directions = extract_answer_directions(completion)
        pred_path = directions_to_path(start, directions) if directions else []
        img_path = os.path.join(vis_dir, f"maze_{seed}.png")
        if pred_path:
            visualize_and_save(maze, pred_path, start, end, img_path)
        entry = {
            'data_source': 'morin/maze',
            'prompt': [{'role': 'user', 'content': prompt}],
            'ability': 'maze_solving',
            'reward_model': {'ground_truth': gt_dirs, 'style': 'rule'},
            'extra_info': {
                'answer': completion,
                'index': seed,
                'question': question,
                'split': None,
                'vis_path': img_path
            }
        }
        samples.append(entry)
    random.shuffle(samples)
    n_val = int(len(samples) * val_ratio)
    for i, s in enumerate(samples):
        s['extra_info']['split'] = 'test' if i < n_val else 'train'
    train = [s for s in samples if s['extra_info']['split']=='train']
    val = [s for s in samples if s['extra_info']['split']=='test']
    return train, val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--width', type=int, default=11)
    parser.add_argument('--height', type=int, default=11)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--out_dir', type=str, default='./data')
    args = parser.parse_args()
    train_samples, val_samples = generate_samples(
        args.num_samples, args.width, args.height, args.val_ratio, args.out_dir
    )
    os.makedirs(args.out_dir, exist_ok=True)
    train_ds = Dataset.from_list(train_samples)
    val_ds = Dataset.from_list(val_samples)
    train_path = os.path.join(args.out_dir, 'maze_train.parquet')
    val_path = os.path.join(args.out_dir, 'maze_val.parquet')
    train_ds.to_parquet(train_path)
    val_ds.to_parquet(val_path)
    print(f"Wrote Parquet files:\n  Train -> {train_path}\n  test -> {val_path}\nVisualizations in {os.path.join(args.out_dir, 'vis')}")

if __name__ == '__main__':
    main()
