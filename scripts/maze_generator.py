import numpy as np
import random
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import deque
import os
import csv


def generate_maze(width, height, seed=None):
    """
    Generate a maze, returning the maze matrix, start point, end point, and path.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    maze = np.ones((height, width), dtype=int)
    start = (1, 1)
    end = (width - 2, height - 2)
    maze[start[1], start[0]] = 0
    maze[end[1], end[0]] = 0

    # Depth-first generation
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

    # Ensure the main path exists
    path = find_path(maze, start, end)
    if not path:
        return generate_maze(width, height, seed)

    # Randomly block some non-main paths
    path_set = set(path)
    for iy in range(height):
        for ix in range(width):
            if maze[iy, ix] == 0 and (ix, iy) not in path_set and (ix, iy) not in (start, end):
                if random.random() < 0.3:
                    maze[iy, ix] = 1
    return maze, start, end, path


def find_path(maze, start, end):
    """BFS pathfinding, returns path list."""
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


def visualize_and_save(maze, path, start, end, filename):
    """Generate visualization from structured data."""
    # Mark the path
    solution = maze.copy()
    for x, y in path:
        solution[y, x] = 2

    cmap = ListedColormap(['white', 'black', 'blue'])
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    # Original maze
    axes[0].imshow(maze, cmap='binary')
    axes[0].plot(start[0], start[1], 'go')
    axes[0].plot(end[0], end[1], 'ro')
    axes[0].set_title('Maze')
    axes[0].axis('off')
    # Solution map
    axes[1].imshow(solution, cmap=cmap, vmin=0, vmax=2)
    axes[1].plot(start[0], start[1], 'go')
    axes[1].plot(end[0], end[1], 'ro')
    axes[1].set_title('Solution')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)


def path_to_directions(path):
    """Convert coordinate path to direction string list."""
    dir_map = {(1, 0): 'right', (-1, 0): 'left', (0, 1): 'down', (0, -1): 'up'}
    return [dir_map.get((x1 - x0, y1 - y0), '') for (x0, y0), (x1, y1) in zip(path, path[1:])]


def build_chain_of_thought(start, path):
    steps = [f"We start at {start}."]
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        dir_name = path_to_directions([(x0, y0), (x1, y1)])[0]
        steps.append(f"Move {dir_name} to {(x1, y1)}.")
    steps.append(f"Thus we reach the end at {path[-1]}.")
    final_dirs = ', '.join(path_to_directions(path))
    steps.append(f"#### {final_dirs}")
    return '\n'.join(steps)


def format_maze_matrix(maze):
    """Format maze as a multi-line string."""
    return '\n'.join(' '.join(str(cell) for cell in row) for row in maze)


def generate_dataset(num_samples, width, height,
                     output_jsonl='maze_dataset.jsonl',
                     image_dir='maze_images_from_jsonl'):
    """
    Generate structured data and visualize.
    """
    os.makedirs(image_dir, exist_ok=True)
    with open(output_jsonl, 'w') as jfile:
        for seed in range(num_samples):
            maze, start, end, path = generate_maze(width, height, seed)
            sample = {
                'maze': maze.tolist(),
                'start': list(start),
                'end': list(end),
                'path': [list(p) for p in path]
            }
            jfile.write(json.dumps(sample, ensure_ascii=False) + '\n')
    # Visualization
    with open(output_jsonl, 'r') as jfile:
        for idx, line in enumerate(jfile):
            sample = json.loads(line)
            maze = np.array(sample['maze'], dtype=int)
            start = tuple(sample['start'])
            end = tuple(sample['end'])
            path = [tuple(p) for p in sample['path']]
            img_path = os.path.join(image_dir, f"maze_{idx}.png")
            visualize_and_save(maze, path, start, end, img_path)
    print(f"Generated {num_samples} samples. Visualizations in '{image_dir}'. Data in '{output_jsonl}'.")


def convert_to_prompt_completion(input_jsonl,
                                 train_jsonl='train.jsonl',
                                 val_jsonl='val.jsonl',
                                 train_csv='train.csv',
                                 val_csv='val.csv',
                                 val_ratio=0.2):
    """
    Read JSONL, generate {prompt,completion} for each sample and split into train/val.
    """
    samples = []
    with open(input_jsonl, 'r') as f:
        for line in f:
            data = json.loads(line)
            maze = np.array(data['maze'], dtype=int)
            start = tuple(data['start'])
            path = [tuple(p) for p in data['path']]
            maze_text = format_maze_matrix(maze)
            prompt = (
                f"Q: Given the following maze (0=passage,1=wall):\n"
                f"{maze_text}\n"
                f"Start: {start}, End: {tuple(data['end'])}.\n"
                "Let's think step by step.\n"
                "At the end, please output **only** the final path steps in one line, "
                "prefixed by `#### `, as comma‑separated directions (up/down/left/right).\n"
                "For example: `#### down, down, right, right, up`.\n"
                "A:"
            )
            completion = build_chain_of_thought(start, path)
            samples.append({'prompt': prompt, 'completion': completion})

    # Split train/val
    random.shuffle(samples)
    n_val = int(len(samples) * val_ratio)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    # Write to files using the provided file paths
    file_paths = {
        'train': (train_jsonl, train_csv, train_samples),
        'val': (val_jsonl, val_csv, val_samples)
    }
    
    for name, (jsonl_path, csv_path, subset) in file_paths.items():
        with open(jsonl_path, 'w') as jf, open(csv_path, 'w', newline='') as cf:
            writer = csv.writer(cf)
            writer.writerow(['prompt', 'completion'])
            for s in subset:
                jf.write(json.dumps(s, ensure_ascii=False) + '\n')
                writer.writerow([s['prompt'], s['completion']])
    print(f"Saved {len(train_samples)} train and {len(val_samples)} val samples to JSONL/CSV.")

if __name__ == '__main__':
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Set output paths to data directory
    output_jsonl = 'data/maze_dataset.jsonl'
    image_dir = 'data/maze_images'
    train_jsonl = 'data/train.jsonl'
    val_jsonl = 'data/val.jsonl'
    train_csv = 'data/train.csv'
    val_csv = 'data/val.csv'
    
    generate_dataset(num_samples=1000, width=5, height=5, output_jsonl=output_jsonl, image_dir=image_dir)
    convert_to_prompt_completion(output_jsonl, train_jsonl=train_jsonl, val_jsonl=val_jsonl, 
                                train_csv=train_csv, val_csv=val_csv, val_ratio=0.2)
