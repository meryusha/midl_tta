import os

def generate_train_val_txt(path_to_shards, out_path, val_ratio=0.1, seed=0):
    """
    Generate train.txt and val.txt from the shards
    """
    import random
    random.seed(seed)
    shards = os.listdir(path_to_shards)
    random.shuffle(shards)
    n_shards = len(shards)
    n_val = int(n_shards * val_ratio)
    n_train = n_shards - n_val
    with open(f'{out_path}/train_shards.txt', 'w') as f:
        for shard in shards[:n_train]:
            f.write(f'{path_to_shards}/{shard}' + '\n')
    with open(f'{out_path}/val_shards.txt', 'w') as f:
        for shard in shards[n_train:]:
            f.write(f'{path_to_shards}/{shard}' + '\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_shards', type=str, default='/home/username/shards')
    parser.add_argument('--out_path', type=str, default='/home/username/shards_split')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    generate_train_val_txt(args.path_to_shards, args.out_path, args.val_ratio, args.seed)