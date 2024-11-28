# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------
import argparse
import json
import os
import os.path as osp
import random
import sys
import zipfile
import glob
import numpy as np
import pandas as pd
import webdataset as wds
from tqdm import tqdm
from sklearn.utils import shuffle


def write_dataset(args):

    # This is the output pattern under which we write shards.
    pattern = os.path.join(args.shards, f'EPIC-sound-{args.split}-%06d.tar')
    video_to_path = {}
    path_to_class = {}
    annotations_data = pd.read_csv(open(f'/home/username/epic-sounds-annotations/EPIC_Sounds_{args.split}.csv'))
    video_files_path = "/home/username/epic_sounds/clips/*"
    video_files = glob.glob(video_files_path)
    for v in video_files:
        video_to_path[os.path.basename(v).replace('.mp4', '')] = v
    annotations_data = shuffle(annotations_data)

    for i, clip_data in annotations_data.iterrows():

        clip_id = clip_data['annotation_id']
        video_id = clip_data['video_id']
        if clip_id not in video_to_path:
            print(f'Skipped {clip_id}')
            continue
        path_to_class[video_to_path[clip_id]] = clip_data['class_id']

    with wds.ShardWriter(
            pattern, maxsize=int(args.maxsize),
            maxcount=int(args.maxcount)) as sink:
        sink.verbose = 0
        all_keys = set()
        skipped = 0
        for idx, video_file in tqdm(enumerate(path_to_class.keys()), desc='total', total=len(path_to_class)):
            if not osp.exists(video_file):
                skipped += 1
                tqdm.write(
                    f'Skipping {video_file}, {skipped}/{len(path_to_class)}')
                continue
            # Construct a unique key from the filename.
            key = os.path.splitext(os.path.basename(video_file))[0]

            # Useful check.
            if key in all_keys:
                tqdm.write(f'duplicate: {video_file}')
                continue
            assert key not in all_keys
            all_keys.add(key)

            # text = str(df.loc[fname]['caption'])

            # Construct a sample.
            xkey = key
            with open(f"{video_file}", "rb") as stream:
                video_data = stream.read()

            sample = {'__key__': xkey, 'mp4': video_data, 'cls': path_to_class[video_file] }

            # Write the sample to the sharded tar archives.
            sink.write(sample)
        print(f'skipped: {skipped}/{len(path_to_class)}')
        print(f'total keys: {len(path_to_class)}')


def parse_args():
    parser = argparse.ArgumentParser(
        """Generate sharded dataset from original ImageNet data.""")
    parser.add_argument('--maxsize', type=float, default=1e9)
    parser.add_argument('--maxcount', type=float, default=256)
    parser.add_argument('--shards', help='directory where shards are written')
    parser.add_argument('--root', help='data root path', default="")
    parser.add_argument('--split', help='split', default="train")
    args = parser.parse_args()

    assert args.maxsize > 10000000
    assert args.maxcount < 1000000
    return args


def main():
    args = parse_args()

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if not os.path.isdir(os.path.join(args.shards, '.')):
        print(
            f'{args.shards}: should be a writable destination directory for shards',
            file=sys.stderr)
        sys.exit(1)

    write_dataset(args=args)


if __name__ == '__main__':
    main()
