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
# import mmcv
from sklearn.utils import shuffle

def write_dataset(args):

    video_files_path = '/home/username/clips/*'
    video_files = glob.glob(video_files_path)
    video_names = [os.path.basename(video_file).split('_')[0] for video_file in video_files]
    unique_video_names = set(video_names)
    print(f"creating shards for {len(unique_video_names)} unique videos")
    arrays_video = []
    for video_name in unique_video_names:
        sorted_list = sorted([filename for filename in video_files if video_name in filename ], key = lambda x: int(os.path.basename(x).split('_')[1].split('.mp4')[0]))
        # print(len(sorted_list))
        arrays_video.extend(sorted_list[:-1][0::2])
        # print(sorted_list[:-1][0::2])
    print(f"creating shards for {len(arrays_video)} unique videos")
    arrays_video = shuffle(arrays_video)
    # This is the output pattern under which we write shards.
    pattern = os.path.join(args.shards, f'ego4d-10sec-%06d.tar')

    with wds.ShardWriter(
            pattern, maxsize=int(args.maxsize),
            maxcount=int(args.maxcount)) as sink:
        sink.verbose = 0
        all_keys = set()

        skipped = 0
        for idx, video_file in tqdm(enumerate(arrays_video), desc='total', total=len(arrays_video)):
            if not osp.exists(video_file):
                skipped += 1
                tqdm.write(f'Skipping {video_file}, {skipped}/{len(video_file)}')
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
            # imu_data = np.load(imu_file)
            with open(f"{video_file}", "rb") as stream:
                video_data = stream.read()  
            sample = {'__key__': xkey, 'mp4': video_data}

            # Write the sample to the sharded tar archives.
            sink.write(sample)
        print(f'skipped: {skipped}/{len(arrays_video)}')
        print(f'total keys: {len(all_keys)}')


def parse_args():
    parser = argparse.ArgumentParser(
        """Generate sharded dataset from original ImageNet data.""")
    parser.add_argument('--maxsize', type=float, default=1e9)
    parser.add_argument('--maxcount', type=float, default=960)
    parser.add_argument('--shards', help='directory where shards are written')
    parser.add_argument('--root', help='data root path', default="")
    parser.add_argument('--info', help='tsv path', default="")
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
