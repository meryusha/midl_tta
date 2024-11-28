import os
import json
import argparse
import tqdm
import decord
import glob

def check_corrupted_videos(path, output_path, minlen):
    """Use decord to check if videos are corrupted or not."""
    assert os.path.exists(path), f"{path} does not exist"
    corrupted_videos = {}
    videos = glob.glob(os.path.join(path, "**/*.mp4"), recursive=True)
    for video in tqdm.tqdm(videos):
        try:
            tmp = decord.VideoReader(video)
            corrupted_videos[os.path.basename(video)] = len(tmp)
        except:
            corrupted_videos[os.path.basename(video)] = None
    with open(output_path, "w") as f:
        json.dump(corrupted_videos, f)
    print(f"Corrupted videos: {len(corrupted_videos)}")
    return corrupted_videos

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--minlen", type=int, default=10)
    args = parser.parse_args()
    check_corrupted_videos(args.path, args.output, args.minlen)