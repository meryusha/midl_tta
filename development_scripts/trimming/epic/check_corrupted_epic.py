import os
import json

def check_trimmed_clips(path, json_anns_path='EPIC_100_train.json', min_size=10000):
    """ Check if all the trimmed clips are there and are not empty"""
    json_annotations = json.load(open(f'tmp/{json_anns_path}', 'r'))
    all_clip_ids = list(json_annotations.keys())
    non_existent_clips = []
    corrupted_clips = []
    for idx, clip_id in enumerate(all_clip_ids):
        clip_path = f'{path}/{clip_id}.mp4'
        if not os.path.exists(clip_path):
            print(f'Clip {clip_id} does not exist')
            non_existent_clips.append((idx, clip_id))
            continue
        if os.stat(clip_path).st_size < min_size:
            print(f'Clip {clip_id} is empty')
            corrupted_clips.append((idx, clip_id))
        else:
            pass
    return dict(non_existent_clips=non_existent_clips, corrupted_clips=corrupted_clips)

if __name__ == "__main__":
    path = '/home/username/clips'
    set_ = 'validation'
    json_anns_path = f'EPIC_100_{set_}.json'
    missing_clips = check_trimmed_clips(path, json_anns_path)
    with open(f'tmp/missing_clips_{set_}.json', 'w') as f:
        json.dump(missing_clips, f)