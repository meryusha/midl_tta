import pandas as pd
import os
import numpy as np
import argparse
import json

def trim_epic_kitchens_video(input_path, ann_dict, clip_name, outpath):
    """ Trim videos according to the annotations json
        and save them to the outpath """
    start, end = ann_dict['time_stamps'][0]
    folder_id = ann_dict['participant_id']
    video_id = ann_dict['video_id']
    input_video = f"{input_path}/{video_id}.MP4"
    out_path_clip = f'{outpath}/{clip_name}.mp4'
    if not os.path.exists(out_path_clip):
        print(f'Trimming video: {clip_name} -ss {start} -to {end}')
        try:
            os.system(f"ffmpeg  -i {input_video} -ss {start} -to {end} -c:v libx264 -vf scale=-2:256  -c:a copy {out_path_clip}")
        except:
            print(f"Error trimming video: {clip_name}")
        pass


def group_annotations_per_narration(annotations):
    """ Group the annotations per video and return a dict with the video name as key
        and a list of tuples (start, end) as value """
    annotations_df = pd.read_csv(annotations)
    grouped = annotations_df.groupby('annotation_id')
    out_name = annotations.split('/')[-1].split('.')[0]
    if os.path.exists(f'tmp/{out_name}.json'):
        print(f'{out_name}.json already exists')
        print(f'Loading annotations from file: tmp/{out_name}.json')
        json_anns = json.load(open(f'tmp/{out_name}.json', 'r'))
        return json_anns
    else:
        json_anns = {key: {'time_stamps':list(zip(grouped.get_group(key)['start_timestamp'].values, grouped.get_group(key)['stop_timestamp'].values)),
                  'participant_id': grouped.get_group(key)['participant_id'].values[0],
                  'video_id': grouped.get_group(key)['video_id'].values[0]} 
            for key in list(grouped.groups.keys())}
        with open(f'tmp/{out_name}.json', 'w') as f:
            json.dump(json_anns, f)
        return json_anns
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/home/username/epic-sounds/videos')
    parser.add_argument('--annotations', type=str, default='/home/username/epic-sounds-annotations/EPIC_Sounds_train.csv')
    parser.add_argument('--outpath', type=str, default='/home/username/epic_sounds/clips')
    parser.add_argument('--video_idx', type=int, required=True)
    args = parser.parse_args()
    
    json_annotations = group_annotations_per_narration(args.annotations)
    all_clip_ids = list(json_annotations.keys())
    clip_name = all_clip_ids[args.video_idx]
    trim_epic_kitchens_video(args.input_path, json_annotations[clip_name], clip_name, args.outpath)