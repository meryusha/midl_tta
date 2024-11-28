import json
import os
import tqdm
import argparse
import datetime
import random
#generates log files for every mp4 file to see if any files are corrupted
# find . -name "*.mp4" -exec sh -c "ffmpeg -v error -i '{}' -map 0:1 -f null - 2>'{}.log'" \;

#this scrip iterated over videos and either reads the log files or crop the videos 

annotations_data = json.load(open("/home/username/Ego4D/ego4d_data/v1/annotations/narration.json"))
in_path = "/home/username/Ego4D/ego4d_data/v1/full_scale/"
out_path = "/home/username/ego4d_clips/"
video_count = 0
narration_count = 0
print(len(annotations_data))
videos_shuffle = list(annotations_data.keys()).copy()
random.shuffle(videos_shuffle)
for video_id in videos_shuffle:
    if narration_count > 75:
        break
    video_data = annotations_data[video_id]
    # print(video_data)
    if 'narration_pass_1' not in video_data:
        print(f'No narration pass 1 in {video_id}')
        continue
    narrations_pass_1 = video_data['narration_pass_1']
    video_count = video_count + 1
    narration_in_video_id = 0
    for narration in narrations_pass_1['narrations']:
        narration_in_video_id = narration_in_video_id + 1
        # annotation_id = narration['annotation_uid']
        narration_id = video_id  + '_' + str( narration_in_video_id)
        in_path_clip = os.path.join(in_path, video_id + ".mp4")
        out_path_clip = os.path.join(out_path, narration_id + ".mp4")
        # print(in_path_clip, out_path_clip)
        if narration['timestamp_sec'] < 0:
           narration['timestamp_sec'] = 0
        start = str(datetime.timedelta(seconds=narration['timestamp_sec']))
        end = str(datetime.timedelta(seconds=narration['timestamp_sec'] + 10 ))
        # crop the video from start to end
        if not os.path.exists(out_path_clip):
            narration_count = narration_count + 1
            os.system(f"ffmpeg  -i {in_path_clip} -ss {start} -to {end} -c:v libx264 -vf scale=w='if(lte(iw,ih),256,-1)':h='if(lte(iw,ih),-1,256)'  -c:a copy {out_path_clip}")

        