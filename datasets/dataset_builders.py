import numpy as np
import warnings
import webdataset as wds
import torch
import os, math
import tempfile
import pandas as pd
import torchaudio
from model.model_builders import get_modalities_from_string
import torchvision.io

def collate_func(batch, modalities_string, extra_keys=['cls'], single_class=True):
    modalities = get_modalities_from_string(modalities_string)
    return_dict = {}
    for modality_string, modality in zip(modalities_string, modalities):
        if modality_string == 'video':
            return_dict[modality] = torch.stack(
                [item['mp4'] for item in batch], 0)
            return_dict['video_attention'] = torch.stack(
                [torch.tensor(item['video_attention']) for item in batch], 0)
        if modality_string == 'imu':
            return_dict[modality] = torch.stack(
                [item['npy'] for item in batch], 0)
        if modality_string == 'audio':
            return_dict[modality] = torch.stack(
                [item['spec'] for item in batch], 0)
            return_dict['audio_attention'] = torch.stack(
                [torch.tensor(item['audio_attention']) for item in batch], 0)
        if 'cls' in extra_keys:
            if single_class:
                #DIRTY HACK - for TTA only
                if type(batch[0]['cls']) == dict:
                    return_dict['cls'] = torch.stack(
                        [torch.tensor(item['cls']['noun']) for item in batch], 0)
                else:
                    return_dict['cls'] = torch.stack(
                        [torch.tensor(item['cls']) for item in batch], 0)
            # if single_class:
            #     return_dict['cls'] = torch.stack(
            #         [torch.tensor(item['cls']) for item in batch], 0)
            else:
                keys = batch[0]['cls'].keys()
                return_dict['cls'] = {}
                for k in list(keys):
                    return_dict['cls'][k] = torch.stack(
                        [torch.tensor(item['cls'][k]) for item in batch], 0)

    if '__key__' in extra_keys:
        return_dict['__key__'] = np.stack(
            [(item['__key__']) for item in batch], 0)

    return return_dict


def warn_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning,
    and continue."""
    print(type(exn))
    print(exn)
    warnings.warn(repr(exn))
    return True


def filter_videos(sample, metadata, target_clip_size, filter_criteria, multiview=True, metadata_file=None):
    metadata_clip = metadata[sample['__key__']]
    if multiview:
        metadata_clip = metadata_clip[2]
    else:
        metadata_clip = metadata_clip[0]

    if 'duration_sec' in filter_criteria: #old functionality, now we resample clips that are too short in the finetuning
        if "constant_duration_sec" not in metadata and metadata[sample['__key__']]['duration_sec'] < target_clip_size:
            return False
    if 'has_audio' in filter_criteria:
        if not metadata_clip['has_audio']:
            return False
    elif 'no_audio'  in filter_criteria:
        if metadata_clip['has_audio']:
            return False
    elif 'has_video' in filter_criteria:
        if not metadata_clip['has_video']:
            return False
    elif 'no_video'  in filter_criteria:
        if metadata_clip['has_video']:
            return False
    elif 'is_modal_complete' in filter_criteria:
        if not (metadata_clip['has_audio'] and metadata_clip['has_video']):
            return False
    elif 'is_modal_incomplete' in filter_criteria:
        if metadata_clip['has_audio'] and metadata_clip['has_video']:
            return False
    # if '00210b16-9556-4501-a163' not in sample['__key__']:
    #     return False
    return True


def augment_views(samples, num_views, num_spatial_crops):
    for sample in samples:
        for i in range(num_views):
            for j in range(num_spatial_crops):
                sample['spatial_crop_id'] = j if num_spatial_crops > 1 else 1
                sample['view_id'] = i
                yield sample


def build_dataset_inference(input_files, config, video_transform, imu_transform, audio_transform, is_train=False):
    modalities_string = config.DATA.modalities
    target_clip_size = 0 # not used for now config.DATA.target_clip_duration
    metadata_file = config.DATA.metadata_file
    dataset_size = config.INFERENCE.data_size
    biased_sampling = config.DATA.biased_sample_clip
    num_views = config.INFERENCE.num_views
    num_spatial_crops = config.INFERENCE.num_spatial_crops

    # decode_all = config.DATA.decode_all
    if metadata_file:
        if num_views :
            metadata = {}
            metadata_groups = pd.read_csv(metadata_file).groupby('video_id')
            for video_id, group in metadata_groups:
                metadata[video_id] = group.set_index('view_id').to_dict('index')
        else:
            metadata = pd.read_csv(metadata_file).set_index(
                'video_id').to_dict('index')
    else:
        metadata = {"constant_duration_sec": config.DATA.input_clip_duration}

    dataset = (
        wds.WebDataset(input_files, repeat=False,
                       handler=warn_and_continue, nodesplitter=wds.split_by_node)
        .select(lambda sample: filter_videos(sample, metadata, target_clip_size, config.DATA.filter_criteria, metadata_file=metadata_file, multiview=num_views > 1))
        .shuffle(dataset_size)
        .decode(handler=warn_and_continue)
        .compose(lambda sample: augment_views(sample, num_views, num_spatial_crops))
        .map(lambda sample: custom_video_imu_decoder(sample, metadata, config, random_sampling=False, biased_sampling=biased_sampling, is_train=is_train), handler=warn_and_continue)
        .map_dict(mp4=lambda video: video, spec=audio_transform, cls=lambda x: x,  handler=warn_and_continue)
        .map(video_transform)
        .batched(config.INFERENCE.batch_size_per_gpu, lambda batch: collate_func(batch, modalities_string, extra_keys=['cls', '__key__'], single_class=config.MODEL.single_class))
        )
    return dataset


def build_dataset_finetune(input_files, config, video_transform, imu_transform, audio_transform, is_train=True):
    modalities_string = config.DATA.modalities
    metadata_file = config.DATA.metadata_file
    dataset_size = config.TRAIN.data_size if is_train else config.VALIDATION.data_size
    random_sampling = config.DATA.random_sample_clip and is_train
    biased_sampling = config.DATA.biased_sample_clip
    if metadata_file:
        metadata = pd.read_csv(metadata_file).set_index(
            'video_id').to_dict('index')
    else:
        metadata = {"constant_duration_sec": config.DATA.input_clip_duration}
    buffer_size = (config.TRAIN.batch_size_per_gpu * 100) if is_train else 0
    epochs_number_of_iterations = int(np.ceil(config.TRAIN.data_size / config.TRAIN.effective_batch_size) // max(config.TRAIN.num_workers, 1)) if is_train else int(
        np.ceil(config.VALIDATION.data_size / config.VALIDATION.effective_batch_size) // config.VALIDATION.num_workers)
    dataset = (
        wds.WebDataset(input_files, repeat=False,
                       handler=warn_and_continue, nodesplitter=wds.split_by_node)
        .select(lambda sample: filter_videos(sample, metadata, 0, config.DATA.filter_criteria, multiview=False))
        .shuffle(buffer_size)
        .decode(handler=warn_and_continue)
        # video decoding part
        .map(lambda sample: custom_video_imu_decoder(sample, metadata, config, random_sampling=random_sampling, biased_sampling = biased_sampling, is_train=is_train), handler=warn_and_continue)
        .map_dict(mp4=lambda video: video_transform(video), spec=audio_transform, cls=lambda x: x,  handler=warn_and_continue)
        .batched(config.TRAIN.batch_size_per_gpu if is_train else config.VALIDATION.batch_size_per_gpu, lambda batch: collate_func(batch, modalities_string, extra_keys=['cls'], single_class=config.MODEL.single_class))
        # .with_length(dataset_size)

        ).with_epoch(epochs_number_of_iterations)
    return dataset


def build_dataset_train(input_files, config, video_transform, imu_transform, audio_transform, is_train=True):
    modalities_string = config.DATA.modalities
    metadata_file = config.DATA.metadata_file
    dataset_size = config.TRAIN.data_size if is_train else config.VALIDATION.data_size
    if metadata_file:
        metadata = pd.read_csv(metadata_file).set_index(
            'video_id').to_dict('index')
    else:
        metadata = {"constant_duration_sec": config.DATA.input_clip_duration}
    buffer_size = (config.TRAIN.batch_size_per_gpu *
                   10) if is_train else (config.TRAIN.batch_size_per_gpu * 4)
    epochs_number_of_iterations = int(np.ceil(config.TRAIN.data_size / config.TRAIN.effective_batch_size) // config.TRAIN.num_workers) if is_train else int(
        np.ceil(config.VALIDATION.data_size / config.VALIDATION.effective_batch_size) // config.VALIDATION.num_workers)
    dataset = (wds.WebDataset(input_files, resampled=True,
                              handler=warn_and_continue, nodesplitter=wds.split_by_node)
        .shuffle(buffer_size)
        .decode(handler=warn_and_continue)  # to decode npy files
        .map(lambda sample: custom_video_imu_decoder(sample, metadata, config, random_sampling=config.DATA.random_sample_clip), handler=warn_and_continue)
        #    .map_dict(mp4=lambda video: video_transform(video), npy=imu_transform, handler=warn_and_continue)
        .map_dict(mp4=lambda video: video_transform(video), spec=audio_transform, npy=imu_transform, handler=warn_and_continue)
        .batched(config.TRAIN.batch_size_per_gpu if is_train else config.VALIDATION.batch_size_per_gpu, lambda batch: collate_func(batch, modalities_string, extra_keys=['__key__']), partial=False)
        #    .with_length(dataset_size)
        ).with_epoch(epochs_number_of_iterations)
    return dataset


def custom_video_imu_decoder(sample, metadata, config, random_sampling=True, aligned_sampling=True, biased_sampling = False, is_train = True, decode_all = False):
    def sample_start_end( metadata_clip, duration, target_clip_size, random_sampling, biased_sampling, decode_all):
        if biased_sampling:
            #samples the points from a specific region
            if not math.isnan(metadata_clip['sample_clip']):
                sample_window = np.random.random() * target_clip_size if random_sampling else target_clip_size / 2.0
                sampled_start = (metadata_clip['sample_clip'] - target_clip_size) + sample_window
                sampled_start =  min(duration - target_clip_size, sampled_start)
                start_time = max(0, sampled_start)
            else:
                # print('no sample clip metadata')
                start_time = max(0, np.random.random() * 
                             (duration - target_clip_size)) if random_sampling else max(0, (duration - target_clip_size) / 2.0)

            end_time = min(start_time + target_clip_size, duration)
        elif random_sampling:
            # sampling the start and end points randomly
            # should be used in training
            start_time = max(0, np.random.random() *
                             (duration - target_clip_size))
            end_time = min(start_time + target_clip_size, duration)
        elif decode_all:
            start_time = 0
            end_time = None 
        else:
            # print('Performing central crop')
            # or doing the central crop, should be used in val & inference
            start_time = max(0, (duration - target_clip_size) / 2.0)
            end_time = min(start_time + target_clip_size, duration)
        return start_time, end_time
    """Decode video using the torchvideo library. And updates the dictionary

    :param sample: dictionary with all the inputs (keys: [__key__, mp4, npy])
    :param metadadata: a dictionary containing the metadata about the input clips (keys : [duration_sec])
    :random_sampling: should be True if want to randomly crop a temporal clip
    :aligned_sample: have imu and video centered at the same point 
    """
    modalities_string = config.DATA.modalities
    key = sample['__key__']
    # json in class for militclass
    cls = sample['cls'] if 'cls' in sample else sample['json'] if 'json' in sample else torch.tensor([
    ])
    output_sample = {'__key__': key, 'spec': torch.tensor(
        []), 'mp4': torch.tensor([]), 'npy': torch.tensor([]), 'cls': cls}
    if 'spatial_crop_id' in sample:
        output_sample['spatial_crop_id'] = sample['spatial_crop_id']

    if 'video' in modalities_string or 'audio' in modalities_string:
        metadata_clip = metadata[key][sample['view_id']] if config.INFERENCE.enabled else metadata[key]
        # constant clip duration or need to query by clip name
        duration = metadata['constant_duration_sec'] if 'constant_duration_sec' in metadata else metadata_clip['duration_sec']
        video_target_clip_size = config.DATA.video.target_clip_duration if 'video' in modalities_string else 0
        audio_target_clip_size = config.DATA.audio.target_clip_duration if 'audio' in modalities_string else 0
        data = sample['mp4']
        #audio is usually longer than video so we need to decoder more frames
        start_time, end_time = sample_start_end( metadata_clip, duration, max(video_target_clip_size, audio_target_clip_size), random_sampling, biased_sampling, decode_all)
        with tempfile.TemporaryDirectory() as dirname:
            fname = os.path.join(dirname, f"file.mp4")
            with open(fname, "wb") as stream:
                stream.write(data)
            stream_data = torchvision.io.read_video(
                fname, start_pts=start_time, end_pts=end_time, pts_unit="sec") #returns vframes (Tensor[T, H, W, C]
            if 'video' in modalities_string:
                video_input_fps, video_output_fps = config.DATA.video.input_fps, config.DATA.video.output_fps
                target_num_frames = video_output_fps * video_target_clip_size
                # if config.INFERENCE.enabled and not metadata_clip['has_video']:
                if  not metadata_clip['has_video']:
                    output_sample['video_attention'] = 0
                    video_data = torch.zeros((target_num_frames, 300, 300, 3), dtype=torch.uint8)
                    output_sample['mp4'] = video_data
                else:
                    output_sample['video_attention'] = 1
                    video_data = stream_data[0]  # only video stream
                    # in epic the fps is not fixed so we will double check
                    video_input_fps = stream_data[2]['video_fps']
                    #as the sampled audio can be longer, we need to slice the middle of the video
                    current_num_frames = video_data.shape[0]
                    sample_video_start = max(0, int((current_num_frames - video_input_fps * video_target_clip_size) // 2))
                    video_data = video_data[sample_video_start : sample_video_start + int(video_input_fps * video_target_clip_size), ...]
                    # if too short - need to resample
                    if duration < video_target_clip_size:
                        real_duration = video_data.shape[0] / video_input_fps
                        video_output_fps = np.ceil(target_num_frames / real_duration)
                        # print(f'need to change the fps for too short! output fps {output_fps} and duration {duration}')
                    new_fps_idx = resample_video_idx(
                        target_num_frames, video_input_fps, video_output_fps)
                    # make sure there are no extra frames
                    output_sample['mp4'] = video_data[new_fps_idx,
                                                    ...][:target_num_frames, ...]
                assert output_sample['mp4'].size(
                    0) == target_num_frames, f'Expected temporal size of {target_num_frames} but got {output_sample["mp4"].size(0)}'
            if 'audio' in modalities_string:
                num_mel_bins = config.DATA.audio.num_mel_bins
                output_spectogram_time = config.DATA.audio.output_spectogram_time
                prob_zero_audio = config.DATA.audio.add_missing
                assert prob_zero_audio >= 0 and prob_zero_audio <= 1, f'prob_zero_audio should be between 0 and 1 but got {prob_zero_audio}'
                if is_train and np.random.choice([0, 1], 1, p =[prob_zero_audio, 1 - prob_zero_audio])[0] == 0:
                    audio_data = torch.zeros((1, output_spectogram_time, num_mel_bins))
                    output_sample['audio_attention'] = 0
                elif config.DATA.audio.pad_missing and 'audio_fps' not in stream_data[2]:
                    audio_data = torch.zeros((1, output_spectogram_time, num_mel_bins))
                    output_sample['audio_attention'] = 0
                # elif config.INFERENCE.enabled and not metadata_clip['has_audio']:
                elif not metadata_clip['has_audio']:
                    output_sample['audio_attention'] = 0
                    audio_data = torch.zeros((1, output_spectogram_time, num_mel_bins))
                else:
                    frame_shift = config.DATA.audio.frame_shift
                    spectogram_padding = config.DATA.audio.spectogram_padding
                    audio_fps = stream_data[2]['audio_fps']
                    audio_data = stream_data[1]
                    audio_data = audio_data[:, :audio_fps * audio_target_clip_size]
                    if audio_data.shape[0] > 1:
                        audio_data = audio_data[0].unsqueeze(0)
                        # kaldi returns (208, 128) - T x F
                    audio_data = torchaudio.compliance.kaldi.fbank(audio_data, htk_compat=True, sample_frequency=audio_fps, use_energy=False,
                                                                window_type='hanning', num_mel_bins=num_mel_bins, dither=0.0, frame_shift=frame_shift)
                    current_length = audio_data.shape[0]
                    to_pad_missing = output_spectogram_time - current_length
                    audio_data = audio_data.unsqueeze(0)
                    if to_pad_missing > 0:
                        if spectogram_padding == 'zero':
                            audio_data = torch.nn.functional.pad(
                                audio_data, (0, 0, to_pad_missing, 0))
                        elif spectogram_padding == 'repeat':
                            offset = np.random.randint(
                                    0, output_spectogram_time ) if random_sampling else 0
                            num_repeats = torch.div(output_spectogram_time + current_length + offset - 1,
                                                            current_length, rounding_mode='floor')
                            repeated_spectogram = audio_data.repeat([1, num_repeats, 1])
                            audio_data = repeated_spectogram[:, offset:offset + output_spectogram_time, :]
                        else:
                            raise ValueError(
                                f'Unknown spectogram padding {spectogram_padding}')
                        # else:
                            #     rate = audio_data.shape[0] / output_spectogram_time
                            #     stretch = torchaudio.transforms.TimeStretch(n_freq  = num_mel_bins)
                            #     audio_data = stretch(audio_data, rate)

                    output_sample['audio_attention'] = 1
                    # print(audio_data.shape)
                output_sample['spec'] = audio_data.transpose(1,2)  # final output : (1, 128, 208)

    if 'imu' in modalities_string:
        imu_target_clip_size = config.DATA.imu.target_clip_size
        # sampling the imu
        if not aligned_sampling:
            # generating new time for the second modality
            start_time = np.random.random() * (duration - imu_target_clip_size)
            end_time = start_time + imu_target_clip_size
        data = sample['npy']
        duration_imu = data.shape[0]
        start_index = int(round(start_time * duration_imu / duration))
        end_index = int(round(end_time * duration_imu / duration))
        output_sample['npy'] = data[start_index: end_index + 1, ...]

    return output_sample


def resample_video_idx(num_frames, original_fps, new_fps):
    # https://github.com/HumamAlwassel/TSP/blob/e0b3f6ef41ac40dd8b33445a769ae769c9f6a1d5/train/untrimmed_video_dataset.py#L135
    # Credit: resampling the video with original_fps to match the new_fps

    step = float(original_fps) / new_fps
    if step.is_integer():
        # optimization: if step is integer, don't need to perform
        # advanced indexing
        step = int(step)
        return slice(None, None, step)
    idxs = torch.arange(num_frames, dtype=torch.float32) * step
    idxs = idxs.floor().to(torch.int64)
    return idxs
