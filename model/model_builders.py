from templates import Modality, MAEJoint, MAEBottleneck, DownstreamModelJoint, DownstreamModelNotJoint,  DownstreamModelBottleneck
from backbones import VisualBackboneVariants, IMUBackboneVariants, AudioBackboneVariants, MultimodalBackboneVariants, build_imu_backbone, build_visual_backbone, build_audio_backbone, build_multimodal_backbone
from heads import HeadVariants, build_head
from preprocessors import VideoPreprocessEncoder, VideoPreprocessDecoder, ImagePreprocessEncoder, ImagePreprocessDecoder, VideoPreprocessEncoderDownstream, ImagePreprocessEncoderDownstream
from templates import Modality
from losses import build_loss, LossVariant
import os
import torch
# import numpy as np
from model.pos_embed import interpolate_pos_embed_video, interpolate_pos_embed_audio


def get_modalities_from_string(modalities_string):
    return [Modality(modality_string) for modality_string in modalities_string]


def load_pre_trained_for_finetune(net, checkpoint_to_finetune, config, modalities):

    checkpoint = torch.load(checkpoint_to_finetune, map_location='cpu')
    current_net_state = net.state_dict()
    # checkpoint_dict_filtered = {
    #     k: v for k, v in checkpoint.items() if k in current_net_state}
    # fiter the missing weights
    if config.checkpoint_load_multimodal.mode == 'default':
        checkpoint_dict_filtered = {k.replace('.encoder', '.backbones').replace('model.', ''): v for k, v in checkpoint['state_dict'].items(
        ) if k.replace('.encoder', '.backbones').replace('model.', '') in current_net_state}
        for k in ['preprocess_encoder.video.patch_embed.proj.weight']:
            if k in checkpoint_dict_filtered:
                if checkpoint_dict_filtered[k].shape != current_net_state[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_dict_filtered[k]
        if config.encoder.random_init_bottle:
             del checkpoint_dict_filtered['bottleneck']
             print('deleting the bottleneck learned weights')
        if config.encoder.missing_from_mask_token:
            # print(checkpoint['state_dict'].keys())
            missing_token = torch.nn.functional.interpolate(
                checkpoint['state_dict']['model.preprocess_decoder.audio.mask_token'], size=768, mode='linear', align_corners=False)
            checkpoint_dict_filtered['learn_tokens_missing.audio'] = missing_token
            # print(config.encoder.missing_freeze)
            # if config.encoder.missing_freeze:
            #     print('freezing the masked token')
            #     checkpoint_dict_filtered['learn_tokens_missing.audio'].requires_grad = False
            print('missing token added')
    elif config.checkpoint_load_multimodal.mode == 'separate_to_joint':
        modality_to_load = config.checkpoint_load_multimodal.load_modality
        checkpoint_dict_filtered = {k.replace(f'.encoder.{modality_to_load}', '.backbones').replace('model.', ''): v for k, v in checkpoint['state_dict'].items(
        ) if k.replace(f'.encoder.{modality_to_load}', '.backbones').replace('model.', '') in current_net_state}
    elif config.checkpoint_load_multimodal.mode == 'joint_to_separate':
        checkpoint_dict_filtered = {}
        for modality in modalities:
            checkpoint_dict_filtered.update({k.replace('.encoder', f'.backbones.{modality}').replace('model.', ''): v for k, v in checkpoint['state_dict'].items(
            ) if k.replace('.encoder', f'.backbones.{modality}').replace('model.', '') in current_net_state})
    elif config.checkpoint_load_multimodal.mode == 'inference_with_one_mod':
        modality_to_load = config.checkpoint_load_multimodal.load_modality
        checkpoint_dict_filtered = {k.replace(f'.{modality_to_load}.blocks', '.blocks').replace('model.', ''): v for k, v in checkpoint['state_dict'].items(
        ) if k.replace(f'.{modality_to_load}.blocks', '.blocks').replace('model.', '') in current_net_state}

        checkpoint_dict_filtered.update({k.replace(f'model.head._net.{modality_to_load}', 'head._net.joint'): v for k, v in checkpoint['state_dict'].items(
        ) if k.replace(f'model.head._net.{modality_to_load}', 'head._net.joint') in current_net_state})

        checkpoint_dict_filtered.update({k.replace(f'model.backbones.{modality_to_load}.norm', 'backbones.norm'): v for k, v in checkpoint['state_dict'].items(
        ) if k.replace(f'model.backbones.{modality_to_load}.norm', 'backbones.norm') in current_net_state})
    elif config.checkpoint_load_multimodal.mode == 'imagenet':  # loading from image-net pre-trained
        checkpoint_dict_filtered = {
            'backbones.' + k: v for k, v in checkpoint.items() if 'backbones.' + k in current_net_state}
        # print(checkpoint_dict_filtered.keys())
        start = 0
        for comp in 'qkv':
            print(start)
            checkpoint_dict_filtered.update({('backbones.' + k).replace('qkv', comp): v[start:start + 768] for k, v in checkpoint.items(
            ) if (('backbones.' + k).replace('qkv', comp) in current_net_state and 'qkv' in k)})
            start = start + 768
    # loading from https://github.com/facebookresearch/mae_st pre-trained
    elif config.checkpoint_load_multimodal.mode == 'kinetics_video_mae':
        checkpoint_dict_filtered = {
            'backbones.' + k: v for k, v in checkpoint['model_state'].items() if 'backbones.' + k in current_net_state}
        checkpoint_dict_filtered.update({'preprocess_encoder.video.' + k: v for k,
                                        v in checkpoint['model_state'].items() if 'preprocess_encoder.video.' + k in current_net_state})

    else:
        checkpoint_dict_filtered = {
            k: v for k, v in checkpoint['state_dict'].items() if k in current_net_state}
    print(
        f'The keys of the weights that were NOT loaded {set(current_net_state.keys()) - set(checkpoint_dict_filtered.keys()) }')
    # overwrite entries in the existing state dict & load the new state dict
    current_net_state.update(checkpoint_dict_filtered)
    print(f'Loading the weights from {checkpoint_to_finetune}')
    interpolate_pos_embed_video(net, current_net_state)
    interpolate_pos_embed_audio(net, current_net_state)
    net.load_state_dict(current_net_state, strict=True)
    # print(net)
    return net


def get_model(config):

    if 'MAE' in config.MODEL.name:
        return get_model_mae(config)

    def build_backbones_from_modalities(modalities_string, if_joint, if_pretrained, m_config):
        modalities = get_modalities_from_string(modalities_string)
        if if_joint:
            return modalities, build_multimodal_backbone(
                MultimodalBackboneVariants(m_config.backbone_joint), **m_config)
        backbones = {}

        for i, modality_string in enumerate(modalities_string):
            if modality_string == 'video':
                backbones[modalities[i]] = build_visual_backbone(
                    VisualBackboneVariants(m_config.backbone_vis), pretrained=if_pretrained, **m_config)
            elif modality_string == 'imu':
                backbones[modalities[i]] = build_imu_backbone(
                    IMUBackboneVariants(m_config.backbone_imu), **m_config)
            elif modality_string == 'audio':
                backbones[modalities[i]] = build_audio_backbone(
                    AudioBackboneVariants(m_config.backbone_audio), **m_config)
            else:
                raise Exception(f'Unsupported modality: {modality_string}')
        return modalities, backbones

    def build_preprocessor_from_modalities(modalities_string, p_config, d_config):
        preprocessor = {}
        modalities = get_modalities_from_string(modalities_string)
        for i, modality_string in enumerate(modalities_string):
            if modality_string == 'video':
                preprocessor[modalities[i]
                             ] = VideoPreprocessEncoderDownstream(**p_config.video)
            elif modality_string == 'imu':
                preprocessor[modalities[i]] = ImagePreprocessEncoderDownstream(
                    **p_config.imu)
            elif modality_string == 'audio':
                p_config.audio.img_size = (
                    d_config.audio.num_mel_bins, d_config.audio.output_spectogram_time)
                preprocessor[modalities[i]] = ImagePreprocessEncoderDownstream(
                    **p_config.audio)
            else:
                raise Exception(f'Unsupported modality: {modality_string}')
        return preprocessor

    def build_head_for_backbone(modalities, backbones, logits_size, config):
        if_predict_from_all = config.MODEL.encoder.joint or config.MODEL.encoder.predict_from_all
        if if_predict_from_all:
            return build_head(variant=HeadVariants(config.MODEL.head_variant),
                              modalities=[Modality('joint')],
                              feature_sizes={
                Modality('joint'): config.MODEL.encoder.embed_dim},
                logits_size=logits_size)
        else:
            return build_head(variant=HeadVariants(config.MODEL.head_variant),
                              modalities=modalities,
                              feature_sizes={m: b.feature_size for m,
                                             b in backbones.items()},
                              logits_size=logits_size)

    modalities_string = config.DATA.modalities
    preprocessors, learn_tokens_missing = None, None
    if_pretrained = False
    if_joint = config.MODEL.encoder.joint
    if_shared = config.MODEL.encoder.shared
    if_predict_from_all = if_joint or config.MODEL.encoder.predict_from_all
    if config.MODEL.preprocessor:
        print('Buidling model with preprocessors')
        preprocessors = build_preprocessor_from_modalities(
            modalities_string, config.MODEL.preprocessor, config.DATA)

    if config.TRAIN.checkpoint_to_finetune:
        if_pretrained = True if config.TRAIN.checkpoint_to_finetune == 'kinetics' else False
        logits_size = config.MODEL.num_classes if config.MODEL.single_class else dict(
            config.MODEL.num_classes)
    else:
        logits_size = 128

    modalities, backbones = build_backbones_from_modalities(
        modalities_string, if_joint or if_shared, if_pretrained, config.MODEL.encoder)
    head = build_head_for_backbone(modalities, backbones, logits_size, config)
    if config.MODEL.encoder.missing_modality_token is not None:
        freeze_mm_token = config.MODEL.encoder.missing_freeze
        print('---')
        print(f'Training with the missing modality : {config.MODEL.encoder.missing_modality_token}, is frozen: {freeze_mm_token}' )
        learn_tokens_missing = {}
        modality_token = torch.nn.Parameter(
            torch.zeros(1, 1, config.MODEL.encoder.embed_dim), requires_grad = not freeze_mm_token)
        if freeze_mm_token:
            torch.nn.init.zeros_(modality_token)
            # torch.nn.init.normal_(modality_token, std=.02)
        else:
            # torch.nn.init.zeros_(modality_token)
            torch.nn.init.normal_(modality_token, std=.02)
        missing_modality = config.MODEL.encoder.missing_modality_token
        learn_tokens_missing[missing_modality] = {'token': modality_token,
                                                  'mask_missing_ratio': eval(f'config.DATA.{missing_modality}.mask_missing_ratio'),
                                                  'ratio_probs': eval(f'config.DATA.{missing_modality}.ratio_probs')}
        # print(learn_tokens_missing)
    if 'bottleneck' in config.MODEL.name and config.MODEL.encoder.bottleneck_num:
        bottleneck = torch.nn.Parameter(
            torch.zeros(1, config.MODEL.encoder.bottleneck_num, config.MODEL.encoder.embed_dim))
        torch.nn.init.normal_(bottleneck, std=.02)
        net = DownstreamModelBottleneck(preprocessors=preprocessors, modalities=modalities,
                                        backbones=backbones, head=head, bottleneck=bottleneck, shared=if_shared, predict_from_all=if_predict_from_all, learn_tokens_missing=learn_tokens_missing)
    elif if_joint:
        net = DownstreamModelJoint(preprocessors=preprocessors, modalities=modalities,
                                   backbones=backbones, head=head)
    else:
        net = DownstreamModelNotJoint(preprocessors=preprocessors, modalities=modalities,
                                      backbones=backbones, head=head)
    # print(net)
    if config.TRAIN.checkpoint_to_finetune not in ['random', 'kinetics'] or  config.MODEL.checkpoint_load_multimodal.mode == 'inference_with_one_mod':
        path_pre_train = config.INFERENCE.checkpoint if config.INFERENCE.enabled else config.TRAIN.checkpoint_to_finetune 
        net = load_pre_trained_for_finetune(
            net, path_pre_train, config.MODEL, modalities_string)
    return net



def get_loss_func(config, class_weights_for_loss = None):
    modalities = get_modalities_from_string(config.DATA.modalities)
    if 'MAE' not in config.MODEL.name and (config.MODEL.encoder.joint or config.MODEL.encoder.predict_from_all):
        modalities = [Modality('joint')]
    loss_func = build_loss(variant=LossVariant(
        config.MODEL.loss_variant), modalities=modalities, class_weights_for_loss = class_weights_for_loss, **config.TRAIN.mixup)
    return loss_func
