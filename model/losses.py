from enum import Enum
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn.functional as F
from templates import Loss, Modality
from typing import List, Dict, Union, Tuple
import numpy as np
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
# import diffdist.functional as diff_dist
# import torch.distributed as dist


class LossVariant(Enum):
    CONTRASTIVE_LOSS = 'contractive_loss'
    CROSS_ENTROPY_LOSS = 'cross_entropy_loss'
    CROSS_ENTROPY_MULTI_HEAD_LOSS = 'cross_entropy_multi_head_loss'
    RECONSTRUCTION = 'recostruction_loss'
    def __str__(self):
        return str(self.value)


def build_loss(variant: LossVariant, **kwargs):
    
    if variant == LossVariant.CONTRASTIVE_LOSS:
        return ContrastiveLoss(**kwargs)
    if variant == LossVariant.CROSS_ENTROPY_LOSS:
        return CrossEntropyLoss(**kwargs, mixup_enabled=kwargs['enabled'])
    if variant == LossVariant.CROSS_ENTROPY_MULTI_HEAD_LOSS:
        return CrossEntropyMultiHeadLoss(**kwargs, mixup_enabled=kwargs['enabled'])
    if variant == LossVariant.RECONSTRUCTION:
        return ReconstructionLoss(**kwargs)


class ContrastiveLoss(Loss):
    # adapted from https://github.com/NVlabs/GroupViT/blob/13b786155a1dfffe4703f40d028c92be58e1178d/models/multi_label_contrastive.py (line 128)
    def __init__(self, modalities: List[Modality], contrast_temperature=0.07, clamp_max=100):
        super().__init__(modalities)
        self.modalities = modalities
        self.logit_scale = torch.nn.Parameter(
            torch.ones([]) * np.log(1 / contrast_temperature))
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.clamp_max = clamp_max

    def forward(self, inputs: Union[Tuple[List[torch.tensor], Dict[Modality, torch.tensor]],
                                    Tuple[List[torch.tensor]]], **kwargs) -> torch.tensor:
        assert len(
            self.modalities) == 2, f'Need to have 2 modalities, got {len(self.modalities)}'
        logits = inputs['logits']
        logits_mod_1 = logits[0]
        logits_mod_2 = logits[1]
        batch_size = logits_mod_1.size(0)
        labels = torch.arange(batch_size, dtype=torch.long,
                              device=logits_mod_1.device)  # + batch_size  # * dist.get_rank()
        logits_mod_1 = F.normalize(logits_mod_1, dim=-1)
        logits_mod_2 = F.normalize(logits_mod_2, dim=-1)
        logits_per_mod_1 = logits_mod_1 @ logits_mod_2.t()  # dist_collect(text_x).t()
        logits_per_mod_2 = logits_mod_2 @ logits_mod_1.t()   # dist_collect(image_x).t()
        logit_scale = torch.clamp(self.logit_scale.exp(), max=self.clamp_max)
        loss_mod_1 = self.cross_entropy(logits_per_mod_1 * logit_scale, labels)
        loss_mod_2 = self.cross_entropy(logits_per_mod_2 * logit_scale, labels)
        loss = 0.5 * (loss_mod_1 + loss_mod_2)
        loss_outputs = {str(LossVariant.CONTRASTIVE_LOSS): loss}
        return loss_outputs


class CrossEntropyLoss(Loss):
    def __init__(self, modalities: List[Modality], mixup_enabled=False, label_smoothing=0.0, **kwargs):
        super().__init__(modalities)
        self.modalities = modalities
        
        if mixup_enabled:
            self.cross_entropy = SoftTargetCrossEntropy()
        else:
            self.cross_entropy = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        self.loss_val = torch.nn.CrossEntropyLoss()
        
    def forward(self, inputs: Union[Tuple[List[torch.tensor], Dict[Modality, torch.tensor]],
                                    Tuple[List[torch.tensor]]], labels, stage='train') -> torch.tensor:
        # assert len(
        #     self.modalities) == 1, f'Need to have 1 modalities, got {len(self.modalities)}'
        loss_outputs = {str(LossVariant.CROSS_ENTROPY_LOSS): 0}
        for i, modality in enumerate(self.modalities):
            logits = inputs['logits'][i]
            if stage == 'train':
                loss = self.cross_entropy(logits, labels)
            else:
                loss = self.loss_val(logits, labels)
            loss_outputs[str(LossVariant.CROSS_ENTROPY_LOSS) + '_' + str(modality)] =  loss
            loss_outputs[str(LossVariant.CROSS_ENTROPY_LOSS)] += loss
        return loss_outputs


class CrossEntropyMultiHeadLoss(Loss):
    def __init__(self, modalities: List[Modality], class_weights_for_loss: Dict,  mixup_enabled=False, label_smoothing=0.0, **kwargs):
        super().__init__(modalities)
        self.modalities = modalities
        self.cross_entropy, self.loss_val = torch.nn.ModuleDict(), torch.nn.ModuleDict()
        if class_weights_for_loss:
            for k, v in class_weights_for_loss.items():
                print(k, v)
                if mixup_enabled:
                    self.cross_entropy[k] = SoftTargetCrossEntropy()
                else:
                    self.cross_entropy[k] = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight = v)
            
                    self.loss_val[k] = torch.nn.CrossEntropyLoss(weight = v)
        # else:

        
    def forward(self, inputs: Union[Tuple[List[torch.tensor], Dict[Modality, torch.tensor]],
                                    Tuple[List[torch.tensor]]], labels, stage='train') -> torch.tensor:
        # assert len(
        #     self.modalities) == 1, f'Need to have 1 modalities, got {len(self.modalities)}'
        loss_outputs = {str(LossVariant.CROSS_ENTROPY_MULTI_HEAD_LOSS): 0}
        for i, modality in enumerate(self.modalities):
            logits_per_prediction = inputs['logits']
            for c, n in logits_per_prediction.items():
                if stage == 'train':
                    loss = self.cross_entropy[c](n[i], labels[c])
                else:
                    loss = self.loss_val[c](n[i], labels[c])
                loss_outputs[str(LossVariant.CROSS_ENTROPY_MULTI_HEAD_LOSS) + '_' + str(modality) + '_' + str(c)] =  loss
                loss_outputs[str(LossVariant.CROSS_ENTROPY_MULTI_HEAD_LOSS)] += loss
        return loss_outputs

    
class ReconstructionLoss(Loss):
    def __init__(self, modalities: List[Modality], norm_pix_loss=False, **kwargs):
        super().__init__(modalities)
        self.modalities = modalities
        self.norm_pix_loss = norm_pix_loss

    def forward(self, inputs: Union[Tuple[List[torch.tensor], Dict[Modality, torch.tensor]],
                                    Tuple[List[torch.tensor]]]) -> torch.tensor:
        decoder_outputs = inputs['x']
        mask_input = inputs['mask']
        patched_input = inputs['patched_input']
        loss_outputs = {}
        loss_outputs[str(LossVariant.RECONSTRUCTION)] = 0
        for modality in self.modalities:
            target = patched_input[modality]
            pred = decoder_outputs[modality]
            mask = mask_input[modality]
            if self.norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.0e-6) ** 0.5
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
            mask = mask.view(loss.shape)

            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
            loss_outputs[str(LossVariant.RECONSTRUCTION) + '_' + str(modality)] = loss
            loss_outputs[str(LossVariant.RECONSTRUCTION)] = loss_outputs[str(LossVariant.RECONSTRUCTION)] + loss
        
        return loss_outputs



