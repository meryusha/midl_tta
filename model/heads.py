import enum
from typing import List, Dict
from templates import Head, Modality

import torch
import torch.nn as nn


class HeadVariants(enum.Enum):
    HEAD_FOR_CONCAT_MODALITIES = 'head_for_concat_modalities'
    ONE_HEAD_PER_MODALITY = 'one_head_per_modality'
    TWO_HEADS_PER_MODALITY = 'two_heads_per_modality'
    # NO_HEAD = 'no_head'

    def __str__(self):
        return str(self.value)


def build_head(variant: HeadVariants, modalities: List[Modality], feature_sizes: Dict[Modality, int], logits_size: int, **kwargs):
    if variant == HeadVariants.HEAD_FOR_CONCAT_MODALITIES:
        return HEAD_FOR_CONCAT_MODALITIES(modalities, feature_sizes, logits_size, **kwargs)
    elif variant == HeadVariants.ONE_HEAD_PER_MODALITY:
        return ONE_HEAD_PER_MODALITY(modalities, feature_sizes, logits_size, **kwargs)
    elif variant == HeadVariants.TWO_HEADS_PER_MODALITY:
        return TWO_HEADS_PER_MODALITY(modalities, feature_sizes, logits_size, **kwargs)
    # elif variant == HeadVariants.NO_HEAD:
    #     return NO_HEAD(modalities, feature_sizes, logits_size, **kwargs)
    else:
        raise ValueError(f'Unknown value: {variant}')


class HEAD_FOR_CONCAT_MODALITIES(Head):
    def __init__(self, modalities: List[Modality], feature_sizes: Dict[Modality, int], logits_size: int, **kwargs):
        super().__init__(modalities, feature_sizes, logits_size, **kwargs)
        self._net = nn.Linear(sum(feature_sizes.values()), logits_size)

    def forward(self, features: Dict[Modality, torch.tensor], **kwargs) -> List[torch.tensor]:
        concate_features = torch.cat([features[modality] for modality in self.modalities], dim=-1)
        return [self._net(concate_features)]


class ONE_HEAD_PER_MODALITY(Head):
    def __init__(self, modalities: List[Modality], feature_sizes: Dict[Modality, int], logits_size: int, **kwargs):
        super().__init__(modalities, feature_sizes, logits_size, **kwargs)
        fcs = {str(m): nn.Linear(s, logits_size) for m, s in feature_sizes.items()}
        self._net = nn.ModuleDict(fcs)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        #take from VIT
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.02)

    def forward(self, features: Dict[Modality, torch.tensor], **kwargs) -> List[torch.tensor]:
        logits = []
        for modality, feature in features.items():
            logits.append(self._net[str(modality)](feature))

        return logits


class TWO_HEADS_PER_MODALITY(Head):
    def __init__(self, modalities: List[Modality], feature_sizes: Dict[Modality, int], logits_size: Dict[str, int], **kwargs):
        super().__init__(modalities, feature_sizes, logits_size, **kwargs)
        print(feature_sizes,logits_size)
        fcs = {str(m) + '_' + str(class_name) : nn.Linear(s, num_classes) for m, s in feature_sizes.items() for class_name, num_classes in logits_size.items()}
        self._net = nn.ModuleDict(fcs)
        self._logits_size = logits_size
        self.apply(self._init_weights)

    def _init_weights(self, m):
        #take from VIT
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.02)

    def forward(self, features: Dict[Modality, torch.tensor], **kwargs) -> List[torch.tensor]:
        logits = {class_name: [] for class_name in self._logits_size.keys()}
        for modality, feature in features.items():
            for class_name in self._logits_size.keys():
                logits[class_name].append(self._net[str(modality) + '_' + str(class_name)](feature))
        return logits

# class NO_HEAD(Head):
#     def __init__(self, modalities: List[Modality], feature_sizes: Dict[Modality, int], logits_size: int, **kwargs):
#         super().__init__(modalities, feature_sizes, logits_size, **kwargs)
#         self.identity = nn.Identity()


#     def forward(self, features: Dict[Modality, torch.tensor], **kwargs) -> List[torch.tensor]:
#         return self.identity(features)

