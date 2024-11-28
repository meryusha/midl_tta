import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from templates import MultimodalModel, Modality
from backbones import VisualBackboneVariants, IMUBackboneVariants, build_imu_backbone, build_visual_backbone
from heads import HeadVariants, build_head
from torchsummary import summary
import torch

modalities = [Modality('video'), Modality('imu')]
backbones = {modalities[0]: build_visual_backbone(VisualBackboneVariants('r2plus1d_34'), pretrained=True),
             modalities[1]: build_imu_backbone(IMUBackboneVariants('vgg_16'), feature_size=240, hidden_size=1024, n_fft=127, hop_length=1)}
head = build_head(variant=HeadVariants('one_head_per_modality'),
                  modalities=modalities,
                  feature_sizes={m: b.feature_size for m, b in backbones.items()},
                  logits_size=128)

model = MultimodalModel(modalities=modalities, backbones=backbones, head=head)

def mock_data(num_examples):
    for _ in range(num_examples):
        inputs = {modalities[0]: torch.rand(1, 3, 32, 32, 32),
                  modalities[1]: torch.rand(1, 6, 198*2)}
        yield inputs

print(model)
for inputs in mock_data(3):
    print('input', set(inputs.keys()))
    logits, features = model(inputs, return_features=True)
    print('output')
    print('logits', [(l.shape, l.dtype) for l in logits])
    print('features', [(k, f.shape, f.dtype) for k, f in features.items()])
summary(backbones[modalities[1]].to('cuda:0'), (6, 198*2))

