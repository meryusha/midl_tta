import enum
from templates import Backbone
import torchaudio
import torch
import torch.nn as nn
from functools import partial
from torchvision.models.video import r2plus1d_18
from torchvision.models.video import r3d_18
from torchvision.models.video.resnet import VideoResNet, R2Plus1dStem, BasicBlock
from torchvision.models import vgg16
from transformer_extra.video_vit import  Attention, Block
from preprocessors import PatchEmbed_3D


class VisualBackboneVariants(enum.Enum):
    R2PLUS1D_34 = 'r2plus1d_34'
    R2PLUS1D_18 = 'r2plus1d_18'
    R3D_18 = 'r3d_18'
    VIT_LARGE = 'vit_large_patch16'
    VIT_BASE = 'vit_base_patch16'
    VIT_HUGE = 'vit_huge_patch32'
    def __str__(self):
        return str(self.value)


class IMUBackboneVariants(enum.Enum):
    VGG_16 = 'vgg_16'
    VIT_LARGE = 'vit_large_patch16'
    VIT_BASE = 'vit_base_patch16'
    VIT_HUGE = 'vit_huge_patch32'

    def __str__(self):
        return str(self.value)


class AudioBackboneVariants(enum.Enum):
    VGG_16 = 'vgg_16'
    VIT_LARGE = 'vit_large_patch16'
    VIT_BASE = 'vit_base_patch16'
    VIT_HUGE = 'vit_huge_patch32'

    def __str__(self):
        return str(self.value)


class MultimodalBackboneVariants(enum.Enum):
    VIT_LARGE = 'vit_large_patch16'
    VIT_BASE = 'vit_base_patch16'
    VIT_HUGE = 'vit_huge_patch32'
    def __str__(self):
        return str(self.value)


def build_visual_backbone(variant: VisualBackboneVariants, pretrained: bool = False, **kwargs):
    if variant == VisualBackboneVariants.R2PLUS1D_34:
        return R2Plus1D(depth=34, pretrained=pretrained, **kwargs)
    elif variant == VisualBackboneVariants.R2PLUS1D_18:
        return R2Plus1D(depth=18, pretrained=pretrained, **kwargs)
    elif variant == VisualBackboneVariants.R3D_18:
        return R3D(pretrained=pretrained, **kwargs)
    elif variant in (VisualBackboneVariants.VIT_LARGE, VisualBackboneVariants.VIT_BASE, VisualBackboneVariants.VIT_HUGE):
        return VisionTransformer( **kwargs)
        # return VisionTransformer(pretrained=pretrained, **kwargs)
    else:
        raise ValueError(f'Unknown value: {variant}')


def build_imu_backbone(variant: IMUBackboneVariants, pretrained: bool = False, **kwargs):
    if variant == IMUBackboneVariants.VGG_16:
        return VGG(pretrained=pretrained, channels=6,  **kwargs)
    else:
        raise ValueError(f'Unknown value: {variant}')


def build_audio_backbone(variant: AudioBackboneVariants, pretrained: bool = False, **kwargs):
    if variant == AudioBackboneVariants.VGG_16:
        return VGG(pretrained=pretrained, channels=3,  **kwargs)
    elif variant in (AudioBackboneVariants.VIT_LARGE, AudioBackboneVariants.VIT_BASE, AudioBackboneVariants.VIT_HUGE):
        return VisionTransformer( **kwargs)
    else:
        raise ValueError(f'Unknown value: {variant}')


def build_multimodal_backbone(variant: MultimodalBackboneVariants, pretrained: bool = False, **kwargs):
    return VisionTransformer( **kwargs)


class R2Plus1D(Backbone):
    _R2PLUS1D_34_MODEL_URL = "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip8_ft_kinetics_from_ig65m-0aa0550b.pth"

    def __init__(self, depth: int, pretrained: bool = False,  **kwargs):
        assert depth == 18 or depth == 34, 'Invalid depth value.'
        super().__init__()

        if depth == 18:
            self._net = r2plus1d_18(pretrained=pretrained, **kwargs)
        elif depth == 34:
            self._net = VideoResNet(
                block=BasicBlock,
                conv_makers=[Conv2Plus1D] * 4,
                layers=[3, 4, 6, 3],
                stem=R2Plus1dStem,
                **kwargs,
            )

            # We need exact Caffe2 momentum for BatchNorm scaling
            for m in self._net.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eps = 1e-3
                    m.momentum = 0.9

            if pretrained:
                state_dict = torch.hub.load_state_dict_from_url(
                    R2Plus1D._R2PLUS1D_34_MODEL_URL)
                self._net.load_state_dict(state_dict)

        # remove the FC layer
        self.feature_size = self._net.fc.in_features
        self._net.fc = nn.Sequential()

    def forward(self, input: torch.tensor, **kwargs) -> torch.tensor:
        return self._net(input)


class Conv2Plus1D(nn.Sequential):
    def __init__(self, in_planes, out_planes, midplanes, stride=1, padding=1):

        midplanes = (in_planes * out_planes * 3 * 3 * 3) // (
            in_planes * 3 * 3 + 3 * out_planes
        )
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(
                in_planes,
                midplanes,
                kernel_size=(1, 3, 3),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=False,
            ),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                midplanes,
                out_planes,
                kernel_size=(3, 1, 1),
                stride=(stride, 1, 1),
                padding=(padding, 0, 0),
                bias=False,
            ),
        )

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


class R3D(Backbone):
    def __init__(self, pretrained: bool = False, **kwargs):
        super().__init__()
        self._net = r3d_18(pretrained=pretrained, **kwargs)
        # remove the FC layer
        self.feature_size = self._net.fc.in_features
        self._net.fc = nn.Sequential()

    def forward(self, input: torch.tensor, **kwargs) -> torch.tensor:
        return self._net(input)


class VGG(Backbone):
    def __init__(self, pretrained: bool = False, channels: int = 6,  feature_size=512, hidden_size=2048, n_fft=127, hop_length=1, **kwargs):
        super().__init__()
        self._spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, hop_length=hop_length)
        self._net = vgg16(pretrained=pretrained, **kwargs)
        # accept input with 6 channels
        self._net.features[0] = nn.Conv2d(
            channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # remove the last ReLU+Dropout+FC layers and reduce the feature size down to 512
        self.feature_size = feature_size
        # in_features = self._net.classifier[0].in_features
        self._net.avgpool = nn.Sequential()
        self._net.classifier = nn.Sequential(
            nn.Linear(12288, hidden_size),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, self.feature_size),
        )

    def forward(self, input: torch.tensor, **kwargs) -> torch.tensor:
        return self._net(self._spectrogram(input))


class VisionTransformer(Backbone):
    def __init__(self,
                 embed_dim=1024,
                 num_heads=16,
                 mlp_ratio=4.0,
                 no_qkv_bias=False,
                 norm_layer=nn.LayerNorm,
                 depth=24,
                 drop_path_rate=0.0,
                 dropout=0.5,
                 cls_embed = False,
                 pretrained=False,
                 **kwargs
                 ):
        super().__init__()       
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.feature_size = embed_dim
        self.norm = norm_layer(embed_dim)
        self.cls_embed = cls_embed
        # --------------------------------------------------------------------------

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, reshape_dims=None):
        requires_t_shape = (
            len(self.blocks) > 0  # support empty decoder
            and hasattr(self.blocks[0].attn, "requires_t_shape")
            and self.blocks[0].attn.requires_t_shape
        )
       
        if requires_t_shape: # we do not really pass reshape_dims now, so we expect this is False
            N, T, L, C = reshape_dims
            x = x.view([N, T, L, C])

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        if requires_t_shape:
            x = x.view([N, T * L, C])

        # classifier
        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :].mean(dim=1)  # global pool
        else:
            x = x[:, :, :].mean(dim=1)
        x = self.norm(x)
        # x = self.fc_norm(x)
        x = self.dropout(x)
        # x = self.head(x)

        return x
    
    
# class VisionTransformer(Backbone):
#     """Vision Transformer with support for global average pooling"""

#     def __init__(
#         self,
#         num_frames = 16,
#         t_patch_size = 4,
#         img_size=224,
#         patch_size=16,
#         in_chans=3,
#         # num_classes=400,
#         embed_dim=768,
#         depth=12,
#         num_heads=12,
#         mlp_ratio=4.0,
#         no_qkv_bias=False,
#         qk_scale=None,
#         drop_rate=0.0,
#         attn_drop_rate=0.0,
#         drop_path_rate=0.0,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         dropout=0.5,
#         sep_pos_embed=False,
#         cls_embed=False,
#         pretrained = False,
#         **kwargs,
#     ):
#         super().__init__()
#         print(locals())

#         self.sep_pos_embed = sep_pos_embed
#         # --------------------------------------------------------------------------
#         # MAE encoder specifics
#         self.patch_embed = PatchEmbed_3D(
#             img_size, patch_size, in_chans, embed_dim, num_frames, t_patch_size
#         )
#         num_patches = self.patch_embed.num_patches
#         input_size = self.patch_embed.input_size
#         self.input_size = input_size
#         self.cls_embed = cls_embed
#         self.feature_size = embed_dim # for our template
#         if self.cls_embed:
#             self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

#         if sep_pos_embed:
#             self.pos_embed_spatial = nn.Parameter(
#                 torch.zeros(1, input_size[1] * input_size[2], embed_dim)
#             )
#             self.pos_embed_temporal = nn.Parameter(
#                 torch.zeros(1, input_size[0], embed_dim)
#             )
#             if self.cls_embed:
#                 self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         else:
#             if self.cls_embed:
#                 _num_patches = num_patches + 1
#             else:
#                 _num_patches = num_patches

#             self.pos_embed = nn.Parameter(
#                 torch.zeros(1, _num_patches, embed_dim), requires_grad=True
#             )  # fixed or not?

#         dpr = [
#             x.item() for x in torch.linspace(0, drop_path_rate, depth)
#         ]  # stochastic depth decay rule

#         self.blocks = nn.ModuleList(
#             [
#                 Block(
#                     embed_dim,
#                     num_heads,
#                     mlp_ratio,
#                     qkv_bias=not no_qkv_bias,
#                     qk_scale=None,
#                     norm_layer=norm_layer,
#                     drop_path=dpr[i],
#                     attn_func=partial(
#                         Attention,
#                         input_size=self.patch_embed.input_size,
#                     ),
#                 )
#                 for i in range(depth)
#             ]
#         )
#         self.norm = norm_layer(embed_dim)
#         # --------------------------------------------------------------------------

#         self.dropout = nn.Dropout(dropout)
#         # self.head = nn.Linear(embed_dim, num_classes)

#         # torch.nn.init.normal_(self.head.weight, std=0.02)

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {
#             "cls_token",
#             "pos_embed",
#             "pos_embed_spatial",
#             "pos_embed_temporal",
#             "pos_embed_class",
#         }

#     def forward(self, x):
#         # embed patches
#         x = self.patch_embed(x)
#         N, T, L, C = x.shape  # T: temporal; L: spatial

#         x = x.view([N, T * L, C])

#         # append cls token
#         if self.cls_embed:
#             cls_token = self.cls_token
#             cls_tokens = cls_token.expand(x.shape[0], -1, -1)
#             x = torch.cat((cls_tokens, x), dim=1)

#         if self.sep_pos_embed:
#             pos_embed = self.pos_embed_spatial.repeat(
#                 1, self.input_size[0], 1
#             ) + torch.repeat_interleave(
#                 self.pos_embed_temporal,
#                 self.input_size[1] * self.input_size[2],
#                 dim=1,
#             )
#             if self.cls_embed:
#                 pos_embed = torch.cat(
#                     [
#                         self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
#                         pos_embed,
#                     ],
#                     1,
#                 )
#         else:
#             pos_embed = self.pos_embed[:, :, :]
#         x = x + pos_embed

#         # reshape to [N, T, L, C] or [N, T*L, C]
#         requires_t_shape = (
#             len(self.blocks) > 0  # support empty decoder
#             and hasattr(self.blocks[0].attn, "requires_t_shape")
#             and self.blocks[0].attn.requires_t_shape
#         )
#         if requires_t_shape:
#             x = x.view([N, T, L, C])

#         # apply Transformer blocks
#         for blk in self.blocks:
#             x = blk(x)

#         if requires_t_shape:
#             x = x.view([N, T * L, C])

#         # classifier
#         x = x[:, 1:, :].mean(dim=1)  # global pool
#         x = self.norm(x)
#         # x = self.fc_norm(x)
#         x = self.dropout(x)
#         # x = self.head(x)

#         return x