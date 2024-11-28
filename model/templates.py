import enum
from typing import List, Dict, Union, Tuple
import torch
from torch import nn


class Modality(enum.Enum):
    VIDEO = 'video'
    IMU = 'imu'
    AUDIO = 'audio'
    JOINT = 'joint'

    def __str__(self):
        return str(self.value)


class Backbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, input: torch.tensor, **kwargs) -> torch.tensor:
        pass


class Head(nn.Module):
    def __init__(self, modalities: List[Modality], feature_sizes: Dict[Modality, int], logits_size: int, **kwargs):
        super().__init__()
        self.modalities = modalities
        self.feature_sizes = feature_sizes
        self.logits_size = logits_size

    def forward(self, features: Dict[Modality, torch.tensor], **kwargs) -> List[torch.tensor]:
        pass


class Preprocesssor(nn.Module):
    def __init__(self,  **kwargs):
        super().__init__()

    def forward(self, inputs: torch.tensor, **kwargs) -> torch.tensor:
        pass


class DownstreamMultimodalModel(nn.Module):
    def __init__(self,
                 preprocessors: List[Modality],
                 modalities: List[Modality],
                 head: Head, **kwargs):

        super().__init__()
        self.modalities = modalities
        self.preprocess_encoder = nn.ModuleDict(
            {str(k): v for k, v in preprocessors.items()}) if preprocessors else None
        self.head = head

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "preprocess_encoder.video.cls_token",
            "preprocess_encoder.video.pos_embed",
            "preprocess_encoder.video.pos_embed_spatial",
            "preprocess_encoder.video.pos_embed_temporal",
            "preprocess_encoder.video.pos_embed_class",
        }

    def forward(self,
                inputs: Dict[Modality, torch.tensor],
                return_features: bool = False,
                **kwargs) -> Dict[str, List[torch.tensor]]:
        pass


class DownstreamModelNotJoint(DownstreamMultimodalModel):
    def __init__(self,
                 preprocessors: List[Modality],
                 modalities: List[Modality],
                 backbones: Dict[Modality, Backbone],
                 head: Head):

        super().__init__(modalities=modalities, preprocessors=preprocessors, head=head)
        self.backbones = nn.ModuleDict(
            {str(k): v for k, v in backbones.items()})

    def forward(self,
                inputs: Dict[Modality, torch.tensor],
                return_features: bool = False, **kwargs
                ) -> Dict[str, List[torch.tensor]]:
        assert set(self.modalities).issubset(set(inputs.keys(
        ))), f'Missing or unexpected input modalities. Got {set(inputs.keys())}, but should be {self.modalities}.'
        features = {}
        for modality in self.modalities:
            if self.preprocess_encoder:  # extras should have pathc_embed
                x = self.preprocess_encoder[str(modality)](inputs[modality])
            else:
                x = inputs[modality]
            x = self.backbones[str(modality)](x)
            features[modality] = x

        logits = self.head(features)
        outputs = {'logits': logits}
        if return_features:
            outputs['features'] = features
        return outputs


class DownstreamModelJoint(DownstreamMultimodalModel):
    def __init__(self,
                 preprocessors: List[Modality],
                 modalities: List[Modality],
                 backbones: Dict[Modality, Backbone],
                 head: Head):

        super().__init__(modalities=modalities, preprocessors=preprocessors, head=head)
        self.backbones = backbones  # should be just one

    def forward(self,
                inputs: Dict[Modality, torch.tensor],
                return_features: bool = False, **kwargs
                ) -> Dict[str, List[torch.tensor]]:
        assert set(self.modalities).issubset(set(inputs.keys(
        ))), f'Missing or unexpected input modalities. Got {set(inputs.keys())}, but should be {self.modalities}.'
        features = {}
        all_encoder_tokens = []
        for modality in self.modalities:
            if self.preprocess_encoder:  # extras should have pathc_embed
                x = self.preprocess_encoder[str(modality)](inputs[modality])
            else:
                x = inputs[modality]
            all_encoder_tokens.append(x)
        x = self.backbones(torch.cat(all_encoder_tokens, 1))

        features[Modality('joint')] = x

        logits = self.head(features)
        outputs = {'logits': logits}
        if return_features:
            outputs['features'] = features
        return outputs


class DownstreamModelBottleneck(DownstreamMultimodalModel):
    def __init__(self,
                 preprocessors: List[Modality],
                 modalities: List[Modality],
                 backbones: Dict[Modality, Backbone],
                 head: Head,
                 bottleneck: List,
                 shared: bool,
                 predict_from_all: bool,
                 learn_tokens_missing: Dict[Modality, Dict],
                 ):
        super().__init__(modalities=modalities,
                         preprocessors=preprocessors, head=head,  backbones=backbones)
        self.backbones = backbones if shared else nn.ModuleDict(
            {str(k): v for k, v in backbones.items()})
        self.bottleneck = bottleneck
        self.shared = shared
        self.predict_from_all = predict_from_all
        self.learn_tokens_missing = nn.ParameterDict(
            {str(k): v['token'] for k, v in learn_tokens_missing.items()}) if learn_tokens_missing else None
        self.mask_missing_ratio = {str(k):v['mask_missing_ratio'] for k, v in learn_tokens_missing.items()} if learn_tokens_missing else None 
        self.ratio_probs = {str(k):v['ratio_probs'] for k, v in learn_tokens_missing.items()} if learn_tokens_missing else None

    def forward(self,
                inputs: Dict[Modality, torch.tensor],
                return_features: bool = False, fusion_layer: int = 0, is_train : bool = True,  **kwargs
                ) -> Dict[str, List[torch.tensor]]:
        assert set(self.modalities).issubset(set(inputs.keys(
        ))), f'Missing or unexpected input modalities. Got {set(inputs.keys())}, but should be {self.modalities}.'
        features = {}
        outputs_encoder = {}
        for modality in self.modalities:
            if self.preprocess_encoder:  # extras should have patch_embed
                missing_modality_mask, missing_modality_token, mask_missing_ratio, ratio_probs = None, None, None, None
                if self.learn_tokens_missing and str(modality) in self.learn_tokens_missing.keys():
                    if str(modality) + '_attention' in inputs.keys() :
                        missing_modality_mask = inputs[str(modality) + '_attention']
                    missing_modality_token = self.learn_tokens_missing[str(modality)]
                    if is_train:
                        #only need to mask tokens randomly during training
                        mask_missing_ratio = self.mask_missing_ratio[str(modality)]
                        ratio_probs = self.ratio_probs[str(modality)]
                outputs_encoder[modality] = self.preprocess_encoder[str(
                    modality)](inputs[modality], missing_modality_mask = missing_modality_mask, missing_modality_token = missing_modality_token, mask_missing_ratio = mask_missing_ratio, ratio_probs = ratio_probs)
            else:
                outputs_encoder[modality] = inputs[modality]
        bottlenecks = self.bottleneck.repeat(
            outputs_encoder[modality].shape[0], 1, 1)
        num_blocks = len(self.backbones.blocks) if self.shared else len(self.backbones[str(modality)].blocks)
        i = 0
        while i < num_blocks:    
            bottleneck_accum = []
            for modality in self.modalities:
                tokens = outputs_encoder[modality]
                num_modality_tokens = tokens.shape[1]
                # if self.learn_tokens_missing and str(modality) in self.learn_tokens_missing.keys():
                #     if str(modality) + '_attention' in inputs.keys() :
                #         tokens[inputs[str(modality) + '_attention']] = self.learn_tokens_missing[modality]
                        # # need to index the inputs for which the modality tokens are not padded (exist)
                        # tokens = tokens[inputs[str(modality) + '_attention']]
                current_encoder = self.backbones if self.shared else self.backbones[str(modality)]
                if i < fusion_layer:
                    tokens = current_encoder.blocks[i](tokens)
                else:
                    tokens = current_encoder.blocks[i](
                        torch.cat([tokens, bottlenecks], 1))
                    bottleneck_accum.append(tokens[:, num_modality_tokens:])
                    tokens = tokens[:, :-self.bottleneck.shape[1], :]
                assert tokens.shape[1] == num_modality_tokens
                #-----
                if i == num_blocks - 1:
                    tokens = current_encoder.norm(tokens)
                    tokens = current_encoder.dropout(tokens)
                #----
                outputs_encoder[modality] = tokens 
            if i >= fusion_layer:
                bottlenecks = torch.mean(torch.stack(bottleneck_accum, -1), -1)
            i = i + 1
        all_encoder_tokens = []
        for modality in self.modalities:
            tokens = outputs_encoder[modality]
            if current_encoder.cls_embed:
                # remove cls token
                tokens = tokens[:, 1:, :]
            else:
                tokens = tokens[:, :, :]
            if self.predict_from_all:
                all_encoder_tokens.append(tokens)
            else:
                tokens = tokens.mean(dim=1)
                features[modality] = tokens

        if self.predict_from_all:
            tokens = torch.cat(all_encoder_tokens, 1)
            tokens = tokens.mean(dim=1)
            features[Modality('joint')] = tokens

        ## if self.shared:
        ##     tokens = self.backbones.norm(tokens)
        ##     tokens = self.backbones.dropout(tokens)
        ## else:
        ##     tokens = self.backbones[str(self.modalities[0])].norm(tokens) #takes the norm layer from the first modality
        ##     tokens = self.backbones[str(self.modalities[0])].dropout(tokens)  
        # features[Modality('joint')] = torch.mean(bottlenecks, dim=1)

        logits = self.head(features)
        outputs = {'logits': logits}
        if return_features:
            outputs['features'] = features
        return outputs


class MAEModel(nn.Module):
    def __init__(self,
                 modalities: List[Modality],
                 decoder: Dict[Modality, Backbone],
                 preprocess_encoder: Dict[Modality, Preprocesssor],
                 preprocess_decoder: Dict[Modality, Preprocesssor],
                 ):
        super().__init__()
        self.modalities = modalities
        self.preprocess_encoder = nn.ModuleDict(
            {str(k): v for k, v in preprocess_encoder.items()})  # if preprocess_encoder else None
        self.preprocess_decoder = nn.ModuleDict(
            {str(k): v for k, v in preprocess_decoder.items()}) if preprocess_decoder else None
        self.decoder = nn.ModuleDict(
            {str(k): v for k, v in decoder.items()}) if decoder else None

    def forward(self,
                inputs: Dict[Modality, torch.tensor],
                return_features: bool = False, return_reconstruct: bool = False,
                mask_ratio: dict = {'audio': 0.8, 'video': 0.8}, fusion_layer: int = 0) -> Union[Tuple[List[torch.tensor], Dict[Modality, torch.tensor]],
                                                                          Tuple[List[torch.tensor]]]:
        pass


class MAEJoint(MAEModel):
    def __init__(self,
                 modalities: List[Modality],
                 encoder: Dict[Modality, Backbone],
                 decoder: Dict[Modality, Backbone],
                 preprocess_encoder: Dict[Modality, Preprocesssor],
                 preprocess_decoder: Dict[Modality, Preprocesssor],
                 ):
        super().__init__(modalities=modalities, preprocess_encoder=preprocess_encoder,
                         preprocess_decoder=preprocess_decoder, decoder=decoder)
        self.encoder = encoder  # early fusion - just 1 encoder

    def forward(self,
                inputs: Dict[Modality, torch.tensor],
                return_features: bool = False, return_reconstruct: bool = False,
                mask_ratio: dict = {'audio': 0.8, 'video': 0.8}, fusion_layer: int = 0) -> Union[Tuple[List[torch.tensor], Dict[Modality, torch.tensor]],
                                                                          Tuple[List[torch.tensor]]]:
        assert set(self.modalities).issubset(set(inputs.keys(
        ))), f'Missing or unexpected input modalities. Got {set(inputs.keys())}, but should be {self.modalities}.'
        outputs_encoder_preprocessors = {}
        all_encoder_tokens = []
        for modality in self.modalities:
            outputs_encoder_preprocessors[modality] = self.preprocess_encoder[str(modality)](
                inputs[modality], mask_ratio=mask_ratio[str(modality)])  # returns a dictionary {x, mask, ids_restore}
            tokens = outputs_encoder_preprocessors[modality]['x']
            outputs_encoder_preprocessors[modality]['num_tokens'] = tokens.shape[1]
            all_encoder_tokens.append(tokens)

        outputs_encoder = self.encoder(torch.cat(all_encoder_tokens, 1))
        if self.decoder:
            modality_token_start = 0
            outputs = {}
            outputs_decoder = {}
            outputs_masks = {}
            outputs_patch = {}
            outputs_reconstruct = {}
            for modality in self.modalities:
                modality_token_end = outputs_encoder_preprocessors[
                    modality]['num_tokens'] + modality_token_start
                ids_restore_in_modality = outputs_encoder_preprocessors[modality]['ids_restore']
                preprocess_decoder_tokens = self.preprocess_decoder[str(
                    modality)](outputs_encoder[:, modality_token_start: modality_token_end, :], ids_restore_in_modality)
                outputs_decoder[modality] = self.decoder[str(
                    modality)](preprocess_decoder_tokens)
                outputs_masks[modality] = outputs_encoder_preprocessors[modality]['mask']
                outputs_patch[modality] = outputs_encoder_preprocessors[modality]['patched_input']
                if return_reconstruct:
                    # used for visualization of the reconstructions
                    outputs_reconstruct[modality] = self.preprocess_encoder[str(
                        modality)].unpatchify(outputs_decoder[modality])
                modality_token_start = modality_token_end

            outputs['reconstructed'] = outputs_reconstruct
            outputs['x'] = outputs_decoder
            outputs['mask'] = outputs_masks
            outputs['patched_input'] = outputs_patch

            return outputs
        else:
            return outputs_encoder

        # outputs = self.head(outputs_backbone, outputs_preprocessors[head_keys], )


class MAEBottleneck(MAEModel):
    def __init__(self,
                 modalities: List[Modality],
                 encoder: Dict[Modality, Backbone],
                 decoder: Dict[Modality, Backbone],
                 preprocess_encoder: Dict[Modality, Preprocesssor],
                 preprocess_decoder: Dict[Modality, Preprocesssor],
                 bottleneck: List,
                 shared: bool):
        super().__init__(modalities=modalities, preprocess_encoder=preprocess_encoder,
                         preprocess_decoder=preprocess_decoder, decoder=decoder)
        self.encoder = encoder if shared else nn.ModuleDict(
            {str(k): v for k, v in encoder.items()})
        self.bottleneck = bottleneck
        self.shared = shared

    def forward(self,
                inputs: Dict[Modality, torch.tensor],
                return_features: bool = False, return_reconstruct: bool = False,
                mask_ratio: dict = {'audio': 0.8, 'video': 0.8}, fusion_layer: int = 0) -> Union[Tuple[List[torch.tensor], Dict[Modality, torch.tensor]],
                                                                          Tuple[List[torch.tensor]]]:
        assert set(self.modalities).issubset(set(inputs.keys(
        ))), f'Missing or unexpected input modalities. Got {set(inputs.keys())}, but should be {self.modalities}.'
        outputs_encoder_preprocessors = {}

        for modality in self.modalities:
            outputs_encoder_preprocessors[modality] = self.preprocess_encoder[str(modality)](
                inputs[modality], mask_ratio=mask_ratio[str(modality)])  # returns a dictionary {x, mask, ids_restore}
            tokens = outputs_encoder_preprocessors[modality]['x']
            # outputs_encoder_preprocessors[modality]['num_tokens'] = tokens.shape[1]

        outputs_encoder = {
            modality: outputs_encoder_preprocessors[modality]['x'] for modality in self.modalities}
        bottlenecks = self.bottleneck.repeat(tokens.shape[0], 1, 1)
        num_blocks = len(self.encoder.blocks) if self.shared else len(self.encoder[str(modality)].blocks) 
        i = 0
        while i < num_blocks:
            bottleneck_accum = []
            for modality in self.modalities:
                tokens = outputs_encoder[modality]
                num_modality_tokens = tokens.shape[1]
                current_encoder = self.encoder if self.shared else self.encoder[str(modality)]
                if i < fusion_layer:
                    tokens = current_encoder.blocks[i](tokens)
                else:
                    tokens = current_encoder.blocks[i](torch.cat([tokens, bottlenecks], 1))
                    bottleneck_accum.append(tokens[:, num_modality_tokens:])
                    tokens = tokens[:, :-self.bottleneck.shape[1], :]
                assert tokens.shape[1] == num_modality_tokens
                if i == num_blocks - 1:
                    # remove cls token
                    tokens = tokens[:, 1:, :] if current_encoder.cls_embed else tokens[:, :, :]
                    tokens = current_encoder.norm(tokens)
                    # tokens = current_encoder.norm(tokens) if self.shared else tokens
                outputs_encoder[modality] = tokens
            if i >= fusion_layer:
                bottlenecks = torch.mean(torch.stack(bottleneck_accum, -1), -1)
            i = i + 1

        if self.decoder:
            modality_token_start = 0
            outputs = {}
            outputs_decoder = {}
            outputs_masks = {}
            outputs_patch = {}
            outputs_reconstruct = {}
            for modality in self.modalities:
                # modality_token_end = outputs_encoder_preprocessors[
                #     modality]['num_tokens'] + modality_token_start
                ids_restore_in_modality = outputs_encoder_preprocessors[modality]['ids_restore']
                preprocess_decoder_tokens = self.preprocess_decoder[str(
                    modality)](outputs_encoder[modality], ids_restore_in_modality)
                outputs_decoder[modality] = self.decoder[str(
                    modality)](preprocess_decoder_tokens)
                outputs_masks[modality] = outputs_encoder_preprocessors[modality]['mask']
                outputs_patch[modality] = outputs_encoder_preprocessors[modality]['patched_input']
                if return_reconstruct:
                    # used for visualization of the reconstructions
                    outputs_reconstruct[modality] = self.preprocess_encoder[str(
                        modality)].unpatchify(outputs_decoder[modality])
                # modality_token_start = modality_token_end

            outputs['reconstructed'] = outputs_reconstruct
            outputs['x'] = outputs_decoder
            outputs['mask'] = outputs_masks
            outputs['patched_input'] = outputs_patch

            return outputs
        else:
            return outputs_encoder


class Loss(nn.Module):
    def __init__(self, modalities: List[Modality], **kwargs):
        super().__init__()
        self.modalities = modalities

    def forward(self, inputs: Union[Tuple[List[torch.tensor], Dict[Modality, torch.tensor]],
                                    Tuple[List[torch.tensor]]], **kwargs
                ) -> torch.tensor:
        pass
