import pytorch_lightning as pl
import torchvision
from common import transforms as T
from common.mixup import MixUp as MixVideo
from model.model_builders import get_model, get_loss_func
from datasets.dataset_builders import build_dataset_train, build_dataset_finetune, build_dataset_inference
import torch, torchaudio
import torch.optim as optim
import webdataset as wds
from model.templates import Modality
import wandb
from pytorch_lightning.utilities import rank_zero_only
import numpy as np
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy
import logging, json
# from torchsummary import summary
from misc.lr_utils import param_groups_lrd, SchedulerMAE, add_weight_decay
import pandas as pd
from pytorchvideo.transforms import create_video_transform
from pytorchvideo.transforms.functional import uniform_crop
# configure logging at the root level of Lightning
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# configure logging on module level, redirect to file
logger = logging.getLogger("pytorch_lightning.core")
logger.addHandler(logging.FileHandler("core.log"))
import os
import os.path as osp
import pickle
import math

def build_inference_video_transforms(config, sample):
    if 'video' not in config.modalities:
        return sample
    v_config = config.video
    #this video transform supports mutli-crop inference
    if 'mp4' in sample:
        video = sample['mp4']
        spatial_crop_id = sample['spatial_crop_id']
        assert spatial_crop_id in (1, 2, 0)
        size = v_config.inference_crop_size
        normalize = T.NormalizeVideo(
                        mean=v_config.augmentations.normalize.mean,
                        std=v_config.augmentations.normalize.std)
        
        transform = torchvision.transforms.Compose([T.ToTensorTransform(), normalize])
        video_T = transform(video)
        sample['mp4'] = uniform_crop(images = video_T, size = size, spatial_idx = spatial_crop_id)
    return sample


def build_video_tranform(config, is_train):
    if 'video' not in config.modalities:
        return torch.nn.Identity()
    v_config = config.video
    if v_config.augmentations.aug_type == 'default':
        aug_paras = None
    else:
        aug_paras = eval(f'v_config.augmentations.{v_config.augmentations.aug_type}')

    if is_train:

        transform = create_video_transform(mode='train',
                    num_samples=v_config.target_clip_duration * v_config.output_fps,
                    crop_size=v_config.augmentations.random_crop_size,
                    video_mean=v_config.augmentations.normalize.mean,
                    video_std=v_config.augmentations.normalize.std,
                    aug_type=v_config.augmentations.aug_type,
                    aug_paras=aug_paras
        )
        transform = torchvision.transforms.Compose([T.ToTensorVideo(), transform])
    else:

        transform = create_video_transform(mode='val',
                    num_samples=v_config.target_clip_duration * v_config.output_fps,
                    crop_size=v_config.augmentations.random_crop_size,
                    video_mean=v_config.augmentations.normalize.mean,
                    video_std=v_config.augmentations.normalize.std,
        )
        transform = torchvision.transforms.Compose([T.ToTensorVideo(), transform])

    return transform


def build_imu_transform(config):
    if 'imu' not in config.modalities:
        return torch.nn.Identity()
    i_config = config.imu
    # imu : num_samples x 6 (gyro_(x, y, z) + accel_(x, y, z))
    # TODO: replace with the config.DATA.clip_duration
    input_imu_size = i_config.input_clip_duration * i_config.frequency
    target_imu_size = i_config.target_clip_duration * i_config.frequency
    transform_train = torchvision.transforms.Compose([
        T.ToTensorIMU(),
        T.ResizeIMU(target_imu_size),
        # T.CropIMU(target_imu_size),
        # normalize
    ])
    return transform_train


def build_audio_transform(config, is_train):
    if 'audio' not in config.modalities:
        return torch.nn.Identity()
    a_config = config.audio
    normalize =  torchvision.transforms.Normalize(
        mean=a_config.augmentations.normalize.mean,
        std=a_config.augmentations.normalize.std)
    if not is_train or a_config.augmentations.aug_type == 'default':
        return normalize
    freqm = torchaudio.transforms.FrequencyMasking(a_config.augmentations.spec_augment.freq_masking)
    timem = torchaudio.transforms.TimeMasking(a_config.augmentations.spec_augment.time_masking)
    transform = torchvision.transforms.Compose([freqm, timem, normalize])
    return transform
        # mean=-4.2677393,
        # std=4.5689974)
    #epic average and std: -6.3594, 4.57392226902641
    #OSCC average and std: -7.6017, 4.871833359673115 
    

def build_denormalize_video(config):
    mean = config.video.augmentations.normalize.mean
    std = config.video.augmentations.normalize.std
    return torchvision.transforms.Compose([T.NormalizeVideo(mean=[0., 0., 0.], 
                                                            std=[1.0 / c for c in std]),
                                        T.NormalizeVideo(mean= [-1 * c for c in mean], 
                                                            std=[1., 1., 1.]),
                                        ])


class generic_lightning_model(pl.LightningModule):
 
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = get_model(config)
        self.loss_func = get_loss_func(config)
        self.save_hyperparameters()
        
        with open(self.config.TRAIN.train_file) as f:
            self._train_files = [line.strip() for line in f.readlines()]
        
        with open(config.VALIDATION.val_file) as f:
            self._val_files = [line.strip() for line in f.readlines()]
            
        self._video_transform_train = build_video_tranform(self.config.DATA, is_train=True)
        self._video_transform_val = build_video_tranform(self.config.DATA, is_train=False)
        self._imu_transform = build_imu_transform(self.config.DATA)
        self._audio_transform_train = build_audio_transform(self.config.DATA, is_train=True)
        self._audio_transform_val = build_audio_transform(self.config.DATA, is_train=False)
        self._denormalize_video = build_denormalize_video(self.config.DATA)

        self.effective_batch_size = None
        self.base_lr = None
        self.num_iterations_per_epoch = None
        
    def wandb_avail(self):
        """ Property to check if wandb is available
            for logging.
        """
        return 'Wandb' in self.logger.name
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass
    
    def get_params():
        pass
    
    def configure_optimizers(self):
        pass
    
    def train_dataloader(self):
        dataset_train = build_dataset_train(
                            self._train_files, self.config, 
                            self._video_transform_train, 
                            self._imu_transform, 
                            self._audio_transform_train)

        return torch.utils.data.DataLoader(
                                    dataset_train, 
                                    batch_size=None, 
                                    pin_memory=True,
                                    num_workers=self.config.TRAIN.num_workers, 
                                    collate_fn=None, 
                                    drop_last=False)
    
    def val_dataloader(self):
        dataset_val = build_dataset_train(
                    self._val_files, self.config, 
                    self._video_transform_val, 
                    self._imu_transform,
                    self._audio_transform_val,
                    is_train=False)
        
        return torch.utils.data.DataLoader(dataset_val, 
                                           batch_size=None, 
                                           pin_memory=True,
                                           num_workers=self.config.TRAIN.num_workers, 
                                           collate_fn=None, 
                                           drop_last=False)

    def test_dataloader(self):
        pass    


class mae_lightning_model(generic_lightning_model):
    
    def __init__(self, config):
        super().__init__(config)
        self.mask_ratio = {}
        for modality in self.config.DATA.modalities:
            self.mask_ratio[modality] = eval(f"self.config.MODEL.preprocessor.{modality}.mask_ratio")
        self.fusion_layer = self.config.MODEL.encoder.fusion_layer

    def forward(self, x):
        return self.model(x)
    
    def log_outputs(self, inputs, outputs, stage):
        if stage == 'train':
            bs = inputs[list(inputs.keys())[0]].shape[0]
            index = torch.randint(0, bs, (1,)).item()
        else:
            index = 0
        for modality in self.model.modalities:
            if str(modality) == 'video':
                self.log_video(inputs[modality][index], outputs[modality][index], stage)
            elif str(modality) == 'imu':
                self.log_imu(inputs[modality][index], outputs[modality][index], stage)
            elif str(modality) == 'audio':
                self.log_audio(inputs[modality][index], outputs[modality][index], stage)
            else:
                logger.error(f"Unknown modality {modality}")
                raise NotImplementedError
        pass
    
    def log_video(self, input_video, output_video, stage):
        input_vis = ((self._denormalize_video(input_video).detach().permute(1, 0, 2, 3).cpu())*255).to(torch.uint8)
        output_vis = ((self._denormalize_video(output_video).detach().permute(1, 0, 2, 3).cpu())*255).to(torch.uint8)
                
        self.logger.log_video(f"visualize/{stage}/video",
                    videos=[wandb.Video(input_vis, fps=self.config.DATA.video.output_fps, format="gif", caption="input"),
                        wandb.Video(output_vis, fps=self.config.DATA.video.output_fps, format="gif", caption="output")], 
                    step=self.global_step)
        
    def log_audio(self, input_audio, output_audio, stage):
        self.logger.log_image(f"visualize/{stage}/audio",
                               images= [wandb.Image(input_audio.detach().cpu(), caption="input"),
                                     wandb.Image(output_audio.detach().cpu(), caption="output")],
                               step=self.global_step)
    
    def log_imu(self, input_imu, output_imu,  step):
        pass
    
    def training_step(self, batch, batch_idx):
        inputs = batch

        # Needed as it is for the scheduler
        self.epoch_progress = batch_idx/self.num_iterations_per_epoch + self.current_epoch
        
        del inputs['__key__'] # Should be removed in the future
        visualize=False
        if batch_idx % self.config.TRAIN.visualize_every == 0 and self.local_rank == 0 and self.wandb_avail():
            visualize = True
        
        inputs = {k: v for k, v in inputs.items()}
        outputs = self.model(inputs, return_reconstruct=visualize, mask_ratio = self.mask_ratio, fusion_layer = self.fusion_layer)
        if visualize:
            self.log_outputs(inputs, outputs['reconstructed'], stage='train')
        
        loss = self.loss_func(outputs)
        for k,v in loss.items():
            self.log(f'train/{k}', v, rank_zero_only=True)
        self.log('train/loss', loss[self.config.MODEL.loss_variant], prog_bar=True, rank_zero_only=True)
        self.log('train/epoch_progress', self.epoch_progress-self.current_epoch, prog_bar=True, rank_zero_only=True)
        return loss[self.config.MODEL.loss_variant]

    def validation_step(self, batch, batch_idx):
        inputs = batch
        del inputs['__key__'] # Should be removed in the future
        visualize=False
        if batch_idx % self.config.VALIDATION.visualize_every == 0 and self.local_rank == 0 and self.wandb_avail():
            visualize = True
        inputs = {k: v for k, v in inputs.items()}
        outputs = self.model(inputs, return_reconstruct=visualize, mask_ratio = self.mask_ratio, fusion_layer = self.fusion_layer)
        if visualize:
            self.log_outputs(inputs, outputs['reconstructed'], stage='val')
            
        loss = self.loss_func(outputs)
        for k,v in loss.items():
            self.log(f'val/{k}', v, rank_zero_only=True)
        self.log('val/loss', loss[self.config.MODEL.loss_variant], prog_bar=True, rank_zero_only=True)
        return loss[self.config.MODEL.loss_variant]

    def get_params(self):
        param_groups = add_weight_decay(
                        self.model,
                        self.config.TRAIN.weight_decay,
                        bias_wd=self.config.TRAIN.bias_wd,
                        )
        
        return param_groups
    
    def lr_scheduler_step(self, scheduler, epoch, metric):
        scheduler.step(epoch=(self.epoch_progress))
        
    def configure_optimizers(self):
        param_groups = self.get_params()
        
        # taken from https://github.com/facebookresearch/mae_st/blob/dc072aaaf640d06892e23a33b42223a994efe272/main_pretrain.py#L274
        
        self.effective_batch_size = self.config.TRAIN.batch_size_per_gpu * self.trainer.num_devices * self.config.TRAIN.num_nodes
        self.base_lr = self.config.TRAIN.learning_rate*(self.effective_batch_size/64)
        self.num_iterations_per_epoch = self.config.TRAIN.data_size // self.effective_batch_size 
        
        optimizer = torch.optim._multi_tensor.AdamW(param_groups,
                                    lr=self.base_lr,
                                    betas=self.config.TRAIN.betas,
                                    )
        
        scheduler = SchedulerMAE(
                        optimizer, 
                        lr=self.base_lr, 
                        min_lr=self.config.TRAIN.min_lr,
                        num_epochs=self.config.TRAIN.num_epochs, 
                        warmup_epochs=self.config.TRAIN.lr_warmup_epochs)
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    

class finetune_lightning_model(generic_lightning_model):

    def __init__(self, config):
        super().__init__(config)

        self.mixup_enabled = None
        self.mixup_fn = self._create_mixup_fn()
        self._init_accuracy()
        self.fusion_layer = self.config.MODEL.encoder.fusion_layer
        #uploading the weights for the classes: TRAIN time only
        # if not self.config.INFERENCE.enabled:
        # self._make_tensors_for_loss_weights()
        #test init
        self._video_transforms_test = lambda sample: build_inference_video_transforms(self.config.DATA, sample)
        self.predictions = {}
        self.test_number_samples = 0
        with open(config.INFERENCE.inference_file) as f:
            self._test_files = [line.strip() for line in f.readlines()]

    def _make_tensors_for_loss_weights(self):
        #needs a separate method for Pytorch lightning # https://lightning.ai/docs/pytorch/stable/accelerators/accelerator_prepare.html

        class_weights_for_loss_file = self.config.MODEL.class_weights_for_loss
        num_classes = self.config.MODEL.num_classes
        if not self.config.MODEL.single_class:
            class_weights_for_loss = {}
            for k, v in num_classes.items():
                if class_weights_for_loss_file[k] is not None:
                    self.register_buffer(f'class_weight_{k}', torch.zeros(v))
                    class_weights_for_loss[k] =  eval(f"self.class_weight_{k}")
                    with open(class_weights_for_loss_file[k], 'rb') as f:
                        dict_weights = pickle.load(f)
                        for i in range(v):
                            if i in dict_weights:
                                class_weights_for_loss[k][i] = dict_weights[i]
                else:
                    class_weights_for_loss[k] = None
        else:
            self.register_buffer('class_weight', torch.zeros(self.config.MODEL.num_classes))
            class_weights_for_loss = self.class_weight

        self.class_weights_for_loss = class_weights_for_loss
        self.loss_func = get_loss_func(self.config, self.class_weights_for_loss)

    def _init_accuracy(self):
        accuracy_dict = {}
        accuracy_dict_train = {}
        # self.accuracy = BinaryAccuracy() if config.MODEL.num_classes == 2 else MulticlassAccuracy(num_classes=self.config.MODEL.num_classes)
        if self.config.MODEL.single_class: 
            for i, modality in enumerate(self.model.head.modalities):
                accuracy_dict[f'{modality}'] = MulticlassAccuracy(num_classes=self.config.MODEL.num_classes, average = 'micro')
                accuracy_dict_train[f'{modality}'] = MulticlassAccuracy(num_classes=self.config.MODEL.num_classes, average = 'micro')
            accuracy_dict['Multimodal'] = MulticlassAccuracy(num_classes=self.config.MODEL.num_classes, average = 'micro')
            self.id_to_label = pd.read_csv(self.config.DATA.id_to_label_file).set_index('id').to_dict()
        else:
            self.id_to_label = {}
            for c, n in self.config.MODEL.num_classes.items():
                for i, modality in enumerate(self.model.head.modalities):
                    accuracy_dict[f'{modality}-{c}'] =  MulticlassAccuracy(num_classes=n, average = 'micro')
                    accuracy_dict_train[f'{modality}-{c}'] =  MulticlassAccuracy(num_classes=n, average = 'micro')
                accuracy_dict[f'Multimodal-{c}'] =  MulticlassAccuracy(num_classes=n, average = 'micro')
                self.id_to_label[c] = pd.read_csv(self.config.DATA.id_to_label_file[c]).set_index('id').to_dict()
        self.accuracy = torch.nn.ModuleDict(accuracy_dict)
        self.train_accuracy = torch.nn.ModuleDict(accuracy_dict_train)

        if self.config.MODEL.single_class:
            self.best_val_acc = -1
        else:
            self.best_val_acc = {}
            for c, n in self.config.MODEL.num_classes.items():
                self.best_val_acc[c] = -1

    def _create_mixup_fn(self):
        
        self.mixup_enabled = self.config.TRAIN.mixup.enabled
        if self.mixup_enabled:
            
            if self.config.MODEL.single_class:
                num_classes = [self.config.MODEL.num_classes]
            else:
                num_classes = [self.config.MODEL.num_classes[c] for c in self.config.MODEL.num_classes.keys()]
            
            return MixVideo(
                            modalities=self.model.modalities,
                            mixup_alpha=self.config.TRAIN.mixup.mixup_alpha,
                            cutmix_alpha=self.config.TRAIN.mixup.cutmix_alpha,
                            mix_prob=self.config.TRAIN.mixup.mixup_prob,
                            switch_prob=self.config.TRAIN.mixup.switch_prob,
                            label_smoothing=self.config.TRAIN.mixup.label_smoothing,
                            num_classes=num_classes,
                        )    
        else:
            return None
    
    def train_single_head(self, batch, batch_idx):
        
        inputs = batch
        labels = inputs['cls']
        del inputs['cls']
        
        # Needed as it is for the scheduler
        self.epoch_progress = batch_idx/self.num_iterations_per_epoch + self.current_epoch
        self.log('train/epoch_progress', self.epoch_progress-self.current_epoch, prog_bar=True)
        
        # Mixup
        if self.mixup_fn is not None:
            inputs, labels = self.mixup_fn(inputs, {'labels':labels})
            labels = labels['labels']
                    
        outputs = self.model(inputs, fusion_layer = self.fusion_layer)
        if batch_idx % self.config.TRAIN.visualize_every == 0 and self.local_rank == 0 and self.wandb_avail():
            self.log_inputs(inputs, labels,  stage='train', single_class = self.config.MODEL.single_class, outputs = outputs)

        loss = self.loss_func(outputs, labels)
        for k,v in loss.items():
            self.log(f'train/{k}', v)
        self.log('train/loss', loss[self.config.MODEL.loss_variant], prog_bar=True, on_step=True, on_epoch=True)
        
        if self.mixup_fn is None:
            for i, modality in enumerate(self.model.head.modalities):
                self.train_accuracy[f'{modality}'](outputs['logits'][i], labels)
                self.log(f'train/accuracy_{modality}', self.train_accuracy[f'{modality}'], 
                    prog_bar=True, 
                    on_epoch = True)
    
        return loss[self.config.MODEL.loss_variant]
    
    def train_multi_head(self, batch, batch_idx):
        
        inputs = batch
        labels = inputs['cls']
        
        del inputs['cls']
        self.epoch_progress = batch_idx/self.num_iterations_per_epoch + self.current_epoch
        self.log('train/epoch_progress', self.epoch_progress-self.current_epoch, prog_bar=True)
        
        # Mixup
        if self.mixup_fn is not None:
            inputs, labels = self.mixup_fn(inputs, labels)
        
        outputs = self.model(inputs, fusion_layer = self.fusion_layer)
        if batch_idx % self.config.TRAIN.visualize_every == 0 and self.local_rank == 0 and self.wandb_avail():
            self.log_inputs(inputs, labels,  stage='train', single_class = self.config.MODEL.single_class, outputs = outputs)

        loss = self.loss_func(outputs, labels)
        for k,v in loss.items():
            self.log(f'train/{k}', v)
        self.log('train/loss', loss[self.config.MODEL.loss_variant], prog_bar=True, on_step=True, on_epoch=True)

        if self.mixup_fn is None:
            for c, n in self.config.MODEL.num_classes.items():
                for i, modality in enumerate(self.model.head.modalities):
                    self.train_accuracy[f'{modality}-{c}'](outputs['logits'][c][i], labels[c])
                    self.log(f'train/accuracy_{modality}-{c}', self.train_accuracy[f'{modality}-{c}'], 
                                            prog_bar=True, 
                                            on_epoch=True,)
                
        return loss[self.config.MODEL.loss_variant]
      
    def training_step(self, batch, batch_idx):
        if self.config.MODEL.single_class:
            return self.train_single_head(batch, batch_idx)
        else:
            return self.train_multi_head(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        inputs = batch
        labels = inputs['cls']
        del inputs['cls']
        
        outputs = self.model(inputs, fusion_layer = self.fusion_layer, is_train = False)
        if batch_idx % self.config.VALIDATION.visualize_every == 0 and self.local_rank == 0 and self.wandb_avail():
            self.log_inputs(inputs, labels, stage='val', single_class = self.config.MODEL.single_class, outputs = outputs )
        
        loss = self.loss_func(outputs, labels, stage='val')
        
        for k,v in loss.items():
            self.log(f'val/{k}', v, on_epoch=True)
        self.log('val/loss', loss[self.config.MODEL.loss_variant], prog_bar=True, on_epoch=True)
        multimodal = []
        if self.config.MODEL.single_class:
            for i, modality in enumerate(self.model.head.modalities):
                self.accuracy[f'{modality}'].update(outputs['logits'][i], labels)
                multimodal.append(outputs['logits'][i])
            self.accuracy['Multimodal'].update(torch.mean(torch.stack(multimodal), dim=0), labels)
        else:
            for c, n in self.config.MODEL.num_classes.items():
                multimodal = []
                for i, modality in enumerate(self.model.head.modalities):
                    self.accuracy[f'{modality}-{c}'].update(outputs['logits'][c][i], labels[c])
                    multimodal.append(outputs['logits'][c][i])
                self.accuracy[f'Multimodal-{c}'].update(torch.mean(torch.stack(multimodal), dim=0), labels[c])
        return loss[self.config.MODEL.loss_variant]

    def on_validation_epoch_end(self):
        # update best accuracy
        if self.config.MODEL.single_class: 
            for i, modality in enumerate(self.model.head.modalities):
                this_acc = self.accuracy[f'{modality}'].compute()
                self.log(f'val/accuracy_{modality}', this_acc, 
                        prog_bar=True,
                        sync_dist=True)
                self.accuracy[f'{modality}'].reset()
            this_acc = self.accuracy['Multimodal'].compute()
            self.log(f'val/accuracy', this_acc, 
                    prog_bar=True,
                    sync_dist=True)
            if this_acc > self.best_val_acc:
                self.best_val_acc = this_acc
            self.log('val/best_accuracy', self.best_val_acc, 
                    prog_bar=True,
                    sync_dist=True)
            self.accuracy['Multimodal'].reset()
        else:
            for c, n in self.config.MODEL.num_classes.items():
                for i, modality in enumerate(self.model.head.modalities):
                    this_acc = self.accuracy[f'{modality}-{c}'].compute()
                    self.log(f'val/accuracy_{modality}-{c}', this_acc, 
                                        sync_dist=True, 
                                        prog_bar=True)
                    self.accuracy[f'{modality}-{c}'].reset()
                this_acc = self.accuracy[f'Multimodal-{c}'].compute()
                self.log(f'val/accuracy_{c}', this_acc, 
                                    sync_dist=True, 
                                    prog_bar=True)                
                if this_acc > self.best_val_acc[c]:
                    self.best_val_acc[c] = this_acc
                self.log(f'val/best_accuracy_{c}', self.best_val_acc[c],
                         prog_bar=True,
                         sync_dist=True)
                self.accuracy[f'Multimodal-{c}'].reset()
                
    def test_step(self, batch, batch_idx):
        def update_predictions_log(keys, predictions, labels, cl = None):
            if cl:
                for k, p, l  in zip(keys, torch.argmax(predictions, axis = 1).cpu().numpy(), labels.cpu().numpy()):
                # for k, p, l  in zip(keys, predictions.cpu().numpy(), labels.cpu().numpy()):
                    if k not in self.predictions:
                        # self.predictions[k] = {cl:{'pred': p.tolist(), 'label': int(l)}}
                        self.predictions[k] = {cl:{'pred': int(p), 'label': int(l)}}
                    else:
                        # self.predictions[k].update({cl:{'pred': p.tolist(), 'label': int(l)}})
                        self.predictions[k].update({cl:{'pred': int(p), 'label': int(l)}})
            else:
                # self.predictions.update({k:{'pred': p.tolist(), 'label': int(l)} for k, p, l in zip(keys, predictions.cpu().numpy(), labels.cpu().numpy())})
                self.predictions.update({k:{'pred': int(p), 'label': int(l)} for k, p, l in zip(keys, torch.argmax(predictions, axis = 1).cpu().numpy(), labels.cpu().numpy())})
            
        inputs = batch
        softmax = torch.nn.Softmax(dim=1).cuda()
        labels = inputs['cls']
        keys = inputs['__key__']
        del inputs['cls']
        del inputs['__key__']
        num_samples = labels.shape[0] if self.config.MODEL.single_class else labels['noun'].shape[0]
        self.test_number_samples += num_samples
        outputs = self.model(inputs, fusion_layer = self.fusion_layer, is_train = False)

        if batch_idx % self.config.INFERENCE.visualize_every == 0 and self.local_rank == 0 and self.wandb_avail():
            for i in range(num_samples):
                self.log_inputs(inputs, labels, stage='test', single_class = self.config.MODEL.single_class, outputs = outputs, index = i )

        if self.config.INFERENCE.num_views > 1 or self.config.INFERENCE.num_spatial_crops > 1:
            multimodal = []
            if self.config.MODEL.single_class:
                for i, modality in enumerate(self.model.head.modalities):
                    labels = torch.unsqueeze(labels[0], 0)
                    predictions = torch.mean(outputs['logits'][0], dim = 0, keepdim = True)
                    multimodal.append(predictions)
                    self.accuracy[f'{modality}'].update(predictions, labels)
                p = torch.mean(torch.stack(multimodal), dim=0)
                self.accuracy['Multimodal'].update(p, labels)
                update_predictions_log(keys, p, labels)
            else:
                for c, n in self.config.MODEL.num_classes.items():
                    multimodal = []
                    for i, modality in enumerate(self.model.head.modalities):
                        # if str(modality) == "audio":
                        #     print(outputs['logits'][c][i])
                        predictions = torch.mean(softmax(outputs['logits'][c][i]), dim = 0, keepdim = True)
                        # predictions = torch.max(outputs['logits'][c][0], dim = 0, keepdim = True)[0]
                        multimodal.append(predictions)
                        labels_c = torch.unsqueeze(labels[c][0], 0)
                        self.accuracy[f'{modality}-{c}'].update(predictions, labels_c)
                    p = torch.mean(torch.stack(multimodal), dim=0)
                    self.accuracy[f'Multimodal-{c}'].update(p, labels_c)
                    update_predictions_log(keys, p, labels_c, c)
        else:
            # take the first output, corresponding to the first modality
            if self.config.MODEL.single_class:
                multimodal = []
                for i, modality in enumerate(self.model.head.modalities):
                    self.accuracy[f'{modality}'].update(outputs['logits'][i], labels)
                    multimodal.append(outputs['logits'][i])
                p = torch.mean(torch.stack(multimodal), dim=0)
                self.accuracy['Multimodal'].update(p, labels)
                update_predictions_log(keys, p, labels)
            else:
                for c, n in self.config.MODEL.num_classes.items():
                    multimodal = []
                    for i, modality in enumerate(self.model.head.modalities):
                        self.accuracy[f'{modality}-{c}'].update(outputs['logits'][c][i], labels[c])
                        multimodal.append(outputs['logits'][c][i])
                    p = torch.mean(torch.stack(multimodal), dim=0)
                    self.accuracy[f'Multimodal-{c}'].update(p, labels[c])
                    update_predictions_log(keys, p, labels[c], c)

    def on_test_epoch_end(self):
        # update best accuracy
        print(f'finished inference for {self.test_number_samples} instances')
        if self.config.MODEL.single_class:
            for i, modality in enumerate(self.model.head.modalities):
                this_acc = self.accuracy[f'{modality}'].compute()
                self.log(f'test/accuracy_{modality}', this_acc, 
                        prog_bar=True,
                        sync_dist=True)
            this_acc = self.accuracy['Multimodal'].compute()
            self.log(f'test/accuracy', this_acc, 
                    prog_bar=True,
                    sync_dist=True)
        else:
            for c, n in self.config.MODEL.num_classes.items():
                for i, modality in enumerate(self.model.head.modalities):
                    this_acc = self.accuracy[f'{modality}-{c}'].compute()
                    self.log(f'test/accuracy_{modality}-{c}', this_acc, 
                                        sync_dist=True, 
                                        prog_bar=True)
                this_acc = self.accuracy[f'Multimodal-{c}'].compute()
                self.log(f'test/accuracy-{c}', this_acc, 
                                    sync_dist=True, 
                                    prog_bar=True)
        
        if not self.config.INFERENCE.predictions_dir:
            flags_optional = "" if self.config.DATA.filter_criteria is None else "_" + self.config.DATA.filter_criteria
            filename = osp.join(self.config.exp_dir, self.config.exp_name, osp.splitext(osp.basename(self.config.DATA.metadata_file))[0] + flags_optional + '.txt')
        else:
            filename = self.config.INFERENCE.predictions_dir
            if not osp.exists(osp.dirname(filename)):
                os.makedirs(osp.dirname(filename), exist_ok=True)
        with open(filename,  "w") as fp:
            json.dump(self.predictions, fp)
            print(filename)

    def log_inputs(self, inputs, labels, stage, single_class, outputs, index = None):
        bs = inputs[list(inputs.keys())[0]].shape[0]
        if index is None:
            index = torch.randint(0, bs, (1,)).item()
              
        if stage == 'val' or stage == 'test':
            if single_class:
                pred_label = self.id_to_label["key"][int(outputs["logits"][0][index].argmax().cpu())]
                target_label = self.id_to_label["key"][int(labels[index].cpu())]
                label_to_print = f'Target: {target_label}, Pred: {pred_label}'
            else:
                label_to_print = ''
                for k, v in labels.items():
                    pred_label = self.id_to_label[k]["key"][int(outputs["logits"][k][0][index].argmax().cpu())]
                    target_label = self.id_to_label[k]["key"][int(v[index].cpu())]
                    label_to_print = f'{label_to_print}  {k}: Target: {target_label}, Pred: {pred_label} \n'
        else:
            if single_class:
                label_to_print = 'mixup' if self.mixup_fn is not None else f'Target: {labels[index]}'
            else:
                if self.mixup_fn is not None:
                    label_to_print = 'mixup'
                else:
                    label_to_print = ''
                    for k, v in labels.items():
                        target_label = self.id_to_label[k]["key"][int(v[index].cpu())]
                        label_to_print = f'{label_to_print}  {k}: Target: {target_label} \n'

        for modality in self.model.modalities:
            if str(modality) == 'video':
                self.log_video(inputs[modality][index], label_to_print, stage)
            elif str(modality) == 'imu':
                self.log_imu(inputs[modality][index], label_to_print, stage)
            elif str(modality) == 'audio':
                self.log_audio(inputs[modality][index], label_to_print, stage)
            else:
                logger.error(f"Unknown modality {modality}")
                raise NotImplementedError
        pass
    
    def log_video(self, video, label, stage):
        input_vis = ((self._denormalize_video(video).detach().permute(1, 0, 2, 3).cpu())*255).to(torch.uint8)
        self.logger.log_video(f"visualize/{stage}/video",
                    videos=[wandb.Video(input_vis, fps=self.config.DATA.video.output_fps, format="gif", caption=label)], 
                    step=self.global_step)

    def log_audio(self, audio, label, stage):
        self.logger.log_image(f"visualize/{stage}/audio",
                               images= [wandb.Image(audio.detach().cpu(), caption=label)],
                               step=self.global_step)
        
    def train_dataloader(self):
        dataset_train = build_dataset_finetune(
                            self._train_files, self.config, 
                            self._video_transform_train, 
                            self._imu_transform, 
                            self._audio_transform_train)

        return torch.utils.data.DataLoader(
                                    dataset_train, 
                                    batch_size=None, 
                                    pin_memory=True,
                                    num_workers=self.config.TRAIN.num_workers, 
                                    collate_fn=None, 
                                    drop_last=False)

    def val_dataloader(self):
        dataset_val = build_dataset_finetune(
                    self._val_files, self.config, 
                    self._video_transform_val, 
                    self._imu_transform,
                    self._audio_transform_val,
                    is_train=False)
        
        return torch.utils.data.DataLoader(dataset_val, 
                                           batch_size=None, 
                                           pin_memory=True,
                                           num_workers=self.config.VALIDATION.num_workers, 
                                           collate_fn=None, 
                                           drop_last=False)

    def test_dataloader(self):
        dataset_test = build_dataset_inference(self._test_files, self.config, 
                    self._video_transforms_test, 
                    self._imu_transform,
                    self._audio_transform_val,
                    )
        return torch.utils.data.DataLoader(dataset_test, 
                                           batch_size=None, 
                                           pin_memory=True,
                                           num_workers=self.config.INFERENCE.num_workers, 
                                           collate_fn=None, 
                                           drop_last=False)

    def get_params(self):
        param_groups = param_groups_lrd(
                        self.model,
                        self.config.TRAIN.weight_decay,
                        no_weight_decay_list=self.model.no_weight_decay(),
                        layer_decay=self.config.TRAIN.layer_decay,
                        )
        return param_groups
    
    def lr_scheduler_step(self, scheduler, epoch, metric):
        scheduler.step(epoch=(self.epoch_progress))
        
    def configure_optimizers(self):
        param_groups = self.get_params()
        
        # taken from https://github.com/facebookresearch/mae_st/blob/dc072aaaf640d06892e23a33b42223a994efe272/main_pretrain.py#L274
        self.effective_batch_size = self.config.TRAIN.batch_size_per_gpu * self.trainer.num_devices * self.config.TRAIN.num_nodes
        self.base_lr = self.config.TRAIN.learning_rate*(self.effective_batch_size/64)
        self.num_iterations_per_epoch = self.config.TRAIN.data_size // self.effective_batch_size 
        
        if self.config.TRAIN.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(param_groups,
                                    lr=self.base_lr)
        elif self.config.TRAIN.optimizer == 'sgd':
            optimizer = torch.optim.SGD(param_groups,
                                        lr=self.base_lr)
        scheduler = SchedulerMAE(
                        optimizer, 
                        lr=self.base_lr, 
                        min_lr=self.config.TRAIN.min_lr,
                        num_epochs=self.config.TRAIN.num_epochs, 
                        warmup_epochs=self.config.TRAIN.lr_warmup_epochs)
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class tta_lightning_model(finetune_lightning_model):

    def __init__(self, config):
        super().__init__(config)
        self.configure_model(update_bn_only = config.INFERENCE.update_bn_only)
        params, _ = self.collect_params()
        base_lr = self.config.INFERENCE.tta_lr if 'tta_lr' in self.config.INFERENCE else 0.00025
        lr = (base_lr / 64) * self.config.TRAIN.batch_size_per_gpu * 2 if self.config.TRAIN.batch_size_per_gpu < 32 else 0.00025
        self.optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
        if self.config.INFERENCE.TTA_METHOD == 'midl':
            self.original_model_predictions = json.load(open(f'{self.config.INFERENCE.original_model_predictions}', 'r'))
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
        # self.log_softmax = torch.nn.LogSoftmax()

    def collect_params(self):
        """Collect the affine scale + shift parameters from batch norms.
        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(m, torch.nn.LayerNorm):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def configure_model(self, update_bn_only=True):
        """Configure model for use with tent."""
        # train mode, because shot optimizes the model to minimize entropy
        # SHOT updates all parameters of the feature extractor, excluding the last FC layers
        # Original SHOT implementation
        if not update_bn_only:
            # self.model.train()
            # is this needed? review later
            # self.model.requires_grad_(True)
            # Freeze classifier layers 
            for m in self.model.modules():
                if isinstance(m, torch.nn.Linear):
                    m.requires_grad_(False)
            # for name, param in self.model.named_parameters():
            #     if 'head' in name:
            #         param.requires_grad = False
        else:
            # In case we want shot to update only the BN layers
            # disable grad, to (re-)enable only what tent updates (originally not used by shot but other papers use it when using shot)
            self.model.requires_grad_(False)
            # configure norm for tent updates: enable grad + force batch statisics
            for m in self.model.modules():
                if isinstance(m, torch.nn.LayerNorm):
                    m.requires_grad_(True)
                    # force use of batch stats in train and eval modes
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
        # return model

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def test_step(self, batch, batch_idx):
        # torch.set_grad_enabled(True)
        labels = batch['cls']
        keys = batch['__key__']
        del batch['cls']
        del batch['__key__'] 
        #from https://github.com/Lightning-AI/pytorch-lightning/issues/14497

        with torch.inference_mode(mode=False):
            def update_predictions_log(keys, predictions, labels, cl = None):
                if cl:
                    for k, p, l  in zip(keys, torch.argmax(predictions, axis = 1).cpu().numpy(), labels.cpu().numpy()):
                        if k not in self.predictions:
                            self.predictions[k] = {cl:{'pred': int(p), 'label': int(l)}}
                        else:
                            self.predictions[k].update({cl:{'pred': int(p), 'label': int(l)}})
                else:
                    self.predictions.update({k:{'pred': int(p), 'label': int(l)} for k, p, l in zip(keys, torch.argmax(predictions, axis = 1).cpu().numpy(), labels.cpu().numpy())})
            inputs = batch
            softmax = torch.nn.Softmax(dim=1).cuda()
            num_samples = labels.shape[0] if self.config.MODEL.single_class else labels['noun'].shape[0]
            self.test_number_samples += num_samples
            outputs = self.model(inputs, fusion_layer = self.fusion_layer, is_train = False)

            if batch_idx % self.config.INFERENCE.visualize_every == 0 and self.local_rank == 0 and self.wandb_avail():
                for i in range(num_samples):
                    self.log_inputs(inputs, labels, stage='test', single_class = self.config.MODEL.single_class, outputs = outputs, index = i )

            if self.config.INFERENCE.num_views > 3 or self.config.INFERENCE.num_spatial_crops > 1:
                multimodal = []
                if self.config.MODEL.single_class:
                    for i, modality in enumerate(self.model.head.modalities):
                        labels = torch.unsqueeze(labels[0], 0)
                        predictions = torch.mean(outputs['logits'][i], dim = 0, keepdim = True)
                        multimodal.append(predictions)
                        self.accuracy[f'{modality}'].update(predictions, labels)
                    p = torch.mean(torch.stack(multimodal), dim=0)
                    self.accuracy['Multimodal'].update(p, labels)
                    update_predictions_log(keys, p, labels)
                else:
                    for c, n in self.config.MODEL.num_classes.items():
                        multimodal = []
                        for i, modality in enumerate(self.model.head.modalities):
                            predictions = torch.mean(softmax(outputs['logits'][c][i]), dim = 0, keepdim = True)
                            # predictions = torch.max(outputs['logits'][c][0], dim = 0, keepdim = True)[0]
                            multimodal.append(predictions)
                            labels_c = torch.unsqueeze(labels[c][0], 0)
                            self.accuracy[f'{modality}-{c}'].update(predictions, labels_c)
                        p = torch.mean(torch.stack(multimodal), dim=0)
                        self.accuracy[f'Multimodal-{c}'].update(p, labels_c)
                        update_predictions_log(keys, p, labels_c, c)
            else:
                '''
                TODO: Organize this piece of code
                this whole piece of code after the else is where
                TTA is happening, so it should be organized in a better way
                '''
                if self.config.MODEL.single_class:
                    '''
                    TODO: Create a cleaner way to do this
                    Epic Sounds adaptation step,
                    single head
                    '''
                    multimodal = []
                    for i, modality in enumerate(self.model.head.modalities):
                        self.accuracy[f'{modality}'].update(outputs['logits'][i], labels)
                        multimodal.append(outputs['logits'][i])
                    pred_average_of_mod = torch.mean(torch.stack(multimodal), dim=0)
                        
                    # SHOT-IM https://github.com/MotasemAlfarra/Online_Test_Time_Adaptation/blob/1ec1979744bae6edab5ec8be36650b96bff38d11/tta_methods/shot_im.py#L69
                    # Tent https://arxiv.org/pdf/2006.10726.pdf
                    # ETA: https://github.com/MotasemAlfarra/Online_Test_Time_Adaptation/blob/main/tta_methods/eata.py

                    if self.config.INFERENCE.TTA_METHOD == 'midl':
                        #i = 0 - > erase video, i = 1 -> erase audio
                        if sum(inputs['audio_attention']) == 0:
                            #only video is here, so we take index 1 or 2
                            pred_average_of_mod = torch.unsqueeze(pred_average_of_mod[1], 0)
                        elif sum(inputs['video_attention']) == 0:
                            #only audio is here, so we take index 0 or 2
                            pred_average_of_mod = torch.unsqueeze(pred_average_of_mod[0], 0)
                        softmax_out = pred_average_of_mod.softmax(1)
                        msoftmax = softmax_out.mean(0)

                        l_ent = (softmax_out * torch.log(softmax_out + 1e-5)).sum(-1).mean(0)
                        l_div = -(msoftmax * torch.log(msoftmax + 1e-5)).sum(-1)
                        assert all(x == keys[0] for x in keys)
                        if keys[0] in self.original_model_predictions:
                            orginal_preds = torch.tensor([self.original_model_predictions[k]['pred'] for k in keys])

                            B = pred_average_of_mod.shape[0]

                            ce_loss = self.kl_loss(pred_average_of_mod.log_softmax(1)[-1:], orginal_preds[-1:].to(pred_average_of_mod.device).softmax(1))
                            loss =  self.config.INFERENCE.lambda_mi * (l_ent + l_div) + self.config.INFERENCE.lambda_kl *  ce_loss

                            #MUTLI-SOURCE-MUTUAL-INFOR-MIN
                            
                    else: # conducting adaptation with Shotim, Tent or ETA
                        softmax_out = pred_average_of_mod.softmax(1)
                        msoftmax = softmax_out.mean(0)
                        entropys = -(softmax_out * torch.log(softmax_out + 1e-5)).sum(-1)
                        if self.config.INFERENCE.TTA_METHOD == 'tent':
                            loss = entropys.mean(0)
                        elif self.config.INFERENCE.TTA_METHOD == 'shot-im':
                            l_div = (msoftmax * torch.log(msoftmax + 1e-5)).sum(-1)
                            loss = entropys.mean(0) + l_div
                        elif self.config.INFERENCE.TTA_METHOD == 'eta':
                            margin = self.config.INFERENCE.ETA.MARGIN_MULTIPLIER * math.log(self.config.MODEL.num_classes)
                            filter_ids_1 = torch.where(entropys < margin)
                            entropys = entropys[filter_ids_1]
                            coeff = 1 / (torch.exp(entropys.clone().detach() - margin))
                            loss = (entropys.mul(coeff)).mean(0)
                        else:
                            print("Method not implemented")
                            raise ValueError
                            
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.config.INFERENCE.TTA_METHOD == 'midl':
                        self.accuracy['Multimodal'].update(pred_average_of_mod[-1:], labels[-1:])
                        update_predictions_log(keys, pred_average_of_mod[-1:], labels[-1:])
                    else:
                        self.accuracy['Multimodal'].update(pred_average_of_mod, labels)
                        update_predictions_log(keys, pred_average_of_mod, labels)
                else:

                    loss_all_labels = 0
                    for c, n in self.config.MODEL.num_classes.items():
                        multimodal = []
                        for i, modality in enumerate(self.model.head.modalities):
                            self.accuracy[f'{modality}-{c}'].update(outputs['logits'][c][i], labels[c])
                            multimodal.append(outputs['logits'][c][i])
                        pred_average_of_mod = torch.mean(torch.stack(multimodal), dim=0)

                        if self.config.INFERENCE.TTA_METHOD == 'midl':
                            #i = 0 - > erase video, i = 1 -> erase audio
                            if sum(inputs['audio_attention']) == 0:
                                #only video is here, so we take index 1 or 2
                                # assert pred_average_of_mod[1] == pred_average_of_mod[2]
                                pred_average_of_mod = torch.unsqueeze(pred_average_of_mod[1], 0)
                            elif sum(inputs['video_attention']) == 0:
                                #only audio is here, so we take index 0 or 2
                                # assert pred_average_of_mod[0] == pred_average_of_mod[2]
                                pred_average_of_mod = torch.unsqueeze(pred_average_of_mod[0], 0)

                            softmax_out = pred_average_of_mod.softmax(1)
                            msoftmax = softmax_out.mean(0)
                            l_ent = (softmax_out * torch.log(softmax_out + 1e-5)).sum(-1).mean(0)
                            l_div = -(msoftmax * torch.log(msoftmax + 1e-5)).sum(-1)
                            assert all(x == keys[0] for x in keys)
                            if keys[0] in self.original_model_predictions:
                                orginal_preds = torch.tensor([self.original_model_predictions[k][c]['pred'] for k in keys])

                                B = pred_average_of_mod.shape[0]

                                ce_loss = self.kl_loss(pred_average_of_mod.log_softmax(1)[-1:], orginal_preds[-1:].to(pred_average_of_mod.device).softmax(1))
                                loss =  self.config.INFERENCE.lambda_mi * (l_ent + l_div) + self.config.INFERENCE.lambda_kl *  ce_loss
                                loss_all_labels = loss_all_labels + loss
                                print(f"l_ent: {l_ent}, l_div: {l_div}, ce_loss: {ce_loss},  loss {loss}")
                                #MUTLI-SOURCE-MUTUAL-INFOR-MIN
                        else:
                            softmax_out = pred_average_of_mod.softmax(1)
                            msoftmax = softmax_out.mean(0)
                            entropys = -(softmax_out * torch.log(softmax_out + 1e-5)).sum(-1)
                            if self.config.INFERENCE.TTA_METHOD == 'tent':
                                loss_all_labels += entropys.mean(0)
                            elif self.config.INFERENCE.TTA_METHOD == 'shot-im':
                                l_div = (msoftmax * torch.log(msoftmax + 1e-5)).sum(-1)
                                loss_all_labels += entropys.mean(0) + l_div
                            elif self.config.INFERENCE.TTA_METHOD == 'eta':
                                num_classes = n
                                margin = self.config.INFERENCE.ETA.MARGIN_MULTIPLIER * math.log(num_classes)
                                filter_ids_1 = torch.where(entropys < margin)
                                entropys = entropys[filter_ids_1]
                                coeff = 1 / (torch.exp(entropys.clone().detach() - margin))
                                loss_all_labels += (entropys.mul(coeff)).mean(0)
                            else:
                                print("Method not implemented")
                                raise ValueError
                        if self.config.INFERENCE.TTA_METHOD == 'midl':
                            self.accuracy[f'Multimodal-{c}'].update(pred_average_of_mod[-1:], labels[c][-1:])
                            update_predictions_log(keys, pred_average_of_mod[-1:], labels[c][-1:], c)
                        else:
                            self.accuracy[f'Multimodal-{c}'].update(pred_average_of_mod, labels[c])
                            update_predictions_log(keys, pred_average_of_mod, labels[c], c)

                    loss_all_labels.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

    def on_test_epoch_end(self):
        # update best accuracy
        print(f'finished inference for {self.test_number_samples} instances')
        if self.config.MODEL.single_class:
            for i, modality in enumerate(self.model.head.modalities):
                this_acc = self.accuracy[f'{modality}'].compute()
                self.log(f'test/accuracy_{modality}', this_acc, 
                        prog_bar=True,
                        sync_dist=True)
            this_acc = self.accuracy['Multimodal'].compute()
            self.log(f'test/accuracy', this_acc, 
                    prog_bar=True,
                    sync_dist=True)
        else:
            for c, n in self.config.MODEL.num_classes.items():
                for i, modality in enumerate(self.model.head.modalities):
                    this_acc = self.accuracy[f'{modality}-{c}'].compute()
                    self.log(f'test/accuracy_{modality}-{c}', this_acc, 
                                        sync_dist=True, 
                                        prog_bar=True)
                this_acc = self.accuracy[f'Multimodal-{c}'].compute()
                self.log(f'test/accuracy-{c}', this_acc, 
                                    sync_dist=True, 
                                    prog_bar=True)

        if self.config.INFERENCE.save_predictions:
            if not self.config.INFERENCE.predictions_dir:
                flags_optional = "" if self.config.DATA.filter_criteria is None else "_" + self.config.DATA.filter_criteria
                filename = osp.join(self.config.exp_dir, self.config.exp_name, osp.splitext(osp.basename(self.config.DATA.metadata_file))[0] + flags_optional + '_tta.txt')
            else:
                filename = self.config.INFERENCE.predictions_dir
                if not osp.exists(osp.dirname(filename)):
                    os.makedirs(osp.dirname(filename), exist_ok=True)
            with open(filename, "w") as fp:
                json.dump(self.predictions, fp)
                print(filename)
            
    # def on_test_model_eval(self, *args, **kwargs):
    #     super().on_test_model_eval(*args, **kwargs)
    #     torch.set_grad_enabled(True)

    # def on_validation_model_eval(self, *args, **kwargs):
    #     super().on_validation_model_eval(*args, **kwargs)
    #     torch.set_grad_enabled(True)