from lightning_module import  finetune_lightning_model, tta_lightning_model

import pytorch_lightning as pl
from misc.custom_logger import WandbVideoLogger
# import torch
import os.path as osp
from misc.setup_dirs import parse_option, generate_exp_directory, generate_exp_name
import logging

# configure logging at the root level of Lightning
logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

# configure logging on module level, redirect to file
logger = logging.getLogger("pytorch_lightning.core")
logger.addHandler(logging.FileHandler("core.log"))

def get_initialization_name(config):
    if 'random' in config.TRAIN.checkpoint_to_finetune:
        return 'random'
    if 'video_mae' in config.TRAIN.checkpoint_to_finetune:
        return 'visual'
    if 'audiovisual' in config.TRAIN.checkpoint_to_finetune:
        return 'audiovisual'
    if 'bottleneck' in config.TRAIN.checkpoint_to_finetune:
        return 'bottleneck'


def get_checkpoint(config):
    print(f'Loading the checkpoint from {config.ckpt_dir}')
    return config.INFERENCE.checkpoint


def get_previous_wandb_runid(config):
    "Return the wandb folder name, if no previous run, return None"
    exp_folder = osp.join(config.exp_dir, config.exp_name)
    wandb_folder = osp.join(exp_folder, 'wandb')
    # get wandb id from the wandb/latest-run symlink
    wandb_id = osp.realpath(osp.join(wandb_folder, 'latest-run'))
    if not osp.exists(wandb_id):
        return None
    # parse wandb id
    return wandb_id.split('/')[-1].split('-')[-1]


def main(config):
    # -- Seed --
    pl.seed_everything(config.seed, workers=True)
    
    # -- Logger --

    if config.wandb.use_wandb:
        wandbid = get_previous_wandb_runid(config)
        import wandb
        if wandbid is None:
            print(
                f'Creating a new wandb run (from {osp.join(config.exp_dir, config.exp_name)})')
            wandb_logger = WandbVideoLogger(
                project=config.wandb.project, name=config.exp_name, save_dir=osp.join(config.exp_dir, config.exp_name))
           
        else:
            print(
                f'Loading the previous wandb run id: {wandbid} (from {osp.join(config.exp_dir, config.exp_name)})')
            wandb_logger = WandbVideoLogger(project=config.wandb.project, name=config.exp_name, save_dir=osp.join(
                config.exp_dir, config.exp_name), id=wandbid, resume='must')

    inference_mode = False if config.INFERENCE.type  == 'tta' else True
    trainer = pl.Trainer(devices=1,
                         accelerator="gpu",
                         num_nodes=1,
                         log_every_n_steps=5,
                         logger=wandb_logger if config.wandb.use_wandb else True,
                        #  profiler="simple",
                         num_sanity_val_steps=0,
                         enable_progress_bar=True,
                         deterministic=True,
                         inference_mode=inference_mode
                         )
    if 'tta' in config.INFERENCE.type:
        trainer.test(model=tta_lightning_model(config), ckpt_path=get_checkpoint(config))
        if config.INFERENCE.save_checkpoint:
            # print(osp.join(config.exp_dir, config.exp_name, config.INFERENCE.save_ckpt_name))
            # trainer.save_checkpoint(osp.join(config.exp_dir, config.exp_name, config.INFERENCE.save_ckpt_name))
            print(osp.join(config.INFERENCE.save_ckpt_name))
            trainer.save_checkpoint(config.INFERENCE.save_ckpt_name)
    elif config.MODEL.checkpoint_load_multimodal.mode == 'inference_with_one_mod':
        trainer.test(model=finetune_lightning_model(config), ckpt_path=None)
    else:
        trainer.test(model=finetune_lightning_model(config), ckpt_path=get_checkpoint(config))


if __name__ == '__main__':
    args, opts, config = parse_option()
    not_showing_tags = ['TRAIN.num_epochs_per_job',
                        'TRAIN.checkpoint_to_finetune', 
                        'TRAIN.num_workers',
                        'setup_id',
                        'VALIDATION.val_check_interval',
                        'MODEL.checkpoint_load_multimodal.mode',
                        'wandb.entity', 'INFERENCE.checkpoint']
    import torch
    checkpoint = torch.load(get_checkpoint(config))
    if config.INFERENCE.load_train_config:
        print('Loading the previous run config')
        config_prev_run = checkpoint['hyper_parameters']['config']
        opt, opts, config = parse_option(config_prev_run)

    # print(config)
    tags = [f'modelname-{config.MODEL.name}',]
    print(opts)
    tags.extend([f"{opts[i].split('.')[-1]}-{opts[i+1]}" for i in range(0,len(opts),2) if opts[i] not in not_showing_tags])
    
    if 'MAE' in config.MODEL.name:
        tags.extend([f'mask_ratio_audio-{config.MODEL.preprocessor.audio.mask_ratio}',
                     f'mask_ratio_visual-{config.MODEL.preprocessor.video.mask_ratio}'])

    if 'finetune' in config.setup_id:
        initialization = get_initialization_name(config)
        tags.extend([f'init-{initialization}'])
        
    # if not config.INFERENCE.load_train_config:
    #     generate_exp_name(config, tags)
    #     generate_exp_directory(config)
    main(config)
