import argparse
from .config import config
import os
import glob
# import time
# import shortuuid
import pathlib
import os.path as osp


def get_experiment_version(config):
    versions = glob.glob(
        f'{os.getcwd()}/{config.exp_dir}/{config.exp_name}/version_*')
    version_id = len(versions)
    if config.INFERENCE.enabled:
        if type(config.INFERENCE.version) == int:
            version_id = config.INFERENCE.version
        else:
            version_id = version_id - 1
    return version_id


def parse_option(previous_config=None):
    parser = argparse.ArgumentParser('CLASSIFIER')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    args, opts = parser.parse_known_args()
    if previous_config:
        config.update(previous_config)
    else:
        config.load(args.cfg, recursive=True)
    config.update(opts)
    return args, opts, config


def generate_exp_directory(config):
    """Function to create checkpoint folder.
    Args:
        config:
    Returns:
        the expname, jobname, and folders into config
    """

    # modal_string = f'modal'
    # Generate experiment names and directories.
    # for modality in config.DATA.modality:
    #     modal_string = modal_string + '-' + modality
    modal_string = f'{config.setup_id}'
    config.exp_dir = f'{config.base_exp_dir}/{modal_string}'
    # if config.TRAIN.checkpoint_to_finetune:
    #     config.log_dir = os.path.join(config.exp_dir, config.TRAIN.checkpoint_to_finetune)
    #     config.version_num = 'F'
    if config.TRAIN.resume_version:
        config.version_num = config.TRAIN.resume_version
        config.log_dir = osp.join(
            config.exp_dir, config.exp_name, f'version_{config.version_num}')      
    else:
        config.version_num = get_experiment_version(config)
        pathlib.Path(config.exp_dir).mkdir(parents=True, exist_ok=True)
        config.log_dir = osp.join(
            config.exp_dir, config.exp_name, f'version_{config.version_num}')
    config.ckpt_dir = os.path.join(config.log_dir, 'checkpoints')
    config.output_dir = os.path.join(config.log_dir, 'outputs')
    pathlib.Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(config.ckpt_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(config.output_dir).mkdir(parents=True, exist_ok=True)


def generate_exp_name(config, tags, name=None, logname=None):
    """Function to create checkpoint folder.
    Args:
        config:
        tags: tags for saving and generating the expname
        name: specific name for experiment (optional)
        logname: the name for the current run. None if auto
    Returns:
        the expname
    """
    if logname is None:
        logname = '_'.join(tags)
        if name:
            logname = '-'.join([name])
    config.exp_name = logname