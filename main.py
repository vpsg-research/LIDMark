import os
import argparse
import sys

startup_messages = []
config_map = {
    'train_distortions': './configurations/train_distortions.yaml',
    'tune_deepfakes': './configurations/tune_deepfakes.yaml',
    'test': './configurations/test.yaml',
}
temp_parser = argparse.ArgumentParser(add_help=False)
temp_subparsers = temp_parser.add_subparsers(dest='mode')
for mode in config_map.keys():
    temp_subparsers.add_parser(mode)
args, _ = temp_parser.parse_known_args()

# Dynamic Configuration Loading:
# We parse the 'mode' argument first to locate and load the corresponding configuration file.
# This allows us to set environment variables (like CUDA_VISIBLE_DEVICES) based on the config
# before initializing the main application and importing heavy libraries.
if args.mode and args.mode in config_map:
    config_path = config_map[args.mode]
    if os.path.exists(config_path):
        from utils import Config
        temp_configs_loader = Config()
        temp_configs_loader.load_config_file(config_path)
        configs = temp_configs_loader.__config__
        
        if 'gpu_ids' in configs and configs['gpu_ids'] is not None:
            gpu_ids = str(configs['gpu_ids'])
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        else:
            startup_messages.append(f"'gpu_ids' not found in '{config_path}'. Using default GPU settings.")
    else:
        startup_messages.append(f"Warning: Config file not found at '{config_path}'.")

import numpy as np
import random
import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm

from utils import Config, make_loader, format_time, get_random_images, save_images, concatenate_images, setup_logger, update_config_resolution
from trainer import TrainerImg
from tester import TesterImg
from test import run_unified_test

sys.path.insert(0, './model/SimSwap')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
startup_messages.append(f"Using device: {device}")

totensor = transforms.ToTensor()
norm = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)


def define_result_dict(configs, ret_type):
    """
    Initializes dictionaries to track performance metrics and loss values.
    
    Args:
        configs: Configuration object containing model settings.
        ret_type (str): 'value' for accumulating current batch sums, 'list' for storing history.
        
    Returns:
        dict: A dictionary structure for logging keys like 'landmark_aed', 'id_ber', 'psnr', 'ssim', etc.
    """
    result_dict = {}
    total_dict = {}

    result_dict['landmark_aed'] = 0.0
    result_dict['id_ber'] = 0.0
    result_dict['psnr'] = 0.0
    result_dict['ssim'] = 0.0
    result_dict['g_loss'] = 0.0
    result_dict['g_loss_enc'] = 0.0
    result_dict['g_loss_dec'] = 0.0
    result_dict['g_loss_dec_landmark'] = 0.0
    result_dict['g_loss_dec_id'] = 0.0
    result_dict['g_loss_dis'] = 0.0
    result_dict['d_loss_raw'] = 0.0
    result_dict['d_loss_wmd'] = 0.0

    total_dict['landmark_aed'] = []
    total_dict['id_ber'] = []
    total_dict['psnr'] = []
    total_dict['ssim'] = []
    total_dict['g_loss'] = []
    total_dict['g_loss_enc'] = []
    total_dict['g_loss_dec'] = []
    total_dict['g_loss_dec_landmark'] = []
    total_dict['g_loss_dec_id'] = []
    total_dict['g_loss_dis'] = []
    total_dict['d_loss_raw'] = []
    total_dict['d_loss_wmd'] = []

    if configs.manipulation_mode == 'deepfake':
        result_dict['g_loss_gen'] = 0.0
        result_dict['g_loss_id'] = 0.0
        total_dict['g_loss_gen'] = []
        total_dict['g_loss_id'] = []

    if ret_type == 'value':
        return result_dict
    elif ret_type == 'list':
        return total_dict
    else:
        return result_dict, total_dict

def train_distortions(logger, res=256):
    """
    Main execution loop for the pre-training phase.
    Focuses on training the watermark framework to be robust against common 
    image distortions (e.g., JPEG, Blur, Resize) defined in 'train_distortions.yaml'.
    """
    configs = Config()
    configs.load_config_file('./configurations/train_distortions.yaml')
    update_config_resolution(configs, res)
    results_dir = f'./results/{res}_152'
    logger.info("--- Current pre-trained configuration ---")
    for key, value in configs.get_items():
        logger.info(f"{key}: {value}")
    logger.info("----------------------------")
    
    trainer = TrainerImg(configs, device, logger)
    writer = SummaryWriter(log_dir=os.path.join(results_dir, 'runs/pretrain_distortions'))
    logger.info("--- Start general noise pre-training ---")
    checkpoint_dir = os.path.join(configs.weight_path, 'checkpoints_distortions')
    next_epoch_to_run = 1
    if hasattr(configs, 'resume') and configs.resume.enable:
        latest_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{configs.resume.epoch}.pth')
        logger.info(f"Try to recover from a checkpoint: {latest_checkpoint_path}")
        completed_epoch = trainer.load_checkpoint(latest_checkpoint_path)
        next_epoch_to_run = completed_epoch + 1
    else:
        logger.info("Start new training from scratch...")

    train_loader = make_loader(configs, model_mode='train', shuffle=True)
    val_loader = make_loader(configs, model_mode='val', shuffle=False)

    loss_record_train = define_result_dict(configs, 'list')
    loss_record_val = define_result_dict(configs, 'list')

    logger.info('Training on going ...')
    for epoch in range(next_epoch_to_run, configs.epochs + 1):
        start_time = time.time()
        running_result = define_result_dict(configs, 'value')

        batch_count = 0
        for imgs, watermarks in tqdm(train_loader):
            result = trainer.train_batch_common(imgs, watermarks)

            for key in running_result:
                running_result[key] += float(result[key])
            batch_count += 1

        logger.info(f"Epoch {epoch}/{configs.epochs} | Time: {format_time(time.time() - start_time)}")
        for key in running_result:
            avg_value = running_result[key] / batch_count
            logger.info(f"  [Train] {key}: {avg_value:.6f}")
            writer.add_scalar(f"Train/{key}", avg_value, epoch)
            loss_record_train[key].append(float(avg_value))

        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        trainer.save_checkpoint(epoch, checkpoint_path)

        if hasattr(configs, 'validation') and configs.validation.enable:
            start_time = time.time()
            running_result_val = define_result_dict(configs, 'value')

            images_dir = os.path.join(results_dir, 'images_val_distortions')
            os.makedirs(images_dir, exist_ok=True)

            save_iters = np.random.choice(np.arange(len(val_loader)), size=configs.validation.save_count, replace=False)
            save_imgs = None
            batch_count_val = 0
            for imgs, watermarks in tqdm(val_loader):
                result, output_lst = trainer.val_batch_common(imgs, watermarks)
                for key in running_result_val:
                    running_result_val[key] += float(result[key])

                if batch_count_val in save_iters:
                    if save_imgs is None:
                        save_imgs = get_random_images(output_lst[0], output_lst[1], output_lst[2])
                    else:
                        save_imgs = concatenate_images(save_imgs, output_lst[0], output_lst[1], output_lst[2])
                batch_count_val += 1

            logger.info(f"Validation finished in {format_time(time.time() - start_time)}.")
            for key in running_result_val:
                avg_value_val = running_result_val[key] / batch_count_val

                logger.info(f"  [Val] {key}: {avg_value_val:.6f}")

                writer.add_scalar(f"Validation/{key}", avg_value_val, epoch)
                loss_record_val[key].append(float(avg_value_val))

            if save_imgs is not None:
                logger.info(f"Save the verification picture to {images_dir}/epoch-{epoch}.png")
                save_images(save_imgs, epoch, images_dir, resize_to=(configs.img_size, configs.img_size))
    writer.close()
    logger.info("--- Training completed ---")


def tune_deepfakes(logger, res=256):
    """
    Main execution loop for the fine-tuning phase.
    Loads the pre-trained distortion model and fine-tunes it against specific 
    Deepfake attacks (e.g., SimSwap, StarGAN) defined in 'tune_deepfakes.yaml'.
    """
    configs = Config()
    configs.load_config_file('./configurations/tune_deepfakes.yaml')
    update_config_resolution(configs, res)
    results_dir = f'./results/{res}_152'
    logger.info("--- Configuration of current fine-tuning training ---")
    for key, value in configs.get_items():
        logger.info(f"{key}: {value}")
    logger.info("----------------------------")
    
    trainer = TrainerImg(configs, device, logger)
    writer = SummaryWriter(log_dir=os.path.join(results_dir, 'runs/tune_deepfakes'))
    logger.info("--- Start Deepfake fine-tuning training ---")

    checkpoint_dir = os.path.join(configs.weight_path, 'checkpoints_deepfakes')
    next_epoch_to_run = 1

    if hasattr(configs, 'resume') and configs.resume.enable:
        latest_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{configs.resume.epoch}.pth')
        logger.info(f"Continue training from the last fine-tuning checkpoint: {latest_checkpoint_path}")
        completed_epoch = trainer.load_checkpoint(latest_checkpoint_path)
        next_epoch_to_run = completed_epoch + 1
    else:
        # Transfer Learning setup:
        # If not resuming a fine-tuning session, load weights from the pre-trained distortions model.
        logger.info(f"Start a new fine-tuning training and load the pre-training model weights (epoch: {configs.epoch})...")
        pretrain_checkpoint_path = os.path.join(configs.weight_path, 'checkpoints_distortions', f'checkpoint_epoch_{configs.epoch}.pth')
        trainer.load_model_for_finetune(pretrain_checkpoint_path)

    train_loader = make_loader(configs, model_mode='train', shuffle=True)
    val_loader = make_loader(configs, model_mode='val', shuffle=False)

    loss_record_train = define_result_dict(configs, 'list')
    loss_record_val = define_result_dict(configs, 'list')

    logger.info('Fine-tuning on going ...')
    for epoch in range(next_epoch_to_run, configs.epochs + 1):
        start_time = time.time()
        running_result = define_result_dict(configs, 'value')

        batch_count = 0
        for imgs, watermarks in tqdm(train_loader):
            result = trainer.train_batch_deepfake(imgs, watermarks)
            for key in running_result:
                running_result[key] += float(result[key])
            batch_count += 1

        logger.info(f"Epoch {epoch}/{configs.epochs} | Time: {format_time(time.time() - start_time)}")
        for key in running_result:
            avg_value = running_result[key] / batch_count
            logger.info(f"  [Train] {key}: {avg_value:.6f}")
            writer.add_scalar(f"Train/{key}", avg_value, epoch)
            loss_record_train[key].append(float(avg_value))
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        trainer.save_checkpoint(epoch, checkpoint_path)

        if hasattr(configs, 'validation') and configs.validation.enable:
            start_time = time.time()
            running_result_val = define_result_dict(configs, 'value')
            images_dir = os.path.join(results_dir, 'images_val_deepfakes')
            os.makedirs(images_dir, exist_ok=True)
            save_iters = np.random.choice(np.arange(len(val_loader)), size=configs.validation.save_count, replace=False)
            save_imgs = None
            batch_count_val = 0
            for imgs, watermarks in tqdm(val_loader):
                result, output_lst = trainer.val_batch_deepfake(imgs, watermarks)
                for key in running_result_val:
                    running_result_val[key] += float(result[key])

                if batch_count_val in save_iters:

                    if save_imgs is None:
                        save_imgs = get_random_images(output_lst[0], output_lst[1], output_lst[2])
                    else:
                        save_imgs = concatenate_images(save_imgs, output_lst[0], output_lst[1], output_lst[2])
                
                batch_count_val += 1

            logger.info(f"Validation finished in {format_time(time.time() - start_time)}.")
            for key in running_result_val:
                avg_value_val = running_result_val[key] / batch_count_val
                logger.info(f"  [Val] {key}: {avg_value_val:.6f}")
                writer.add_scalar(f"Validation/{key}", avg_value_val, epoch)
                loss_record_val[key].append(float(avg_value_val))
            
            if save_imgs is not None:
                logger.info(f"Save the verification picture to {images_dir}/epoch-{epoch}.png")
                save_images(save_imgs, epoch, images_dir, resize_to=(configs.img_size, configs.img_size), manipulated_is_denormalized=False)
    writer.close()
    logger.info("--- Fine tuning completed ---")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    """
    Main Entry Point.
    Parses command-line arguments to determine the execution mode (train, tune, or test),
    sets global random seeds for reproducibility, and dispatches control to the appropriate function.
    """
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--res', type=int, default=256, choices=[128, 256], 
                        help='Image resolution (128 or 256). Default is 256.')

    parser = argparse.ArgumentParser(description="MultiFunction Watermarking Model Training and Testing")
    
    subparsers = parser.add_subparsers(dest='mode', help='Select the mode to run', required=True)
    
    parser_train_common = subparsers.add_parser('train_distortions', parents=[parent_parser], help='Run pre-training with common manipulations.')
    
    parser_tune_deepfake = subparsers.add_parser('tune_deepfakes', parents=[parent_parser], help='Run fine-tuning with deepfake manipulations.')
    
    parser_test_unified = subparsers.add_parser('test', parents=[parent_parser], help='Run the unified test for invisibility, common noise, and deepfake robustness.')

    final_args = parser.parse_args()

    config_file_path = config_map[final_args.mode]
    temp_configs = Config()
    if os.path.exists(config_file_path):
        temp_configs.load_config_file(config_file_path)
        if hasattr(temp_configs, 'seed'):
            set_seed(temp_configs.seed)
            startup_messages.append(f"The global random seed has been set to: {temp_configs.seed}")
        else:
            default_seed = 42
            set_seed(default_seed)
            startup_messages.append(f"Warning: The' seed' field was not found in the configuration file. Default seed used: {default_seed}")
            
    results_dir = f'./results/{final_args.res}_152'
    os.makedirs(results_dir, exist_ok=True)

    log_filename = os.path.join(results_dir, f'run_{final_args.mode}.log')
    
    if final_args.mode == 'test':
        log_filename = os.path.join(results_dir, 'run_unified_test.log')

    logger = setup_logger(log_file=log_filename)

    for msg in startup_messages:
        logger.info(msg)

    # Dispatch execution based on the selected mode
    if final_args.mode == 'train_distortions':
        train_distortions(logger, res=final_args.res)
    elif final_args.mode == 'tune_deepfakes':
        tune_deepfakes(logger, res=final_args.res)
    elif final_args.mode == 'test':
        run_unified_test(logger, res=final_args.res)
    else:
        parser.print_help()
