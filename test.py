import os
import torch
from tqdm import tqdm

from utils import Config, make_loader, save_image_test_distortions, save_image_test_deepfakes, save_image_test_distortions_batch, save_image_test_deepfakes_batch, update_config_resolution
from tester import TesterImg

def run_unified_test(logger, res=256):
    """
    Executes the comprehensive unified testing pipeline.
    This function evaluates the model's performance across three key dimensions:
    1. Watermark Invisibility (PSNR/SSIM).
    2. Robustness against common image distortions.
    3. Robustness against deepfake manipulations.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    configs = Config()
    configs.load_config_file('./configurations/test.yaml')
    update_config_resolution(configs, res)
    results_dir = f'./results/{res}_152'
    logger.info("--- Start unified and comprehensive testing ---")
    logger.info("--- Current test configuration ---")
    for key, value in configs.get_items():
        logger.info(f"{key}: {value}")
    logger.info("----------------------------")

    tester = TesterImg(configs, device, logger)
    # Load the specific checkpoint derived from the Deepfake fine-tuning stage.
    # This model is expected to be robust against both common noises (from pre-training) 
    # and deepfake attacks (from fine-tuning).
    checkpoint_path = os.path.join(configs.weight_path, 'checkpoints_deepfakes', f'checkpoint_epoch_{configs.epoch}.pth')
    logger.info(f"Load the fine-tuned model for unified testing: {checkpoint_path}")
    tester.load_checkpoint_for_test(checkpoint_path)

    test_loader = make_loader(configs, model_mode='test', shuffle=False)
    batch_count = len(test_loader)

    logger.info("\n--- [Part 1] Start testing the robustness of common distortions and the invisibility of watermark ---")
    
    # Iterate through the test dataset to evaluate standard robustness and visual quality.
    # For 'Identity()', we calculate PSNR/SSIM (Invisibility).
    # For other distortions (JPEG, Resize, etc.), we calculate AED/BER (Robustness).
    common_manipulation_lst = configs.common_manipulation_layers
    aed_dict_common = {m: 0.0 for m in common_manipulation_lst}
    ber_dict_common = {m: 0.0 for m in common_manipulation_lst}
    total_psnr = 0.0
    total_ssim = 0.0

    for batch_idx, (imgs, watermarks) in enumerate(tqdm(test_loader, desc="Testing Distortions Manipulations")):
        batch_visuals_list_for_grid = []

        for manipulation in common_manipulation_lst:
            if manipulation == 'Identity()':
                psnr, ssim, aed, ber, visuals = tester.test_one_manipulation(imgs, watermarks, manipulation)
                total_psnr += float(psnr)
                total_ssim += float(ssim)
                aed_dict_common[manipulation] += float(aed)
                ber_dict_common[manipulation] += float(ber)
            else:
                aed, ber, visuals = tester.test_one_manipulation(imgs, watermarks, manipulation)
                aed_dict_common[manipulation] += float(aed)
                ber_dict_common[manipulation] += float(ber)

            if hasattr(configs, 'save_samples') and hasattr(configs.save_samples, 'common') and configs.save_samples.common:
                save_image_test_distortions(
                    original_images=visuals['original'],
                    watermarked_images=visuals['watermarked'],
                    manipulated_images=visuals['manipulated'],
                    gt_watermarks=watermarks,
                    manipulation_name=manipulation,
                    batch_idx=batch_idx + 1,
                    configs=configs,
                    save_folder=os.path.join(results_dir, 'images_test/distortions')
                )

            if hasattr(configs, 'save_batches') and hasattr(configs.save_batches, 'common') and configs.save_batches.common:
                batch_visuals_list_for_grid.append(visuals)

        if hasattr(configs, 'save_batches') and hasattr(configs.save_batches, 'common') and configs.save_batches.common:
                save_image_test_distortions_batch(
                    batch_visuals=batch_visuals_list_for_grid,
                    gt_watermarks=watermarks,
                    manipulation_names=common_manipulation_lst,
                    batch_idx=batch_idx + 1,
                    configs=configs,
                    save_folder=os.path.join(results_dir, 'images_test_batch/distortions')
                )

    logger.info("\n--- [Part 2] Start testing the robustness of deepfake manipulations ---")
    
    running_metrics_deepfake = {}

    # Evaluate robustness against various Deepfake models (SimSwap, UniFace, etc.)
    # The 'tester.run_deepfake_tests' method handles the inference for all enabled deepfake models 
    # defined in the configuration.    
    for batch_idx, (imgs, watermarks) in enumerate(tqdm(test_loader, desc="Testing Deepfakes Manipulations")):
        results, visuals = tester.run_deepfake_tests(imgs, watermarks)
        for name, metrics in results.items():
            if name not in running_metrics_deepfake:
                running_metrics_deepfake[name] = {'landmark_aed': 0.0, 'id_ber': 0.0}
            running_metrics_deepfake[name]['landmark_aed'] += float(metrics['landmark_aed'])
            running_metrics_deepfake[name]['id_ber'] += float(metrics['id_ber'])
            
        if hasattr(configs, 'save_samples') and hasattr(configs.save_samples, 'deepfake') and configs.save_samples.deepfake:
            for name, visual_data in visuals.items():
                save_image_test_deepfakes(
                    original_images=visual_data['original'],
                    watermarked_images=visual_data['watermarked'],
                    manipulated_images=visual_data['manipulated'],
                    gt_watermarks=visual_data['gt_watermarks'],
                    deepfake_name=name,
                    batch_idx=batch_idx + 1,
                    configs=configs,
                    save_folder=os.path.join(results_dir, 'images_test/deepfakes')
                )

        if hasattr(configs, 'save_batches') and hasattr(configs.save_batches, 'deepfake') and configs.save_batches.deepfake:
            active_deepfake_names = list(visuals.keys())
            if active_deepfake_names:
                save_image_test_deepfakes_batch(
                    batch_visuals=visuals,
                    gt_watermarks=watermarks,
                    deepfake_names=active_deepfake_names,
                    batch_idx=batch_idx + 1,
                    configs=configs,
                    save_folder=os.path.join(results_dir, 'images_test_batch/deepfakes')
                )

    logger.info("\n--- [Final Results] Unified summary of comprehensive test results ---")
    
    logger.info("\n------ Watermark invisibility index ------")
    # Calculate and log average perceptual quality metrics across the entire test set.
    avg_psnr = total_psnr / batch_count
    avg_ssim = total_ssim / batch_count
    logger.info(f"Avg PSNR: {avg_psnr:.4f}")
    logger.info(f"Avg SSIM: {avg_ssim:.4f}")

    logger.info("\n------ General noise robustness-Landmark AED (Average Euclidean Distance) ------")
    total_aed_common = 0
    for m in common_manipulation_lst:
        avg_aed = aed_dict_common[m] / batch_count
        total_aed_common += avg_aed
        logger.info(f"{m:<25}: {avg_aed:.4f}")
    avg_total_aed = total_aed_common / len(common_manipulation_lst)
    logger.info(f"--- Avg Landmark AED: {avg_total_aed:.4f} ---")

    logger.info("\n------ General noise robustness-ID BER (Bit Error Rate) ------")
    total_ber_common = 0
    for m in common_manipulation_lst:
        avg_ber = ber_dict_common[m] / batch_count
        total_ber_common += avg_ber
        logger.info(f"{m:<25}: {avg_ber:.6f}")
    avg_total_ber = total_ber_common / len(common_manipulation_lst)
    logger.info(f"--- Avg ID BER: {avg_total_ber:.6f} ---")

    logger.info("\n------ Robustness of Deepfake model ------")
    if not running_metrics_deepfake:
        logger.warning("No Deepfake tests are enabled in test.yaml")
    else:
        total_aed_deepfake = 0
        total_ber_deepfake = 0
        num_deepfake_models = len(running_metrics_deepfake)

        for name, total_metrics in running_metrics_deepfake.items():
            avg_aed = total_metrics['landmark_aed'] / batch_count
            avg_ber = total_metrics['id_ber'] / batch_count

            total_aed_deepfake += avg_aed
            total_ber_deepfake += avg_ber
            
            logger.info(f"--- Robustness against {name} ---")
            logger.info(f"  Landmark AED: {avg_aed:.4f}")
            logger.info(f"  ID BER:       {avg_ber:.6f}")
        
        if num_deepfake_models > 0:
            avg_total_aed_deepfake = total_aed_deepfake / num_deepfake_models
            avg_total_ber_deepfake = total_ber_deepfake / num_deepfake_models
            logger.info(f"--- Avg Landmark AED: {avg_total_aed_deepfake:.4f} ---")
            logger.info(f"--- Avg ID BER: {avg_total_ber_deepfake:.6f} ---")