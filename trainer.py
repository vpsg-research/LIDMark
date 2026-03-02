import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
import os
import random

from utils import calculate_metrics
from model.lidmark import LIDMarkEncoder, FHD, LIDMark
from model.discriminator import Discriminator
from model.distortions import DistortionSimulator
from model.losses import LandmarkL2Loss, PSNRLoss, SSIMLoss


class TrainerImg:
    """
    Main trainer class managing the training and validation lifecycles.
    It handles model initialization, data flow, loss computation, and backpropagation 
    for both the pre-training (common distortions) and fine-tuning (deepfake Manipulations) phases.
    """
    def __init__(self, configs, device, logger):
        super(TrainerImg, self).__init__()
        self.configs = configs
        self.logger = logger

        self.img_size = configs.img_size
        self.wm_len = configs.watermark_length
        self.enc_c = configs.encoder_channels
        self.enc_blocks = configs.encoder_blocks
        self.dec_c = configs.decoder_channels
        self.dec_blocks = configs.decoder_blocks
        self.dis_c = configs.discriminator_channels
        self.dis_blocks = configs.discriminator_blocks

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs
        self.lr = configs.lr
        self.device = device

        # Initialize core models: Encoder for embedding, Decoder (FHD) for extraction.
        # Support initializing them separately or as a unified LIDMark wrapper.
        if self.configs.sep_model:
            self.logger.info('Initializing encoder and decoder separately.')
            self.encoder = LIDMarkEncoder(self.img_size, self.enc_c, self.enc_blocks, self.wm_len).to(self.device)
            self.decoder = FHD(self.img_size, self.dec_c, self.dec_blocks, self.wm_len).to(self.device)
        else:
            self.logger.info('Initializing encoder and decoder together.')
            self.model = LIDMark(
                self.img_size, self.enc_c, self.enc_blocks, self.dec_c, self.dec_blocks, self.wm_len, self.device,
                self.configs.manipulation_layers).to(self.device)
        self.discriminator = Discriminator(self.dis_c, self.dis_blocks).to(self.device)

        if self.configs.manipulation_mode == 'common':
            if self.configs.sep_model:
                self._init_common_manipulation()
        elif self.configs.manipulation_mode == 'deepfake':
            self._init_deepfake_manipulations()
            seed = self.configs.seed if hasattr(self.configs, 'seed') else 42
            self.deepfake_settings = {}
            if hasattr(configs, 'deepfake_manipulation_layers'):
                for layer in configs.deepfake_manipulation_layers:
                    self.deepfake_settings.update(layer)

            if self.deepfake_settings.get('simswap'):
                from model.deepfakes import SimSwapModel
                self.simswap = SimSwapModel(self.img_size, mode='train').to(self.device)
            if self.deepfake_settings.get('uniface'):
                from model.deepfakes import UniFaceModel
                self.uniface = UniFaceModel(self.device, mode='train').to(self.device)
            if self.deepfake_settings.get('cscs'):
                from model.deepfakes import CSCSModel
                self.cscs = CSCSModel(self.device, mode='train').to(self.device)
            if self.deepfake_settings.get('stargan_v2'):
                from model.deepfakes import StarGANModel
                self.stargan_v2 = StarGANModel(self.img_size, mode='train', seed=seed).to(self.device)
            if self.deepfake_settings.get('infoswap'):
                from model.deepfakes import InfoSwapModel
                self.infoswap = InfoSwapModel(self.device).to(self.device)
        else:
            raise Exception('Manipulation mode must be one of "common" and "deepfake".')

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1 and self.device != 'cpu':
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs ...")
            if self.configs.sep_model:
                self.encoder = nn.DataParallel(self.encoder)
                self.decoder = nn.DataParallel(self.decoder)
            else:
                self.model = nn.DataParallel(self.model)
            self.discriminator = nn.DataParallel(self.discriminator)
            if self.configs.manipulation_mode == 'common':
                if self.configs.sep_model:
                    self._common_manipulation_multi_gpu()
            elif self.configs.manipulation_mode == 'deepfake':
                self._deepfake_manipulation_multi_gpu()
            else:
                raise Exception('Manipulation mode must be one of "common" and "deepfake".')

        self.labels_raw = torch.full((self.batch_size, 1), 1, dtype=torch.float, device=self.device)
        self.labels_wmd = torch.full((self.batch_size, 1), 0, dtype=torch.float, device=self.device)

        if self.configs.sep_model:
            self.opt_model = Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr)
            self.opt_encoder = Adam(self.encoder.parameters(), lr=self.lr)
            self.opt_decoder = Adam(self.decoder.parameters(), lr=self.lr)
        else:
            self.opt_model = Adam(self.model.parameters(), lr=self.lr)
        self.opt_discriminator = Adam(self.discriminator.parameters(), lr=self.lr)

        # Initialize Loss Functions:
        # - BCE: For Discriminator and ID classification.
        # - MSE: For image reconstruction.
        # - LandmarkL2: For geometric alignment of recovered landmarks.
        # - PSNR/SSIM: For perceptual quality assessment.
        self.criterion_BCE = nn.BCEWithLogitsLoss().to(self.device)
        self.criterion_MSE = nn.MSELoss().to(self.device)
        self.criterion_Landmark = LandmarkL2Loss().to(self.device)
        self.criterion_PSNR = PSNRLoss(max_val=1.0).to(self.device)
        self.criterion_SSIM = SSIMLoss(window_size=5, reduction='mean').to(self.device)

        self.enc_w = configs.encoder_weight
        self.landmark_w = configs.landmark_loss_weight
        self.id_w = configs.id_loss_weight
        self.dis_w = configs.discriminator_weight

        self.norm_imgnet = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.denorm_imgnet = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
        self.norm = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        self.denorm = transforms.Normalize(
            mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
            std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
        )

    def _init_common_manipulation(self):
        self.common_manipulation = DistortionSimulator(self.configs.manipulation_layers).to(self.device)

    def _init_deepfake_manipulations(self):
        self.simswap = None
        self.uniface = None
        self.cscs = None
        self.stargan_v2 = None
        self.infoswap = None

        self.gen_w = self.configs.generative_weight

    def _common_manipulation_multi_gpu(self):
        self.common_manipulation = nn.DataParallel(self.common_manipulation)

    def _deepfake_manipulation_multi_gpu(self):
        if self.simswap: self.simswap = nn.DataParallel(self.simswap)
        if self.uniface: self.uniface = nn.DataParallel(self.uniface)
        if self.cscs: self.cscs = nn.DataParallel(self.cscs)
        if self.stargan_v2: self.stargan_v2 = nn.DataParallel(self.stargan_v2)
        if self.infoswap: self.infoswap = nn.DataParallel(self.infoswap)

    def train_batch_common(self, imgs, wms):
        """
        Executes a single training step for the pre-training phase (Common Distortions).
        Updates the Discriminator to distinguish watermarked images and the Generator (Encoder/Decoder) 
        to minimize perceptual loss and maximize watermark recovery.
        """
        self.model.train()
        self.discriminator.train()

        with torch.enable_grad():
            imgs, wms = imgs.to(self.device), wms.to(self.device)
            imgs_wm, manipulated_wm_img, wms_recover_landmarks, wms_recover_id = self.model(imgs, wms)

            self.opt_discriminator.zero_grad()
            d_label_raw = self.discriminator(imgs)
            d_raw_loss = self.criterion_BCE(d_label_raw, self.labels_raw[:d_label_raw.shape[0]])
            d_raw_loss.backward()
            d_label_wmd = self.discriminator(imgs_wm.detach())
            d_wmd_loss = self.criterion_BCE(d_label_wmd, self.labels_wmd[:d_label_wmd.shape[0]])
            d_wmd_loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), 5)
            self.opt_discriminator.step()

            self.opt_model.zero_grad()
            g_label_decoded = self.discriminator(imgs_wm)
            g_loss_dis = self.criterion_BCE(g_label_decoded, self.labels_raw[:g_label_decoded.shape[0]])
            g_loss_enc = self.criterion_MSE(imgs_wm, imgs)
            gt_landmarks = wms[:, :136]
            gt_id = wms[:, 136:]

            g_loss_dec_landmarks = self.criterion_Landmark(wms_recover_landmarks, gt_landmarks)

            gt_id_mapped = (gt_id + 1) / 2
            g_loss_dec_id = self.criterion_BCE(wms_recover_id, gt_id_mapped)
            
            # Calculate Generator Loss:
            # Weighted sum of Encoder reconstruction loss, Decoder recovery loss (Landmarks + ID), 
            # and Adversarial loss to fool the discriminator.
            g_loss_dec = self.landmark_w * g_loss_dec_landmarks + self.id_w * g_loss_dec_id
            g_loss_total = self.enc_w * g_loss_enc + g_loss_dec + self.dis_w * g_loss_dis
            g_loss_total.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.opt_model.step()

            psnr = self.criterion_PSNR(self.denorm(imgs_wm.detach()), self.denorm(imgs.detach())) * (-1)
            ssim = 1 - 2 * self.criterion_SSIM(self.denorm(imgs_wm.detach()), self.denorm(imgs.detach()))
            landmark_aed, id_ber = calculate_metrics(wms, wms_recover_landmarks, wms_recover_id, self.img_size)

            result = {
                'landmark_aed': landmark_aed, 'id_ber': id_ber, 'psnr': psnr, 'ssim': ssim, 'g_loss': g_loss_total,
                'g_loss_enc': g_loss_enc, 'g_loss_dec': g_loss_dec, 'g_loss_dec_landmark': g_loss_dec_landmarks, 'g_loss_dec_id': g_loss_dec_id, 'g_loss_dis': g_loss_dis,
                'd_loss_raw': d_raw_loss, 'd_loss_wmd': d_wmd_loss
            }
            return result

    def val_batch_common(self, imgs, wms):
        self.model.eval()
        self.discriminator.eval()

        with torch.no_grad():
            imgs, wms = imgs.to(self.device), wms.to(self.device)
            imgs_wm, manipulated_wm_img, wms_recover_landmarks, wms_recover_id = self.model(imgs, wms)

            d_label_raw = self.discriminator(imgs)
            d_raw_loss = self.criterion_BCE(d_label_raw, self.labels_raw[:d_label_raw.shape[0]])
            d_label_wmd = self.discriminator(imgs_wm.detach())
            d_wmd_loss = self.criterion_BCE(d_label_wmd, self.labels_wmd[:d_label_wmd.shape[0]])

            g_label_decoded = self.discriminator(imgs_wm)
            g_loss_dis = self.criterion_BCE(g_label_decoded, self.labels_raw[:g_label_decoded.shape[0]])
            g_loss_enc = self.criterion_MSE(imgs_wm, imgs)
            gt_landmarks = wms[:, :136]
            gt_id = wms[:, 136:]

            g_loss_dec_landmarks = self.criterion_Landmark(wms_recover_landmarks, gt_landmarks)

            gt_id_mapped = (gt_id + 1) / 2
            g_loss_dec_id = self.criterion_BCE(wms_recover_id, gt_id_mapped)
            
            g_loss_dec = self.landmark_w * g_loss_dec_landmarks + self.id_w * g_loss_dec_id
            g_loss_total = self.enc_w * g_loss_enc + g_loss_dec + self.dis_w * g_loss_dis

            psnr = self.criterion_PSNR(self.denorm(imgs_wm.detach()), self.denorm(imgs.detach())) * (-1)
            ssim = 1 - 2 * self.criterion_SSIM(self.denorm(imgs_wm.detach()), self.denorm(imgs.detach()))
            landmark_aed, id_ber = calculate_metrics(wms, wms_recover_landmarks, wms_recover_id, self.img_size)

            result = {
                'landmark_aed': landmark_aed, 'id_ber': id_ber, 'psnr': psnr, 'ssim': ssim, 'g_loss': g_loss_total,
                'g_loss_enc': g_loss_enc, 'g_loss_dec': g_loss_dec, 'g_loss_dec_landmark': g_loss_dec_landmarks, 'g_loss_dec_id': g_loss_dec_id, 'g_loss_dis': g_loss_dis,
                'd_loss_raw': d_raw_loss, 'd_loss_wmd': d_wmd_loss
            }
            return result, [imgs, imgs_wm, manipulated_wm_img]

    def train_batch_deepfake(self, imgs, wms):
        """
        Orchestrates the fine-tuning phase by randomly selecting one of the enabled 
        Deepfake models (e.g., SimSwap, StarGAN) to simulate a real-world attack 
        during the current training iteration.
        """
        options = []
        if self.deepfake_settings.get('simswap'):
            options.append('simswap')
        if self.deepfake_settings.get('uniface'):
            options.append('uniface')
        if self.deepfake_settings.get('cscs'):
            options.append('cscs')
        if self.deepfake_settings.get('stargan_v2'):
            options.append('stargan_v2')
        if self.deepfake_settings.get('infoswap'):
            options.append('infoswap')

        if not options:
            raise ValueError("At least one deepfake manipulation must be enabled for tuning.")

        choice = random.choice(options)
        
        if choice == 'simswap':
            return self.tune_batch_simswap(imgs, wms)
        elif choice == 'uniface':
            return self.tune_batch_uniface(imgs, wms)
        elif choice == 'cscs':
            return self.tune_batch_cscs(imgs, wms)
        elif choice == 'stargan_v2':
            return self.tune_batch_stargan(imgs, wms)
        elif choice == 'infoswap':
            return self.tune_batch_infoswap(imgs, wms)

    def val_batch_deepfake(self, imgs, wms):
        options = []
        if self.deepfake_settings.get('simswap'):
            options.append('simswap')
        if self.deepfake_settings.get('uniface'):
            options.append('uniface')
        if self.deepfake_settings.get('cscs'):
            options.append('cscs')
        if self.deepfake_settings.get('stargan_v2'):
            options.append('stargan_v2')
        if self.deepfake_settings.get('infoswap'):
            options.append('infoswap')

        if not options:
            self.encoder.eval()
            self.decoder.eval()
            with torch.no_grad():
                imgs_wm = self.encoder(imgs.to(self.device), wms.to(self.device))
            return {}, [imgs, imgs_wm, imgs_wm]

        choice = random.choice(options)

        if choice == 'simswap':
            return self.val_batch_simswap(imgs, wms)
        elif choice == 'uniface':
            return self.val_batch_uniface(imgs, wms)
        elif choice == 'cscs':
            return self.val_batch_cscs(imgs, wms)
        elif choice == 'stargan_v2':
            return self.val_batch_stargan(imgs, wms)
        elif choice == 'infoswap':
            return self.val_batch_infoswap(imgs, wms)

    def tune_batch_simswap(self, imgs, wms):
        """
        Performs a fine-tuning step using the SimSwap model.
        It generates a watermarked image, applies face swapping using a reference identity, 
        and trains the decoder to recover the original watermark from the face-swapped result.
        """
        self.encoder.train()
        self.decoder.train()
        self.simswap.eval()
        self.discriminator.train()

        with torch.enable_grad():
            imgs, wms = imgs.to(self.device), wms.to(self.device)  

            imgs_wm = self.encoder(imgs, wms)
            identity_img = torch.roll(imgs, 1, 0)
            # Apply SimSwap to both watermarked and original images
            swapped_img_wm = self.simswap([imgs_wm, identity_img, self.device])
            swapped_img = self.simswap([imgs, identity_img, self.device])
            
            wms_recover_landmarks, wms_recover_id = self.decoder(swapped_img_wm)

            self.opt_discriminator.zero_grad()
            d_label_raw = self.discriminator(imgs)
            d_raw_loss = self.criterion_BCE(d_label_raw, self.labels_raw[:d_label_raw.shape[0]])
            d_raw_loss.backward()
            d_label_wmd = self.discriminator(imgs_wm.detach())
            d_wmd_loss = self.criterion_BCE(d_label_wmd, self.labels_wmd[:d_label_wmd.shape[0]])
            d_wmd_loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), 5)
            self.opt_discriminator.step()

            self.opt_model.zero_grad()
            g_label_decoded = self.discriminator(imgs_wm)
            g_loss_dis = self.criterion_BCE(g_label_decoded, self.labels_raw[:g_label_decoded.shape[0]])
            g_loss_enc = self.criterion_MSE(imgs_wm, imgs)
            gt_landmarks = wms[:, :136]
            gt_id = wms[:, 136:]

            g_loss_dec_landmarks = self.criterion_Landmark(wms_recover_landmarks, gt_landmarks)
            gt_id_mapped = (gt_id + 1) / 2
            g_loss_dec_id = self.criterion_BCE(wms_recover_id, gt_id_mapped)
            
            g_loss_dec = self.landmark_w * g_loss_dec_landmarks + self.id_w * g_loss_dec_id
            
            # Generative Consistency Loss:
            # Ensures that the watermarked image, when swapped, produces a result 
            # visually similar to swapping the original non-watermarked image.
            g_loss_gen = self.criterion_MSE(swapped_img_wm, swapped_img)
            wms_identity_landmarks, wms_identity_id = self.decoder(imgs_wm)
            g_loss_id_landmarks = self.criterion_Landmark(wms_identity_landmarks, gt_landmarks)
            g_loss_id_id = self.criterion_BCE(wms_identity_id, gt_id_mapped)
            g_loss_id = g_loss_id_landmarks + g_loss_id_id

            g_loss_total = (self.dis_w * g_loss_dis + self.enc_w * g_loss_enc + g_loss_dec + 
                            self.gen_w * g_loss_gen + g_loss_id * self.landmark_w)
            g_loss_total.backward()

            nn.utils.clip_grad_norm_(self.encoder.parameters(), 5)
            nn.utils.clip_grad_norm_(self.decoder.parameters(), 5)
            self.opt_model.step()

            psnr = self.criterion_PSNR(self.denorm(imgs_wm.detach()), self.denorm(imgs.detach())) * (-1)
            ssim = 1 - 2 * self.criterion_SSIM(self.denorm(imgs_wm.detach()), self.denorm(imgs.detach()))
            landmark_aed, id_ber = calculate_metrics(wms, wms_recover_landmarks, wms_recover_id, self.img_size)

            result = {
                'landmark_aed': landmark_aed, 'id_ber': id_ber, 'psnr': psnr, 'ssim': ssim,
                'g_loss': g_loss_total, 'g_loss_enc': g_loss_enc, 'g_loss_dec': g_loss_dec,
                'g_loss_dec_landmark': g_loss_dec_landmarks, 'g_loss_dec_id': g_loss_dec_id,
                'g_loss_gen': g_loss_gen, 'g_loss_id': g_loss_id, 'g_loss_dis': g_loss_dis,
                'd_loss_raw': d_raw_loss, 'd_loss_wmd': d_wmd_loss
            }
            return result

    def val_batch_simswap(self, imgs, wms):
        self.encoder.eval()
        self.decoder.eval()
        self.simswap.eval()

        with torch.no_grad():
            imgs, wms = imgs.to(self.device), wms.to(self.device)

            imgs_wm = self.encoder(imgs, wms)
            identity_img = torch.roll(imgs, 1, 0)

            swapped_img_wm = self.simswap([imgs_wm, identity_img, self.device])
            swapped_img = self.simswap([imgs, identity_img, self.device])
            
            wms_recover_landmarks, wms_recover_id = self.decoder(swapped_img_wm)

            d_label_raw = self.discriminator(imgs)
            d_raw_loss = self.criterion_BCE(d_label_raw, self.labels_raw[:d_label_raw.shape[0]])
            d_label_wmd = self.discriminator(imgs_wm.detach())
            d_wmd_loss = self.criterion_BCE(d_label_wmd, self.labels_wmd[:d_label_wmd.shape[0]])

            g_label_decoded = self.discriminator(imgs_wm)
            g_loss_dis = self.criterion_BCE(g_label_decoded, self.labels_raw[:g_label_decoded.shape[0]])
            g_loss_enc = self.criterion_MSE(imgs_wm, imgs)
            gt_landmarks = wms[:, :136]
            gt_id = wms[:, 136:]
            g_loss_dec_landmarks = self.criterion_Landmark(wms_recover_landmarks, gt_landmarks)
            gt_id_mapped = (gt_id + 1) / 2
            g_loss_dec_id = self.criterion_BCE(wms_recover_id, gt_id_mapped)
            g_loss_dec = self.landmark_w * g_loss_dec_landmarks + self.id_w * g_loss_dec_id
            
            g_loss_gen = self.criterion_MSE(swapped_img_wm, swapped_img)
            wms_identity_landmarks, wms_identity_id = self.decoder(imgs_wm)
            g_loss_id_landmarks = self.criterion_Landmark(wms_identity_landmarks, gt_landmarks)
            g_loss_id_id = self.criterion_BCE(wms_identity_id, gt_id_mapped)
            g_loss_id = g_loss_id_landmarks + g_loss_id_id

            g_loss_total = (self.dis_w * g_loss_dis + self.enc_w * g_loss_enc + g_loss_dec + 
                            self.gen_w * g_loss_gen + g_loss_id * self.landmark_w)

            psnr = self.criterion_PSNR(self.denorm(imgs_wm.detach()), self.denorm(imgs.detach())) * (-1)
            ssim = 1 - 2 * self.criterion_SSIM(self.denorm(imgs_wm.detach()), self.denorm(imgs.detach()))
            landmark_aed, id_ber = calculate_metrics(wms, wms_recover_landmarks, wms_recover_id, self.img_size)

            result = {
                'landmark_aed': landmark_aed, 'id_ber': id_ber, 'psnr': psnr, 'ssim': ssim,
                'g_loss': g_loss_total, 'g_loss_enc': g_loss_enc, 'g_loss_dec': g_loss_dec,
                'g_loss_dec_landmark': g_loss_dec_landmarks, 'g_loss_dec_id': g_loss_dec_id,
                'g_loss_gen': g_loss_gen, 'g_loss_id': g_loss_id, 'g_loss_dis': g_loss_dis,
                'd_loss_raw': d_raw_loss, 'd_loss_wmd': d_wmd_loss
            }
            return result, [imgs, imgs_wm, swapped_img_wm]

    def tune_batch_uniface(self, imgs, wms):
        self.encoder.train()
        self.decoder.train()
        self.uniface.eval()
        self.discriminator.train()

        with torch.enable_grad():
            imgs, wms = imgs.to(self.device), wms.to(self.device)
            imgs_wm = self.encoder(imgs, wms)
            identity_img = torch.roll(imgs, 1, 0)
            
            swapped_img_wm = self.uniface([imgs_wm, identity_img, self.device])
            swapped_img_clean = self.uniface([imgs, identity_img, self.device])
            
            wms_recover_landmarks, wms_recover_id = self.decoder(swapped_img_wm)

            self.opt_discriminator.zero_grad()
            d_label_raw = self.discriminator(imgs)
            d_raw_loss = self.criterion_BCE(d_label_raw, self.labels_raw[:d_label_raw.shape[0]])
            d_raw_loss.backward()
            d_label_wmd = self.discriminator(imgs_wm.detach())
            d_wmd_loss = self.criterion_BCE(d_label_wmd, self.labels_wmd[:d_label_wmd.shape[0]])
            d_wmd_loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), 5)
            self.opt_discriminator.step()

            self.opt_model.zero_grad()
            g_label_decoded = self.discriminator(imgs_wm)
            g_loss_dis = self.criterion_BCE(g_label_decoded, self.labels_raw[:g_label_decoded.shape[0]])
            g_loss_enc = self.criterion_MSE(imgs_wm, imgs)

            gt_landmarks = wms[:, :136]
            gt_id = wms[:, 136:]
            g_loss_dec_landmarks = self.criterion_Landmark(wms_recover_landmarks, gt_landmarks)
            gt_id_mapped = (gt_id + 1) / 2
            g_loss_dec_id = self.criterion_BCE(wms_recover_id, gt_id_mapped)
            g_loss_dec = self.landmark_w * g_loss_dec_landmarks + self.id_w * g_loss_dec_id

            g_loss_gen = self.criterion_MSE(swapped_img_wm, swapped_img_clean.detach())
            
            wms_identity_landmarks, wms_identity_id = self.decoder(imgs_wm)
            g_loss_id_landmarks = self.criterion_Landmark(wms_identity_landmarks, gt_landmarks)
            g_loss_id_id = self.criterion_BCE(wms_identity_id, gt_id_mapped)
            g_loss_id = g_loss_id_landmarks + g_loss_id_id

            g_loss_total = (self.dis_w * g_loss_dis + self.enc_w * g_loss_enc + g_loss_dec + 
                            self.gen_w * g_loss_gen + g_loss_id * self.landmark_w)
            g_loss_total.backward()

            nn.utils.clip_grad_norm_(self.encoder.parameters(), 5)
            nn.utils.clip_grad_norm_(self.decoder.parameters(), 5)
            self.opt_model.step()

            psnr = self.criterion_PSNR(self.denorm(imgs_wm.detach()), self.denorm(imgs.detach())) * (-1)
            ssim = 1 - 2 * self.criterion_SSIM(self.denorm(imgs_wm.detach()), self.denorm(imgs.detach()))
            landmark_aed, id_ber = calculate_metrics(wms, wms_recover_landmarks, wms_recover_id, self.img_size)

            result = {
                'landmark_aed': landmark_aed, 'id_ber': id_ber, 'psnr': psnr, 'ssim': ssim,
                'g_loss': g_loss_total, 'g_loss_enc': g_loss_enc, 'g_loss_dec': g_loss_dec,
                'g_loss_dec_landmark': g_loss_dec_landmarks, 'g_loss_dec_id': g_loss_dec_id,
                'g_loss_gen': g_loss_gen, 'g_loss_id': g_loss_id, 'g_loss_dis': g_loss_dis,
                'd_loss_raw': d_raw_loss, 'd_loss_wmd': d_wmd_loss
            }
            return result

    def val_batch_uniface(self, imgs, wms):
        self.encoder.eval()
        self.decoder.eval()
        self.uniface.eval()

        with torch.no_grad():
            imgs, wms = imgs.to(self.device), wms.to(self.device)
            imgs_wm = self.encoder(imgs, wms)
            identity_img = torch.roll(imgs, 1, 0)
            
            swapped_img_wm = self.uniface([imgs_wm, identity_img, self.device])
            swapped_img_clean = self.uniface([imgs, identity_img, self.device])
            
            wms_recover_landmarks, wms_recover_id = self.decoder(swapped_img_wm)

            d_label_raw = self.discriminator(imgs)
            d_raw_loss = self.criterion_BCE(d_label_raw, self.labels_raw[:d_label_raw.shape[0]])
            d_label_wmd = self.discriminator(imgs_wm.detach())
            d_wmd_loss = self.criterion_BCE(d_label_wmd, self.labels_wmd[:d_label_wmd.shape[0]])

            g_label_decoded = self.discriminator(imgs_wm)
            g_loss_dis = self.criterion_BCE(g_label_decoded, self.labels_raw[:g_label_decoded.shape[0]])
            g_loss_enc = self.criterion_MSE(imgs_wm, imgs)
            
            gt_landmarks = wms[:, :136]
            gt_id = wms[:, 136:]
            g_loss_dec_landmarks = self.criterion_Landmark(wms_recover_landmarks, gt_landmarks)
            gt_id_mapped = (gt_id + 1) / 2
            g_loss_dec_id = self.criterion_BCE(wms_recover_id, gt_id_mapped)
            g_loss_dec = self.landmark_w * g_loss_dec_landmarks + self.id_w * g_loss_dec_id
            
            g_loss_gen = self.criterion_MSE(swapped_img_wm, swapped_img_clean.detach())
            wms_identity_landmarks, wms_identity_id = self.decoder(imgs_wm)
            g_loss_id_landmarks = self.criterion_Landmark(wms_identity_landmarks, gt_landmarks)
            g_loss_id_id = self.criterion_BCE(wms_identity_id, gt_id_mapped)
            g_loss_id = g_loss_id_landmarks + g_loss_id_id

            g_loss_total = (self.dis_w * g_loss_dis + self.enc_w * g_loss_enc + g_loss_dec + 
                            self.gen_w * g_loss_gen + g_loss_id * self.landmark_w)
            
            psnr = self.criterion_PSNR(self.denorm(imgs_wm.detach()), self.denorm(imgs.detach())) * (-1)
            ssim = 1 - 2 * self.criterion_SSIM(self.denorm(imgs_wm.detach()), self.denorm(imgs.detach()))
            landmark_aed, id_ber = calculate_metrics(wms, wms_recover_landmarks, wms_recover_id, self.img_size)

            result = {
                'landmark_aed': landmark_aed, 'id_ber': id_ber, 'psnr': psnr, 'ssim': ssim,
                'g_loss': g_loss_total, 'g_loss_enc': g_loss_enc, 'g_loss_dec': g_loss_dec,
                'g_loss_dec_landmark': g_loss_dec_landmarks, 'g_loss_dec_id': g_loss_dec_id,
                'g_loss_gen': g_loss_gen, 'g_loss_id': g_loss_id, 'g_loss_dis': g_loss_dis,
                'd_loss_raw': d_raw_loss, 'd_loss_wmd': d_wmd_loss
            }
            return result, [imgs, imgs_wm, swapped_img_wm]

    def tune_batch_cscs(self, imgs, wms):
        self.encoder.train()
        self.decoder.train()
        self.cscs.eval()
        self.discriminator.train()

        with torch.enable_grad():
            imgs, wms = imgs.to(self.device), wms.to(self.device)

            imgs_wm = self.encoder(imgs, wms)
            identity_img = torch.roll(imgs, 1, 0)

            swapped_img_wm = self.cscs([imgs_wm, identity_img, self.device])
            swapped_img = self.cscs([imgs, identity_img, self.device])
            
            wms_recover_landmarks, wms_recover_id = self.decoder(swapped_img_wm)

            self.opt_discriminator.zero_grad()
            d_label_raw = self.discriminator(imgs)
            d_raw_loss = self.criterion_BCE(d_label_raw, self.labels_raw[:d_label_raw.shape[0]])
            d_raw_loss.backward()
            d_label_wmd = self.discriminator(imgs_wm.detach())
            d_wmd_loss = self.criterion_BCE(d_label_wmd, self.labels_wmd[:d_label_wmd.shape[0]])
            d_wmd_loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), 5)
            self.opt_discriminator.step()

            self.opt_model.zero_grad()
            g_label_decoded = self.discriminator(imgs_wm)
            g_loss_dis = self.criterion_BCE(g_label_decoded, self.labels_raw[:g_label_decoded.shape[0]])
            g_loss_enc = self.criterion_MSE(imgs_wm, imgs)

            gt_landmarks = wms[:, :136]
            gt_id = wms[:, 136:]
            g_loss_dec_landmarks = self.criterion_Landmark(wms_recover_landmarks, gt_landmarks)
            gt_id_mapped = (gt_id + 1) / 2
            g_loss_dec_id = self.criterion_BCE(wms_recover_id, gt_id_mapped)
            
            g_loss_dec = self.landmark_w * g_loss_dec_landmarks + self.id_w * g_loss_dec_id
            
            g_loss_gen = self.criterion_MSE(swapped_img_wm, swapped_img)
            wms_identity_landmarks, wms_identity_id = self.decoder(imgs_wm)
            g_loss_id_landmarks = self.criterion_Landmark(wms_identity_landmarks, gt_landmarks)
            g_loss_id_id = self.criterion_BCE(wms_identity_id, gt_id_mapped)
            g_loss_id = g_loss_id_landmarks + g_loss_id_id

            g_loss_total = (self.dis_w * g_loss_dis + self.enc_w * g_loss_enc + g_loss_dec + 
                            self.gen_w * g_loss_gen + g_loss_id * self.landmark_w)
            g_loss_total.backward()

            nn.utils.clip_grad_norm_(self.encoder.parameters(), 5)
            nn.utils.clip_grad_norm_(self.decoder.parameters(), 5)
            self.opt_model.step()

            psnr = self.criterion_PSNR(self.denorm(imgs_wm.detach()), self.denorm(imgs.detach())) * (-1)
            ssim = 1 - 2 * self.criterion_SSIM(self.denorm(imgs_wm.detach()), self.denorm(imgs.detach()))
            landmark_aed, id_ber = calculate_metrics(wms, wms_recover_landmarks, wms_recover_id, self.img_size)

            result = {
                'landmark_aed': landmark_aed, 'id_ber': id_ber, 'psnr': psnr, 'ssim': ssim,
                'g_loss': g_loss_total, 'g_loss_enc': g_loss_enc, 'g_loss_dec': g_loss_dec,
                'g_loss_dec_landmark': g_loss_dec_landmarks, 'g_loss_dec_id': g_loss_dec_id,
                'g_loss_gen': g_loss_gen, 'g_loss_id': g_loss_id, 'g_loss_dis': g_loss_dis,
                'd_loss_raw': d_raw_loss, 'd_loss_wmd': d_wmd_loss
            }
            return result
    
    def val_batch_cscs(self, imgs, wms):
        self.encoder.eval()
        self.decoder.eval()
        self.cscs.eval()

        with torch.no_grad():
            imgs, wms = imgs.to(self.device), wms.to(self.device)

            imgs_wm = self.encoder(imgs, wms)
            identity_img = torch.roll(imgs, 1, 0)

            swapped_img_wm = self.cscs([imgs_wm, identity_img, self.device])
            swapped_img = self.cscs([imgs, identity_img, self.device])
            
            wms_recover_landmarks, wms_recover_id = self.decoder(swapped_img_wm)

            d_label_raw = self.discriminator(imgs)
            d_raw_loss = self.criterion_BCE(d_label_raw, self.labels_raw[:d_label_raw.shape[0]])
            d_label_wmd = self.discriminator(imgs_wm.detach())
            d_wmd_loss = self.criterion_BCE(d_label_wmd, self.labels_wmd[:d_label_wmd.shape[0]])

            g_label_decoded = self.discriminator(imgs_wm)
            g_loss_dis = self.criterion_BCE(g_label_decoded, self.labels_raw[:g_label_decoded.shape[0]])
            g_loss_enc = self.criterion_MSE(imgs_wm, imgs)
            gt_landmarks = wms[:, :136]
            gt_id = wms[:, 136:]
            g_loss_dec_landmarks = self.criterion_Landmark(wms_recover_landmarks, gt_landmarks)
            gt_id_mapped = (gt_id + 1) / 2
            g_loss_dec_id = self.criterion_BCE(wms_recover_id, gt_id_mapped)
            g_loss_dec = self.landmark_w * g_loss_dec_landmarks + self.id_w * g_loss_dec_id
            
            g_loss_gen = self.criterion_MSE(swapped_img_wm, swapped_img)
            wms_identity_landmarks, wms_identity_id = self.decoder(imgs_wm)
            g_loss_id_landmarks = self.criterion_Landmark(wms_identity_landmarks, gt_landmarks)
            g_loss_id_id = self.criterion_BCE(wms_identity_id, gt_id_mapped)
            g_loss_id = g_loss_id_landmarks + g_loss_id_id

            g_loss_total = (self.dis_w * g_loss_dis + self.enc_w * g_loss_enc + g_loss_dec + 
                            self.gen_w * g_loss_gen + g_loss_id * self.landmark_w)

            psnr = self.criterion_PSNR(self.denorm(imgs_wm.detach()), self.denorm(imgs.detach())) * (-1)
            ssim = 1 - 2 * self.criterion_SSIM(self.denorm(imgs_wm.detach()), self.denorm(imgs.detach()))
            landmark_aed, id_ber = calculate_metrics(wms, wms_recover_landmarks, wms_recover_id, self.img_size)

            result = {
                'landmark_aed': landmark_aed, 'id_ber': id_ber, 'psnr': psnr, 'ssim': ssim,
                'g_loss': g_loss_total, 'g_loss_enc': g_loss_enc, 'g_loss_dec': g_loss_dec,
                'g_loss_dec_landmark': g_loss_dec_landmarks, 'g_loss_dec_id': g_loss_dec_id,
                'g_loss_gen': g_loss_gen, 'g_loss_id': g_loss_id, 'g_loss_dis': g_loss_dis,
                'd_loss_raw': d_raw_loss, 'd_loss_wmd': d_wmd_loss
            }
            return result, [imgs, imgs_wm, swapped_img_wm]

    def tune_batch_stargan(self, imgs, wms):
        self.encoder.train()
        self.decoder.train()
        self.stargan_v2.eval()
        self.discriminator.train()

        with torch.enable_grad():
            imgs, wms = imgs.to(self.device), wms.to(self.device)
            
            imgs_wm = self.encoder(imgs, wms)
            identity_img = torch.roll(imgs, 1, 0)
            reenacted_img_wm = self.stargan_v2([imgs_wm, identity_img, self.device])
            wms_recover_landmarks, wms_recover_id = self.decoder(reenacted_img_wm)

            reenacted_img_clean = self.stargan_v2([imgs, identity_img, self.device])

            self.opt_discriminator.zero_grad()
            d_label_raw = self.discriminator(imgs)
            d_raw_loss = self.criterion_BCE(d_label_raw, self.labels_raw[:d_label_raw.shape[0]])
            d_raw_loss.backward()
            d_label_wmd = self.discriminator(imgs_wm.detach())
            d_wmd_loss = self.criterion_BCE(d_label_wmd, self.labels_wmd[:d_label_wmd.shape[0]])
            d_wmd_loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), 5)
            self.opt_discriminator.step()

            self.opt_model.zero_grad()
            g_label_decoded = self.discriminator(imgs_wm)
            g_loss_dis = self.criterion_BCE(g_label_decoded, self.labels_raw[:g_label_decoded.shape[0]])
            g_loss_enc = self.criterion_MSE(imgs_wm, imgs)

            gt_landmarks = wms[:, :136]
            gt_id = wms[:, 136:]
            g_loss_dec_landmarks = self.criterion_Landmark(wms_recover_landmarks, gt_landmarks)
            gt_id_mapped = (gt_id + 1) / 2
            g_loss_dec_id = self.criterion_BCE(wms_recover_id, gt_id_mapped)
            g_loss_dec = self.landmark_w * g_loss_dec_landmarks + self.id_w * g_loss_dec_id

            g_loss_gen = self.criterion_MSE(reenacted_img_wm, reenacted_img_clean.detach())
            
            wms_identity_landmarks, wms_identity_id = self.decoder(imgs_wm)
            g_loss_id_landmarks = self.criterion_Landmark(wms_identity_landmarks, gt_landmarks)
            g_loss_id_id = self.criterion_BCE(wms_identity_id, gt_id_mapped)
            g_loss_id = g_loss_id_landmarks + g_loss_id_id

            g_loss_total = (self.dis_w * g_loss_dis + self.enc_w * g_loss_enc + g_loss_dec + 
                            self.gen_w * g_loss_gen + g_loss_id * self.landmark_w)
            g_loss_total.backward()

            nn.utils.clip_grad_norm_(self.encoder.parameters(), 5)
            nn.utils.clip_grad_norm_(self.decoder.parameters(), 5)
            self.opt_model.step()

            psnr = self.criterion_PSNR(self.denorm(imgs_wm.detach()), self.denorm(imgs.detach())) * (-1)
            ssim = 1 - 2 * self.criterion_SSIM(self.denorm(imgs_wm.detach()), self.denorm(imgs.detach()))
            landmark_aed, id_ber = calculate_metrics(wms, wms_recover_landmarks, wms_recover_id, self.img_size)

            result = {
                'landmark_aed': landmark_aed, 'id_ber': id_ber, 'psnr': psnr, 'ssim': ssim,
                'g_loss': g_loss_total, 'g_loss_enc': g_loss_enc, 'g_loss_dec': g_loss_dec,
                'g_loss_dec_landmark': g_loss_dec_landmarks, 'g_loss_dec_id': g_loss_dec_id,
                'g_loss_gen': g_loss_gen, 'g_loss_id': g_loss_id, 'g_loss_dis': g_loss_dis,
                'd_loss_raw': d_raw_loss, 'd_loss_wmd': d_wmd_loss
            }
            return result

    def val_batch_stargan(self, imgs, wms):
        self.encoder.eval()
        self.decoder.eval()
        self.stargan_v2.eval()

        with torch.no_grad():
            imgs, wms = imgs.to(self.device), wms.to(self.device)
            imgs_wm = self.encoder(imgs, wms)
            identity_img = torch.roll(imgs, 1, 0)
            reenacted_img_wm = self.stargan_v2([imgs_wm, identity_img, self.device])
            wms_recover_landmarks, wms_recover_id = self.decoder(reenacted_img_wm)
            reenacted_img_clean = self.stargan_v2([imgs, identity_img, self.device])

            d_label_raw = self.discriminator(imgs)
            d_raw_loss = self.criterion_BCE(d_label_raw, self.labels_raw[:d_label_raw.shape[0]])
            d_label_wmd = self.discriminator(imgs_wm.detach())
            d_wmd_loss = self.criterion_BCE(d_label_wmd, self.labels_wmd[:d_label_wmd.shape[0]])

            g_label_decoded = self.discriminator(imgs_wm)
            g_loss_dis = self.criterion_BCE(g_label_decoded, self.labels_raw[:g_label_decoded.shape[0]])
            g_loss_enc = self.criterion_MSE(imgs_wm, imgs)
            
            gt_landmarks = wms[:, :136]
            gt_id = wms[:, 136:]
            g_loss_dec_landmarks = self.criterion_Landmark(wms_recover_landmarks, gt_landmarks)
            gt_id_mapped = (gt_id + 1) / 2
            g_loss_dec_id = self.criterion_BCE(wms_recover_id, gt_id_mapped)
            g_loss_dec = self.landmark_w * g_loss_dec_landmarks + self.id_w * g_loss_dec_id
            
            g_loss_gen = self.criterion_MSE(reenacted_img_wm, reenacted_img_clean.detach())
            wms_identity_landmarks, wms_identity_id = self.decoder(imgs_wm)
            g_loss_id_landmarks = self.criterion_Landmark(wms_identity_landmarks, gt_landmarks)
            g_loss_id_id = self.criterion_BCE(wms_identity_id, gt_id_mapped)
            g_loss_id = g_loss_id_landmarks + g_loss_id_id

            g_loss_total = (self.dis_w * g_loss_dis + self.enc_w * g_loss_enc + g_loss_dec + 
                            self.gen_w * g_loss_gen + g_loss_id * self.landmark_w)
            
            psnr = self.criterion_PSNR(self.denorm(imgs_wm.detach()), self.denorm(imgs.detach())) * (-1)
            ssim = 1 - 2 * self.criterion_SSIM(self.denorm(imgs_wm.detach()), self.denorm(imgs.detach()))
            landmark_aed, id_ber = calculate_metrics(wms, wms_recover_landmarks, wms_recover_id, self.img_size)

            result = {
                'landmark_aed': landmark_aed, 'id_ber': id_ber, 'psnr': psnr, 'ssim': ssim,
                'g_loss': g_loss_total, 'g_loss_enc': g_loss_enc, 'g_loss_dec': g_loss_dec,
                'g_loss_dec_landmark': g_loss_dec_landmarks, 'g_loss_dec_id': g_loss_dec_id,
                'g_loss_gen': g_loss_gen, 'g_loss_id': g_loss_id, 'g_loss_dis': g_loss_dis,
                'd_loss_raw': d_raw_loss, 'd_loss_wmd': d_wmd_loss
            }
            return result, [imgs, imgs_wm, reenacted_img_wm]

    def tune_batch_infoswap(self, imgs, wms):
        self.encoder.train()
        self.decoder.train()
        self.infoswap.eval()
        self.discriminator.train()

        with torch.enable_grad():
            imgs, wms = imgs.to(self.device), wms.to(self.device)
            
            imgs_wm = self.encoder(imgs, wms)
            identity_img = torch.roll(imgs, 1, 0)
            
            swapped_img_wm = self.infoswap([imgs_wm, identity_img, self.device])
            swapped_img = self.infoswap([imgs, identity_img, self.device])
            
            resize_back = transforms.Resize((self.img_size, self.img_size))
            swapped_img_wm_resized = resize_back(swapped_img_wm)
            swapped_img_resized = resize_back(swapped_img)
            
            wms_recover_landmarks, wms_recover_id = self.decoder(swapped_img_wm_resized)

            self.opt_discriminator.zero_grad()
            d_label_raw = self.discriminator(imgs)
            d_raw_loss = self.criterion_BCE(d_label_raw, self.labels_raw[:d_label_raw.shape[0]])
            d_raw_loss.backward()
            d_label_wmd = self.discriminator(imgs_wm.detach())
            d_wmd_loss = self.criterion_BCE(d_label_wmd, self.labels_wmd[:d_label_wmd.shape[0]])
            d_wmd_loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), 5)
            self.opt_discriminator.step()

            self.opt_model.zero_grad()
            g_label_decoded = self.discriminator(imgs_wm)
            g_loss_dis = self.criterion_BCE(g_label_decoded, self.labels_raw[:g_label_decoded.shape[0]])
            g_loss_enc = self.criterion_MSE(imgs_wm, imgs)

            gt_landmarks = wms[:, :136]
            gt_id = wms[:, 136:]
            g_loss_dec_landmarks = self.criterion_Landmark(wms_recover_landmarks, gt_landmarks)
            gt_id_mapped = (gt_id + 1) / 2
            g_loss_dec_id = self.criterion_BCE(wms_recover_id, gt_id_mapped)
            
            g_loss_dec = self.landmark_w * g_loss_dec_landmarks + self.id_w * g_loss_dec_id

            g_loss_gen = self.criterion_MSE(swapped_img_wm_resized, swapped_img_resized.detach())
            
            wms_identity_landmarks, wms_identity_id = self.decoder(imgs_wm)
            g_loss_id_landmarks = self.criterion_Landmark(wms_identity_landmarks, gt_landmarks)
            g_loss_id_id = self.criterion_BCE(wms_identity_id, gt_id_mapped)
            g_loss_id = g_loss_id_landmarks + g_loss_id_id

            g_loss_total = (self.dis_w * g_loss_dis + self.enc_w * g_loss_enc + g_loss_dec + 
                            self.gen_w * g_loss_gen + g_loss_id * self.landmark_w)
            g_loss_total.backward()

            nn.utils.clip_grad_norm_(self.encoder.parameters(), 5)
            nn.utils.clip_grad_norm_(self.decoder.parameters(), 5)
            self.opt_model.step()

            psnr = self.criterion_PSNR(self.denorm(imgs_wm.detach()), self.denorm(imgs.detach())) * (-1)
            ssim = 1 - 2 * self.criterion_SSIM(self.denorm(imgs_wm.detach()), self.denorm(imgs.detach()))
            landmark_aed, id_ber = calculate_metrics(wms, wms_recover_landmarks, wms_recover_id, self.img_size)

            result = {
                'landmark_aed': landmark_aed, 'id_ber': id_ber, 'psnr': psnr, 'ssim': ssim,
                'g_loss': g_loss_total, 'g_loss_enc': g_loss_enc, 'g_loss_dec': g_loss_dec,
                'g_loss_dec_landmark': g_loss_dec_landmarks, 'g_loss_dec_id': g_loss_dec_id,
                'g_loss_gen': g_loss_gen, 'g_loss_id': g_loss_id, 'g_loss_dis': g_loss_dis,
                'd_loss_raw': d_raw_loss, 'd_loss_wmd': d_wmd_loss
            }
            return result
    
    def val_batch_infoswap(self, imgs, wms):
        self.encoder.eval()
        self.decoder.eval()
        self.infoswap.eval()

        with torch.no_grad():
            imgs, wms = imgs.to(self.device), wms.to(self.device)
            imgs_wm = self.encoder(imgs, wms)
            identity_img = torch.roll(imgs, 1, 0)
            
            swapped_img_wm = self.infoswap([imgs_wm, identity_img, self.device])
            swapped_img = self.infoswap([imgs, identity_img, self.device])

            resize_back = transforms.Resize((self.img_size, self.img_size))
            swapped_img_wm_resized = resize_back(swapped_img_wm)
            swapped_img_resized = resize_back(swapped_img)
            
            wms_recover_landmarks, wms_recover_id = self.decoder(swapped_img_wm_resized)

            d_label_raw = self.discriminator(imgs)
            d_raw_loss = self.criterion_BCE(d_label_raw, self.labels_raw[:d_label_raw.shape[0]])
            d_label_wmd = self.discriminator(imgs_wm.detach())
            d_wmd_loss = self.criterion_BCE(d_label_wmd, self.labels_wmd[:d_label_wmd.shape[0]])

            g_label_decoded = self.discriminator(imgs_wm)
            g_loss_dis = self.criterion_BCE(g_label_decoded, self.labels_raw[:g_label_decoded.shape[0]])
            g_loss_enc = self.criterion_MSE(imgs_wm, imgs)
            
            gt_landmarks = wms[:, :136]
            gt_id = wms[:, 136:]
            g_loss_dec_landmarks = self.criterion_Landmark(wms_recover_landmarks, gt_landmarks)
            gt_id_mapped = (gt_id + 1) / 2
            g_loss_dec_id = self.criterion_BCE(wms_recover_id, gt_id_mapped)
            g_loss_dec = self.landmark_w * g_loss_dec_landmarks + self.id_w * g_loss_dec_id
            g_loss_gen = self.criterion_MSE(swapped_img_wm_resized, swapped_img_resized)
            
            wms_identity_landmarks, wms_identity_id = self.decoder(imgs_wm)
            g_loss_id_landmarks = self.criterion_Landmark(wms_identity_landmarks, gt_landmarks)
            g_loss_id_id = self.criterion_BCE(wms_identity_id, gt_id_mapped)
            g_loss_id = g_loss_id_landmarks + g_loss_id_id

            g_loss_total = (self.dis_w * g_loss_dis + self.enc_w * g_loss_enc + g_loss_dec + 
                            self.gen_w * g_loss_gen + g_loss_id * self.landmark_w)

            psnr = self.criterion_PSNR(self.denorm(imgs_wm.detach()), self.denorm(imgs.detach())) * (-1)
            ssim = 1 - 2 * self.criterion_SSIM(self.denorm(imgs_wm.detach()), self.denorm(imgs.detach()))
            landmark_aed, id_ber = calculate_metrics(wms, wms_recover_landmarks, wms_recover_id, self.img_size)

            result = {
                'landmark_aed': landmark_aed, 'id_ber': id_ber, 'psnr': psnr, 'ssim': ssim,
                'g_loss': g_loss_total, 'g_loss_enc': g_loss_enc, 'g_loss_dec': g_loss_dec,
                'g_loss_dec_landmark': g_loss_dec_landmarks, 'g_loss_dec_id': g_loss_dec_id,
                'g_loss_gen': g_loss_gen, 'g_loss_id': g_loss_id, 'g_loss_dis': g_loss_dis,
                'd_loss_raw': d_raw_loss, 'd_loss_wmd': d_wmd_loss
            }
            return result, [imgs, imgs_wm, swapped_img_wm_resized]

    def save_checkpoint(self, epoch, checkpoint_path):

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        

        if self.configs.sep_model:
            model_state_dict = {
                'encoder_state_dict': self.encoder.module.state_dict() if self.num_gpus > 1 else self.encoder.state_dict(),
                'decoder_state_dict': self.decoder.module.state_dict() if self.num_gpus > 1 else self.decoder.state_dict()
            }
        else:
            model_state_dict = self.model.module.state_dict() if self.num_gpus > 1 else self.model.state_dict()


        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'discriminator_state_dict': self.discriminator.module.state_dict() if self.num_gpus > 1 else self.discriminator.state_dict(),
            'optimizer_model_state_dict': self.opt_model.state_dict(),
            'optimizer_discriminator_state_dict': self.opt_discriminator.state_dict(),
            'configs': self.configs.get_items() 
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            self.logger.info(f"Warning: Checkpoint file not found at {checkpoint_path}. Starting from scratch.")
            return 0


        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        

        if self.configs.sep_model:

            encoder_state_dict = checkpoint['model_state_dict']['encoder_state_dict']
            decoder_state_dict = checkpoint['model_state_dict']['decoder_state_dict']
            if self.num_gpus > 1:
                self.encoder.module.load_state_dict(encoder_state_dict)
                self.decoder.module.load_state_dict(decoder_state_dict)
            else:
                self.encoder.load_state_dict(encoder_state_dict)
                self.decoder.load_state_dict(decoder_state_dict)
        else:

            model_state_dict = checkpoint['model_state_dict']
            if self.num_gpus > 1:
                self.model.module.load_state_dict(model_state_dict)
            else:
                self.model.load_state_dict(model_state_dict)


        discriminator_state_dict = checkpoint['discriminator_state_dict']
        if self.num_gpus > 1:
            self.discriminator.module.load_state_dict(discriminator_state_dict)
        else:
            self.discriminator.load_state_dict(discriminator_state_dict)


        self.opt_model.load_state_dict(checkpoint['optimizer_model_state_dict'])
        self.opt_discriminator.load_state_dict(checkpoint['optimizer_discriminator_state_dict'])
        

        start_epoch = checkpoint['epoch']
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}. Resuming from epoch {start_epoch + 1}.")
        
        return start_epoch

    def load_model_for_finetune(self, pretrain_checkpoint_path):
        """
        Loads pre-trained weights (usually from the common distortion phase) to initialize 
        the model for the deepfake fine-tuning phase, facilitating transfer learning.
        """
        if not os.path.exists(pretrain_checkpoint_path):
            raise FileNotFoundError(f"Pre-trained checkpoint not found at {pretrain_checkpoint_path}")

        checkpoint = torch.load(pretrain_checkpoint_path, map_location=self.device)
        

        model_dict = checkpoint['model_state_dict']
        if 'encoder.conv_head_img.conv.weight' in model_dict:
             enc_dict = {k.replace('encoder.', ''): v for k, v in model_dict.items() if k.startswith('encoder.')}
             dec_dict = {k.replace('decoder.', ''): v for k, v in model_dict.items() if k.startswith('decoder.')}
        else:
             enc_dict = model_dict['encoder_state_dict']
             dec_dict = model_dict['decoder_state_dict']
        

        if self.num_gpus > 1:
            self.encoder.module.load_state_dict(enc_dict)
            self.decoder.module.load_state_dict(dec_dict)
        else:
            self.encoder.load_state_dict(enc_dict)
            self.decoder.load_state_dict(dec_dict)
        

        discriminator_dict = checkpoint['discriminator_state_dict']
        if self.num_gpus > 1:
            self.discriminator.module.load_state_dict(discriminator_dict)
        else:
            self.discriminator.load_state_dict(discriminator_dict)

        self.logger.info(f'Finished loading pre-trained weights for fine-tuning from {pretrain_checkpoint_path}')
