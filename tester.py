import torch
import torch.nn as nn
from torchvision import transforms
import os

from utils import calculate_metrics
from model.lidmark import LIDMarkEncoder, FHD
from model.distortions import DistortionSimulator
from model.losses import PSNRLoss, SSIMLoss


class TesterImg:
    """
    Test manager class responsible for evaluating the model's performance.
    It handles model initialization, data loading, and execution of both 
    standard distortion tests and deepfake robustness tests.
    """
    def __init__(self, configs, device, logger):
        super(TesterImg, self).__init__()
        self.configs = configs
        self.logger = logger

        self.img_size = configs.img_size
        self.wm_len = configs.watermark_length
        self.enc_c = configs.encoder_channels
        self.enc_blocks = configs.encoder_blocks
        self.dec_c = configs.decoder_channels
        self.dec_blocks = configs.decoder_blocks

        self.batch_size = configs.batch_size
        self.device = device

        # Initialize evaluation metrics:
        # PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity)
        # are used to measure the visual quality (invisibility) of the watermarked images.
        self.criterion_PSNR = PSNRLoss(max_val=1.0).to(self.device)
        self.criterion_SSIM = SSIMLoss(window_size=5, reduction='mean').to(self.device)

        # Initialize the core watermark Encoder and Decoder networks
        self.encoder = LIDMarkEncoder(self.img_size, self.enc_c, self.enc_blocks, self.wm_len).to(self.device)
        self.decoder = FHD(self.img_size, self.dec_c, self.dec_blocks, self.wm_len).to(self.device)

        self._init_common_manipulation()
        self._init_deepfake_manipulations()

        seed = self.configs.seed if hasattr(self.configs, 'seed') else 42
        self.deepfake_settings = {}
        if hasattr(configs, 'deepfake_manipulation_layers'):
            for layer in configs.deepfake_manipulation_layers:
                self.deepfake_settings.update(layer)

        if self.deepfake_settings.get('simswap'):
            from model.deepfakes import SimSwapModel
            self.simswap = SimSwapModel(self.img_size, mode='test').to(self.device)
        if self.deepfake_settings.get('uniface'):
            from model.deepfakes import UniFaceModel
            self.uniface = UniFaceModel(self.device, mode='test').to(self.device)
        if self.deepfake_settings.get('cscs'):
            from model.deepfakes import CSCSModel
            self.cscs = CSCSModel(self.device, mode='test').to(self.device)
        if self.deepfake_settings.get('stargan_v2'):
            from model.deepfakes import StarGANModel
            self.stargan_v2 = StarGANModel(self.img_size, mode='test', seed=seed).to(self.device)
        if self.deepfake_settings.get('infoswap'):
            from model.deepfakes import InfoSwapModel
            self.infoswap = InfoSwapModel(self.device, mode='test').to(self.device)

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1 and self.device != 'cpu':
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs ...")
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)
            self._common_manipulation_multi_gpu()
            self._deepfake_manipulation_multi_gpu()

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
        if hasattr(self.configs, 'common_manipulation_layers'):
            layers = self.configs.common_manipulation_layers
        else:
            layers = []
            self.logger.warning("Common_manipulation_layers' or' manipulation_layers' were not found in the configuration file. The common distortion test will not be performed.")
        
        self.common_manipulation = DistortionSimulator(layers).to(self.device)

    def _init_deepfake_manipulations(self):
        self.simswap = None
        self.uniface = None
        self.cscs = None
        self.stargan_v2 = None
        self.infoswap = None

    def _common_manipulation_multi_gpu(self):
        self.common_manipulation = nn.DataParallel(self.common_manipulation)

    def _deepfake_manipulation_multi_gpu(self):
        if self.simswap: self.simswap = nn.DataParallel(self.simswap)
        if self.uniface: self.uniface = nn.DataParallel(self.uniface)
        if self.cscs: self.cscs = nn.DataParallel(self.cscs)
        if self.stargan_v2: self.stargan_v2 = nn.DataParallel(self.stargan_v2)
        if self.infoswap: self.infoswap = nn.DataParallel(self.infoswap)

    def run_deepfake_tests(self, imgs, wms):
        """
        Executes robustness tests against enabled Deepfake models.
        It generates watermarked images, subjects them to face-swapping attacks, 
        and evaluates the recovery of landmarks and identity information.
        
        Args:
            imgs (Tensor): Original input images.
            wms (Tensor): Ground truth watermarks.
            
        Returns:
            tuple: A dictionary of metrics (results) and a dictionary of images (visuals) for logging.
        """
        results = {}
        visuals = {}
        test_map = {
            'simswap': ('SimSwap', self.simswap),
            'uniface': ('UniFace', self.uniface),
            'cscs': ('CSCS', self.cscs),
            'stargan_v2': ('StarGAN', self.stargan_v2),
            'infoswap': ('InfoSwap', self.infoswap)
        }

        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            imgs_tensor, wms_tensor = imgs.to(self.device), wms.to(self.device)
            imgs_wm = self.encoder(imgs_tensor, wms_tensor)
            identity_img = torch.roll(imgs_tensor, 1, 0)

            for flag, (name, model_instance) in test_map.items():
                if self.deepfake_settings.get(flag) and model_instance is not None:
                    model_instance.eval()

                    manipulated_img_wm = model_instance([imgs_wm, identity_img, self.device])

                    if name == 'InfoSwap':
                        resize_back = transforms.Resize((self.img_size, self.img_size))
                        manipulated_img_wm = resize_back(manipulated_img_wm)

                    wms_recover_landmarks, wms_recover_id = self.decoder(manipulated_img_wm)
                    
                    landmark_aed, id_ber = calculate_metrics(wms_tensor, wms_recover_landmarks, wms_recover_id, self.img_size)
                    results[name] = {'landmark_aed': landmark_aed, 'id_ber': id_ber}
                    
                    visuals[name] = {
                        'original': imgs_tensor.cpu(),
                        'watermarked': imgs_wm.cpu(),
                        'manipulated': manipulated_img_wm.cpu(),
                        'gt_watermarks': wms_tensor.cpu()
                    }

        return results, visuals

    def test_one_manipulation(self, imgs, wms, manipulation):
        """
        Evaluates the model's robustness against a specific, single distortion.
        It calculates metrics for both watermark invisibility (PSNR/SSIM) 
        and recovery accuracy (AED/BER).
        """
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            imgs, wms = imgs.to(self.device), wms.to(self.device)
            imgs_wm = self.encoder(imgs, wms)
            
            self.common_manipulation = DistortionSimulator([manipulation]).to(self.device)
            manipulated_wm_img = self.common_manipulation([imgs_wm, imgs, self.device])
            
            wms_recover_landmarks, wms_recover_id = self.decoder(manipulated_wm_img)

            landmark_aed, id_ber = calculate_metrics(wms, wms_recover_landmarks, wms_recover_id, self.img_size)

            visuals = {
                'original': imgs.cpu(),
                'watermarked': imgs_wm.cpu(),
                'manipulated': manipulated_wm_img.cpu()
            }

            if manipulation == 'Identity()':
                psnr = self.criterion_PSNR(self.denorm(imgs_wm.detach()), self.denorm(imgs.detach())) * (-1)
                ssim = 1 - 2 * self.criterion_SSIM(self.denorm(imgs_wm.detach()), self.denorm(imgs.detach()))
                return psnr, ssim, landmark_aed, id_ber, visuals
            else:
                return landmark_aed, id_ber, visuals

    def load_checkpoint_for_test(self, checkpoint_path):
        """
        Loads model weights from a specified checkpoint for testing.
        It includes logic to handle state dictionaries saved from both 
        single-GPU and multi-GPU (DataParallel) training setups.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model_state_dict = checkpoint['model_state_dict']
        # Handle potential key mismatch due to 'module.' prefix from DataParallel training
        if 'encoder_state_dict' in model_state_dict: 
             encoder_state_dict = model_state_dict['encoder_state_dict']
             decoder_state_dict = model_state_dict['decoder_state_dict']
        else: 
             encoder_state_dict = {k.replace('encoder.', ''): v for k, v in model_state_dict.items() if k.startswith('encoder.')}
             decoder_state_dict = {k.replace('decoder.', ''): v for k, v in model_state_dict.items() if k.startswith('decoder.')}
            
        if self.num_gpus > 1:
            self.encoder.module.load_state_dict(encoder_state_dict)
            self.decoder.module.load_state_dict(decoder_state_dict)
        else:
            self.encoder.load_state_dict(encoder_state_dict)
            self.decoder.load_state_dict(decoder_state_dict)
        
        self.logger.info(f"Finished loading model weights for testing from {checkpoint_path}")