import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import os
import argparse
from argparse import Namespace
import random
from PIL import Image
import yaml
import sys

sys.path.insert(0, './SimSwap')
sys.path.insert(0, './UniFace')
sys.path.insert(0, './CSCS')
sys.path.insert(0, './StarGAN')
sys.path.insert(0, './InfoSwap')

class SimSwapModel(nn.Module):
    """
    Wrapper class for the SimSwap face swapping model.
    It encapsulates the model initialization, argument parsing, and inference logic
    to provide a unified interface for the LIDMark framework.
    """
    def __init__(self, img_size=128, mode='test'):
        super(SimSwapModel, self).__init__()
        from .SimSwap.models.models import create_model
        from .SimSwap.options.test_options import TestOptions

        # Temporarily modify sys.argv to bypass SimSwap's command-line argument parser requirements
        # during class initialization.
        import sys
        original_argv = sys.argv
        sys.argv = [original_argv[0]]
        opt = TestOptions().parse()
        sys.argv = original_argv

        self.img_size = img_size
        if self.img_size == 128:
            opt.crop_size = 224
            opt.image_size = 224
            opt.netG = 'global'
        else:
            opt.crop_size = 224
            opt.image_size = 224
            opt.netG = 'global'
        self.sim_swap = create_model(opt)
        self.sim_swap.eval()
        self.arcface_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.mode = mode

        self.norm = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        self.denorm = transforms.Normalize(
            mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
            std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
        )

    def one_step_swap(self, source, target, device):
        source = self.arcface_norm(source)
        source_downsample = F.interpolate(source, size=(112, 112))
        latent_source = self.sim_swap.netArc(source_downsample)
        latent_source = latent_source.detach().to('cpu')
        latent_source = latent_source / np.linalg.norm(latent_source, axis=1, keepdims=True)
        latent_source = latent_source.to(device)

        swapped_face = self.sim_swap(source, target, latent_source, latent_source, True)
        return swapped_face

    def forward(self, img_wm_device):
        """
        Forward pass for SimSwap inference.
        
        Args:
            img_wm_device (tuple): A tuple containing:
                - target_img (Tensor): The target image (background/body).
                - source_img (Tensor): The source image (identity provider).
                - device (torch.device): The computation device.
        
        Returns:
            Tensor: The face-swapped image resized back to the original input resolution.
        """
        target_img = img_wm_device[0]
        source_img = img_wm_device[1]
        device = img_wm_device[2]

        target_img_denorm = self.denorm(target_img)
        source_img_denorm = self.denorm(source_img)

        if self.img_size == 128:
            resize = transforms.Resize((224, 224))
        else:
            resize = transforms.Resize((224, 224))
        resize_back = transforms.Resize((self.img_size, self.img_size))

        target_img_denorm = resize(target_img_denorm)
        source_img_denorm = resize(source_img_denorm)

        swapped_face_wm_denorm = self.one_step_swap(source_img_denorm, target_img_denorm, device)
        
        swapped_face_wm = self.norm(swapped_face_wm_denorm)
        swapped_face_wm = resize_back(swapped_face_wm)

        return swapped_face_wm

class UniFaceModel(nn.Module):
    def __init__(self, device, mode='test'):
        super(UniFaceModel, self).__init__()
        from .UniFace import generate_swap

        parser = argparse.ArgumentParser()
        parser.add_argument("--mixing_type", type=str, default='examples')
        parser.add_argument("--inter", type=str, default='pair')
        parser.add_argument("--ckpt", type=str, default='./model/UniFace/checkpoints/500000.pt')
        parser.add_argument("--test_path", type=str, default='examples/img/')
        parser.add_argument("--test_txt_path", type=str, default='examples/pair_swap.txt')
        parser.add_argument("--batch", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=1)
        parser.add_argument("--save_image_dir", type=str, default="expr")

        import sys
        original_argv = sys.argv
        sys.argv = [original_argv[0]]
        global args
        args = parser.parse_args()
        sys.argv = original_argv

        if not os.path.exists(args.ckpt):
            raise FileNotFoundError(f"UniFace checkpoint not found at {args.ckpt}")
            
        ckpt = torch.load(args.ckpt, map_location=device)
        train_args = ckpt["train_args"]
        for key in vars(train_args):
            if not (key in vars(args)):
                setattr(args, key, getattr(train_args, key))

        generate_swap.args = args
        self.swap_model = generate_swap.Model(args).half().to(device)
        self.swap_model.g_ema.load_state_dict(ckpt["g_ema"])
        self.swap_model.e_ema.load_state_dict(ckpt["e_ema"])
        self.swap_model.eval()
        self.device = device
        self.mode = mode

        self.resize_256 = transforms.Resize((256, 256))

    def forward(self, img_wm_device):
        target_img = img_wm_device[0]
        source_img = img_wm_device[1]
        
        # UniFace requires inputs to be resized to 256x256
        target_img_input = self.resize_256(target_img)
        source_img_input = self.resize_256(source_img)
        
        img_size = target_img.shape[-1]
        resize_back = transforms.Resize((img_size, img_size))

        # Perform inference using Half precision (fp16) as expected by the UniFace implementation
        _, _, swapped_img_wm = self.swap_model(
            [target_img_input.type(torch.cuda.HalfTensor), source_img_input.type(torch.cuda.HalfTensor)]
        )
        
        swapped_face_wm = resize_back(swapped_img_wm.float())
        return swapped_face_wm

class CSCSModel(nn.Module):
    def __init__(self, device, mode='test', adapter_type='add'):
        super(CSCSModel, self).__init__()
        from .CSCS.model.arcface.iresnet import iresnet100
        from .CSCS.model.arcface.iresnet_adapter import iresnet100_adapter
        from .CSCS.model.faceshifter.layers.faceshifter.layers_arcface import AEI_Net

        self.device = device
        self.mode = mode
        self.adapter_type = adapter_type

        # Load the pre-trained ArcFace backbone for Identity Embedding extraction
        id_emb_model_path = './model/CSCS/model/arcface/ms1mv3_arcface_r100_fp16_backbone.pth'
        weight_path = './model/CSCS/model_34_loss_-0.1688.pth.tar'

        self.ID_emb = iresnet100()
        self.ID_emb.load_state_dict(torch.load(id_emb_model_path, map_location='cpu'))
        self.ID_emb = self.ID_emb.to(device)
        self.ID_emb.eval()

        self.ID_adapter = iresnet100_adapter(type=self.adapter_type)
        model_weight = torch.load(weight_path, map_location='cpu')
        self.ID_adapter.load_state_dict(model_weight['adapter'])
        self.ID_adapter = self.ID_adapter.to(device)
        self.ID_adapter.eval()

        if self.adapter_type == 'concat':
            self.G = AEI_Net(1024)
        else:
            self.G = AEI_Net(512)
        self.G.load_state_dict(model_weight['G'])
        self.G = self.G.to(device)
        self.G.eval()

        self.norm_transform = transforms.Compose([
            transforms.Normalize(0.5, .5)
        ])
        self.resize_T = transforms.Resize(size=(256, 256))

    def forward(self, img_wm_device):
        target_img, source_img, _ = img_wm_device

        h, w = target_img.shape[-2], target_img.shape[-1]

        source_img = self.resize_T(source_img)
        target_img = self.resize_T(target_img)

        with torch.no_grad():
            # Extract and normalize identity embeddings from the source image
            src_id = F.normalize(
                self.ID_emb(F.interpolate(source_img, size=112, mode="bilinear")),
                dim=-1, p=2
            )
            # Apply the adapter to the identity embeddings
            src_id_adapt = F.normalize(
                self.ID_adapter(F.interpolate(source_img, size=112, mode="bilinear")),
                dim=-1, p=2
            )

            # Combine original and adapted identity features based on the adapter type strategy
            if self.adapter_type == 'concat':
                src_id_combined = torch.cat([src_id, src_id_adapt], dim=1)
            elif self.adapter_type == 'add':
                src_id_combined = src_id + src_id_adapt
            elif self.adapter_type == 'replace':
                src_id_combined = src_id_adapt
            else:
                raise ValueError("Invalid adapter type")

            swapped_face, _, _ = self.G(target_img, src_id_combined)
        
        if swapped_face.shape[-1] != w or swapped_face.shape[-2] != h:
            swapped_face = F.interpolate(swapped_face, size=(h, w), mode='bilinear', align_corners=True)

        return swapped_face

class StarGANModel(nn.Module):
    def __init__(self, img_size, mode='test', seed=777):
        super(StarGANModel, self).__init__()
        from .StarGAN.core.solver import Solver
        
        self.img_size = img_size
        self.mode = mode
        parser = argparse.ArgumentParser()
        self.args = self.get_args(parser)
        self.args.seed = seed
        torch.manual_seed(self.args.seed)
        
        print("Initializing StarGAN-v2 solver...")
        solver = Solver(self.args)
        
        # Manually load the EMA (Exponential Moving Average) weights for the generator and encoders
        # to ensure stable inference results.
        checkpoint_path = os.path.join(self.args.checkpoint_dir, f'{self.args.resume_iter:06d}_nets_ema.ckpt')
        print(f"Manually loading checkpoint: {checkpoint_path}...")
        
        if torch.cuda.is_available():
            ckpt = torch.load(checkpoint_path, map_location='cuda')
        else:
            ckpt = torch.load(checkpoint_path, map_location='cpu')

        solver.nets_ema.generator.module.load_state_dict(ckpt['generator'], strict=False)
        
        solver.nets_ema.mapping_network.module.load_state_dict(ckpt['mapping_network'])
        solver.nets_ema.style_encoder.module.load_state_dict(ckpt['style_encoder'])
        
        print("Checkpoint loaded successfully.")

        self.star_gan = solver.nets_ema
        self.resize_up = transforms.Resize((256, 256))
        self.resize_down = transforms.Resize((self.img_size, self.img_size))
        print("StarGAN-v2 model is ready.")

    def get_args(self, parser):
        parser.add_argument('--img_size', type=int, default=256,
                            help='Image resolution')
        parser.add_argument('--num_domains', type=int, default=2,
                            help='Number of domains')
        parser.add_argument('--latent_dim', type=int, default=16,
                            help='Latent vector dimension')
        parser.add_argument('--hidden_dim', type=int, default=512,
                            help='Hidden dimension of mapping network')
        parser.add_argument('--style_dim', type=int, default=64,
                            help='Style code dimension')

        parser.add_argument('--lambda_reg', type=float, default=1,
                            help='Weight for R1 regularization')
        parser.add_argument('--lambda_cyc', type=float, default=1,
                            help='Weight for cyclic consistency loss')
        parser.add_argument('--lambda_sty', type=float, default=1,
                            help='Weight for style reconstruction loss')
        parser.add_argument('--lambda_ds', type=float, default=1,
                            help='Weight for diversity sensitive loss')
        parser.add_argument('--ds_iter', type=int, default=100000,
                            help='Number of iterations to optimize diversity sensitive loss')
        parser.add_argument('--w_hpf', type=float, default=1,
                            help='weight for high-pass filtering')
        parser.add_argument('--resume_iter', type=int, default=100000,
                            help='Iterations to resume training/testing')
        parser.add_argument('--mode', type=str, default='sample',
                            choices=['train', 'sample', 'eval', 'align'],
                            help='This argument is used in solver')
        parser.add_argument('--num_workers', type=int, default=4,
                            help='Number of workers used in DataLoader')
        parser.add_argument('--seed', type=int, default=777,
                            help='Seed for random number generator')
        parser.add_argument('--checkpoint_dir', type=str, default='./model/StarGAN/expr/checkpoints/celeba_hq',
                            help='Directory for saving network checkpoints')
        parser.add_argument('--wing_path', type=str, default='./model/StarGAN/expr/checkpoints/wing.ckpt')
        parser.add_argument('--lm_path', type=str, default='./model/StarGAN/expr/checkpoints/celeba_lm_mean.npz')

        parser.add_argument('--print_every', type=int, default=10)
        parser.add_argument('--sample_every', type=int, default=5000)
        parser.add_argument('--save_every', type=int, default=10000)
        parser.add_argument('--eval_every', type=int, default=50000)

        import sys
        original_argv = sys.argv
        sys.argv = [original_argv[0]]
        args = parser.parse_args()
        sys.argv = original_argv

        return args

    def reenactment(self, source, target, ref):
        """
        Performs face reenactment.
        
        Args:
            source (Tensor): The source image (provides pose/content).
            target (Tensor): The target image (provides style/appearance).
            ref (Tensor): Reference labels (usually domain labels).
            
        Returns:
            Tensor: The generated image with source content and target style.
        """
        masks = self.star_gan.fan.get_heatmap(source) if self.args.w_hpf > 0 else None
        s_ref = self.star_gan.style_encoder(target, ref)
        x_fake = self.star_gan.generator(source, s_ref, masks=masks)
        return x_fake

    def forward(self, img_wm_device):
        pose_image = img_wm_device[0]
        identity_image = img_wm_device[1]
        device = img_wm_device[2]

        pose_image_resized = self.resize_up(pose_image)
        identity_image_resized = self.resize_up(identity_image)

        N = pose_image_resized.shape[0]
        ref = torch.ones(N, dtype=torch.long).to(device)

        reenacted_img = self.reenactment(pose_image_resized, identity_image_resized, ref)

        return self.resize_down(reenacted_img)

class InfoSwapModel(nn.Module):
    def __init__(self, device, mode='test', ib_mode='smooth'):
        super(InfoSwapModel, self).__init__()
        self.device = device
        self.mode = mode
        
        from .InfoSwap.modules.encoder128 import Backbone128
        from .InfoSwap.modules.iib import IIB
        from .InfoSwap.modules.aii_generator import AII512
        from .InfoSwap.modules.decoder512 import UnetDecoder512
        from .InfoSwap.preprocess.mtcnn import MTCNN

        if ib_mode == 'smooth':
            root = './model/InfoSwap/pre-trained_models/checkpoints_512/w_kernel_smooth'
            prefix = 'ckpt_ks_'
        else:
            root = './model/InfoSwap/pre-trained_models/checkpoints_512/wo_kernel_smooth'
            prefix = 'ckpt_'

        pathG = f'{prefix}G.pth'
        pathE = f'{prefix}E.pth'
        pathI = f'{prefix}I.pth'

        self.encoder = Backbone128(50, 0.6, 'ir_se').eval().to(device)
        encoder_path = './model/InfoSwap/pre-trained_models/model_128_ir_se50.pth' 
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=device), strict=True)

        self.G = AII512().eval().to(device)
        self.decoder = UnetDecoder512().eval().to(device)
        self.G.load_state_dict(torch.load(os.path.join(root, pathG), map_location=device), strict=True)
        self.decoder.load_state_dict(torch.load(os.path.join(root, pathE), map_location=device), strict=True)

        self.N = 10
        with torch.no_grad():
            _ = self.encoder(torch.rand(1, 3, 128, 128).to(device), cache_feats=True)
            _readout_feats = self.encoder.features[:(self.N + 1)]
        in_c = sum(map(lambda f: f.shape[-3], _readout_feats))
        out_c_list = [_readout_feats[i].shape[-3] for i in range(self.N)]
        
        # Load the Information Bottleneck (IIB) module and pre-trained weights
        self.iib = IIB(in_c, out_c_list, device, smooth=(ib_mode=='smooth'), kernel_size=1).eval()
        self.iib.load_state_dict(torch.load(os.path.join(root, pathI), map_location=device), strict=(ib_mode=='smooth'))
        
        # Load pre-computed feature statistics (mean, std, active neurons) for the readout layers
        self.param_dict = []
        for i in range(self.N + 1):
            readout_path = f'./model/InfoSwap/modules/weights128/readout_layer{i}.pth'
            state = torch.load(readout_path, map_location=device)
            n_samples = state['n_samples'].float()
            std = torch.sqrt(state['s'] / (n_samples - 1)).to(device)
            neuron_nonzero = state['neuron_nonzero'].float()
            active_neurons = (neuron_nonzero / n_samples) > 0.01
            self.param_dict.append([state['m'].to(device), std, active_neurons])
            
        self.img_transforms = transforms.Compose([
            transforms.Resize((512, 512)),
        ])

    def forward(self, img_wm_device):
        target_img = img_wm_device[0]
        source_img = img_wm_device[1]
        
        source_img_bgr = source_img[:, [2, 1, 0], :, :]
        target_img_bgr = target_img[:, [2, 1, 0], :, :]
    
        Xs = self.img_transforms(source_img_bgr)
        Xt = self.img_transforms(target_img_bgr)
        
        B = Xs.shape[0]
        
        with torch.no_grad():
            input_tensor = torch.cat((Xs, Xt), dim=0)
            input_tensor_128 = F.interpolate(input_tensor[:, :, 37:475, 37:475], size=[128, 128], mode='bilinear', align_corners=True)

            # Extract identity features from the backbone
            X_id = self.encoder(input_tensor_128, cache_feats=True)
            
            min_std = torch.tensor(0.01, device=self.device)
            readout_feats = [(self.encoder.features[i] - self.param_dict[i][0]) / torch.max(self.param_dict[i][1], min_std) 
                             for i in range(self.N + 1)]
            
            X_id_restrict = torch.zeros_like(X_id).to(self.device)
            Xt_feats, Xt_lambda = [], []
            
            # Apply Information Bottleneck (IIB) to filter features and disentangle identity/attributes
            for i in range(self.N):
                R = self.encoder.features[i]
                Z, lambda_, _ = getattr(self.iib, f'iba_{i}')(
                    R, readout_feats,
                    m_r=self.param_dict[i][0], std_r=self.param_dict[i][1],
                    active_neurons=self.param_dict[i][2],
                )
                X_id_restrict += self.encoder.restrict_forward(Z, i)

                Rs, Rt = R[:B], R[B:]
                lambda_s, lambda_t = lambda_[:B], lambda_[B:]

                m_s = torch.mean(Rs, dim=0)
                std_s = torch.std(Rs, dim=0)
                eps_s = torch.randn(size=Rt.shape).to(Rt.device) * std_s + m_s
                feat_t = Rt * (1. - lambda_t) + lambda_t * eps_s

                Xt_feats.append(feat_t)
                Xt_lambda.append(lambda_t)

            X_id_restrict /= float(self.N)
            Xs_id = X_id_restrict[:B]
            
            Xt_feats[0] = Xt
            Xt_attr, Xt_attr_lamb = self.decoder(Xt_feats, lambs=Xt_lambda, use_lambda=True)

            Y = self.G(Xs_id, Xt_attr, Xt_attr_lamb)
            
            self.encoder.features = []
        Y_rgb = Y[:, [2, 1, 0], :, :]
        return Y_rgb