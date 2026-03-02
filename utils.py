import os
import yaml
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from math import exp
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import datetime
import numpy as np
import random
from PIL import Image, ImageDraw
import logging

try:
    import face_alignment
except ImportError:
    print("Warning: The face_alignment library is not installed, and the detected face key points will not be visualized.")
    face_alignment = None

from scipy.spatial.distance import squareform
from torch import linalg as LA


class Config:
    """
    Configuration management class.
    It handles loading settings from YAML files and provides attribute-style access to configuration parameters.
    Recursively converts nested dictionaries into Config objects.
    """
    def __init__(self):
        self.__config__ = None

    def load_config_file(self, path):
        with open(path, 'r') as file:
            self.__config__ = yaml.safe_load(file)
            file.close()

        self.set_items()

    def load_dict(self, config_dict):
        self.__config__ = config_dict

        self.set_items()

    def set_items(self):
        for key in self.__config__:
            value = self.__config__[key]
            if isinstance(value, dict):
                sub_config = Config()
                sub_config.load_dict(value)
                self.__setattr__(key, sub_config)
            else:
                self.__setattr__(key, value)

    def get_items(self):
        items = []
        for key in self.__config__:
            items.append((key, self.__config__[key]))
        return items

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class ImageDataset(Dataset):
    """
    Custom Dataset class for loading image-watermark pairs.
    It handles reading images and their corresponding watermark files (.npy),
    and applies necessary transforms (resize, crop, normalization) based on the mode (train/val).
    """
    def __init__(self, path_img, path_wm, img_size, wm_len, mode='train'):
        super(ImageDataset, self).__init__()
        self.img_size = img_size
        self.wm_len = wm_len
        self.path_img = path_img
        self.path_wm = path_wm
        self.lst_wm = os.listdir(self.path_wm)
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((int(self.img_size * 1.1), int(self.img_size * 1.1))),
                transforms.RandomCrop((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ])

    def transform_image(self, img):
        img = self.transform(img)
        return img

    def __getitem__(self, idx):
        while True:
            wm_name = self.lst_wm[idx]
            img_name = wm_name.replace('npy', 'jpg')
            img = Image.open(os.path.join(self.path_img, img_name)).convert('RGB')
            wm = np.load(os.path.join(self.path_wm, wm_name))
            if img is not None and wm is not None:
                img = self.transform_image(img)
                return img, torch.Tensor(wm)
            print('Somehow skipped the image:', os.path.join(self.path_wm, self.lst_wm[idx]))
            idx += 1

    def __len__(self):
        return len(self.lst_wm)

def make_loader(configs, model_mode='train', shuffle=True):
    dataset = ImageDataset(
        os.path.join(configs.img_path, model_mode),
        os.path.join(configs.wm_path, str(configs.img_size), model_mode),
        configs.img_size,
        configs.watermark_length,
        mode=model_mode
    )
    g = torch.Generator()
    g.manual_seed(configs.seed if hasattr(configs, 'seed') else 42)
    loader = DataLoader(dataset, batch_size=configs.batch_size, num_workers=4, shuffle=shuffle, drop_last=True,
                        worker_init_fn=seed_worker, generator=g)
    return loader


def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def decoded_message_error_rate(msgs, wms_recover):
    length = msgs.shape[0]

    msgs = msgs.gt(0.5)
    wms_recover = wms_recover.gt(0.5)
    error_rate = float(sum(msgs != wms_recover)) / length

    return error_rate


def decoded_message_error_rate_batch(messages, decoded_messages):
    error_rate = 0.0
    batch_size = len(messages)
    for i in range(batch_size):
        error_rate += decoded_message_error_rate(messages[i], decoded_messages[i])
    error_rate /= batch_size
    return error_rate


def calculate_metrics(wms, decoded_landmarks, decoded_id, img_size=256):
    """
    Computes key performance metrics for the watermarking task.
    
    Args:
        wms (Tensor): Ground truth watermark containing flattened landmarks and ID vector.
        decoded_landmarks (Tensor): Predicted landmarks from the decoder.
        decoded_id (Tensor): Predicted ID vector from the decoder.
        img_size (int): Image resolution used for scaling landmark coordinates.
        
    Returns:
        tuple: 
            - landmark_aed: Average Euclidean Distance for landmark alignment.
            - id_ber: Bit Error Rate for identity vector recovery.
    """
    batch_size = wms.shape[0]
    num_landmarks = 68
    
    gt_landmarks_flat = wms[:, :136]
    gt_id = wms[:, 136:]
    
    gt_landmarks_points = gt_landmarks_flat.view(batch_size, num_landmarks, 2)
    decoded_landmarks_points = decoded_landmarks.view(batch_size, num_landmarks, 2)

    gt_landmarks_pixels = gt_landmarks_points * img_size
    decoded_landmarks_pixels = decoded_landmarks_points * img_size

    pixel_diff = decoded_landmarks_pixels - gt_landmarks_pixels
    distances = torch.norm(pixel_diff, p=2, dim=2)

    landmark_aed = torch.mean(distances)

    id_pred_binary = torch.sign(decoded_id)
    id_errors = torch.sum(id_pred_binary != gt_id).float()
    id_ber = id_errors / (batch_size * gt_id.shape[1])
    
    return landmark_aed, id_ber

def get_random_images(imgs, imgs_wm, manipulated_imgs_wm):
    selected_id = np.random.randint(1, imgs.shape[0]) if imgs.shape[0] > 1 else 1
    img = imgs.cpu()[selected_id - 1:selected_id, :, :, :]
    img_wm = imgs_wm.cpu()[selected_id - 1:selected_id, :, :, :]
    manipulated_img_wm = manipulated_imgs_wm.cpu()[selected_id - 1:selected_id, :, :, :]
    return [img, img_wm, manipulated_img_wm]


def concatenate_images(save_imgs, imgs, imgs_wm, manipulated_imgs_wm):
    saved = get_random_images(imgs, imgs_wm, manipulated_imgs_wm)
    if save_imgs[2].shape[2] != saved[2].shape[2]:
        return save_imgs
    save_imgs[0] = torch.cat((save_imgs[0], saved[0]), 0)
    save_imgs[1] = torch.cat((save_imgs[1], saved[1]), 0)
    save_imgs[2] = torch.cat((save_imgs[2], saved[2]), 0)
    return save_imgs


def save_images(imgs, epoch, folder, resize_to=None, manipulated_is_denormalized=False):
    """
    Visualizes and saves a grid of images during training validation.
    The grid includes: Original, Watermarked, Manipulated, Difference (normalized), and Difference (linear/gray).
    This helps in monitoring the visual quality and the impact of distortions epoch by epoch.
    """
    os.makedirs(folder, exist_ok=True)
    original_images, watermarked_images, manipulated_images = imgs

    images = original_images[:original_images.shape[0], :, :, :].cpu()
    watermarked_images = watermarked_images[:watermarked_images.shape[0], :, :, :].cpu()

    images = (images + 1) / 2
    watermarked_images = (watermarked_images + 1) / 2
    if not manipulated_is_denormalized:
        manipulated_images = (manipulated_images + 1) / 2

    if manipulated_images.shape != images.shape:
        resize = nn.UpsamplingNearest2d(size=(images.shape[2], images.shape[3]))
        manipulated_images = resize(manipulated_images)

    if resize_to is not None:
        images = F.interpolate(images, size=resize_to)
        watermarked_images = F.interpolate(watermarked_images, size=resize_to)

    diff_images = (watermarked_images - images + 1) / 2

    diff_images_linear = diff_images.clone()
    R = diff_images_linear[:, 0, :, :]
    G = diff_images_linear[:, 1, :, :]
    B = diff_images_linear[:, 2, :, :]
    diff_images_linear[:, 0, :, :] = 0.299 * R + 0.587 * G + 0.114 * B
    diff_images_linear[:, 1, :, :] = diff_images_linear[:, 0, :, :]
    diff_images_linear[:, 2, :, :] = diff_images_linear[:, 0, :, :]
    diff_images_linear = torch.abs(diff_images_linear * 2 - 1)

    for id in range(diff_images_linear.shape[0]):
        diff_images_linear[id] = (diff_images_linear[id] - diff_images_linear[id].min()) / (
                diff_images_linear[id].max() - diff_images_linear[id].min())

    stacked_images = torch.cat(
        [images.unsqueeze(0), watermarked_images.unsqueeze(0), manipulated_images.unsqueeze(0),
         diff_images.unsqueeze(0), diff_images_linear.unsqueeze(0)], dim=0)
    shape = stacked_images.shape
    stacked_images = stacked_images.permute(0, 3, 1, 4, 2).reshape(shape[3] * shape[0], shape[4] * shape[1], shape[2])
    stacked_images = stacked_images.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
    filename = os.path.join(folder, 'epoch-{}.png'.format(epoch))

    saved_image = Image.fromarray(np.array(stacked_images, dtype=np.uint8)).convert("RGB")
    saved_image.save(filename)

def save_image_test_distortions(original_images, watermarked_images, manipulated_images, gt_watermarks, manipulation_name, batch_idx, configs, save_folder='./results/images_test/distortions'):
    """
    Saves visualization results for specific distortion tests.
    It draws ground truth landmarks (green) and detected landmarks (red, via face_alignment) 
    on the images to visually verify the geometric consistency after distortion.
    """
    if face_alignment is None:
        print("save_image_test_distortions: Skipping image saving because the face_alignment library is not available.")
        return

    os.makedirs(save_folder, exist_ok=True)
    batch_size = original_images.shape[0]
    img_size = original_images.shape[2]

    point_r = 2 if img_size >= 256 else 1.2

    if hasattr(configs, 'save_samples') and hasattr(configs.save_samples, 'count'):
        save_nums = configs.save_samples.count
    elif hasattr(configs, 'save_img_nums'):
        save_nums = configs.save_img_nums
    else:
        print("Warning: 'save_samples.count' or 'save_img_nums' was not found. All images in the batch will be saved by default.")
        save_nums = batch_size

    if save_nums > batch_size:
        print(f"Error: save_img_nums ({save_nums}) > batch_size ({batch_size}), das Testbild kann nicht gespeichert werden.")
        return
    
    selected_indices = np.random.choice(np.arange(batch_size), size=save_nums, replace=False)

    original_images = original_images[selected_indices]
    watermarked_images = watermarked_images[selected_indices]
    manipulated_images = manipulated_images[selected_indices]
    gt_watermarks = gt_watermarks[selected_indices]
    batch_size = save_nums

    try:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda' if torch.cuda.is_available() else 'cpu')
    except Exception as e:
        print(f"Unable to initialize face_alignment: {e}. Skip key point visualization.")
        fa = None
        
    clean_manipulation_name = manipulation_name.split('(')[0]

    originals_pil = [transforms.ToPILImage()((img.cpu() / 2 + 0.5).clamp(0, 1)) for img in original_images]
    watermarked_pil = [transforms.ToPILImage()((img.cpu() / 2 + 0.5).clamp(0, 1)) for img in watermarked_images]
    manipulated_pil = [transforms.ToPILImage()((img.cpu() / 2 + 0.5).clamp(0, 1)) for img in manipulated_images]

    gt_landmarks_flat = gt_watermarks[:, :136].cpu().numpy()
    gt_landmarks_points = gt_landmarks_flat.reshape(batch_size, 68, 2) * img_size
    row4_images = []
    for i in range(batch_size):
        img_copy = originals_pil[i].copy()
        draw = ImageDraw.Draw(img_copy)
        for point in gt_landmarks_points[i]:
            x, y = point
            draw.ellipse((x - point_r, y - point_r, x + point_r, y + point_r), fill='lime', outline='lime')
        row4_images.append(img_copy)

    row5_images = []
    row6_images = []
    for i in range(batch_size):
        img_manipulated_np = np.array(manipulated_pil[i])
        
        img5_copy = manipulated_pil[i].copy()
        draw5 = ImageDraw.Draw(img5_copy)
        img6_blank = Image.new('RGB', (img_size, img_size), 'white')
        draw6 = ImageDraw.Draw(img6_blank)

        for point in gt_landmarks_points[i]:
            x, y = point
            draw6.ellipse((x - point_r, y - point_r, x + point_r, y + point_r), fill='lime', outline='lime')

        if fa:
            preds = fa.get_landmarks(img_manipulated_np)
            if preds is not None and len(preds) > 0:
                detected_points = preds[0]
                for point in detected_points:
                    x, y = point
                    draw5.ellipse((x - point_r, y - point_r, x + point_r, y + point_r), fill='red', outline='red')
                    draw6.ellipse((x - point_r, y - point_r, x + point_r, y + point_r), fill='red', outline='red')
        
        row5_images.append(img5_copy)
        row6_images.append(img6_blank)

    final_image = Image.new('RGB', (img_size * batch_size, img_size * 6))
    all_rows = [originals_pil, watermarked_pil, manipulated_pil, row4_images, row5_images, row6_images]

    for row_idx, image_list in enumerate(all_rows):
        for col_idx, img in enumerate(image_list):
            final_image.paste(img, (col_idx * img_size, row_idx * img_size))

    filename = os.path.join(save_folder, f'batch-{batch_idx}-{clean_manipulation_name}.png')
    final_image.save(filename)
    print(f"Saved test image to {filename}")

def save_image_test_deepfakes(original_images, watermarked_images, manipulated_images, gt_watermarks, deepfake_name, batch_idx, configs, save_folder='./results/images_test/deepfakes'):
    if face_alignment is None:
        print("save_image_test_deepfakes: Skipping image saving because the face_alignment library is not available.")
        return

    os.makedirs(save_folder, exist_ok=True)
    batch_size = original_images.shape[0]
    img_size = original_images.shape[2]

    point_r = 2 if img_size >= 256 else 1.2

    if hasattr(configs, 'save_samples') and hasattr(configs.save_samples, 'count'):
        save_nums = configs.save_samples.count
    elif hasattr(configs, 'save_img_nums'):
        save_nums = configs.save_img_nums
    else:
        print("Warning: 'save_samples.count' or 'save_img_nums' was not found. All images in the batch will be saved by default.")
        save_nums = batch_size

    if save_nums > batch_size:
        print(f"Error: save_img_nums ({save_nums}) > batch_size ({batch_size}), das Testbild kann nicht gespeichert werden.")
        return
    
    selected_indices = np.random.choice(np.arange(batch_size), size=save_nums, replace=False)

    original_images = original_images[selected_indices]
    watermarked_images = watermarked_images[selected_indices]
    manipulated_images = manipulated_images[selected_indices]
    gt_watermarks = gt_watermarks[selected_indices]
    batch_size = save_nums

    try:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda' if torch.cuda.is_available() else 'cpu')
    except Exception as e:
        print(f"Unable to initialize face_alignment: {e}. Skip key point visualization.")
        fa = None

    originals_pil = [transforms.ToPILImage()((img.cpu() / 2 + 0.5).clamp(0, 1)) for img in original_images]
    watermarked_pil = [transforms.ToPILImage()((img.cpu() / 2 + 0.5).clamp(0, 1)) for img in watermarked_images]
    manipulated_pil = [transforms.ToPILImage()((img.cpu() / 2 + 0.5).clamp(0, 1)) for img in manipulated_images]

    gt_landmarks_flat = gt_watermarks[:, :136].cpu().numpy()
    gt_landmarks_points = gt_landmarks_flat.reshape(batch_size, 68, 2) * img_size
    row4_images = []
    for i in range(batch_size):
        img_copy = originals_pil[i].copy()
        draw = ImageDraw.Draw(img_copy)
        for point in gt_landmarks_points[i]:
            x, y = point
            draw.ellipse((x - point_r, y - point_r, x + point_r, y + point_r), fill='lime', outline='lime')
        row4_images.append(img_copy)

    row5_images = []
    row6_images = []
    for i in range(batch_size):
        img_manipulated_np = np.array(manipulated_pil[i])
        
        img5_copy = manipulated_pil[i].copy()
        draw5 = ImageDraw.Draw(img5_copy)

        img6_blank = Image.new('RGB', (img_size, img_size), 'white')
        draw6 = ImageDraw.Draw(img6_blank)

        for point in gt_landmarks_points[i]:
            x, y = point
            draw6.ellipse((x - point_r, y - point_r, x + point_r, y + point_r), fill='lime', outline='lime')

        if fa:
            preds = fa.get_landmarks(img_manipulated_np)
            if preds is not None:
                detected_points = preds[0]
                for point in detected_points:
                    x, y = point
                    draw5.ellipse((x - point_r, y - point_r, x + point_r, y + point_r), fill='red', outline='red')
                    draw6.ellipse((x - point_r, y - point_r, x + point_r, y + point_r), fill='red', outline='red')
        
        row5_images.append(img5_copy)
        row6_images.append(img6_blank)

    final_image = Image.new('RGB', (img_size * batch_size, img_size * 6))
    all_rows = [originals_pil, watermarked_pil, manipulated_pil, row4_images, row5_images, row6_images]

    for row_idx, image_list in enumerate(all_rows):
        for col_idx, img in enumerate(image_list):
            final_image.paste(img, (col_idx * img_size, row_idx * img_size))

    filename = os.path.join(save_folder, f'batch-{batch_idx}-{deepfake_name}.png')
    final_image.save(filename)
    print(f"Saved test image to {filename}")

def save_image_test_distortions_batch(batch_visuals, gt_watermarks, manipulation_names, batch_idx, configs, save_folder='./results/images_test_batch/distortions'):
    if face_alignment is None:
        print("save_image_test_distortions_batch: Skipping image saving because the face_alignment library is not available.")
        return

    os.makedirs(save_folder, exist_ok=True)
    
    num_manipulations = len(manipulation_names)
    if num_manipulations == 0:
        return

    img_size = configs.img_size
    batch_size = gt_watermarks.shape[0]

    point_r = 2 if img_size >= 256 else 1.2

    try:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda' if torch.cuda.is_available() else 'cpu')
    except Exception as e:
        print(f"Unable to initialize face_alignment: {e}. Skip key point visualization.")
        fa = None

    final_image = Image.new('RGB', (img_size * num_manipulations, img_size * 6))

    for col_idx, manipulation_name in enumerate(manipulation_names):
        random_sample_idx = random.randint(0, batch_size - 1)
        
        visuals = batch_visuals[col_idx]
        
        original_img_tensor = visuals['original'][random_sample_idx]
        watermarked_img_tensor = visuals['watermarked'][random_sample_idx]
        manipulated_img_tensor = visuals['manipulated'][random_sample_idx]
        gt_watermark_tensor = gt_watermarks[random_sample_idx]

        original_pil = transforms.ToPILImage()((original_img_tensor.cpu() / 2 + 0.5).clamp(0, 1))
        watermarked_pil = transforms.ToPILImage()((watermarked_img_tensor.cpu() / 2 + 0.5).clamp(0, 1))
        manipulated_pil = transforms.ToPILImage()((manipulated_img_tensor.cpu() / 2 + 0.5).clamp(0, 1))
        
        gt_landmarks_points = gt_watermark_tensor[:136].view(68, 2).cpu().numpy() * img_size
        img_row4 = original_pil.copy()
        draw4 = ImageDraw.Draw(img_row4)
        for point in gt_landmarks_points:
            x, y = point
            draw4.ellipse((x - point_r, y - point_r, x + point_r, y + point_r), fill='lime', outline='lime')

        img_manipulated_np = np.array(manipulated_pil)
        img_row5 = manipulated_pil.copy()
        draw5 = ImageDraw.Draw(img_row5)
        img_row6 = Image.new('RGB', (img_size, img_size), 'white')
        draw6 = ImageDraw.Draw(img_row6)

        for point in gt_landmarks_points:
            x, y = point
            draw6.ellipse((x - point_r, y - point_r, x + point_r, y + point_r), fill='lime', outline='lime')

        if fa:
            preds = fa.get_landmarks(img_manipulated_np)
            if preds is not None and len(preds) > 0:
                detected_points = preds[0]
                for point in detected_points:
                    x, y = point
                    draw5.ellipse((x - point_r, y - point_r, x + point_r, y + point_r), fill='red', outline='red')
                    draw6.ellipse((x - point_r, y - point_r, x + point_r, y + point_r), fill='red', outline='red')

        all_rows_pil = [original_pil, watermarked_pil, manipulated_pil, img_row4, img_row5, img_row6]
        for row_idx, pil_img in enumerate(all_rows_pil):
            final_image.paste(pil_img, (col_idx * img_size, row_idx * img_size))

    filename = os.path.join(save_folder, f'batch-{batch_idx}.png')
    final_image.save(filename)
    print(f"Saved batch aggregation test image to {filename}")

def save_image_test_deepfakes_batch(batch_visuals, gt_watermarks, deepfake_names, batch_idx, configs, save_folder='./results/images_test_batch/deepfakes'):
    if face_alignment is None:
        print("save_image_test_deepfakes_batch: Skipping image saving because the face_alignment library is not available.")
        return

    os.makedirs(save_folder, exist_ok=True)
    
    num_deepfakes = len(deepfake_names)
    if num_deepfakes == 0:
        return

    img_size = configs.img_size
    batch_size = gt_watermarks.shape[0]

    point_r = 2 if img_size >= 256 else 1.2

    try:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda' if torch.cuda.is_available() else 'cpu')
    except Exception as e:
        print(f"Unable to initialize face_alignment: {e}. Skip key point visualization.")
        fa = None

    final_image = Image.new('RGB', (img_size * num_deepfakes, img_size * 6))

    for col_idx, deepfake_name in enumerate(deepfake_names):
        random_sample_idx = random.randint(0, batch_size - 1)
        
        visuals = batch_visuals[deepfake_name]
        
        original_img_tensor = visuals['original'][random_sample_idx]
        watermarked_img_tensor = visuals['watermarked'][random_sample_idx]
        manipulated_img_tensor = visuals['manipulated'][random_sample_idx]
        gt_watermark_tensor = gt_watermarks[random_sample_idx]

        original_pil = transforms.ToPILImage()((original_img_tensor.cpu() / 2 + 0.5).clamp(0, 1))
        watermarked_pil = transforms.ToPILImage()((watermarked_img_tensor.cpu() / 2 + 0.5).clamp(0, 1))
        manipulated_pil = transforms.ToPILImage()((manipulated_img_tensor.cpu() / 2 + 0.5).clamp(0, 1))
        
        gt_landmarks_points = gt_watermark_tensor[:136].view(68, 2).cpu().numpy() * img_size
        img_row4 = original_pil.copy()
        draw4 = ImageDraw.Draw(img_row4)
        for point in gt_landmarks_points:
            x, y = point
            draw4.ellipse((x - point_r, y - point_r, x + point_r, y + point_r), fill='lime', outline='lime')

        img_manipulated_np = np.array(manipulated_pil)
        img_row5 = manipulated_pil.copy()
        draw5 = ImageDraw.Draw(img_row5)
        img_row6 = Image.new('RGB', (img_size, img_size), 'white')
        draw6 = ImageDraw.Draw(img_row6)

        for point in gt_landmarks_points:
            x, y = point
            draw6.ellipse((x - point_r, y - point_r, x + point_r, y + point_r), fill='lime', outline='lime')

        if fa:
            preds = fa.get_landmarks(img_manipulated_np)
            if preds is not None and len(preds) > 0:
                detected_points = preds[0]
                for point in detected_points:
                    x, y = point
                    draw5.ellipse((x - point_r, y - point_r, x + point_r, y + point_r), fill='red', outline='red')
                    draw6.ellipse((x - point_r, y - point_r, x + point_r, y + point_r), fill='red', outline='red')

        all_rows_pil = [original_pil, watermarked_pil, manipulated_pil, img_row4, img_row5, img_row6]
        for row_idx, pil_img in enumerate(all_rows_pil):
            final_image.paste(pil_img, (col_idx * img_size, row_idx * img_size))

    filename = os.path.join(save_folder, f'batch-{batch_idx}.png')
    final_image.save(filename)
    print(f"Saved batch aggregation test image to {filename}")

def setup_logger(name='LampMarkLogger', log_file='./results/training.log'):
    """
    Configures the logging system.
    Sets up both file handler (for persistence) and stream handler (for console output)
    with a standard formatting style.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO) 

    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

def update_config_resolution(configs, res):
    """
    Dynamically updates the configuration based on the selected image resolution (128 or 256).
    It adjusts model depth (blocks), batch sizes, and file paths to match the target resolution,
    facilitating easy switching between different experiment settings.
    """
    res = int(res)
    
    def _update(key, value):
        setattr(configs, key, value)
        
        if hasattr(configs, '_Config__config__') and configs._Config__config__ is not None:
            configs._Config__config__[key] = value
        elif hasattr(configs, '__config__') and configs.__config__ is not None:
            configs.__config__[key] = value

    _update('img_size', res)
    
    if res == 128:
        _update('encoder_blocks', 3)
        _update('decoder_blocks', 1)
    elif res == 256:
        _update('encoder_blocks', 4)
        _update('decoder_blocks', 2)
        
    def adjust_path(path):
        if not path or not isinstance(path, str): return path
        src = '256' if res == 128 else '128'
        dst = str(res)
        return path.replace(src, dst)

    if hasattr(configs, 'img_path'):
        _update('img_path', adjust_path(configs.img_path))
    if hasattr(configs, 'weight_path'):
        _update('weight_path', adjust_path(configs.weight_path))
    if hasattr(configs, 'wm_path'):
        _update('wm_path', adjust_path(configs.wm_path))
    
    # if hasattr(configs, 'batch_size') and configs.batch_size == 8 and res == 128:
    #     _update('batch_size', 16)
        
    return configs