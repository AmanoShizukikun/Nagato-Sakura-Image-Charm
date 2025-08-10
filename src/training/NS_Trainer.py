import os
import gc
import time
import random
import json
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from src.models.NS_ImageEnhancer import ImageQualityEnhancer, MultiScaleDiscriminator

logger = logging.getLogger("Trainer")
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logs_dir = os.path.join(base_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)

torch.backends.cudnn.benchmark = True

class EnhancedPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.register_buffer('laplacian', self._create_laplacian_kernel())
        self.register_buffer('sobel_x', torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3).repeat(3,1,1,1))
        self.register_buffer('sobel_y', torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32).view(1,1,3,3).repeat(3,1,1,1))
        self.ssim = SSIMLoss(window_size=11, size_average=True, channel=3)

    def _create_laplacian_kernel(self):
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        laplacian = F.pad(laplacian, (1, 1, 1, 1), "constant", 0)
        kernel = laplacian.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        return kernel

    def extract_high_freq(self, x):
        return F.conv2d(F.pad(x, (2, 2, 2, 2), mode='reflect'), self.laplacian, padding=0, groups=3)

    def extract_sobel(self, x):
        gx = F.conv2d(F.pad(x, (1,1,1,1), mode='reflect'), self.sobel_x, padding=0, groups=3)
        gy = F.conv2d(F.pad(x, (1,1,1,1), mode='reflect'), self.sobel_y, padding=0, groups=3)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)

    def gram_matrix(self, x):
        B, C, H, W = x.size()
        features = x.view(B, C, -1)
        G = torch.bmm(features, features.transpose(1,2)) / (C*H*W)
        return G

    def fourier_loss(self, x, target):
        x = x.to(torch.float32)
        target = target.to(torch.float32)
        x_fft = torch.fft.fft2(x, norm='ortho')
        t_fft = torch.fft.fft2(target, norm='ortho')
        x_mag = torch.abs(x_fft)
        t_mag = torch.abs(t_fft)
        return self.l1_loss(x_mag, t_mag)

    def local_contrast(self, x, kernel_size=7):
        max_pool = F.max_pool2d(x, kernel_size, stride=1, padding=kernel_size//2)
        min_pool = -F.max_pool2d(-x, kernel_size, stride=1, padding=kernel_size//2)
        return max_pool - min_pool

    def forward(self, x, target):
        mse_loss = self.mse_loss(x, target).clamp(0, 10)
        l1_loss = self.l1_loss(x, target).clamp(0, 10)
        ssim_loss = self.ssim(x, target).clamp(0, 10)
        sobel_loss = self.l1_loss(self.extract_sobel(x), self.extract_sobel(target)).clamp(0, 10)
        high_freq_loss = self.l1_loss(self.extract_high_freq(x), self.extract_high_freq(target)).clamp(0, 10)
        contrast_loss = self.l1_loss(self.local_contrast(x), self.local_contrast(target)).clamp(0, 10)
        gram_loss = self.l1_loss(self.gram_matrix(x), self.gram_matrix(target)).clamp(0, 10)
        fft_loss = self.fourier_loss(x, target).clamp(0, 10)
        color_loss = self.l1_loss(x.mean(dim=[2, 3]), target.mean(dim=[2, 3])).clamp(0, 10)
        pixel_error = torch.abs(x - target)
        mask = (pixel_error > 0.1).float()
        region_loss = ((pixel_error * mask).sum() / (mask.sum() + 1e-6)).clamp(0, 10)
        
        total_loss = (
            mse_loss * 1.0 +
            l1_loss * 1.0 +
            ssim_loss * 1.0 +
            sobel_loss * 0.5 +
            high_freq_loss * 0.5 +
            contrast_loss * 0.05 +
            gram_loss * 0.05 +
            fft_loss * 0.05 +
            region_loss * 0.05 +
            color_loss * 0.01
        )
        return total_loss.clamp(0, 10)

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.register_buffer('window', self._create_window(window_size, self.channel))

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        pad = self.window_size // 2
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
        mu1 = F.conv2d(F.pad(img1, (pad, pad, pad, pad), mode='reflect'), window, padding=0, groups=channel)
        mu2 = F.conv2d(F.pad(img2, (pad, pad, pad, pad), mode='reflect'), window, padding=0, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(F.pad(img1*img1, (pad, pad, pad, pad), mode='reflect'), window, padding=0, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(F.pad(img2*img2, (pad, pad, pad, pad), mode='reflect'), window, padding=0, groups=channel) - mu2_sq
        sigma12 = F.conv2d(F.pad(img1*img2, (pad, pad, pad, pad), mode='reflect'), window, padding=0, groups=channel) - mu1_mu2
        sigma1_sq = F.relu(sigma1_sq) + 1e-8
        sigma2_sq = F.relu(sigma2_sq) + 1e-8
        C1 = (0.01 * 1)**2
        C2 = (0.03 * 1)**2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        loss = 1.0 - ssim_map
        if self.size_average:
            return loss.mean()
        else:
            return loss.mean(1).mean(1).mean(1)


class QualityDataset(Dataset):
    def __init__(self, data_dir, transform=None, crop_size=256, augment=True, cache_images=False,
                 min_quality=10, max_quality=90):
        self.data_dir = data_dir
        self.transform = transform or transforms.ToTensor()
        self.crop_size = crop_size
        self.augment = augment
        self.cache_images = cache_images
        self.image_cache = {}
        self.image_groups = {}
        self.epoch = 0
        self.min_quality = min_quality
        self.max_quality = max_quality
        all_image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                           if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        for path in all_image_paths:
            try:
                base_name_part = os.path.splitext(os.path.basename(path))[0]
                parts = base_name_part.rsplit('_', 1)
                if len(parts) == 2 and parts[1].startswith('q') and parts[1][1:].isdigit():
                    base_name = parts[0]
                    quality_str = parts[1]
                    if base_name not in self.image_groups:
                        self.image_groups[base_name] = {}
                    self.image_groups[base_name][quality_str] = path
            except Exception:
                continue
        self.valid_groups = []
        for base_name, qualities in self.image_groups.items():
            if 'q100' in qualities:
                has_valid_low_quality = False
                for q_str in qualities.keys():
                    if q_str != 'q100':
                        try:
                            q_val = int(q_str[1:])
                            if self.min_quality <= q_val <= self.max_quality:
                                has_valid_low_quality = True
                                break
                        except ValueError:
                            continue
                if has_valid_low_quality:
                    self.valid_groups.append(base_name)
        if not self.valid_groups:
            raise ValueError(f"在目錄 {data_dir} 中找不到有效的圖像組 (需要 q100 和 q{min_quality}-q{max_quality} 範圍內的圖像)")
        if self.cache_images and len(self.valid_groups) > 0:
            for base_name in self.valid_groups:
                qualities = self.image_groups[base_name]
                try:
                    high_quality_path = qualities['q100']
                    if high_quality_path not in self.image_cache:
                        self.image_cache[high_quality_path] = Image.open(high_quality_path).convert("RGB")
                    for q_name, low_quality_path in qualities.items():
                        if q_name != 'q100':
                            q_val = int(q_name[1:])
                            if self.min_quality <= q_val <= self.max_quality:
                                if low_quality_path not in self.image_cache:
                                    self.image_cache[low_quality_path] = Image.open(low_quality_path).convert("RGB")
                except Exception:
                    continue

    def __len__(self):
        return len(self.valid_groups)

    def __getitem__(self, idx):
        base_name = self.valid_groups[idx]
        qualities = self.image_groups[base_name]
        low_quality_options = []
        for q_str, path in qualities.items():
            if q_str != 'q100':
                try:
                    q_val = int(q_str[1:])
                    if self.min_quality <= q_val <= self.max_quality:
                        low_quality_options.append((q_str, path))
                except ValueError:
                    continue
        if not low_quality_options:
            raise RuntimeError(f"資料異常: {base_name}")
        weights = []
        options_paths = []
        for q_name, path in low_quality_options:
            q_num = int(q_name[1:])
            weight = 1.0 / (q_num + 1e-6)
            weights.append(weight)
            options_paths.append(path)
        sum_weight = sum(weights)
        if sum_weight > 0:
            weights = [w / sum_weight for w in weights]
        else:
            weights = [1.0 / len(options_paths)] * len(options_paths)
        low_quality_path = random.choices(options_paths, weights=weights, k=1)[0]
        high_quality_path = qualities['q100']
        try:
            if self.cache_images and low_quality_path in self.image_cache:
                low_quality_image = self.image_cache[low_quality_path].copy()
            else:
                low_quality_image = Image.open(low_quality_path).convert("RGB")
            
            if self.cache_images and high_quality_path in self.image_cache:
                high_quality_image = self.image_cache[high_quality_path].copy()
            else:
                high_quality_image = Image.open(high_quality_path).convert("RGB")
            width, height = low_quality_image.size
            if high_quality_image.size != (width, height):
                high_quality_image = high_quality_image.resize((width, height), Image.LANCZOS)
            crop_w = min(self.crop_size, width)
            crop_h = min(self.crop_size, height)
            if width >= crop_w and height >= crop_h:
                i, j, h, w = transforms.RandomCrop.get_params(
                    low_quality_image, output_size=(crop_h, crop_w)
                )
                low_quality_image = transforms.functional.crop(low_quality_image, i, j, h, w)
                high_quality_image = transforms.functional.crop(high_quality_image, i, j, h, w)
            else:
                low_quality_image = transforms.functional.resize(low_quality_image, (crop_h, crop_w), interpolation=transforms.InterpolationMode.BILINEAR)
                high_quality_image = transforms.functional.resize(high_quality_image, (crop_h, crop_w), interpolation=transforms.InterpolationMode.LANCZOS)
            if self.augment:
                if random.random() > 0.5:
                    low_quality_image = transforms.functional.hflip(low_quality_image)
                    high_quality_image = transforms.functional.hflip(high_quality_image)
                if random.random() > 0.5:
                    low_quality_image = transforms.functional.vflip(low_quality_image)
                    high_quality_image = transforms.functional.vflip(high_quality_image)
                if random.random() > 0.9:
                    angle = random.choice([0, 90, 180, 270])
                    if angle != 0:
                        low_quality_image = transforms.functional.rotate(low_quality_image, angle)
                        high_quality_image = transforms.functional.rotate(high_quality_image, angle)
            if self.transform:
                low_quality_image = self.transform(low_quality_image)
                high_quality_image = self.transform(high_quality_image)
            if low_quality_image.shape[1:] != high_quality_image.shape[1:]:
                raise RuntimeError(f"資料 shape 不符: {base_name}")
            return low_quality_image, high_quality_image
        except Exception as e:
            raise RuntimeError(f"資料異常: {base_name} ({e})")

def format_time(seconds):
    """將秒數格式化為易讀的時間字串"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def calculate_psnr(img1, img2, data_range=1.0):
    """計算兩張圖片的峰值信噪比 (PSNR)"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(data_range / torch.sqrt(mse))
    return psnr.item()

def validate(generator, val_loader, device, max_validate_batches=None, return_images=False, crop_size=256):
    """驗證模型性能並計算PSNR"""
    generator.eval()
    val_psnr_list = []
    val_mse_list = []
    validation_images = []
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            if max_validate_batches is not None and i >= max_validate_batches:
                break
            images, targets = images.to(device), targets.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                fake_images = generator(images)
            fake_images = torch.clamp(fake_images, 0.0, 1.0)
            batch_size = images.size(0)
            for j in range(batch_size):
                psnr = calculate_psnr(fake_images[j], targets[j], data_range=1.0)
                if not math.isinf(psnr) and not math.isnan(psnr):
                    val_psnr_list.append(psnr)
                    val_mse_list.append(F.mse_loss(fake_images[j], targets[j]).item())
                if return_images and i < 1 and j < 4:
                    validation_images.append((
                        transforms.ToPILImage()(images[j].cpu()),
                        transforms.ToPILImage()(fake_images[j].cpu()),
                        transforms.ToPILImage()(targets[j].cpu())
                    ))
    avg_psnr = np.mean(val_psnr_list) if val_psnr_list else 0.0
    avg_mse = np.mean(val_mse_list) if val_mse_list else float('inf')
    logger.info(f"驗證完成. 平均 PSNR: {avg_psnr:.4f} dB, 平均 MSE: {avg_mse:.6f}")
    if return_images:
        return avg_psnr, validation_images
    return avg_psnr

def save_model_with_metadata(model, path, metadata=None):
    """保存模型並附帶元數據"""
    torch.save(model.state_dict(), path)
    logger.info(f"模型已保存至: {path}")
    if metadata:
        metadata_path = os.path.splitext(path)[0] + "_info.json"
        try:
            cleaned_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, dict):
                    cleaned_metadata[key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, torch.Tensor):
                            cleaned_metadata[key][sub_key] = sub_value.item()
                        else:
                            cleaned_metadata[key][sub_key] = sub_value
                elif isinstance(value, torch.Tensor):
                    cleaned_metadata[key] = value.item()
                else:
                    cleaned_metadata[key] = value
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_metadata, f, ensure_ascii=False, indent=4)
            logger.info(f"元數據已保存至: {metadata_path}")
        except Exception as e:
            logger.error(f"保存元數據時出錯: {e}")


class Trainer:
    def __init__(self, data_dir=None, save_dir="./models", log_dir="./logs", 
                batch_size=8, learning_rate=0.0001, num_epochs=50,
                save_options=None, training_args=None):
        """初始化訓練器"""
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        device_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
        logger.info(f"使用設備: {self.device} ({device_name})")
        if torch.cuda.is_available():
            logger.info(f"GPU 記憶體: {device_memory:.2f} GB")
        self.save_options = {
            'save_best_loss': True,
            'save_best_psnr': True,
            'save_final': True,
            'save_checkpoint': False,
            'checkpoint_interval': 10
        }
        if save_options:
            self.save_options.update(save_options)
        self.training_args = {
            'crop_size': 256,
            'min_quality': 10,
            'max_quality': 90,
            'optimizer': 'AdamW',
            'weight_decay': 1e-6,
            'scheduler': 'cosine',
            'plateau_patience': 15,
            'plateau_factor': 0.2,
            'cosine_t_max': 1000,
            'step_size': 100,
            'step_gamma': 0.5,
            'min_lr': 1e-7,
            'd_lr_factor': 0.5,
            'grad_accum': 4,
            'max_grad_norm': 0.5,
            'cache_images': False,
            'validation_interval': 1,
            'fast_validation': False,
            'validate_batches': 50,
            'checkpoint_interval': 50,
            'num_workers': 4,
            'pin_memory': True
        }
        if training_args:
            self.training_args.update(training_args)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

    def prepare_dataset(self, data_dir=None):
        """準備資料集"""
        if data_dir:
            self.data_dir = data_dir
        if not self.data_dir or not os.path.exists(self.data_dir):
            raise ValueError(f"資料集目錄不存在: {self.data_dir}")
        logger.info(f"正在載入資料集: {self.data_dir}")
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = QualityDataset(
            self.data_dir,
            transform=transform,
            crop_size=self.training_args['crop_size'],
            cache_images=self.training_args['cache_images'],
            min_quality=self.training_args['min_quality'],
            max_quality=self.training_args['max_quality']
        )
        dataset_size = len(dataset)
        if dataset_size == 0:
            raise ValueError("資料集為空，請檢查資料目錄和品質範圍設置")
        val_split = 0.1
        val_size = max(1, int(val_split * dataset_size))
        train_size = dataset_size - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        num_workers = min(self.training_args['num_workers'], os.cpu_count() // 2 if os.cpu_count() else 4)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=self.training_args['pin_memory'] and torch.cuda.is_available(),
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=self.training_args['pin_memory'] and torch.cuda.is_available(),
            persistent_workers=True if num_workers > 0 else False
        )
        logger.info(f"數據集大小: {dataset_size}, 訓練集: {train_size}, 驗證集: {val_size}")
        return train_loader, val_loader

    def train(self, progress_callback=None, epoch_callback=None, stop_check_callback=None):
        """訓練模型"""
        try:
            train_loader, val_loader = self.prepare_dataset()
            generator = ImageQualityEnhancer(num_rrdb_blocks=16, features=64)
            discriminator = MultiScaleDiscriminator(num_scales=3, input_channels=3)
            criterion_dict = {
                'perceptual': EnhancedPerceptualLoss().to(self.device),
                'ssim': SSIMLoss().to(self.device)
            }
            if self.training_args['optimizer'] == 'AdamW':
                g_optimizer = torch.optim.AdamW(
                    generator.parameters(),
                    lr=self.learning_rate,
                    betas=(0.9, 0.999),
                    weight_decay=self.training_args['weight_decay']
                )
                d_optimizer = torch.optim.AdamW(
                    discriminator.parameters(),
                    lr=self.learning_rate * self.training_args['d_lr_factor'],
                    betas=(0.9, 0.999),
                    weight_decay=self.training_args['weight_decay']
                )
            else:
                g_optimizer = torch.optim.Adam(
                    generator.parameters(),
                    lr=self.learning_rate,
                    betas=(0.9, 0.999)
                )
                d_optimizer = torch.optim.Adam(
                    discriminator.parameters(),
                    lr=self.learning_rate * self.training_args['d_lr_factor'],
                    betas=(0.9, 0.999)
                )
            if self.training_args['scheduler'] == 'cosine':
                scheduler_g = CosineAnnealingLR(
                    g_optimizer,
                    T_max=self.training_args['cosine_t_max'],
                    eta_min=self.training_args['min_lr']
                )
                scheduler_d = CosineAnnealingLR(
                    d_optimizer,
                    T_max=self.training_args['cosine_t_max'],
                    eta_min=self.training_args['min_lr'] * self.training_args['d_lr_factor']
                )
            elif self.training_args['scheduler'] == 'step':
                scheduler_g = torch.optim.lr_scheduler.StepLR(
                    g_optimizer,
                    step_size=self.training_args['step_size'],
                    gamma=self.training_args['step_gamma']
                )
                scheduler_d = torch.optim.lr_scheduler.StepLR(
                    d_optimizer,
                    step_size=self.training_args['step_size'],
                    gamma=self.training_args['step_gamma']
                )
            else:
                scheduler_g = ReduceLROnPlateau(
                    g_optimizer,
                    mode='max',
                    factor=self.training_args['plateau_factor'],
                    patience=self.training_args['plateau_patience'],
                    verbose=True,
                    min_lr=self.training_args['min_lr']
                )
                scheduler_d = ReduceLROnPlateau(
                    d_optimizer,
                    mode='max',
                    factor=self.training_args['plateau_factor'],
                    patience=self.training_args['plateau_patience'],
                    verbose=True,
                    min_lr=self.training_args['min_lr'] * self.training_args['d_lr_factor']
                )
            generator, discriminator = self._train_model(
                generator, discriminator, train_loader, val_loader, 
                criterion_dict, g_optimizer, d_optimizer, 
                scheduler_g, scheduler_d,
                progress_callback, epoch_callback, stop_check_callback
            )
            return generator
        except Exception as e:
            logger.error(f"訓練過程中發生錯誤: {e}")
            raise

    def _train_model(self, generator, discriminator, train_loader, val_loader, 
                    criterion_dict, g_optimizer, d_optimizer, scheduler_g, scheduler_d,
                    progress_callback=None, epoch_callback=None, stop_check_callback=None):
        """核心訓練函數"""
        generator.to(self.device)
        discriminator.to(self.device)
        scaler = torch.amp.GradScaler(enabled=(self.device.type == 'cuda'))
        best_g_loss = float('inf')
        best_psnr = 0.0
        log_file = os.path.join(self.log_dir, f"training_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        with open(log_file, "w") as f:
            f.write("Epoch,G_Loss,D_Loss,MSE_Loss,L1_Loss,Perceptual_Loss,SSIM_Loss,Adv_Loss,PSNR,Time_Epoch,Time_Total,LR_G,LR_D\n")
        training_start_time = time.time()
        mse_weight = 1.0
        l1_weight = 1.0
        perceptual_weight = 1.0
        ssim_weight = 1.0
        adversarial_weight = 0.0002
        perceptual_criterion = criterion_dict['perceptual']
        ssim_criterion = criterion_dict['ssim']
        model_metadata = {
            "model_name": "NS-IC",
            "version": "NS-IC-v7",
            "architecture": {
                "type": "ImageQualityEnhancer",
                "num_rrdb_blocks": len(generator.rrdb_blocks) if hasattr(generator, 'rrdb_blocks') else 'Unknown',
                "features": generator.conv_first.out_channels if hasattr(generator, 'conv_first') else 'Unknown'
            },
            "training_args": self.training_args,
            "training_info": {
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_size": len(train_loader.dataset) + len(val_loader.dataset),
                "total_epochs_planned": self.num_epochs,
                "quality_range": f"q{self.training_args['min_quality']}-q{self.training_args['max_quality']}",
                "loss_weights": {
                    "mse": mse_weight,
                    "l1": l1_weight,
                    "perceptual": perceptual_weight,
                    "ssim": ssim_weight,
                    "adversarial": adversarial_weight
                }
            },
            "performance": {
                "best_psnr": best_psnr,
                "best_g_loss": best_g_loss,
            }
        }
        for epoch in range(self.num_epochs):
            if stop_check_callback and stop_check_callback():
                logger.info("收到停止信號，中斷訓練")
                break
            generator.train()
            discriminator.train()
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_mse_loss = 0.0
            epoch_l1_loss = 0.0
            epoch_perceptual_loss = 0.0
            epoch_ssim_loss = 0.0
            epoch_adv_loss = 0.0
            epoch_start_time = time.time()
            batch_count = 0
            if hasattr(train_loader.dataset, 'dataset') and hasattr(train_loader.dataset.dataset, 'epoch'):
                train_loader.dataset.dataset.epoch = epoch
            elif hasattr(train_loader.dataset, 'epoch'):
                train_loader.dataset.epoch = epoch
            for i, (images, targets) in enumerate(train_loader):
                if stop_check_callback and stop_check_callback():
                    logger.info("收到停止信號，中斷訓練")
                    break
                if torch.all(images == 0) or torch.all(targets == 0):
                    logger.warning(f"Epoch {epoch+1}, Batch {i+1}: 批次為全 0，跳過。")
                    continue
                images, targets = images.to(self.device), targets.to(self.device)
                if torch.isnan(images).any() or torch.isinf(images).any() or \
                   torch.isnan(targets).any() or torch.isinf(targets).any():
                    logger.warning(f"Epoch {epoch+1}, Batch {i+1}: 輸入數據包含 NaN/Inf 值。跳過此批次。")
                    continue
                if batch_count % self.training_args['grad_accum'] == 0:
                    d_optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                    fake_images = generator(images)
                    if torch.isnan(fake_images).any() or torch.isinf(fake_images).any():
                        logger.warning(f"Epoch {epoch+1}, Batch {i+1}: 生成器輸出包含 NaN/Inf 值。跳過此批次。")
                        continue
                    fake_images_detached = fake_images.detach()
                    real_outputs = discriminator(targets)
                    fake_outputs = discriminator(fake_images_detached)
                    d_loss_real = 0
                    d_loss_fake = 0
                    num_outputs = len(real_outputs)
                    for scale_idx in range(num_outputs):
                        d_loss_real += torch.mean((real_outputs[scale_idx] - 1.0) ** 2)
                        d_loss_fake += torch.mean((fake_outputs[scale_idx] - 0.0) ** 2)
                    d_loss = 0.5 * (d_loss_real + d_loss_fake) / num_outputs
                    d_loss_scaled = d_loss / self.training_args['grad_accum']
                if not (torch.isnan(d_loss_scaled).any() or torch.isinf(d_loss_scaled).any()):
                    scaler.scale(d_loss_scaled).backward()
                    epoch_d_loss += d_loss.item()
                if batch_count % self.training_args['grad_accum'] == 0:
                    g_optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                    fake_outputs_for_g = discriminator(fake_images)
                    mse_loss = F.mse_loss(fake_images, targets).clamp(0, 10)
                    l1_loss = F.l1_loss(fake_images, targets).clamp(0, 10)
                    perceptual_loss = perceptual_criterion(fake_images, targets).clamp(0, 10)
                    ssim_loss = ssim_criterion(fake_images, targets).clamp(0, 10)
                    adversarial_g_loss = 0
                    num_outputs_g = len(fake_outputs_for_g)
                    for scale_idx in range(num_outputs_g):
                        adversarial_g_loss += torch.mean((fake_outputs_for_g[scale_idx] - 1.0) ** 2)
                    adversarial_g_loss /= num_outputs_g
                    g_loss = (mse_weight * mse_loss +
                              l1_weight * l1_loss +
                              perceptual_weight * perceptual_loss +
                              ssim_weight * ssim_loss +
                              adversarial_weight * adversarial_g_loss)
                    
                    g_loss = g_loss.clamp(0, 10)
                    g_loss_scaled = g_loss / self.training_args['grad_accum']
                if not (torch.isnan(g_loss_scaled).any() or torch.isinf(g_loss_scaled).any()):
                    scaler.scale(g_loss_scaled).backward()
                    epoch_g_loss += g_loss.item()
                    epoch_mse_loss += mse_loss.item()
                    epoch_l1_loss += l1_loss.item()
                    epoch_perceptual_loss += perceptual_loss.item()
                    epoch_ssim_loss += ssim_loss.item()
                    epoch_adv_loss += adversarial_g_loss.item()
                batch_count += 1
                if batch_count % self.training_args['grad_accum'] == 0 or (i == len(train_loader) - 1):
                    if not (torch.isnan(d_loss_scaled).any() or torch.isinf(d_loss_scaled).any()):
                        scaler.unscale_(d_optimizer)
                        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=self.training_args['max_grad_norm'])
                        scaler.step(d_optimizer)
                    if not (torch.isnan(g_loss_scaled).any() or torch.isinf(g_loss_scaled).any()):
                        scaler.unscale_(g_optimizer)
                        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=self.training_args['max_grad_norm'])
                        scaler.step(g_optimizer)
                    scaler.update()
                if progress_callback and (i % 10 == 0 or i == len(train_loader) - 1):
                    progress = (i + 1) / len(train_loader)
                    current_g_loss = g_loss.item() if 'g_loss' in locals() and not torch.isnan(g_loss).any() else 0
                    current_d_loss = d_loss.item() if 'd_loss' in locals() and not torch.isnan(d_loss).any() else 0
                    progress_callback(epoch, i, len(train_loader), current_g_loss, current_d_loss)
            epoch_time = time.time() - epoch_start_time
            total_training_time = time.time() - training_start_time
            num_batches = len(train_loader)
            avg_g_loss = epoch_g_loss / num_batches if num_batches > 0 else 0
            avg_d_loss = epoch_d_loss / num_batches if num_batches > 0 else 0
            avg_mse_loss = epoch_mse_loss / num_batches if num_batches > 0 else 0
            avg_l1_loss = epoch_l1_loss / num_batches if num_batches > 0 else 0
            avg_perceptual_loss = epoch_perceptual_loss / num_batches if num_batches > 0 else 0
            avg_ssim_loss = epoch_ssim_loss / num_batches if num_batches > 0 else 0
            avg_adv_loss = epoch_adv_loss / num_batches if num_batches > 0 else 0
            val_psnr = 0.0
            if (epoch + 1) % self.training_args['validation_interval'] == 0:
                logger.info(f"--- 驗證輪數 {epoch+1} ---")
                validate_batches = self.training_args['validate_batches'] if self.training_args['fast_validation'] else None
                val_psnr, validation_images = validate(generator, val_loader, self.device,
                                                     max_validate_batches=validate_batches,
                                                     return_images=True,
                                                     crop_size=self.training_args['crop_size'])
                logger.info(f"--- 驗證完成 ---")
            current_lr_g = g_optimizer.param_groups[0]['lr']
            current_lr_d = d_optimizer.param_groups[0]['lr']
            if isinstance(scheduler_g, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if val_psnr > 0:
                    scheduler_g.step(val_psnr)
                    scheduler_d.step(val_psnr)
            else:
                scheduler_g.step()
                scheduler_d.step()
            new_lr_g = g_optimizer.param_groups[0]['lr']
            new_lr_d = d_optimizer.param_groups[0]['lr']
            with open(log_file, "a") as f:
                f.write(f"{epoch+1},{avg_g_loss:.6f},{avg_d_loss:.6f},"
                        f"{avg_mse_loss:.6f},{avg_l1_loss:.6f},{avg_perceptual_loss:.6f},{avg_ssim_loss:.6f},{avg_adv_loss:.6f},"
                        f"{val_psnr:.6f},{epoch_time:.2f},{total_training_time:.2f},{new_lr_g:.8f},{new_lr_d:.8f}\n")
            model_metadata["training_info"]["last_completed_epoch"] = epoch + 1
            model_metadata["training_info"]["total_time_seconds"] = total_training_time
            model_metadata["performance"]["current_psnr"] = val_psnr
            model_metadata["performance"]["current_g_loss"] = avg_g_loss
            is_best_psnr = False
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                is_best_psnr = True
                model_metadata["performance"]["best_psnr"] = best_psnr
                model_metadata["performance"]["best_psnr_epoch"] = epoch + 1
                if self.save_options['save_best_psnr']:
                    best_psnr_path = os.path.join(self.save_dir, f"NS-IC_best_psnr.pth")
                    save_model_with_metadata(generator, best_psnr_path, model_metadata)
                    logger.info(f"*** 新的最佳 PSNR 模型已保存 (Epoch {epoch+1}, PSNR: {val_psnr:.4f} dB) ***")
            is_best_loss = False
            if avg_g_loss < best_g_loss:
                best_g_loss = avg_g_loss
                is_best_loss = True
                model_metadata["performance"]["best_g_loss"] = best_g_loss
                model_metadata["performance"]["best_g_loss_epoch"] = epoch + 1
                if self.save_options['save_best_loss']:
                    best_loss_path = os.path.join(self.save_dir, f"NS-IC_best_loss.pth")
                    save_model_with_metadata(generator, best_loss_path, model_metadata)
            save_checkpoint = (epoch + 1) % self.training_args['checkpoint_interval'] == 0 or epoch == (self.num_epochs - 1)
            if self.save_options['save_checkpoint'] and save_checkpoint:
                checkpoint = {
                    'epoch': epoch + 1,
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'g_optimizer_state_dict': g_optimizer.state_dict(),
                    'd_optimizer_state_dict': d_optimizer.state_dict(),
                    'g_scheduler_state_dict': scheduler_g.state_dict(),
                    'd_scheduler_state_dict': scheduler_d.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'g_loss': avg_g_loss,
                    'd_loss': avg_d_loss,
                    'psnr': val_psnr,
                    'best_g_loss': best_g_loss,
                    'best_psnr': best_psnr,
                    'metadata': model_metadata,
                    'args': self.training_args
                }
                checkpoint_path = os.path.join(self.save_dir, f"NS-IC_checkpoint_epoch_{epoch+1}.pth")
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"檢查點已保存: {checkpoint_path}")
            if epoch_callback:
                epoch_callback(epoch, avg_g_loss, avg_d_loss, val_psnr)
            if (epoch + 1) % 5 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if self.save_options['save_final']:
            final_path = os.path.join(self.save_dir, f"NS-IC_final_epoch_{self.num_epochs}.pth")
            model_metadata["training_info"]["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            model_metadata["training_info"]["total_time_formatted"] = format_time(time.time() - training_start_time)
            save_model_with_metadata(generator, final_path, model_metadata)
        logger.info("訓練完成！")
        logger.info(f"最佳 PSNR: {best_psnr:.4f} dB")
        logger.info(f"日誌文件保存在: {log_file}")
        return generator, discriminator