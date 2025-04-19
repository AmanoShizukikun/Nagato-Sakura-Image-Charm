import os
import time
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split

from src.models.NS_ImageEnhancer import ImageQualityEnhancer


logger = logging.getLogger("Trainer")
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logs_dir = os.path.join(base_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)
log_file = os.path.join(logs_dir, f"training_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, num_scales=3, input_channels=3):
        super(MultiScaleDiscriminator, self).__init__()
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList()
        
        for _ in range(num_scales):
            self.discriminators.append(self._create_discriminator(input_channels))
            
    def _create_discriminator(self, input_channels):
        return nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        outputs = []
        for discriminator in self.discriminators:
            outputs.append(discriminator(x))
            x = F.avg_pool2d(x, kernel_size=2)
        return outputs

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.loss = nn.L1Loss()
        
    def forward(self, x, y):
        return self.loss(x, y)

class QualityDataset(Dataset):
    def __init__(self, data_dir, high_quality_extensions=None, transform=None):
        self.data_dir = data_dir
        self.transform = transform or transforms.ToTensor()
        self.high_quality_ext = high_quality_extensions or ["_q100.jpg"]
        self.file_pairs = []
        self._scan_and_pair_files()
        
    def _scan_and_pair_files(self):
        all_files = os.listdir(self.data_dir)
        high_quality_files = set()
        for ext in self.high_quality_ext:
            high_quality_files.update([f for f in all_files if f.endswith(ext)])
        for file in all_files:
            if file in high_quality_files:
                continue
            base_name = os.path.splitext(file)[0]
            if "_q" in base_name:
                prefix = base_name.split("_q")[0]
                for ext in self.high_quality_ext:
                    high_quality_file = f"{prefix}{ext}"
                    if high_quality_file in high_quality_files:
                        self.file_pairs.append((file, high_quality_file))
                        break
    
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        low_quality_file, high_quality_file = self.file_pairs[idx]
        low_quality_path = os.path.join(self.data_dir, low_quality_file)
        high_quality_path = os.path.join(self.data_dir, high_quality_file)
        low_quality_img = Image.open(low_quality_path).convert('RGB')
        high_quality_img = Image.open(high_quality_path).convert('RGB')
        if self.transform:
            low_quality_img = self.transform(low_quality_img)
            high_quality_img = self.transform(high_quality_img)
        return low_quality_img, high_quality_img

def format_time(seconds):
    """將秒數格式化為易讀的時間字串"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"

def calculate_psnr(img1, img2):
    """計算兩張圖片的峰值信噪比 (PSNR)"""
    mse = F.mse_loss(img1, img2).item()
    if mse == 0:
        return 100
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def validate(generator, val_loader, device):
    """驗證模型性能並計算PSNR"""
    generator.eval()
    total_psnr = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = generator(images)
            total_psnr += calculate_psnr(outputs, targets)
    return total_psnr / len(val_loader)

class Trainer:
    def __init__(self, data_dir=None, save_dir="./models", log_dir="./logs", 
                batch_size=8, learning_rate=0.0001, num_epochs=50,
                save_options=None):
        """初始化訓練器
        Args:
            data_dir: 訓練資料集目錄
            save_dir: 模型儲存目錄
            log_dir: 日誌儲存目錄
            batch_size: 批次大小
            learning_rate: 學習率
            num_epochs: 訓練週期數
            save_options: 模型保存選項字典，包含以下鍵值：
                - save_best_loss: 是否保存最佳損失模型
                - save_best_psnr: 是否保存最佳PSNR模型
                - save_final: 是否保存最終模型
                - save_checkpoint: 是否定期保存檢查點
                - checkpoint_interval: 檢查點保存間隔 (以epoch為單位)
        """
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
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

    def prepare_dataset(self, data_dir=None):
        """準備資料集"""
        if data_dir:
            self.data_dir = data_dir
        if not self.data_dir or not os.path.exists(self.data_dir):
            raise ValueError(f"資料集目錄不存在: {self.data_dir}")   
        logger.info(f"正在載入資料集: {self.data_dir}")
        transform = transforms.Compose([
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        dataset = QualityDataset(
            self.data_dir,
            high_quality_extensions=["_q100.jpg"],
            transform=transform
        )
        dataset_size = len(dataset)
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )
        logger.info(f"數據集大小: {dataset_size}, 訓練集: {train_size}, 驗證集: {val_size}")
        
        return train_loader, val_loader

    def train(self, progress_callback=None, epoch_callback=None, stop_check_callback=None):
        """訓練模型
        Args:
            progress_callback: 每批次更新的回調函數，接收參數 (epoch, batch, total_batches, loss)
            epoch_callback: 每週期完成後的回調函數，接收參數 (epoch, g_loss, d_loss, psnr)
            stop_check_callback: 檢查是否中止訓練的回調函數
            
        Returns:
            訓練完成的生成器模型
        """
        train_loader, val_loader = self.prepare_dataset()
        
        # 初始化模型
        generator = ImageQualityEnhancer(num_rrdb_blocks=16)
        discriminator = MultiScaleDiscriminator(num_scales=3)
        perceptual_criterion = PerceptualLoss()
        
        # 優化器
        g_optimizer = torch.optim.Adam(generator.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        
        # 學習率調度
        scheduler_g = CosineAnnealingLR(g_optimizer, T_max=self.num_epochs, eta_min=1e-7)
        scheduler_d = CosineAnnealingLR(d_optimizer, T_max=self.num_epochs, eta_min=1e-7)
        generator.to(self.device)
        discriminator.to(self.device)
        scaler = torch.amp.GradScaler()
        best_g_loss = float('inf')
        best_psnr = 0.0
        log_file = os.path.join(self.log_dir, f"training_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
        with open(log_file, "w") as f:
            f.write("Epoch,G_Loss,D_Loss,PSNR,Time\n")
        training_start_time = time.time()
        
        # 設定損失權重
        mse_weight = 1.0
        perceptual_weight = 0.1
        adversarial_weight = 0.01
        
        # 開始訓練
        for epoch in range(self.num_epochs):
            if stop_check_callback and stop_check_callback():
                logger.info("訓練已中止")
                break
            generator.train()
            discriminator.train()
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_start_time = time.time()
            for i, (images, targets) in enumerate(train_loader):
                if stop_check_callback and stop_check_callback():
                    break
                images, targets = images.to(self.device), targets.to(self.device)
                d_optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
                    real_outputs = discriminator(targets)
                    fake_images = generator(images)
                    fake_outputs = discriminator(fake_images.detach())
                    d_loss_real = 0
                    d_loss_fake = 0
                    for real_output, fake_output in zip(real_outputs, fake_outputs):
                        d_loss_real += torch.mean(-torch.log(real_output + 1e-8))
                        d_loss_fake += torch.mean(-torch.log(1 - fake_output + 1e-8))
                    d_loss = (d_loss_real + d_loss_fake) / len(real_outputs)
                scaler.scale(d_loss).backward()
                scaler.step(d_optimizer)
                scaler.update()
                g_optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
                    fake_outputs = discriminator(fake_images)
                    mse_loss = F.mse_loss(fake_images, targets)
                    perceptual_loss = perceptual_criterion(fake_images, targets)
                    adversarial_loss = 0
                    for fake_output in fake_outputs:
                        adversarial_loss += torch.mean(-torch.log(fake_output + 1e-8))
                    adversarial_loss /= len(fake_outputs)
                    g_loss = (mse_weight * mse_loss + 
                              perceptual_weight * perceptual_loss + 
                              adversarial_weight * adversarial_loss)
                scaler.scale(g_loss).backward()
                scaler.step(g_optimizer)
                scaler.update()
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                if progress_callback:
                    progress_callback(epoch, i, len(train_loader), g_loss.item(), d_loss.item())
                if i % 5 == 0 or i == len(train_loader) - 1:
                    progress = (i + 1) / len(train_loader)
                    percentage = progress * 100
                    elapsed_time = time.time() - epoch_start_time
                    eta = elapsed_time / progress - elapsed_time if progress > 0 else 0
                    fill_length = int(50 * progress)
                    space_length = 50 - fill_length
                    logger.info(f"Epoch [{epoch+1}/{self.num_epochs}] "
                          f"Progress: {percentage:3.0f}%|{'█' * fill_length}{' ' * space_length}| "
                          f"[{format_time(elapsed_time)}<{format_time(eta)}] "
                          f"G: {g_loss.item():.4f}, D: {d_loss.item():.4f}")
            if stop_check_callback and stop_check_callback():
                break
            scheduler_g.step()
            scheduler_d.step()
            avg_g_loss = epoch_g_loss / len(train_loader)
            avg_d_loss = epoch_d_loss / len(train_loader)
            val_psnr = validate(generator, val_loader, self.device)
            total_training_time = time.time() - training_start_time
            logger.info(f"\nEpoch [{epoch+1}/{self.num_epochs}], G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}, "
                  f"PSNR: {val_psnr:.2f} dB [{format_time(total_training_time)} elapsed]")
            with open(log_file, "a") as f:
                f.write(f"{epoch+1},{avg_g_loss:.6f},{avg_d_loss:.6f},{val_psnr:.6f},{total_training_time:.2f}\n")
            if self.save_options['save_best_loss'] and avg_g_loss < best_g_loss:
                best_g_loss = avg_g_loss
                torch.save(generator.state_dict(), os.path.join(self.save_dir, "best_loss_generator.pth"))
                logger.info(f"最佳損失模型已保存於 Epoch {epoch+1}，G Loss: {avg_g_loss:.4f}")
            if self.save_options['save_best_psnr'] and val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(generator.state_dict(), os.path.join(self.save_dir, "best_psnr_generator.pth"))
                logger.info(f"最佳PSNR模型已保存於 Epoch {epoch+1}，PSNR: {val_psnr:.2f} dB")
            if self.save_options['save_checkpoint'] and \
               (epoch + 1) % self.save_options['checkpoint_interval'] == 0:
                checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'g_optimizer_state_dict': g_optimizer.state_dict(),
                    'd_optimizer_state_dict': d_optimizer.state_dict(),
                    'g_loss': avg_g_loss,
                    'psnr': val_psnr,
                }, checkpoint_path)
                logger.info(f"週期檢查點已保存: {checkpoint_path}")
            if epoch_callback:
                epoch_callback(epoch+1, avg_g_loss, avg_d_loss, val_psnr)
        if self.save_options['save_final']:
            torch.save(generator.state_dict(), os.path.join(self.save_dir, "final_generator.pth"))
            logger.info("訓練完成！最終模型已保存！")
        return generator