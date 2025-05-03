import os
import re
import gc
import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


logger = logging.getLogger(__name__)

def should_use_amp(device):
    """自動檢測是否適合使用混合精度計算"""
    if device.type != 'cuda':
        return False
    gpu_name = torch.cuda.get_device_name(device)
    logger.info(f"正在檢測GPU '{gpu_name}' 是否適合使用混合精度...")
    try:
        cuda_version = torch.version.cuda
        if cuda_version:
            cuda_major = int(cuda_version.split('.')[0])
            if cuda_major < 10:
                logger.info("CUDA版本低於10.0，禁用混合精度計算")
                return False
    except Exception as e:
        logger.warning(f"無法獲取CUDA版本信息: {e}")
    excluded_gpus = ['1650', '1660', 'MX', 'P4', 'P40', 'K80', 'M4']
    for model in excluded_gpus:
        if model in gpu_name:
            logger.info(f"檢測到GPU型號 {model} 在排除列表中，禁用混合精度")
            return False
    amp_supported_gpus = ['RTX', 'A100', 'A10', 'V100', 'T4', '30', '40', 'TITAN V']
    for model in amp_supported_gpus:
        if model in gpu_name:
            logger.info(f"檢測到GPU型號 {model} 支持混合精度計算")
            return True
    cc_match = re.search(r'compute capability: (\d+)\.(\d+)', gpu_name.lower())
    if cc_match:
        major = int(cc_match.group(1))
        minor = int(cc_match.group(2))
        compute_capability = float(f"{major}.{minor}")
        if compute_capability >= 7.0:
            logger.info(f"GPU計算能力 {compute_capability} >= 7.0，啟用混合精度")
            return True
    logger.info("無法確定GPU是否支持混合精度，為安全起見禁用")
    return False

class DepthwiseSeparableConv(nn.Module):
    """深度可分離卷積層，將標準卷積分解為逐深度卷積和逐點卷積"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class NagatoSakuraImageQualityClassificationCNN(nn.Module):
    """
    輕量化CNN模型，使用深度可分離卷積、批量歸一化和全局平均池化減少參數量。
    """
    def __init__(self, dropout_rate: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            # 第一區塊
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二區塊
            DepthwiseSeparableConv(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第三區塊
            DepthwiseSeparableConv(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第四區塊
            DepthwiseSeparableConv(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ImageQualityScorer:
    """使用AI模型評估圖像品質分數"""
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.use_amp = False
        if self.device.type == 'cuda':
            self.use_amp = should_use_amp(self.device)
    
    def load_model(self):
        """按需載入模型"""
        if self.model is not None:
            logger.info("模型已經載入，無需重複載入")
            return True
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            potential_paths = [
                os.path.join(base_dir, "extensions","Nagato-Sakura-Image-Quality-Assessment", "models", "NS-IQA.pth")
            ]
            model_path = None
            for path in potential_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            if model_path is None:
                logger.error("找不到品質評分模型文件，請確認模型已放置在extensions資料夾中")
                return False
            self.model = NagatoSakuraImageQualityClassificationCNN(dropout_rate=0)
            try:
                model_state = torch.load(model_path, map_location=self.device, weights_only=True)
                logger.info("使用weights_only=True載入模型")
            except TypeError:
                logger.info("不支持weights_only參數，使用兼容模式載入")
                model_state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(model_state)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"成功載入圖像品質評分模型: {model_path}")
            return True
        except Exception as e:
            logger.error(f"載入圖像品質評分模型失敗: {e}")
            self.model = None
            return False
    
    def unload_model(self):
        """卸載模型釋放資源"""
        if self.model is None:
            return
        try:
            if self.device.type == 'cuda':
                self.model.to('cpu')
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("已成功卸載圖像品質評分模型並釋放資源")
        except Exception as e:
            logger.error(f"卸載模型時出錯: {e}")
    
    def score_image(self, image):
        """評估圖像品質分數
        參數:
            image: PIL.Image 對象
        返回:
            float或None: 品質分數，如果模型未載入則返回None
        """
        model_loaded = self.model is not None or self.load_model()
        if not model_loaded:
            logger.error("無法評分: 模型未成功載入")
            return None 
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                try:
                    if self.device.type == 'cuda' and self.use_amp:
                        with torch.amp.autocast('cuda', enabled=True):
                            prediction = self.model(image_tensor)
                    else:
                        prediction = self.model(image_tensor)
                except AttributeError:
                    if self.device.type == 'cuda' and self.use_amp:
                        with torch.cuda.amp.autocast(enabled=True):
                            prediction = self.model(image_tensor)
                    else:
                        prediction = self.model(image_tensor)
            raw_score = prediction.item()
            logger.info(f"圖像評分 (原始值: {raw_score:.4f}, 轉換後: {raw_score:.2f})")
            return raw_score
        except Exception as e:
            logger.error(f"評分過程中出現錯誤: {e}")
            return None 