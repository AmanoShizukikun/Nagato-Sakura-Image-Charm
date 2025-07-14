import time
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal
import logging
import re


class EnhancerThread(QThread):
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal(Image.Image, float)
    
    def __init__(self, model, image_path, device, block_size=256, overlap=64, 
                 use_weight_mask=True, blending_mode='改進型高斯分佈', strength=1.0, 
                 upscale_factor=1, target_width=0, target_height=0, 
                 maintain_aspect_ratio=False, resize_mode="延伸至適合大小",
                 use_amp=None):
        super().__init__()
        self.model = model
        self.image_path = image_path
        self.device = device
        self.block_size = block_size
        self.overlap = overlap
        self.use_weight_mask = use_weight_mask
        self.blending_mode = blending_mode
        self.strength = strength 
        self.upscale_factor = upscale_factor 
        self.target_width = target_width
        self.target_height = target_height
        self.maintain_aspect_ratio = maintain_aspect_ratio
        self.resize_mode = resize_mode
        self.weight_mask_cache = {}
        try:
            model_device = next(model.parameters()).device
            if model_device != device:
                logging.debug(f"將模型從 {model_device} 移動到 {device}")
                self.model = model.to(device)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as e:
            logging.error(f"檢查模型設備時出錯: {e}")
            self.model = model.to(device)
        if use_amp is None:
            self.use_amp = self._should_use_amp(device)
        else:
            self.use_amp = use_amp
        if device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(device)
            logging.info(f"使用GPU: {gpu_name}")
            logging.info(f"CUDA版本: {torch.version.cuda}")
            logging.info(f"混合精度計算: {'啟用' if self.use_amp else '禁用'}")
        elif device.type == 'mps':
            logging.info(f"使用MPS設備: Apple Metal GPU")
            logging.info(f"混合精度計算: 禁用 (MPS不支援)")
        else:
            logging.info(f"使用CPU作為計算設備")
    
    def _should_use_amp(self, device):
        """自動檢測是否應使用混合精度計算"""
        if device.type != 'cuda':
            if device.type == 'mps':
                logging.info("檢測到MPS設備，MPS不支援混合精度計算")
            return False
        gpu_name = torch.cuda.get_device_name(device)
        logging.info(f"正在檢測GPU '{gpu_name}' 是否適合使用混合精度...")
        try:
            cuda_version = torch.version.cuda
            if cuda_version:
                cuda_major = int(cuda_version.split('.')[0])
                if cuda_major < 10:
                    logging.info("CUDA版本低於10.0，禁用混合精度計算")
                    return False
        except Exception as e:
            logging.warning(f"無法獲取CUDA版本信息: {e}")
        excluded_gpus = ['1650', '1660', 'MX', 'P4', 'P40', 'K80', 'M4']
        for model in excluded_gpus:
            if model in gpu_name:
                logging.info(f"檢測到GPU型號 {model} 在排除列表中，禁用混合精度")
                return False
        amp_supported_gpus = ['RTX', 'A100', 'A10', 'V100', 'T4', '30', '40', 'TITAN V']
        for model in amp_supported_gpus:
            if model in gpu_name:
                logging.info(f"檢測到GPU型號 {model} 支持混合精度計算")
                return True
        cc_match = re.search(r'compute capability: (\d+)\.(\d+)', gpu_name.lower())
        if cc_match:
            major = int(cc_match.group(1))
            minor = int(cc_match.group(2))
            compute_capability = float(f"{major}.{minor}")
            if compute_capability >= 7.0:
                logging.info(f"GPU計算能力 {compute_capability} >= 7.0，啟用混合精度")
                return True
        try:
            test_tensor = torch.randn(1, 3, 64, 64, device=device, dtype=torch.float32)
            with torch.no_grad():
                try:
                    with torch.amp.autocast(device_type='cuda'):
                        _ = self.model(test_tensor)
                    logging.info("混合精度測試成功，啟用混合精度計算")
                    return True
                except Exception as e:
                    logging.warning(f"混合精度測試失敗: {e}")
                    return False
        except Exception as e:
            logging.warning(f"混合精度功能測試出錯: {e}")
        logging.info("無法確定GPU是否支持混合精度，為安全起見禁用")
        return False
        
    def run(self):
        try:
            enhanced_image = self.enhance_image_by_blocks()
            self.finished_signal.emit(enhanced_image, self.elapsed_time)
        except Exception as e:
            logging.error(f"處理圖片時出錯: {str(e)}")
            try:
                image = Image.open(self.image_path).convert("RGB")
                self.elapsed_time = 0
                self.finished_signal.emit(image, self.elapsed_time)
            except Exception as open_err:
                logging.error(f"無法開啟原圖: {str(open_err)}")
                error_img = Image.new("RGB", (400, 300), color=(50, 50, 50))
                self.finished_signal.emit(error_img, 0)
    
    def create_weight_mask(self, height, width, device, mode='改進型高斯分佈'):
        cache_key = f"{height}x{width}_{mode}"
        if cache_key in self.weight_mask_cache:
            cached_mask = self.weight_mask_cache[cache_key]
            if cached_mask.device != device:
                cached_mask = cached_mask.to(device)
                self.weight_mask_cache[cache_key] = cached_mask
            return cached_mask
        max_dim = max(height, width)
        x = torch.linspace(-1, 1, max_dim)
        y = torch.linspace(-1, 1, max_dim)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        if mode == '高斯分佈':
            sigma = 0.5
            weight = torch.exp(-(X**2 + Y**2) / (2 * sigma**2))
        elif mode == '改進型高斯分佈':
            sigma_center = 0.7  
            sigma_edge = 0.3  
            dist = torch.sqrt(X**2 + Y**2)
            sigma = torch.ones_like(dist) * sigma_center
            edge_mask = (dist > 0.5) 
            sigma[edge_mask] = sigma_edge
            weight = torch.exp(-(X**2 + Y**2) / (2 * sigma**2))
            weight = torch.clamp(weight, min=0.1)
        elif mode == '線性分佈':
            dist = torch.sqrt(X**2 + Y**2)
            weight = 1.0 - dist
            weight = torch.clamp(weight, min=0.1)
        elif mode == '餘弦分佈':
            dist = torch.sqrt(X**2 + Y**2).clamp(max=1)
            weight = torch.cos(dist * torch.pi/2)
            weight = torch.clamp(weight, min=0.1)
        elif mode == '泊松分佈':
            dist = torch.sqrt(X**2 + Y**2)
            weight = torch.exp(-dist * 3)
            weight = torch.clamp(weight, min=0.1)
        else:
            sigma_center = 0.7
            sigma_edge = 0.3
            dist = torch.sqrt(X**2 + Y**2)
            sigma = torch.ones_like(dist) * sigma_center
            edge_mask = (dist > 0.5)
            sigma[edge_mask] = sigma_edge
            weight = torch.exp(-(X**2 + Y**2) / (2 * sigma**2))
            weight = torch.clamp(weight, min=0.1)
        if height != max_dim or width != max_dim:
            weight = weight[:height, :width]
        weight = weight.view(1, 1, height, width).to(dtype=torch.float32)
        weight = weight.to(device=device)
        self.weight_mask_cache[cache_key] = weight
        return weight
    
    def poisson_blend_borders(self, tensor, block_size, step, x_blocks, y_blocks):
        height, width = tensor.shape[2], tensor.shape[3]
        blend_result = tensor.clone()
        device = tensor.device
        for y in range(y_blocks-1):
            y_pos = min((y+1) * step, height - 1)
            y_range = slice(max(0, y_pos - 10), min(height, y_pos + 10))
            kernel_size = 5
            kernel = torch.ones(3, 1, kernel_size, 1, device=device) / kernel_size
            smoothed = F.conv2d(
                tensor[:, :, y_range, :], 
                kernel, 
                padding=(kernel_size//2, 0),
                groups=3
            )
            blend_result[:, :, y_range, :] = smoothed
        for x in range(x_blocks-1):
            x_pos = min((x+1) * step, width - 1)
            x_range = slice(max(0, x_pos - 10), min(width, x_pos + 10))
            kernel_size = 5
            kernel = torch.ones(3, 1, 1, kernel_size, device=device) / kernel_size
            smoothed = F.conv2d(
                tensor[:, :, :, x_range], 
                kernel, 
                padding=(0, kernel_size//2),
                groups=3
            )
            blend_result[:, :, :, x_range] = smoothed
        return blend_result
    
    def resize_image(self, image, output_tensor):
        """根據設置的參數調整圖像大小"""
        if self.upscale_factor == 1.0 and self.target_width == 0 and self.target_height == 0:
            return image
        if self.upscale_factor > 1.0:
            width = int(image.width * self.upscale_factor)
            height = int(image.height * self.upscale_factor)
            return image.resize((width, height), Image.LANCZOS)
        if self.target_width > 0 and self.target_height > 0:
            if not self.maintain_aspect_ratio:
                return image.resize((self.target_width, self.target_height), Image.LANCZOS)
            else:
                orig_width, orig_height = image.size
                orig_ratio = orig_width / orig_height
                target_ratio = self.target_width / self.target_height
                if self.resize_mode == "延伸至適合大小":
                    if orig_ratio > target_ratio:
                        new_width = self.target_width
                        new_height = int(new_width / orig_ratio)
                    else:
                        new_height = self.target_height
                        new_width = int(new_height * orig_ratio)
                    resized = image.resize((new_width, new_height), Image.LANCZOS)
                    left = (new_width - self.target_width) // 2 if new_width > self.target_width else 0
                    top = (new_height - self.target_height) // 2 if new_height > self.target_height else 0
                    right = left + self.target_width if new_width > self.target_width else new_width
                    bottom = top + self.target_height if new_height > self.target_height else new_height
                    if new_width >= self.target_width and new_height >= self.target_height:
                        return resized.crop((left, top, right, bottom))
                    else:
                        result = Image.new("RGB", (self.target_width, self.target_height))
                        paste_left = (self.target_width - new_width) // 2
                        paste_top = (self.target_height - new_height) // 2
                        result.paste(resized, (paste_left, paste_top))
                        return result
                else:
                    if orig_ratio > target_ratio:
                        new_height = self.target_height
                        new_width = int(new_height * orig_ratio)
                    else:
                        new_width = self.target_width
                        new_height = int(new_width / orig_ratio)
                    resized = image.resize((new_width, new_height), Image.LANCZOS)
                    result = Image.new("RGB", (self.target_width, self.target_height))
                    paste_left = (self.target_width - new_width) // 2
                    paste_top = (self.target_height - new_height) // 2
                    result.paste(resized, (paste_left, paste_top))
                    return result  
        return image
    
    def enhance_image_by_blocks(self):
        start_time = time.time()
        self.model = self.model.to(self.device)
        image = Image.open(self.image_path).convert("RGB")
        width, height = image.size
        self.block_size = min(self.block_size, width, height)
        self.overlap = min(self.overlap, self.block_size // 2)
        transform = transforms.ToTensor()
        image_tensor = transform(image).unsqueeze(0).to(self.device, dtype=torch.float32)
        output_tensor = torch.zeros_like(image_tensor)
        if self.use_weight_mask:
            weight_tensor = torch.zeros((1, 1, height, width), device=self.device, dtype=torch.float32)
        step = max(1, self.block_size - self.overlap)
        x_blocks = (width - self.overlap) // step + (1 if (width - self.overlap) % step != 0 else 0)
        y_blocks = (height - self.overlap) // step + (1 if (height - self.overlap) % step != 0 else 0)
        x_blocks = max(1, x_blocks)
        y_blocks = max(1, y_blocks)
        self.model.eval()
        total_blocks = x_blocks * y_blocks
        blocks_processed = 0
        with torch.no_grad():
            for y in range(y_blocks):
                for x in range(x_blocks):
                    x_start = min(x * step, width - self.block_size)
                    y_start = min(y * step, height - self.block_size)
                    x_end = min(x_start + self.block_size, width)
                    y_end = min(y_start + self.block_size, height)
                    x_start = max(0, x_start)
                    y_start = max(0, y_start)
                    x_end = max(x_start + 1, x_end) 
                    y_end = max(y_start + 1, y_end) 
                    block_width = x_end - x_start
                    block_height = y_end - y_start
                    block = image_tensor[:, :, y_start:y_end, x_start:x_end].to(self.device)
                    needs_padding = block_width < self.block_size or block_height < self.block_size
                    if needs_padding:
                        padded_block = torch.zeros(1, 3, self.block_size, self.block_size, 
                                                 device=self.device, dtype=torch.float32)
                        padded_block[:, :, :block_height, :block_width] = block
                        block = padded_block
                    try:
                        block = block.to(self.device, dtype=torch.float32)
                        if self.use_amp and self.device.type == 'cuda':
                            with torch.amp.autocast(device_type='cuda'):
                                enhanced_block = self.model(block)
                                enhanced_block = enhanced_block.to(dtype=torch.float32)
                        else:
                            enhanced_block = self.model(block)
                        enhanced_block = enhanced_block.to(self.device)
                        if enhanced_block.abs().sum().item() < 1e-6:
                            logging.warning(f"警告: 區塊 {blocks_processed+1}/{total_blocks} 輸出接近零. 使用原始區塊.")
                            enhanced_block = block
                        if blocks_processed == 0:
                            logging.info(f"首個區塊處理完成，輸入形狀: {block.shape}, 輸出形狀: {enhanced_block.shape}")
                            logging.info(f"輸入設備: {block.device}, 輸出設備: {enhanced_block.device}")
                    except Exception as e:
                        logging.error(f"處理區塊出錯: {str(e)}, 使用原始區塊替代")
                        enhanced_block = block.to(self.device)
                    if needs_padding:
                        enhanced_block = enhanced_block[:, :, :block_height, :block_width]
                    if self.use_weight_mask:
                        try:
                            current_mask = self.create_weight_mask(
                                block_height, 
                                block_width, 
                                self.device, 
                                mode=self.blending_mode
                            )
                            if current_mask.shape[2] != enhanced_block.shape[2] or current_mask.shape[3] != enhanced_block.shape[3]:
                                logging.warning(f"調整遮罩大小從 {current_mask.shape} 到 {enhanced_block.shape}")
                                current_mask = F.interpolate(
                                    current_mask,
                                    size=(enhanced_block.shape[2], enhanced_block.shape[3]),
                                    mode='bilinear',
                                    align_corners=False
                                )
                            if current_mask.device != enhanced_block.device:
                                logging.debug(f"將掩碼從 {current_mask.device} 移到 {enhanced_block.device}")
                                current_mask = current_mask.to(enhanced_block.device)
                            weighted_block = enhanced_block * current_mask
                            weighted_block = weighted_block.to(self.device)
                            current_mask = current_mask.to(self.device)
                            output_tensor[:, :, y_start:y_end, x_start:x_end] += weighted_block
                            weight_tensor[:, :, y_start:y_end, x_start:x_end] += current_mask
                        except Exception as mask_err:
                            logging.error(f"掩碼應用出錯: {mask_err}, 使用無掩碼區塊")
                            enhanced_block = enhanced_block.to(self.device)
                            output_tensor[:, :, y_start:y_end, x_start:x_end] = enhanced_block
                    else:
                        enhanced_block = enhanced_block.to(self.device)
                        output_tensor[:, :, y_start:y_end, x_start:x_end] = enhanced_block
                    blocks_processed += 1
                    self.progress_signal.emit(blocks_processed, total_blocks)
                    if blocks_processed % 10 == 0 and self.device.type == 'cuda':
                        torch.cuda.empty_cache()
        try:
            if self.use_weight_mask:
                weight_tensor = weight_tensor.to(device=output_tensor.device)
                weight_tensor = torch.clamp(weight_tensor, min=1e-8)
                output_tensor = output_tensor / weight_tensor.repeat(1, 3, 1, 1)
            if self.blending_mode == '泊松分佈':
                output_tensor = self.poisson_blend_borders(output_tensor, self.block_size, step, x_blocks, y_blocks)
            if self.strength < 1.0:
                image_tensor = image_tensor.to(device=output_tensor.device)
                output_tensor = image_tensor * (1 - self.strength) + output_tensor * self.strength
            output_tensor = output_tensor.squeeze(0).cpu().clamp(0, 1)
            enhanced_image = transforms.ToPILImage()(output_tensor)
            enhanced_image = self.resize_image(enhanced_image, output_tensor)
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            self.weight_mask_cache.clear()
            self.elapsed_time = time.time() - start_time
            return enhanced_image
        except Exception as e:
            logging.error(f"處理圖像最終階段出錯: {e}")
            logging.error(f"錯誤詳情: {str(e)}")
            image = Image.open(self.image_path).convert("RGB")
            self.elapsed_time = time.time() - start_time
            return image