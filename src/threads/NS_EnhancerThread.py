import time
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal
import logging

class EnhancerThread(QThread):
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal(Image.Image, float)
    
    def __init__(self, model, image_path, device, block_size=256, overlap=64, use_weight_mask=True, blending_mode='改進型高斯分佈'):
        super().__init__()
        self.model = model
        self.image_path = image_path
        self.device = device
        self.block_size = block_size
        self.overlap = overlap
        self.use_weight_mask = use_weight_mask
        self.blending_mode = blending_mode
        self.weight_mask_cache = {}
        
    def run(self):
        try:
            enhanced_image = self.enhance_image_by_blocks()
            self.finished_signal.emit(enhanced_image, self.elapsed_time)
        except Exception as e:
            logging.error(f"處理圖片時出錯: {str(e)}")
            image = Image.open(self.image_path).convert("RGB")
            self.elapsed_time = 0
            self.finished_signal.emit(image, self.elapsed_time)
    
    def create_weight_mask(self, height, width, device, mode='改進型高斯分佈'):
        cache_key = f"{height}x{width}_{mode}"
        if cache_key in self.weight_mask_cache:
            return self.weight_mask_cache[cache_key]
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
        weight = weight.view(1, 1, height, width).to(device)
        self.weight_mask_cache[cache_key] = weight
        return weight
    
    def poisson_blend_borders(self, tensor, block_size, step, x_blocks, y_blocks):
        height, width = tensor.shape[2], tensor.shape[3]
        blend_result = tensor.clone()
        for y in range(y_blocks-1):
            y_pos = min((y+1) * step, height - 1)
            y_range = slice(max(0, y_pos - 10), min(height, y_pos + 10))
            kernel_size = 5
            kernel = torch.ones(3, 1, kernel_size, 1).to(tensor.device) / kernel_size
            smoothed = torch.nn.functional.conv2d(
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
            kernel = torch.ones(3, 1, 1, kernel_size).to(tensor.device) / kernel_size
            smoothed = torch.nn.functional.conv2d(
                tensor[:, :, :, x_range], 
                kernel, 
                padding=(0, kernel_size//2),
                groups=3
            )
            blend_result[:, :, :, x_range] = smoothed
        return blend_result
    
    def enhance_image_by_blocks(self):
        start_time = time.time()
        image = Image.open(self.image_path).convert("RGB")
        width, height = image.size
        self.block_size = min(self.block_size, width, height)
        self.overlap = min(self.overlap, self.block_size // 2)
        transform = transforms.ToTensor()
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        output_tensor = torch.zeros_like(image_tensor)
        if self.use_weight_mask:
            weight_tensor = torch.zeros((1, 1, height, width), device=self.device)
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
                    block = image_tensor[:, :, y_start:y_end, x_start:x_end]
                    needs_padding = block_width < self.block_size or block_height < self.block_size
                    if needs_padding:
                        padded_block = torch.zeros(1, 3, self.block_size, self.block_size).to(self.device)
                        padded_block[:, :, :block_height, :block_width] = block
                        block = padded_block
                    with torch.amp.autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
                        enhanced_block = self.model(block)
                    if needs_padding:
                        enhanced_block = enhanced_block[:, :, :block_height, :block_width]
                    if self.use_weight_mask:
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
                        output_tensor[:, :, y_start:y_end, x_start:x_end] += enhanced_block * current_mask
                        weight_tensor[:, :, y_start:y_end, x_start:x_end] += current_mask
                    else:
                        output_tensor[:, :, y_start:y_end, x_start:x_end] = enhanced_block
                    blocks_processed += 1
                    self.progress_signal.emit(blocks_processed, total_blocks)
        if self.use_weight_mask:
            weight_tensor = torch.clamp(weight_tensor, min=1e-8)
            output_tensor = output_tensor / weight_tensor.repeat(1, 3, 1, 1)
        if self.blending_mode == '泊松分佈':
            output_tensor = self.poisson_blend_borders(output_tensor, self.block_size, step, x_blocks, y_blocks)
        output_tensor = output_tensor.squeeze(0).cpu().clamp(0, 1)
        enhanced_image = transforms.ToPILImage()(output_tensor)
        self.elapsed_time = time.time() - start_time
        return enhanced_image