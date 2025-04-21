import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import logging
import re

logger = logging.getLogger(__name__)

def create_weight_mask(size, device, mode='改進型高斯分佈'):
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
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
        
    return weight.view(1, 1, size, size).to(device, dtype=torch.float32)


def poisson_blend_borders(tensor, block_size, step, x_blocks, y_blocks):
    height, width = tensor.shape[2], tensor.shape[3]
    blend_result = tensor.clone()
    for y in range(y_blocks-1):
        y_pos = min((y+1) * step, height - 1)
        y_range = slice(max(0, y_pos - 10), min(height, y_pos + 10))
        kernel_size = 5
        kernel = torch.ones(1, 1, kernel_size, 1).to(tensor.device) / kernel_size
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
        kernel = torch.ones(1, 1, 1, kernel_size).to(tensor.device) / kernel_size
        smoothed = F.conv2d(
            tensor[:, :, :, x_range], 
            kernel, 
            padding=(0, kernel_size//2),
            groups=3
        )
        blend_result[:, :, :, x_range] = smoothed
    return blend_result


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


class ImageBlockProcessor:
    def __init__(self, model, device, block_size, overlap, use_weight_mask, blending_mode, use_amp=None):
        self.model = model
        self.device = device
        self.block_size = block_size
        self.overlap = overlap
        self.use_weight_mask = use_weight_mask
        self.blending_mode = blending_mode
        self.use_amp = use_amp
        
        if self.use_amp is None:
            self.use_amp = should_use_amp(device)
            
        if device.type == 'cuda':
            if self.use_amp:
                logger.debug("區塊處理器使用混合精度計算")
            else:
                logger.debug("區塊處理器使用標準精度計算")
        
        if use_weight_mask:
            self.weight_mask = create_weight_mask(block_size, device, mode=blending_mode)
        else:
            self.weight_mask = None
            
    def process_block(self, block):
        block_height, block_width = block.shape[2], block.shape[3]
        padded = False
        if block_height < self.block_size or block_width < self.block_size:
            padded = True
            padded_block = torch.zeros(1, 3, self.block_size, self.block_size).to(self.device)
            padded_block[:, :, :block_height, :block_width] = block
            block = padded_block
        
        with torch.no_grad():
            if self.use_amp:
                with torch.amp.autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
                    enhanced_block = self.model(block)
                    enhanced_block = enhanced_block.to(dtype=torch.float32)
            else:
                block = block.to(dtype=torch.float32)
                enhanced_block = self.model(block)
                
            if enhanced_block.abs().sum().item() < 1e-6:
                logger.warning(f"警告: 區塊輸出接近零. 使用原始區塊作為備用.")
                enhanced_block = block
        
        if padded:
            enhanced_block = enhanced_block[:, :, :block_height, :block_width]
            
        return enhanced_block
        
    def apply_weight_mask(self, enhanced_block, block_height, block_width):
        if not self.use_weight_mask:
            return enhanced_block 
        current_mask = self.weight_mask
        if block_height < self.block_size or block_width < self.block_size:
            current_mask = self.weight_mask[:, :, :block_height, :block_width]
            
        return enhanced_block * current_mask, current_mask


def process_image_in_patches(model, image, device, block_size=256, overlap=64, 
                           use_weight_mask=True, blending_mode='改進型高斯分佈', use_amp=None):
    return ImagePatchEnhancer(model, image, device, block_size, overlap, 
                            use_weight_mask, blending_mode, use_amp)


class ImagePatchEnhancer:
    def __init__(self, model, image, device, block_size=256, overlap=64, 
                use_weight_mask=True, blending_mode='改進型高斯分佈', use_amp=None):
        self.model = model
        self.image = image
        self.device = device
        self.block_size = block_size
        self.overlap = overlap
        self.use_weight_mask = use_weight_mask
        self.blending_mode = blending_mode
        self.use_amp = use_amp
        transform = transforms.ToTensor()
        self.width, self.height = image.size
        self.image_tensor = transform(image).unsqueeze(0).to(device)
        self.step = block_size - overlap
        self.x_blocks = (self.width - overlap) // self.step + (1 if self.width % self.step != 0 else 0)
        self.y_blocks = (self.height - overlap) // self.step + (1 if self.height % self.step != 0 else 0)
        self.processor = ImageBlockProcessor(model, device, block_size, overlap, 
                                           use_weight_mask, blending_mode, use_amp)
        self.total_blocks = self.x_blocks * self.y_blocks
        self.blocks_processed = 0
        
    def process(self, progress_callback=None):
        model = self.model
        model.eval()
        output_tensor = torch.zeros_like(self.image_tensor)
        if self.use_weight_mask:
            weight_tensor = torch.zeros((1, 1, self.height, self.width), device=self.device, dtype=torch.float32)
        
        try:
            with torch.no_grad():
                for y in range(self.y_blocks):
                    for x in range(self.x_blocks):
                        x_start = min(x * self.step, self.width - self.block_size)
                        y_start = min(y * self.step, self.height - self.block_size)
                        x_end = min(x_start + self.block_size, self.width)
                        y_end = min(y_start + self.block_size, self.height)
                        block_width = x_end - x_start
                        block_height = y_end - y_start
                        block = self.image_tensor[:, :, y_start:y_end, x_start:x_end]
                        
                        try:
                            enhanced_block = self.processor.process_block(block)
                            
                            if y == 0 and x == 0:
                                logger.debug(f"首個區塊處理完成，輸入形狀: {block.shape}, 輸出形狀: {enhanced_block.shape}")
                                logger.debug(f"輸入範圍: {block.min().item():.4f} 到 {block.max().item():.4f}")
                                logger.debug(f"輸出範圍: {enhanced_block.min().item():.4f} 到 {enhanced_block.max().item():.4f}")
                            
                            if self.use_weight_mask:
                                weighted_block, mask = self.processor.apply_weight_mask(
                                    enhanced_block, block_height, block_width)
                                output_tensor[:, :, y_start:y_end, x_start:x_end] += weighted_block
                                weight_tensor[:, :, y_start:y_end, x_start:x_end] += mask
                            else:
                                output_tensor[:, :, y_start:y_end, x_start:x_end] = enhanced_block
                                
                        except Exception as e:
                            logger.error(f"處理區塊時出錯: {e}")
                            if self.use_weight_mask:
                                _, mask = self.processor.apply_weight_mask(
                                    block, block_height, block_width)
                                output_tensor[:, :, y_start:y_end, x_start:x_end] += block * mask
                                weight_tensor[:, :, y_start:y_end, x_start:x_end] += mask
                            else:
                                output_tensor[:, :, y_start:y_end, x_start:x_end] = block
                            
                        self.blocks_processed += 1
                        if progress_callback:
                            progress_callback(self.blocks_processed, self.total_blocks)
        except Exception as e:
            logger.error(f"處理圖像時發生錯誤: {e}")
            return self.image
            
        try:
            if self.use_weight_mask:
                weight_tensor = torch.clamp(weight_tensor, min=1e-8)
                output_tensor = output_tensor / weight_tensor.repeat(1, 3, 1, 1)
                
            if self.blending_mode == '泊松分佈':
                output_tensor = poisson_blend_borders(
                    output_tensor, self.block_size, self.step, self.x_blocks, self.y_blocks)
                    
            output_tensor = output_tensor.squeeze(0).cpu().clamp(0, 1)
            enhanced_image = transforms.ToPILImage()(output_tensor)
            return enhanced_image
            
        except Exception as e:
            logger.error(f"處理圖像最終階段出錯: {e}")
            return self.image