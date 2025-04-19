import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms


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
    return weight.view(1, 1, size, size).to(device)

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

class ImageBlockProcessor:
    def __init__(self, model, device, block_size, overlap, use_weight_mask, blending_mode):
        self.model = model
        self.device = device
        self.block_size = block_size
        self.overlap = overlap
        self.use_weight_mask = use_weight_mask
        self.blending_mode = blending_mode
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
            with torch.amp.autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
                enhanced_block = self.model(block)
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
                            use_weight_mask=True, blending_mode='改進型高斯分佈'):
    return ImagePatchEnhancer(model, image, device, block_size, overlap, 
                             use_weight_mask, blending_mode)

class ImagePatchEnhancer:
    def __init__(self, model, image, device, block_size=256, overlap=64, 
                use_weight_mask=True, blending_mode='改進型高斯分佈'):
        self.model = model
        self.image = image
        self.device = device
        self.block_size = block_size
        self.overlap = overlap
        self.use_weight_mask = use_weight_mask
        self.blending_mode = blending_mode
        transform = transforms.ToTensor()
        self.width, self.height = image.size
        self.image_tensor = transform(image).unsqueeze(0).to(device)
        self.step = block_size - overlap
        self.x_blocks = (self.width - overlap) // self.step + (1 if self.width % self.step != 0 else 0)
        self.y_blocks = (self.height - overlap) // self.step + (1 if self.height % self.step != 0 else 0)
        self.processor = ImageBlockProcessor(model, device, block_size, overlap, 
                                           use_weight_mask, blending_mode)
        self.total_blocks = self.x_blocks * self.y_blocks
        self.blocks_processed = 0
        
    def process(self, progress_callback=None):
        model = self.model
        model.eval()
        output_tensor = torch.zeros_like(self.image_tensor)
        if self.use_weight_mask:
            weight_tensor = torch.zeros((1, 1, self.height, self.width), device=self.device)
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
                    enhanced_block = self.processor.process_block(block)
                    if self.use_weight_mask:
                        weighted_block, mask = self.processor.apply_weight_mask(
                            enhanced_block, block_height, block_width)
                        output_tensor[:, :, y_start:y_end, x_start:x_end] += weighted_block
                        weight_tensor[:, :, y_start:y_end, x_start:x_end] += mask
                    else:
                        output_tensor[:, :, y_start:y_end, x_start:x_end] = enhanced_block
                    self.blocks_processed += 1
                    if progress_callback:
                        progress_callback(self.blocks_processed, self.total_blocks)
        if self.use_weight_mask:
            weight_tensor = torch.clamp(weight_tensor, min=1e-8)
            output_tensor = output_tensor / weight_tensor.repeat(1, 3, 1, 1)
        if self.blending_mode == '泊松分佈':
            output_tensor = poisson_blend_borders(
                output_tensor, self.block_size, self.step, self.x_blocks, self.y_blocks)
        output_tensor = output_tensor.squeeze(0).cpu().clamp(0, 1)
        enhanced_image = transforms.ToPILImage()(output_tensor) 
        return enhanced_image