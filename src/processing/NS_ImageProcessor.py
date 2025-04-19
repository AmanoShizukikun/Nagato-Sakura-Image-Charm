import torch
import torchvision.transforms as transforms
from PIL import Image

from src.processing.NS_PatchProcessor import process_image_in_patches


class ImageProcessor:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def process_image(self, image_path, block_size=256, overlap=64, use_weight_mask=True, blending_mode='改進型高斯分佈'):
        """處理單一圖片"""
        image = Image.open(image_path).convert("RGB")
        enhancer = process_image_in_patches(
            self.model,
            image,
            self.device,
            block_size=block_size,
            overlap=overlap,
            use_weight_mask=use_weight_mask,
            blending_mode=blending_mode
        )
        enhanced_image = enhancer.process()
        return enhanced_image
    
    def process_frame(self, pil_image, block_size=256, overlap=64, use_weight_mask=True, blending_mode='改進型高斯分佈'):
        """處理單一影片幀"""
        enhancer = process_image_in_patches(
            self.model,
            pil_image,
            self.device,
            block_size=block_size,
            overlap=overlap,
            use_weight_mask=use_weight_mask,
            blending_mode=blending_mode
        )
        enhanced_image = enhancer.process()
        return enhanced_image