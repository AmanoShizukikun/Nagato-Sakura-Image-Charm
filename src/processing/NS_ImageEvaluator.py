import numpy as np
import cv2
import torch
import logging
import io
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ImageEvaluator:
    """圖像品質評估處理器"""
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def calculate_psnr(self, img1, img2):
        """計算峰值信噪比 (PSNR)"""
        if not isinstance(img1, np.ndarray):
            img1 = np.array(img1)
        if not isinstance(img2, np.ndarray):
            img2 = np.array(img2)
        if img1.shape != img2.shape:
            min_height = min(img1.shape[0], img2.shape[0])
            min_width = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_height, :min_width]
            img2 = img2[:min_height, :min_width]
        return psnr(img1, img2)

    def calculate_ssim(self, img1, img2):
        """計算結構相似性指數 (SSIM)"""
        if not isinstance(img1, np.ndarray):
            img1 = np.array(img1)
        if not isinstance(img2, np.ndarray):
            img2 = np.array(img2)
        if img1.shape != img2.shape:
            min_height = min(img1.shape[0], img2.shape[0])
            min_width = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_height, :min_width]
            img2 = img2[:min_height, :min_width]
        if img1.ndim == 3 and img1.shape[2] == 3:
            return ssim(img1, img2, channel_axis=2, data_range=255)
        return ssim(img1, img2, data_range=255)

    def calculate_mse(self, img1, img2):
        """計算均方誤差 (MSE)"""
        if not isinstance(img1, np.ndarray):
            img1 = np.array(img1, dtype=np.float32) / 255.0
        if not isinstance(img2, np.ndarray):
            img2 = np.array(img2, dtype=np.float32) / 255.0
        if img1.shape != img2.shape:
            min_height = min(img1.shape[0], img2.shape[0])
            min_width = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_height, :min_width]
            img2 = img2[:min_height, :min_width]
        return np.mean((img1 - img2) ** 2)

    def generate_difference_map(self, img1, img2):
        """生成兩個圖像之間的差異熱力圖"""
        if not isinstance(img1, np.ndarray):
            img1 = np.array(img1, dtype=np.float32) / 255.0
        if not isinstance(img2, np.ndarray):
            img2 = np.array(img2, dtype=np.float32) / 255.0
        if img1.shape != img2.shape:
            min_height = min(img1.shape[0], img2.shape[0])
            min_width = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_height, :min_width]
            img2 = img2[:min_height, :min_width]
        diff = np.abs(img1 - img2)
        diff = np.clip(diff * 5.0, 0, 1.0)
        diff_gray = np.mean(diff, axis=2) if diff.ndim == 3 else diff
        plt.figure(figsize=(8, 6))
        plt.imshow(diff_gray, cmap='jet')
        plt.colorbar(label='Difference')
        plt.title('Image Difference Map')
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        diff_image = Image.open(buf)
        plt.close()
        return diff_image

    def get_image_histograms(self, img):
        """獲取圖像的RGB直方圖"""
        if isinstance(img, np.ndarray):
            img = Image.fromarray((img * 255).astype(np.uint8) if img.dtype == np.float32 else img)
        hist_r = img.histogram()[0:256]
        hist_g = img.histogram()[256:512]
        hist_b = img.histogram()[512:768]
        plt.figure(figsize=(8, 4))
        plt.subplot(3, 1, 1)
        plt.bar(range(256), hist_r, color='red', alpha=0.7)
        plt.title('Red Channel')
        plt.xlim([0, 256])
        plt.subplot(3, 1, 2)
        plt.bar(range(256), hist_g, color='green', alpha=0.7)
        plt.title('Green Channel')
        plt.xlim([0, 256])
        plt.subplot(3, 1, 3)
        plt.bar(range(256), hist_b, color='blue', alpha=0.7)
        plt.title('Blue Channel')
        plt.xlim([0, 256])
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        hist_image = Image.open(buf)
        plt.close()
        return hist_image

    def evaluate_images(self, img1, img2):
        """綜合評估兩個圖像，計算各種品質指標"""
        results = {}
        if img1 is None or img2 is None:
            return {"error": "需要兩張完整的圖像進行比較"}
        results["psnr"] = self.calculate_psnr(img1, img2)
        results["ssim"] = self.calculate_ssim(img1, img2)
        results["mse"] = self.calculate_mse(img1, img2)
        results["difference_map"] = self.generate_difference_map(img1, img2)
        results["histogram_img1"] = self.get_image_histograms(img1)
        results["histogram_img2"] = self.get_image_histograms(img2)
        return results