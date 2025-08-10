import os
import logging
import numpy as np
import cv2
import datetime
import io
import json
import hashlib
import shutil
from PIL import Image, ImageFilter, ImageEnhance
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Callable, Optional, Any, Tuple
from enum import Enum
from pathlib import Path
import threading


class ProcessingMode(Enum):
    """資料處理模式枚舉"""
    JPEG_COMPRESSION = "jpeg"
    NOISE_ADDITION = "noise" 
    PIXELATION = "pixel"
    BLUR_EFFECTS = "blur"
    COLOR_DISTORTION = "color"
    MIXED_DEGRADATION = "mixed"
    CUSTOM_PIPELINE = "custom"


class BlurType(Enum):
    """模糊類型枚舉"""
    GAUSSIAN = "gaussian"
    MOTION = "motion"
    LENS = "lens"
    DEFOCUS = "defocus"


class NoiseType(Enum):
    """雜訊類型枚舉"""
    GAUSSIAN = "gaussian"
    POISSON = "poisson"
    SALT_PEPPER = "salt_pepper"
    SPECKLE = "speckle"
    UNIFORM = "uniform"


class DataProcessor:
    def __init__(self, min_quality=10, max_quality=101, quality_interval=10, 
                 processing_mode=ProcessingMode.JPEG_COMPRESSION, num_workers=None,
                 preserve_metadata=True, generate_previews=True, enable_validation=True):
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.quality_interval = quality_interval
        self.processing_mode = processing_mode
        self.num_workers = num_workers or max(1, os.cpu_count() // 2)
        self.preserve_metadata = preserve_metadata
        self.generate_previews = generate_previews
        self.enable_validation = enable_validation
        self.logger = logging.getLogger("DataProcessor")
        self.processing_config = {
            'output_format': 'jpg',
            'output_quality': 95,
            'resize_images': False,
            'target_size': (512, 512),
            'maintain_aspect_ratio': True,
            'skip_existing': False,
            'create_backup': False,
            'validate_output': True
        }
        
        # JPEG 壓縮設定
        self.jpeg_quality_settings = {
            100: {"quality": 100, "subsampling": 0},  # 原始品質
            90: {"quality": 90, "subsampling": 0},    # 高品質
            80: {"quality": 80, "subsampling": 0},    # 良好品質
            70: {"quality": 70, "subsampling": 0},    # 中高品質
            60: {"quality": 60, "subsampling": 1},    # 中等品質
            50: {"quality": 50, "subsampling": 1},    # 中低品質
            40: {"quality": 40, "subsampling": 1},    # 較低品質
            30: {"quality": 30, "subsampling": 2},    # 低品質
            20: {"quality": 20, "subsampling": 2},    # 很低品質
            10: {"quality": 10, "subsampling": 2}     # 極低品質
        }
        
        # 擴展雜訊參數設定
        self.noise_params = {
            100: {"gaussian_std": 0, "iso_noise": 0, "color_noise": 0, "compression": 0, "poisson": 0},
            90: {"gaussian_std": 0.8, "iso_noise": 1.0, "color_noise": 0.2, "compression": 0.3, "poisson": 0.1},
            80: {"gaussian_std": 1.2, "iso_noise": 1.5, "color_noise": 0.4, "compression": 0.5, "poisson": 0.2},
            70: {"gaussian_std": 1.8, "iso_noise": 2.2, "color_noise": 0.6, "compression": 0.8, "poisson": 0.3},
            60: {"gaussian_std": 2.5, "iso_noise": 3.0, "color_noise": 0.9, "compression": 1.2, "poisson": 0.4},
            50: {"gaussian_std": 3.5, "iso_noise": 4.0, "color_noise": 1.3, "compression": 1.6, "poisson": 0.6},
            40: {"gaussian_std": 4.8, "iso_noise": 5.5, "color_noise": 1.8, "compression": 2.2, "poisson": 0.8},
            30: {"gaussian_std": 6.5, "iso_noise": 7.5, "color_noise": 2.5, "compression": 3.0, "poisson": 1.2},
            20: {"gaussian_std": 8.5, "iso_noise": 10.0, "color_noise": 3.5, "compression": 4.0, "poisson": 1.8},
            10: {"gaussian_std": 11.0, "iso_noise": 13.0, "color_noise": 5.0, "compression": 5.5, "poisson": 2.5}
        }
        
        # 模糊效果參數
        self.blur_params = {
            100: {"gaussian": 0, "motion": 0, "lens": 0, "defocus": 0},
            90: {"gaussian": 0.3, "motion": 0.2, "lens": 0.1, "defocus": 0.2},
            80: {"gaussian": 0.6, "motion": 0.5, "lens": 0.3, "defocus": 0.4},
            70: {"gaussian": 1.0, "motion": 0.8, "lens": 0.5, "defocus": 0.7},
            60: {"gaussian": 1.5, "motion": 1.2, "lens": 0.8, "defocus": 1.0},
            50: {"gaussian": 2.0, "motion": 1.6, "lens": 1.2, "defocus": 1.5},
            40: {"gaussian": 2.8, "motion": 2.2, "lens": 1.8, "defocus": 2.2},
            30: {"gaussian": 3.8, "motion": 3.0, "lens": 2.5, "defocus": 3.0},
            20: {"gaussian": 5.0, "motion": 4.0, "lens": 3.5, "defocus": 4.0},
            10: {"gaussian": 7.0, "motion": 5.5, "lens": 5.0, "defocus": 5.5}
        }
        
        # 色彩失真參數
        self.color_params = {
            100: {"saturation": 1.0, "brightness": 1.0, "contrast": 1.0, "hue_shift": 0},
            90: {"saturation": 0.95, "brightness": 0.98, "contrast": 0.98, "hue_shift": 2},
            80: {"saturation": 0.88, "brightness": 0.95, "contrast": 0.95, "hue_shift": 5},
            70: {"saturation": 0.80, "brightness": 0.90, "contrast": 0.90, "hue_shift": 8},
            60: {"saturation": 0.70, "brightness": 0.85, "contrast": 0.85, "hue_shift": 12},
            50: {"saturation": 0.60, "brightness": 0.80, "contrast": 0.80, "hue_shift": 15},
            40: {"saturation": 0.50, "brightness": 0.75, "contrast": 0.75, "hue_shift": 20},
            30: {"saturation": 0.40, "brightness": 0.70, "contrast": 0.70, "hue_shift": 25},
            20: {"saturation": 0.30, "brightness": 0.65, "contrast": 0.65, "hue_shift": 30},
            10: {"saturation": 0.20, "brightness": 0.60, "contrast": 0.60, "hue_shift": 40}
        }
        
        # 統計資訊
        self.stats = {
            'total_processed': 0,
            'total_generated': 0,
            'total_skipped': 0,
            'total_errors': 0,
            'processing_time': 0,
            'psnr_values': [],
            'ssim_values': [],
            'file_sizes': [],
            'processing_speeds': []
        }
        self._stats_lock = threading.Lock()
        self._stop_event = threading.Event()
    
    def process_images(self, input_dir, output_dir, callback=None):
        """處理圖片集合，轉換為多種品質等級的訓練資料集"""
        start_time = datetime.datetime.now()
        quality_levels = list(range(self.min_quality, self.max_quality, self.quality_interval))
        os.makedirs(output_dir, exist_ok=True)
        image_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'))]
        total_files = len(image_files)
        if total_files == 0:
            self.logger.warning(f"在 {input_dir} 中找不到任何圖片檔案")
            return 0
        self.logger.info(f"開始處理 {total_files} 張圖片，處理模式: {self.processing_mode.value}")
        log_path = self._create_log_file(output_dir)
        processed_files = 0
        if self.processing_mode in [ProcessingMode.NOISE_ADDITION, ProcessingMode.PIXELATION]:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                for file_name in image_files:
                    future = executor.submit(
                        self._process_single_image_multiprocess, 
                        file_name, 
                        input_dir, 
                        output_dir, 
                        quality_levels,
                        self.processing_mode.value,
                        self.noise_params if self.processing_mode == ProcessingMode.NOISE_ADDITION else None
                    )
                    futures.append((future, file_name))
                for future, file_name in futures:
                    try:
                        result = future.result()
                        if result:
                            self._append_to_log(log_path, result)
                            processed_files += 1
                        if callback:
                            callback(processed_files, total_files)
                    except Exception as e:
                        self.logger.error(f"處理 {file_name} 時發生錯誤: {e}")
        else:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                for file_name in image_files:
                    future = executor.submit(
                        self.process_single_image, 
                        file_name, 
                        input_dir, 
                        output_dir, 
                        quality_levels
                    )
                    futures.append(future)
                
                for future in futures:
                    try:
                        result = future.result()
                        if result:
                            processed_files += 1
                        if callback:
                            callback(processed_files, total_files)
                    except Exception as e:
                        self.logger.error(f"處理時發生錯誤: {e}")
        processing_time = datetime.datetime.now() - start_time
        self.stats.update({
            'total_processed': processed_files,
            'total_generated': processed_files * len(quality_levels),
            'processing_time': processing_time.total_seconds()
        })
        self._create_settings_file(output_dir, quality_levels)
        self.logger.info(f"已完成處理 {processed_files} 張圖片，耗時: {processing_time}")
        return processed_files

    def process_single_image(self, file_name, input_dir, output_dir, quality_levels):
        """處理單張圖片，生成多種品質版本"""
        input_path = os.path.join(input_dir, file_name)
        try:
            with Image.open(input_path) as img:
                if img.mode == "RGBA":
                    img = img.convert("RGB")
                base_name, ext = os.path.splitext(file_name)
                if self.processing_mode == ProcessingMode.JPEG_COMPRESSION:
                    return self._process_jpeg_compression(img, base_name, output_dir, quality_levels)
                else:
                    img_array = np.array(img)
                    return self._process_with_effects(img_array, base_name, output_dir, quality_levels)
        except Exception as e:
            self.logger.error(f"處理圖片 {file_name} 時發生錯誤: {e}")
            return None
    
    def _process_jpeg_compression(self, img, base_name, output_dir, quality_levels):
        """處理JPEG壓縮"""
        original_array = np.array(img)
        log_entries = []
        for quality in quality_levels:
            output_file = f"{base_name}_q{quality}.jpg"
            output_path = os.path.join(output_dir, output_file)
            if quality not in self.jpeg_quality_settings:
                settings = {"quality": quality, "subsampling": 2 if quality < 50 else 1}
            else:
                settings = self.jpeg_quality_settings[quality]
            if quality == 100:
                img.save(output_path, "JPEG", quality=100, subsampling=0, dct_method="FLOAT")
                psnr = float('inf')
            else:
                buffer = io.BytesIO()
                img.save(
                    buffer, "JPEG",
                    quality=settings["quality"],
                    subsampling=settings["subsampling"],
                    dct_method="FLOAT" if quality >= 50 else "FAST"
                )
                buffer.seek(0)
                with Image.open(buffer) as compressed_img:
                    compressed_array = np.array(compressed_img)
                    psnr = self._calculate_psnr(original_array, compressed_array)
                    compressed_img.save(output_path, "JPEG", quality=95)
            log_entries.append(f"{base_name}_q{quality}.jpg,{quality},{psnr:.2f}")
            self.stats['psnr_values'].append(psnr)
        return log_entries
    
    def _process_with_effects(self, img_array, base_name, output_dir, quality_levels):
        """處理雜訊添加或其他效果"""
        log_entries = []
        original_array = img_array.copy()
        processed_arrays = []
        for quality in quality_levels:
            if self._stop_event.is_set():
                break
            try:
                if self.processing_mode == ProcessingMode.NOISE_ADDITION:
                    processed_array = self._add_realistic_noise(img_array, quality)
                elif self.processing_mode == ProcessingMode.PIXELATION:
                    processed_array = self._create_pixelation(img_array, quality)
                elif self.processing_mode == ProcessingMode.BLUR_EFFECTS:
                    processed_array = self._apply_blur_effects(img_array, quality)
                elif self.processing_mode == ProcessingMode.COLOR_DISTORTION:
                    processed_array = self._apply_color_distortion(img_array, quality)
                elif self.processing_mode == ProcessingMode.MIXED_DEGRADATION:
                    processed_array = self._apply_mixed_degradation(img_array, quality)
                else:
                    processed_array = img_array.copy()
                processed_arrays.append(processed_array)
                psnr = self._calculate_psnr(original_array, processed_array)
                ssim_value = self._calculate_ssim(original_array, processed_array)
                output_file = f"{base_name}_q{quality}.{self.processing_config['output_format']}"
                output_path = os.path.join(output_dir, output_file)
                processed_img = Image.fromarray(processed_array.astype(np.uint8))
                if self.preserve_metadata and self.processing_config['output_format'] == 'jpg':
                    processed_img.save(output_path, "JPEG", quality=self.processing_config['output_quality'], optimize=True)
                else:
                    processed_img.save(output_path)
                file_size = os.path.getsize(output_path)
                with self._stats_lock:
                    self.stats['psnr_values'].append(psnr)
                    self.stats['ssim_values'].append(ssim_value)
                    self.stats['file_sizes'].append(file_size)
                log_entries.append(f"{output_file},{quality},{psnr:.2f},{ssim_value:.4f},{file_size}")
            except Exception as e:
                self.logger.error(f"處理品質等級 {quality} 時發生錯誤: {e}")
                with self._stats_lock:
                    self.stats['total_errors'] += 1
                continue
        if processed_arrays:
            self._create_preview_image(original_array, processed_arrays, base_name, output_dir)
        return log_entries
    
    def _add_realistic_noise(self, image, noise_level):
        """添加逼真的相機雜訊"""
        if noise_level == 100:
            return image
        params = self.noise_params.get(noise_level, self.noise_params[50])
        noisy_image = image.astype(np.float64)
        if params["gaussian_std"] > 0:
            gaussian_noise = np.random.normal(0, params["gaussian_std"], image.shape)
            noisy_image += gaussian_noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        if params["iso_noise"] > 0:
            noisy_image = self._add_iso_noise(noisy_image, params["iso_noise"])
        if params["color_noise"] > 0 and len(image.shape) == 3:
            noisy_image = self._add_color_noise(noisy_image, params["color_noise"])
        if params["compression"] > 0:
            noisy_image = self._add_compression_artifacts(noisy_image, params["compression"])
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    def _add_iso_noise(self, image, noise_level):
        """添加ISO雜訊"""
        if len(image.shape) == 3:
            brightness = np.mean(image, axis=2)
        else:
            brightness = image.copy()
        brightness_norm = brightness / 255.0
        noise_factor = noise_level * (1.3 - brightness_norm * 0.5)
        if len(image.shape) == 3:
            noise = np.zeros_like(image, dtype=np.float64)
            for i in range(3):
                noise[:, :, i] = np.random.normal(0, noise_factor, image.shape[:2])
        else:
            noise = np.random.normal(0, noise_factor, image.shape)
        noisy_image = image.astype(np.float64) + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    def _add_color_noise(self, image, noise_level):
        """添加色彩雜訊"""
        noisy_image = image.astype(np.float64)
        for i in range(3):
            channel_noise = np.random.normal(0, noise_level * [0.9, 0.8, 1.1][i], image.shape[:2])
            noisy_image[:, :, i] += channel_noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    def _add_compression_artifacts(self, image, intensity):
        """添加輕微的壓縮偽影"""
        quantization_step = max(1, intensity)
        quantized = np.round(image.astype(np.float64) / quantization_step) * quantization_step
        random_variation = np.random.uniform(-intensity * 0.3, intensity * 0.3, image.shape)
        result = quantized + random_variation
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _create_pixelation(self, image, quality_level):
        """創建像素化效果"""
        pixel_size = (100 - quality_level) // 10 + 1
        if pixel_size <= 1:
            return image
        height, width = image.shape[:2]
        small_width = max(1, width // pixel_size)
        small_height = max(1, height // pixel_size)
        small_img = cv2.resize(image, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small_img, (width, height), interpolation=cv2.INTER_NEAREST)
        return pixelated
    
    def _apply_blur_effects(self, image, quality_level):
        """應用模糊效果"""
        if quality_level == 100:
            return image 
        params = self.blur_params.get(quality_level, self.blur_params[50])
        img_pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        if params["gaussian"] > 0:
            img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=params["gaussian"]))
        if params["motion"] > 0:
            img_pil = self._apply_motion_blur(img_pil, params["motion"])
        if params["lens"] > 0:
            img_pil = self._apply_lens_blur(img_pil, params["lens"])
        if params["defocus"] > 0:
            img_pil = self._apply_defocus_blur(img_pil, params["defocus"])
        return np.array(img_pil)
    
    def _apply_motion_blur(self, img_pil, intensity):
        """應用運動模糊"""
        kernel_size = int(intensity * 5) + 1
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        img_array = np.array(img_pil)
        if len(img_array.shape) == 3:
            blurred = np.zeros_like(img_array)
            for i in range(3):
                blurred[:,:,i] = cv2.filter2D(img_array[:,:,i], -1, kernel)
        else:
            blurred = cv2.filter2D(img_array, -1, kernel)
        return Image.fromarray(np.uint8(blurred))
    
    def _apply_lens_blur(self, img_pil, intensity):
        """應用鏡頭模糊"""
        radius = max(1, int(intensity * 3))
        return img_pil.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def _apply_defocus_blur(self, img_pil, intensity):
        """應用散焦模糊"""
        radius = max(1, int(intensity * 2))
        return img_pil.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def _apply_color_distortion(self, image, quality_level):
        """應用色彩失真"""
        if quality_level == 100:
            return image
        params = self.color_params.get(quality_level, self.color_params[50])
        img_pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        enhancer = ImageEnhance.Color(img_pil)
        img_pil = enhancer.enhance(params["saturation"])
        enhancer = ImageEnhance.Brightness(img_pil)
        img_pil = enhancer.enhance(params["brightness"])
        enhancer = ImageEnhance.Contrast(img_pil)
        img_pil = enhancer.enhance(params["contrast"])
        if params["hue_shift"] > 0:
            img_pil = self._apply_hue_shift(img_pil, params["hue_shift"])
        return np.array(img_pil)
    
    def _apply_hue_shift(self, img_pil, shift_amount):
        """應用色相偏移"""
        if img_pil.mode != 'RGB':
            return img_pil
        img_array = np.array(img_pil)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        hsv[:,:,0] = (hsv[:,:,0] + shift_amount) % 180
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return Image.fromarray(rgb)
    
    def _apply_mixed_degradation(self, image, quality_level):
        """應用混合劣化效果"""
        if quality_level == 100:
            return image
        effects = []
        if quality_level <= 80:
            effects.append('noise')
        if quality_level <= 60:
            effects.append('blur')
        if quality_level <= 40:
            effects.append('color')
        if quality_level <= 20:
            effects.append('pixel')
        processed_image = image.copy()
        if 'noise' in effects:
            processed_image = self._add_realistic_noise(processed_image, min(quality_level + 20, 100))
        if 'blur' in effects:
            processed_image = self._apply_blur_effects(processed_image, min(quality_level + 30, 100))
        if 'color' in effects:
            processed_image = self._apply_color_distortion(processed_image, min(quality_level + 25, 100))
        if 'pixel' in effects and quality_level <= 30:
            processed_image = self._create_pixelation(processed_image, quality_level + 50)
        return processed_image
    
    def _calculate_ssim(self, img1, img2):
        """計算SSIM值"""
        try:
            from skimage.metrics import structural_similarity as ssim
            if len(img1.shape) == 3:
                gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            else:
                gray1 = img1
            if len(img2.shape) == 3:
                gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            else:
                gray2 = img2
            return ssim(gray1, gray2)
        except ImportError:
            self.logger.warning("scikit-image 未安裝，無法計算SSIM")
            return 0.0
        except Exception as e:
            self.logger.error(f"計算SSIM時發生錯誤: {e}")
            return 0.0
    
    def _validate_image(self, image_path):
        """驗證圖像是否有效"""
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception as e:
            self.logger.error(f"圖像驗證失敗 {image_path}: {e}")
            return False
    
    def _create_image_hash(self, image_array):
        """創建圖像哈希值用於去重"""
        small_img = cv2.resize(image_array, (8, 8), interpolation=cv2.INTER_AREA)
        if len(small_img.shape) == 3:
            small_img = cv2.cvtColor(small_img, cv2.COLOR_RGB2GRAY)
        avg = small_img.mean()
        hash_str = ''.join(['1' if pixel > avg else '0' for pixel in small_img.flatten()])
        return hash_str
    
    def _should_skip_processing(self, input_path, output_paths):
        """檢查是否應跳過處理"""
        if not self.processing_config['skip_existing']:
            return False
        for output_path in output_paths:
            if not os.path.exists(output_path):
                return False
        input_mtime = os.path.getmtime(input_path)
        for output_path in output_paths:
            if os.path.getmtime(output_path) < input_mtime:
                return False
        return True
    
    def _create_preview_image(self, original_array, processed_arrays, base_name, output_dir):
        """創建預覽圖像"""
        if not self.generate_previews:
            return
        try:
            num_images = len(processed_arrays) + 1
            grid_size = int(np.ceil(np.sqrt(num_images)))
            h, w = original_array.shape[:2]
            preview_size = (128, 128)
            images = [cv2.resize(original_array, preview_size)]
            for processed in processed_arrays:
                images.append(cv2.resize(processed, preview_size))
            grid = np.zeros((grid_size * preview_size[0], grid_size * preview_size[1], 3), dtype=np.uint8)
            for i, img in enumerate(images):
                row = i // grid_size
                col = i % grid_size
                y1, y2 = row * preview_size[0], (row + 1) * preview_size[0]
                x1, x2 = col * preview_size[1], (col + 1) * preview_size[1]
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                grid[y1:y2, x1:x2] = img
            preview_path = os.path.join(output_dir, "previews", f"{base_name}_preview.jpg")
            os.makedirs(os.path.dirname(preview_path), exist_ok=True)
            preview_img = Image.fromarray(grid)
            preview_img.save(preview_path, "JPEG", quality=85)
        except Exception as e:
            self.logger.error(f"創建預覽圖像時發生錯誤: {e}")
    
    def update_config(self, **kwargs):
        """更新處理配置"""
        self.processing_config.update(kwargs)
        self.logger.info(f"更新處理配置: {kwargs}")
    
    def get_config(self):
        """獲取當前配置"""
        return self.processing_config.copy()
    
    def estimate_processing_time(self, input_dir):
        """估算處理時間"""
        try:
            image_files = [f for f in os.listdir(input_dir) 
                          if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'))]
            total_files = len(image_files)
            quality_levels = list(range(self.min_quality, self.max_quality, self.quality_interval))
            total_outputs = total_files * len(quality_levels)
            base_time_per_file = {
                ProcessingMode.JPEG_COMPRESSION: 0.1,
                ProcessingMode.NOISE_ADDITION: 0.3,
                ProcessingMode.PIXELATION: 0.2,
                ProcessingMode.BLUR_EFFECTS: 0.4,
                ProcessingMode.COLOR_DISTORTION: 0.2,
                ProcessingMode.MIXED_DEGRADATION: 0.8,
                ProcessingMode.CUSTOM_PIPELINE: 1.0
            }.get(self.processing_mode, 0.3)
            estimated_seconds = (total_outputs * base_time_per_file) / self.num_workers
            return {
                'total_files': total_files,
                'total_outputs': total_outputs,
                'estimated_seconds': estimated_seconds,
                'estimated_minutes': estimated_seconds / 60,
                'workers': self.num_workers
            }
        except Exception as e:
            self.logger.error(f"估算處理時間時發生錯誤: {e}")
            return None
    
    def cleanup_temp_files(self, output_dir):
        """清理臨時檔案"""
        temp_dir = os.path.join(output_dir, ".temp")
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                self.logger.info("已清理臨時檔案")
            except Exception as e:
                self.logger.error(f"清理臨時檔案時發生錯誤: {e}")
    
    def stop_processing(self):
        """停止處理"""
        self._stop_event.set()
        self.logger.info("處理停止請求已發送")
    
    def _calculate_psnr(self, original_arr, processed_arr):
        """計算PSNR值"""
        original_arr = original_arr.astype(np.float64)
        processed_arr = processed_arr.astype(np.float64)
        mse = np.mean((original_arr - processed_arr) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    def _create_log_file(self, output_dir):
        """建立日誌檔案"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        mode_name = {
            ProcessingMode.JPEG_COMPRESSION: 'jpeg_compression',
            ProcessingMode.NOISE_ADDITION: 'noise_addition',
            ProcessingMode.PIXELATION: 'pixelation'
        }.get(self.processing_mode, 'processing')
        log_path = os.path.join(output_dir, f"{mode_name}_metrics_{timestamp}.csv")
        with open(log_path, "w", encoding='utf-8') as log_file:
            log_file.write("檔案名稱,品質等級,PSNR(dB)\n")
        return log_path
    
    def _append_to_log(self, log_path, entries):
        """追加日誌條目"""
        with open(log_path, "a", encoding='utf-8') as log_file:
            for entry in entries:
                log_file.write(f"{entry}\n")
    
    def _create_settings_file(self, output_dir, quality_levels):
        """建立處理設定說明檔案"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        settings_path = os.path.join(output_dir, f"processing_settings_{timestamp}.txt")
        with open(settings_path, "w", encoding='utf-8') as f:
            f.write(f"資料處理設定 - {self.processing_mode.value}\n")
            f.write(f"處理時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"品質範圍: {self.min_quality} - {self.max_quality - 1}\n")
            f.write(f"品質間隔: {self.quality_interval}\n")
            f.write(f"工作程序數: {self.num_workers}\n\n")
            if self.processing_mode == ProcessingMode.JPEG_COMPRESSION:
                f.write("JPEG壓縮品質參數設定:\n")
                f.write("品質等級,Quality值,Chroma Subsampling,DCT方法\n")
                for q in sorted(quality_levels, reverse=True):
                    if q in self.jpeg_quality_settings:
                        settings = self.jpeg_quality_settings[q]
                        dct_method = "FLOAT" if q >= 50 else "FAST"
                        f.write(f"{q},{settings['quality']},{settings['subsampling']},{dct_method}\n")
            elif self.processing_mode == ProcessingMode.NOISE_ADDITION:
                f.write("雜訊添加參數設定:\n")
                f.write("品質等級,高斯雜訊標準差,ISO雜訊,色彩雜訊,壓縮偽影\n")
                for q in sorted(quality_levels, reverse=True):
                    if q in self.noise_params:
                        params = self.noise_params[q]
                        f.write(f"{q},{params['gaussian_std']},{params['iso_noise']},{params['color_noise']},{params['compression']}\n")
            elif self.processing_mode == ProcessingMode.PIXELATION:
                f.write("像素化效果參數設定:\n")
                f.write("品質等級,像素區塊大小\n")
                for q in sorted(quality_levels, reverse=True):
                    pixel_size = (100 - q) // 10 + 1
                    f.write(f"{q},{pixel_size}\n")
    
    def get_processing_stats(self):
        """獲取處理統計資訊"""
        if self.stats['psnr_values']:
            avg_psnr = np.mean([p for p in self.stats['psnr_values'] if p != float('inf')])
            min_psnr = min([p for p in self.stats['psnr_values'] if p != float('inf')] or [0])
            max_psnr = max([p for p in self.stats['psnr_values'] if p != float('inf')] or [0])
        else:
            avg_psnr = min_psnr = max_psnr = 0
        return {
            'total_processed': self.stats['total_processed'],
            'total_generated': self.stats['total_generated'],
            'processing_time': self.stats['processing_time'],
            'average_psnr': avg_psnr,
            'min_psnr': min_psnr,
            'max_psnr': max_psnr,
            'processing_mode': self.processing_mode.value
        }
    
    def set_processing_mode(self, mode: ProcessingMode):
        """設定處理模式"""
        self.processing_mode = mode
        self.logger.info(f"處理模式已設定為: {mode.value}")
    
    def reset_stats(self):
        """重置統計資訊"""
        self.stats = {
            'total_processed': 0,
            'total_generated': 0,
            'processing_time': 0,
            'psnr_values': []
        }
    
    def _process_single_image_multiprocess(self, file_name, input_dir, output_dir, quality_levels, processing_mode, noise_params=None):
        """為多程序處理準備的方法"""
        return _process_single_image_multiprocess(file_name, input_dir, output_dir, quality_levels, processing_mode, noise_params)


def _process_single_image_multiprocess(file_name, input_dir, output_dir, quality_levels, processing_mode, noise_params=None):
    """多程序處理單張圖片的獨立函數"""
    try:
        input_path = os.path.join(input_dir, file_name)
        img = cv2.imread(input_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_array = img.copy()
        base_name, _ = os.path.splitext(file_name)
        log_entries = []
        for quality in quality_levels:
            try:
                if processing_mode == "noise":
                    processed_array = _add_noise_multiprocess(img, quality, noise_params)
                elif processing_mode == "pixel":
                    processed_array = _create_pixelation_multiprocess(img, quality)
                else:
                    processed_array = img.copy()
                psnr = _calculate_psnr_multiprocess(original_array, processed_array)
                output_file = f"{base_name}_q{quality}.jpg"
                output_path = os.path.join(output_dir, output_file)
                processed_bgr = cv2.cvtColor(processed_array, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, processed_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                log_entries.append(f"{output_file},{quality},{psnr:.2f}")
            except Exception as e:
                continue
        return log_entries
    except Exception as e:
        return None


def _add_noise_multiprocess(image, noise_level, noise_params):
    """多程序雜訊添加"""
    if noise_level == 100 or not noise_params:
        return image
    params = noise_params.get(noise_level, noise_params[50])
    noisy_image = image.astype(np.float64)
    if params["gaussian_std"] > 0:
        gaussian_noise = np.random.normal(0, params["gaussian_std"], image.shape)
        noisy_image += gaussian_noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def _create_pixelation_multiprocess(image, quality_level):
    """多程序像素化"""
    pixel_size = (100 - quality_level) // 10 + 1
    if pixel_size <= 1:
        return image
    height, width = image.shape[:2]
    small_width = max(1, width // pixel_size)
    small_height = max(1, height // pixel_size)
    small_img = cv2.resize(image, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small_img, (width, height), interpolation=cv2.INTER_NEAREST)
    return pixelated


def _calculate_psnr_multiprocess(original_arr, processed_arr):
    """多程序PSNR計算"""
    original_arr = original_arr.astype(np.float64)
    processed_arr = processed_arr.astype(np.float64)
    mse = np.mean((original_arr - processed_arr) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr