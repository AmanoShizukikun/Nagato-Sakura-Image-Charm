import os
import logging
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


class DataProcessor:
    def __init__(self, min_quality=10, max_quality=101, quality_interval=10):
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.quality_interval = quality_interval
        self.logger = logging.getLogger("DataProcessor")
    
    def process_images(self, input_dir, output_dir, callback=None):
        """處理圖片集合，轉換為多種品質等級的訓練資料集
        Args:
            input_dir: 輸入圖片目錄
            output_dir: 輸出目錄
            callback: 進度回呼函式，接收(當前進度, 總數量)參數
        """
        quality_levels = list(range(self.min_quality, self.max_quality, self.quality_interval))
        os.makedirs(output_dir, exist_ok=True)
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        total_files = len(image_files)
        processed_files = 0
        with ThreadPoolExecutor() as executor:
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
                future.result()
                processed_files += 1
                if callback:
                    callback(processed_files, total_files)        
        self.logger.info(f"已完成處理 {processed_files} 張圖片")
        return processed_files

    def process_single_image(self, file_name, input_dir, output_dir, quality_levels):
        """處理單張圖片，生成多種品質版本"""
        input_path = os.path.join(input_dir, file_name)
        try:
            with Image.open(input_path) as img:
                if img.mode == "RGBA":
                    img = img.convert("RGB")
                for quality in quality_levels:
                    base_name, ext = os.path.splitext(file_name)
                    output_file = f"{base_name}_q{quality}.jpg"
                    output_path = os.path.join(output_dir, output_file)
                    adjusted_quality = max(1, int(quality * 1)) 
                    img.save(
                        output_path,
                        "JPEG",
                        quality=adjusted_quality,
                        subsampling=0,
                        dct_method="FAST" 
                    )
        except Exception as e:
            self.logger.error(f"處理圖片 {file_name} 時發生錯誤: {e}")