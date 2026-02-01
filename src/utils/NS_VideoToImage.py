import cv2
import os
import argparse
import glob
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import math
import re
from PIL import Image
import numpy as np
import json
import hashlib
import threading
from pathlib import Path
import logging
from typing import List, Tuple, Optional, Dict, Any
import shutil
import psutil
import gc
from skimage.metrics import structural_similarity as ssim


logger = logging.getLogger(__name__)

class FrameExtractor:
    """影片幀提取器類"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processed_stats = {
            'total_videos': 0,
            'success_videos': 0,
            'failed_videos': 0,
            'total_frames': 0,
            'processing_time': 0,
            'skipped_similar': 0,
            'skipped_duplicate': 0
        }
        self._lock = None
        
    def _get_lock(self):
        """延遲初始化鎖"""
        if self._lock is None:
            self._lock = threading.Lock()
        return self._lock
    
    def sanitize_filename(self, filename: str) -> str:
        """更嚴格處理檔名中的特殊字元"""
        safe_name = re.sub(r'[\\/*?:"<>|＃＆％＠！？\[\]{}()~`]', "_", filename)
        safe_name = re.sub(r'[\s　\t\n\r\f\v]', "_", safe_name)
        safe_name = re.sub(r'_{2,}', "_", safe_name)
        safe_name = safe_name.strip('_')
        if len(safe_name) > 100:
            safe_name = safe_name[:100]
        return safe_name or "unknown"
    
    def get_video_info(self, video_path: str) -> Optional[Dict[str, Any]]:
        """獲取影片資訊"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            info = {
                'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration': 0
            }
            if info['fps'] > 0:
                info['duration'] = info['total_frames'] / info['fps']
            cap.release()
            if info['total_frames'] <= 0 or info['width'] <= 0 or info['height'] <= 0:
                return None
            return info
        except Exception as e:
            logger.error(f"獲取影片資訊失敗 {video_path}: {e}")
            return None
    
    def resize_frame(self, frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        """智能調整幀大小"""
        if not self.config.get('resize', False):
            return frame
        h, w = frame.shape[:2]
        if w == target_width and h == target_height:
            return frame
        if self.config.get('keep_aspect_ratio', True):
            scale = min(target_width / w, target_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            if new_w != target_width or new_h != target_height:
                result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                y_offset = (target_height - new_h) // 2
                x_offset = (target_width - new_w) // 2
                result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
                return result
            else:
                return resized
        else:
            return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    def calculate_frame_hash(self, frame: np.ndarray) -> str:
        """計算幀的哈希值用於完全重複檢測"""
        if not self.config.get('dedup_frames', False):
            return ""
        small_frame = cv2.resize(frame, (64, 64))
        gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        return hashlib.md5(gray_frame.tobytes()).hexdigest()
    
    def calculate_frame_features(self, frame: np.ndarray) -> np.ndarray:
        """計算幀的特徵向量用於相似度比較"""
        if not self.config.get('similarity_check', False):
            return np.array([])
        small_frame = cv2.resize(frame, (128, 128))
        gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        edges = cv2.Canny(gray_frame, 50, 150)
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        texture_features = self.calculate_texture_features(gray_frame)
        color_features = self.calculate_color_features(small_frame)
        features = np.concatenate([
            hist,
            [edge_density],
            texture_features,
            color_features
        ])
        return features
    
    def calculate_texture_features(self, gray_frame: np.ndarray) -> np.ndarray:
        """計算紋理特徵"""
        sobel_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        texture_mean = np.mean(magnitude)
        texture_std = np.std(magnitude)
        texture_energy = np.sum(magnitude**2) / (magnitude.shape[0] * magnitude.shape[1])
        return np.array([texture_mean, texture_std, texture_energy])
    
    def calculate_color_features(self, frame: np.ndarray) -> np.ndarray:
        """計算顏色特徵"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        features = []
        for channel in range(3):
            channel_data = hsv[:, :, channel]
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.median(channel_data)
            ])
        return np.array(features)
    
    def calculate_ssim_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """使用SSIM計算兩幀的相似度"""
        try:
            h, w = min(frame1.shape[0], frame2.shape[0]), min(frame1.shape[1], frame2.shape[1])
            frame1_resized = cv2.resize(frame1, (w, h))
            frame2_resized = cv2.resize(frame2, (w, h))
            gray1 = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2GRAY)
            similarity = ssim(gray1, gray2, data_range=255)
            return similarity
        except Exception as e:
            logger.warning(f"SSIM計算失敗: {e}")
            return 0.0
    
    def calculate_histogram_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """計算直方圖相似度"""
        try:
            h, w = min(frame1.shape[0], frame2.shape[0]), min(frame1.shape[1], frame2.shape[1])
            frame1_resized = cv2.resize(frame1, (w, h))
            frame2_resized = cv2.resize(frame2, (w, h))
            hsv1 = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2HSV)
            hist1 = cv2.calcHist([hsv1], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
            hist2 = cv2.calcHist([hsv2], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
            cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            return max(0.0, correlation)
        except Exception as e:
            logger.warning(f"直方圖相似度計算失敗: {e}")
            return 0.0
    
    def is_solid_color_frame(self, frame: np.ndarray, variance_threshold: float = 100.0) -> bool:
        """檢測純色或近似純色畫面"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            variance = np.var(gray)
            if variance < variance_threshold:
                return True
            b_std = np.std(frame[:, :, 0])
            g_std = np.std(frame[:, :, 1])
            r_std = np.std(frame[:, :, 2])
            if b_std < 10 and g_std < 10 and r_std < 10:
                return True
            return False
        except Exception as e:
            logger.warning(f"純色檢測失敗: {e}")
            return False
    
    def calculate_perceptual_hash(self, frame: np.ndarray) -> str:
        """計算感知哈希（用於檢測結構相似的圖像）"""
        try:
            resized = cv2.resize(frame, (32, 32))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            dct = cv2.dct(np.float32(gray))
            dct_low = dct[:8, :8]
            avg = np.mean(dct_low[1:])
            hash_bits = []
            for i in range(8):
                for j in range(8):
                    if i == 0 and j == 0:
                        continue
                    hash_bits.append('1' if dct_low[i, j] > avg else '0')
            hash_str = ''.join(hash_bits)
            return hex(int(hash_str, 2))[2:].zfill(16)
        except Exception as e:
            logger.warning(f"感知哈希計算失敗: {e}")
            return ""
    
    def calculate_edge_density(self, frame: np.ndarray) -> float:
        """計算邊緣密度"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.shape[0] * edges.shape[1]
            return edge_pixels / total_pixels
        except Exception as e:
            logger.warning(f"邊緣密度計算失敗: {e}")
            return 0.0
    
    def is_frame_similar(self, new_frame: np.ndarray, reference_frames: List[np.ndarray], 
                        similarity_threshold: float = 0.85) -> bool:
        """檢查新幀是否與參考幀過於相似"""
        if not self.config.get('similarity_check', False) or not reference_frames:
            return False
        if self.is_solid_color_frame(new_frame, self.config.get('solid_color_threshold', 100.0)):
            return True
        new_hash = self.calculate_perceptual_hash(new_frame)
        new_edge_density = self.calculate_edge_density(new_frame)
        max_compare_frames = min(len(reference_frames), self.config.get('max_compare_frames', 5))
        recent_frames = reference_frames[-max_compare_frames:]
        for ref_frame in recent_frames:
            if new_hash:
                ref_hash = self.calculate_perceptual_hash(ref_frame)
                if ref_hash:
                    try:
                        hamming_distance = bin(int(new_hash, 16) ^ int(ref_hash, 16)).count('1')
                        if hamming_distance <= self.config.get('perceptual_hash_threshold', 8):
                            return True
                    except:
                        pass
            if self.config.get('use_ssim', True):
                ssim_score = self.calculate_ssim_similarity(new_frame, ref_frame)
                if ssim_score > similarity_threshold:
                    return True
            if self.config.get('use_histogram', True):
                hist_similarity = self.calculate_histogram_similarity(new_frame, ref_frame)
                if hist_similarity > self.config.get('histogram_threshold', 0.9):
                    return True
            if self.config.get('use_edge_density', True):
                ref_edge_density = self.calculate_edge_density(ref_frame)
                edge_diff = abs(new_edge_density - ref_edge_density)
                if edge_diff < self.config.get('edge_density_threshold', 0.02):
                    ssim_score = self.calculate_ssim_similarity(new_frame, ref_frame)
                    if ssim_score > 0.7:
                        return True
            if self.config.get('use_feature_comparison', True):
                ref_features = self.calculate_frame_features(ref_frame)
                new_features = self.calculate_frame_features(new_frame)
                if ref_features.size > 0 and new_features.size > 0:
                    cosine_sim = np.dot(new_features, ref_features) / (
                        np.linalg.norm(new_features) * np.linalg.norm(ref_features) + 1e-8
                    )
                    if cosine_sim > self.config.get('feature_similarity_threshold', 0.9):
                        return True
        return False
    
    def calculate_scene_change_score(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """計算場景變化分數"""
        try:
            h, w = min(frame1.shape[0], frame2.shape[0]), min(frame1.shape[1], frame2.shape[1])
            frame1_resized = cv2.resize(frame1, (w, h))
            frame2_resized = cv2.resize(frame2, (w, h))
            gray1 = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2GRAY)
            hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            return 1.0 - correlation
        except Exception as e:
            logger.warning(f"場景變化分數計算失敗: {e}")
            return 0.0
    
    def save_frame(self, frame: np.ndarray, output_path: str, quality: int = 95) -> bool:
        """保存幀到檔案"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            save_kwargs = {}
            format_lower = self.config['output_format'].lower()
            if format_lower == 'jpg' or format_lower == 'jpeg':
                save_kwargs['quality'] = quality
                save_kwargs['optimize'] = True
            elif format_lower == 'png':
                save_kwargs['optimize'] = True
                save_kwargs['compress_level'] = 6
            abs_output_path = os.path.abspath(output_path)
            img.save(abs_output_path, **save_kwargs)
            return True
        except Exception as e:
            logger.error(f"保存幀失敗 {output_path}: {e}")
            return False
    
    def update_stats(self, frames_saved: int, processing_time: float, 
                    skipped_duplicate: int, skipped_similar: int):
        """更新統計資訊（線程安全）"""
        lock = self._get_lock()
        with lock:
            self.processed_stats['total_frames'] += frames_saved
            self.processed_stats['processing_time'] += processing_time
            self.processed_stats['skipped_duplicate'] += skipped_duplicate
            self.processed_stats['skipped_similar'] += skipped_similar
    
    def extract_frames_from_video(self, video_path: str, output_dir: str) -> Dict[str, Any]:
        """從單個影片提取幀"""
        start_time = time.time()
        video_info = self.get_video_info(video_path)
        if not video_info:
            logger.error(f"無法獲取影片資訊: {video_path}")
            return {'success': False, 'frames': 0, 'time': 0}
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        safe_video_name = self.sanitize_filename(video_name)
        
        logger.info(f"開始處理影片 {safe_video_name}: {video_info['width']}x{video_info['height']}, "
                   f"{video_info['total_frames']} 幀, {video_info['fps']:.2f} FPS, "
                   f"時長 {video_info['duration']:.2f} 秒")
        num_workers = min(self.config['max_workers'], multiprocessing.cpu_count())
        frames_per_worker = max(1, math.ceil(video_info['total_frames'] / num_workers))
        frame_ranges = []
        for i in range(num_workers):
            start_frame = i * frames_per_worker
            if start_frame >= video_info['total_frames']:
                break
            end_frame = min((i + 1) * frames_per_worker, video_info['total_frames'])
            frame_ranges.append((start_frame, end_frame, i))
        if not frame_ranges:
            logger.warning(f"無法為影片 {safe_video_name} 計算有效的幀範圍")
            return {'success': False, 'frames': 0, 'time': 0}
        total_saved_frames = 0
        total_processed_frames = 0
        total_skipped_duplicate = 0
        total_skipped_similar = 0
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for start_frame, end_frame, worker_id in frame_ranges:
                future = executor.submit(
                    process_frame_range_worker,
                    video_path,
                    start_frame,
                    end_frame,
                    output_dir,
                    worker_id,
                    self.config
                )
                futures.append(future)
            for future in as_completed(futures):
                try:
                    result = future.result()
                    total_saved_frames += result['saved_count']
                    total_processed_frames += result['processed_count']
                    total_skipped_duplicate += result['skipped_duplicate']
                    total_skipped_similar += result['skipped_similar']
                    if result['error_count'] > 0:
                        logger.warning(f"Worker {result['worker_id']} 遇到 {result['error_count']} 個錯誤")
                    if result['skipped_duplicate'] > 0:
                        logger.info(f"Worker {result['worker_id']} 跳過 {result['skipped_duplicate']} 個重複幀")
                    if result['skipped_similar'] > 0:
                        logger.info(f"Worker {result['worker_id']} 跳過 {result['skipped_similar']} 個相似幀")
                except Exception as e:
                    logger.error(f"子進程執行錯誤: {e}")
        elapsed_time = time.time() - start_time
        self.update_stats(total_saved_frames, elapsed_time, total_skipped_duplicate, total_skipped_similar)
        logger.info(f"影片 {safe_video_name} 處理完成: {total_saved_frames} 幀已保存, "
                   f"跳過重複: {total_skipped_duplicate}, 跳過相似: {total_skipped_similar}, "
                   f"耗時 {elapsed_time:.2f} 秒")
        return {
            'success': True,
            'frames': total_saved_frames,
            'time': elapsed_time,
            'processed_frames': total_processed_frames,
            'skipped_duplicate': total_skipped_duplicate,
            'skipped_similar': total_skipped_similar
        }
    
    def create_extraction_report(self, output_dir: str):
        """創建提取報告"""
        report = {
            'extraction_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': self.config,
            'stats': self.processed_stats,
            'system_info': {
                'cpu_count': multiprocessing.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3)
            }
        }
        report_path = os.path.join(output_dir, 'extraction_report.json')
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"提取報告已保存至: {report_path}")
        except Exception as e:
            logger.error(f"保存提取報告失敗: {e}")
    
    def process_videos_in_directory(self, input_dir: str, output_dir: str):
        """處理目錄中的所有影片"""
        start_time = time.time()
        os.makedirs(output_dir, exist_ok=True)
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm', '*.m4v']
        video_files = []
        logger.info(f"正在搜尋 {input_dir} 中的影片檔案...")
        for extension in video_extensions:
            if self.config.get('recursive', False):
                pattern = os.path.join(input_dir, '**', extension)
                video_files.extend(glob.glob(pattern, recursive=True))
            else:
                pattern = os.path.join(input_dir, extension)
                video_files.extend(glob.glob(pattern))
        if not video_files:
            logger.error(f"在 {input_dir} 中找不到任何影片檔案")
            return
        if self.config.get('skip_processed', False):
            original_count = len(video_files)
            video_files = [v for v in video_files if not self.is_video_processed(v, output_dir)]
            skipped_count = original_count - len(video_files)
            if skipped_count > 0:
                logger.info(f"跳過已處理的影片: {skipped_count} 個")
        logger.info(f"找到 {len(video_files)} 個影片檔案待處理")
        self.processed_stats['total_videos'] = len(video_files)
        if self.config.get('sort_by_size', False):
            video_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
        for i, video_path in enumerate(video_files, 1):
            logger.info(f"處理進度: {i}/{len(video_files)} - {os.path.basename(video_path)}")
            try:
                result = self.extract_frames_from_video(video_path, output_dir)
                if result['success']:
                    self.processed_stats['success_videos'] += 1
                else:
                    self.processed_stats['failed_videos'] += 1  
            except Exception as e:
                logger.error(f"處理影片 {os.path.basename(video_path)} 時發生嚴重錯誤: {e}")
                self.processed_stats['failed_videos'] += 1
            if i % 5 == 0:
                gc.collect()
        total_time = time.time() - start_time
        self.processed_stats['processing_time'] = total_time
        logger.info("=" * 50)
        logger.info("批量處理完成！")
        logger.info(f"成功處理: {self.processed_stats['success_videos']} 個影片")
        logger.info(f"失敗或跳過: {self.processed_stats['failed_videos']} 個影片")
        logger.info(f"提取總幀數: {self.processed_stats['total_frames']}")
        logger.info(f"跳過重複幀: {self.processed_stats['skipped_duplicate']}")
        logger.info(f"跳過相似幀: {self.processed_stats['skipped_similar']}")
        logger.info(f"總耗時: {total_time:.2f} 秒")
        self.create_extraction_report(output_dir)
    
    def is_video_processed(self, video_path: str, output_dir: str) -> bool:
        """檢查影片是否已經處理過"""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        safe_video_name = self.sanitize_filename(video_name)
        pattern = os.path.join(output_dir, f"{safe_video_name}_frame_*.{self.config['output_format']}")
        return len(glob.glob(pattern)) > 0


def process_frame_range_worker(video_path: str, start_frame: int, end_frame: int, 
                              output_dir: str, worker_id: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """處理影片中指定範圍的幀 - 獨立函數，避免序列化問題"""
    
    def sanitize_filename(filename: str) -> str:
        """檔名處理"""
        safe_name = re.sub(r'[\\/*?:"<>|＃＆％＠！？\[\]{}()~`]', "_", filename)
        safe_name = re.sub(r'[\s　\t\n\r\f\v]', "_", safe_name)
        safe_name = re.sub(r'_{2,}', "_", safe_name)
        safe_name = safe_name.strip('_')
        if len(safe_name) > 100:
            safe_name = safe_name[:100]
        return safe_name or "unknown"
    
    def calculate_frame_hash(frame: np.ndarray) -> str:
        """計算幀哈希"""
        if not config.get('dedup_frames', False):
            return ""
        small_frame = cv2.resize(frame, (64, 64))
        gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        return hashlib.md5(gray_frame.tobytes()).hexdigest()
    
    def calculate_ssim_similarity(frame1: np.ndarray, frame2: np.ndarray) -> float:
        """計算SSIM相似度"""
        try:
            h, w = min(frame1.shape[0], frame2.shape[0]), min(frame1.shape[1], frame2.shape[1])
            frame1_resized = cv2.resize(frame1, (w, h))
            frame2_resized = cv2.resize(frame2, (w, h))
            gray1 = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2GRAY)
            similarity = ssim(gray1, gray2, data_range=255)
            return similarity
        except Exception:
            return 0.0
    
def process_frame_range_worker(video_path: str, start_frame: int, end_frame: int, 
                              output_dir: str, worker_id: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """處理影片中指定範圍的幀 - 獨立函數，避免序列化問題"""
    
    def sanitize_filename(filename: str) -> str:
        """檔名處理"""
        safe_name = re.sub(r'[\\/*?:"<>|＃＆％＠！？\[\]{}()~`]', "_", filename)
        safe_name = re.sub(r'[\s　\t\n\r\f\v]', "_", safe_name)
        safe_name = re.sub(r'_{2,}', "_", safe_name)
        safe_name = safe_name.strip('_')
        if len(safe_name) > 100:
            safe_name = safe_name[:100]
        return safe_name or "unknown"
    
    def calculate_frame_hash(frame: np.ndarray) -> str:
        """計算幀哈希"""
        if not config.get('dedup_frames', False):
            return ""
        small_frame = cv2.resize(frame, (64, 64))
        gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        return hashlib.md5(gray_frame.tobytes()).hexdigest()
    
    def calculate_ssim_similarity(frame1: np.ndarray, frame2: np.ndarray) -> float:
        """計算SSIM相似度"""
        try:
            h, w = min(frame1.shape[0], frame2.shape[0]), min(frame1.shape[1], frame2.shape[1])
            frame1_resized = cv2.resize(frame1, (w, h))
            frame2_resized = cv2.resize(frame2, (w, h))
            gray1 = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2GRAY)
            similarity = ssim(gray1, gray2, data_range=255)
            return similarity
        except Exception:
            return 0.0
    
    def calculate_histogram_similarity(frame1: np.ndarray, frame2: np.ndarray) -> float:
        """計算直方圖相似度"""
        try:
            h, w = min(frame1.shape[0], frame2.shape[0]), min(frame1.shape[1], frame2.shape[1])
            frame1_resized = cv2.resize(frame1, (w, h))
            frame2_resized = cv2.resize(frame2, (w, h))
            hsv1 = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2HSV)
            hist1 = cv2.calcHist([hsv1], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
            hist2 = cv2.calcHist([hsv2], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
            cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            return max(0.0, correlation)
        except Exception:
            return 0.0
    
    def is_solid_color_frame(frame: np.ndarray, variance_threshold: float = 100.0) -> bool:
        """檢測純色或近似純色畫面"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            variance = np.var(gray)
            if variance < variance_threshold:
                return True
            b_std = np.std(frame[:, :, 0])
            g_std = np.std(frame[:, :, 1])
            r_std = np.std(frame[:, :, 2])
            if b_std < 10 and g_std < 10 and r_std < 10:
                return True
            return False
        except Exception:
            return False
    
    def calculate_perceptual_hash(frame: np.ndarray) -> str:
        """計算感知哈希"""
        try:
            resized = cv2.resize(frame, (32, 32))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            dct = cv2.dct(np.float32(gray))
            dct_low = dct[:8, :8]
            avg = np.mean(dct_low[1:])
            hash_bits = []
            for i in range(8):
                for j in range(8):
                    if i == 0 and j == 0:
                        continue
                    hash_bits.append('1' if dct_low[i, j] > avg else '0')
            hash_str = ''.join(hash_bits)
            return hex(int(hash_str, 2))[2:].zfill(16)
        except Exception:
            return ""
    
    def calculate_edge_density(frame: np.ndarray) -> float:
        """計算邊緣密度"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.shape[0] * edges.shape[1]
            return edge_pixels / total_pixels
        except Exception:
            return 0.0
    
    def is_frame_similar(new_frame: np.ndarray, reference_frames: List[np.ndarray], 
                        similarity_threshold: float) -> bool:
        """檢查幀相似度 - 增強版"""
        if not config.get('similarity_check', False) or not reference_frames:
            return False
        if is_solid_color_frame(new_frame, config.get('solid_color_threshold', 100.0)):
            return True
        new_hash = calculate_perceptual_hash(new_frame)
        new_edge_density = calculate_edge_density(new_frame)
        max_compare_frames = min(len(reference_frames), config.get('max_compare_frames', 5))
        recent_frames = reference_frames[-max_compare_frames:]
        for ref_frame in recent_frames:
            if new_hash:
                ref_hash = calculate_perceptual_hash(ref_frame)
                if ref_hash:
                    try:
                        hamming_distance = bin(int(new_hash, 16) ^ int(ref_hash, 16)).count('1')
                        if hamming_distance <= config.get('perceptual_hash_threshold', 8):
                            return True
                    except:
                        pass
            if config.get('use_ssim', True):
                ssim_score = calculate_ssim_similarity(new_frame, ref_frame)
                if ssim_score > similarity_threshold:
                    return True
            if config.get('use_histogram', True):
                hist_similarity = calculate_histogram_similarity(new_frame, ref_frame)
                if hist_similarity > config.get('histogram_threshold', 0.9):
                    return True
            if config.get('use_edge_density', True):
                ref_edge_density = calculate_edge_density(ref_frame)
                edge_diff = abs(new_edge_density - ref_edge_density)
                if edge_diff < config.get('edge_density_threshold', 0.02):
                    ssim_score = calculate_ssim_similarity(new_frame, ref_frame)
                    if ssim_score > 0.7:
                        return True
        return False
    
    def save_frame(frame: np.ndarray, output_path: str, quality: int = 95) -> bool:
        """保存幀"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            save_kwargs = {}
            format_lower = config['output_format'].lower()
            if format_lower == 'jpg' or format_lower == 'jpeg':
                save_kwargs['quality'] = quality
                save_kwargs['optimize'] = True
            elif format_lower == 'png':
                save_kwargs['optimize'] = True
                save_kwargs['compress_level'] = 6
            abs_output_path = os.path.abspath(output_path)
            img.save(abs_output_path, **save_kwargs)
            return True
        except Exception as e:
            logger.error(f"保存幀失敗 {output_path}: {e}")
            return False
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    safe_video_name = sanitize_filename(video_name)
    result = {
        'worker_id': worker_id,
        'saved_count': 0,
        'processed_count': 0,
        'skipped_duplicate': 0,
        'skipped_similar': 0,
        'error_count': 0
    }
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"無法開啟影片 {video_path}")
        return result
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        frame_hashes = set() if config.get('dedup_frames', False) else None
        reference_frames = [] if config.get('similarity_check', False) else None
        similarity_threshold = config.get('similarity_threshold', 0.85)
        pbar = tqdm(
            total=end_frame - start_frame,
            desc=f"Worker {worker_id}: {safe_video_name}",
            position=worker_id % 8,
            leave=False
        )
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"讀取幀失敗 {current_frame}/{end_frame} in {safe_video_name}")
                break
            result['processed_count'] += 1
            if current_frame % config['frame_interval'] == 0:
                try:
                    if frame_hashes is not None:
                        frame_hash = calculate_frame_hash(frame)
                        if frame_hash in frame_hashes:
                            result['skipped_duplicate'] += 1
                            current_frame += 1
                            pbar.update(1)
                            continue
                        frame_hashes.add(frame_hash)
                    if reference_frames is not None:
                        if is_frame_similar(frame, reference_frames, similarity_threshold):
                            result['skipped_similar'] += 1
                            current_frame += 1
                            pbar.update(1)
                            continue
                    output_filename = f"{safe_video_name}_frame_{current_frame:08d}.{config['output_format']}"
                    output_path = os.path.join(output_dir, output_filename)
                    if save_frame(frame, output_path, config.get('quality', 95)):
                        result['saved_count'] += 1
                        if reference_frames is not None:
                            reference_frames.append(frame.copy())
                            max_reference_frames = config.get('max_reference_frames', 10)
                            if len(reference_frames) > max_reference_frames:
                                reference_frames.pop(0)
                    else:
                        result['error_count'] += 1
                except Exception as e:
                    logger.error(f"處理幀 {current_frame} 時發生錯誤: {e}")
                    result['error_count'] += 1
            current_frame += 1
            pbar.update(1)
            if current_frame % 1000 == 0:
                gc.collect()
        pbar.close()
        
    except Exception as e:
        logger.error(f"處理幀範圍時發生錯誤: {e}")
        result['error_count'] += 1
    finally:
        cap.release()
    return result


def load_config(config_path: str) -> Dict[str, Any]:
    """載入配置檔案"""
    default_config = {
        'frame_interval': 72,
        'output_format': 'jpg',
        'quality': 100,
        'resize': False,
        'target_width': 1280,
        'target_height': 720,
        'keep_aspect_ratio': True,
        'max_workers': multiprocessing.cpu_count(),
        'dedup_frames': True,
        'similarity_check': True,
        'similarity_threshold': 0.75,  # 降低SSIM閾值
        'use_ssim': True,
        'use_histogram': True,  # 啟用直方圖檢查
        'use_edge_density': True,  # 啟用邊緣密度檢查
        'use_feature_comparison': True,
        'histogram_threshold': 0.88,  # 直方圖相似度閾值
        'edge_density_threshold': 0.02,  # 邊緣密度差異閾值
        'solid_color_threshold': 150.0,  # 純色檢測閾值
        'perceptual_hash_threshold': 8,  # 感知哈希漢明距離閾值
        'feature_similarity_threshold': 0.88,  # 特徵相似度閾值
        'adaptive_interval': False,
        'min_scene_change': 0.3,
        'max_compare_frames': 8,  # 增加比較幀數
        'max_reference_frames': 15,  # 增加參考幀數
        'recursive': True,
        'skip_processed': True,
        'sort_by_size': False
    }
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            default_config.update(user_config)
            logger.info(f"已載入配置檔案: {config_path}")
        except Exception as e:
            logger.warning(f"載入配置檔案失敗，使用預設配置: {e}")
    return default_config


def save_default_config(config_path: str):
    """保存預設配置檔案"""
    default_config = {
        "frame_interval": 72,
        "output_format": "jpg",
        "quality": 100,
        "resize": False,
        "target_width": 1280,
        "target_height": 720,
        "keep_aspect_ratio": True,
        "max_workers": multiprocessing.cpu_count(),
        "dedup_frames": True,
        "similarity_check": True,
        "similarity_threshold": 0.75,
        "use_ssim": True,
        "use_histogram": True,
        "use_edge_density": True,
        "use_feature_comparison": True,
        "histogram_threshold": 0.88,
        "edge_density_threshold": 0.02,
        "solid_color_threshold": 150.0,
        "perceptual_hash_threshold": 8,
        "feature_similarity_threshold": 0.88,
        "adaptive_interval": False,
        "min_scene_change": 0.3,
        "max_compare_frames": 8,
        "max_reference_frames": 15,
        "recursive": True,
        "skip_processed": True,
        "sort_by_size": False
    }
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        logger.info(f"預設配置檔案已保存至: {config_path}")
    except Exception as e:
        logger.error(f"保存配置檔案失敗: {e}")


def main():
    parser = argparse.ArgumentParser(description="高效能批量影片幀提取工具 - 智能相似度過濾版")
    parser.add_argument("--input-dir", "-i", default="data/inputs", help="輸入影片目錄 (預設: data/inputs)")
    parser.add_argument("--output-dir", "-o", default="data/outputs", help="輸出圖片目錄 (預設: data/outputs)")
    parser.add_argument("--config", "-c", default="v2i_config.json", help="配置檔案路徑 (預設: v2i_config.json)")
    parser.add_argument("--create-config", action="store_true", help="創建預設配置檔案")
    parser.add_argument("--interval", type=int, help="幀提取間隔 (覆蓋配置檔案設定)")
    parser.add_argument("--format", choices=["jpg", "png", "bmp"], help="輸出格式 (覆蓋配置檔案設定)")
    parser.add_argument("--width", type=int, help="目標寬度 (覆蓋配置檔案設定)")
    parser.add_argument("--height", type=int, help="目標高度 (覆蓋配置檔案設定)")
    parser.add_argument("--no-resize", action="store_true", help="不調整圖片大小")
    parser.add_argument("--recursive", action="store_true", help="遞歸搜尋子目錄")
    parser.add_argument("--workers", type=int, help="工作進程數量")
    parser.add_argument("--dedup", action="store_true", help="啟用幀去重")
    parser.add_argument("--similarity-check", action="store_true", help="啟用相似度檢查")
    parser.add_argument("--similarity-threshold", type=float, help="相似度閾值 (0.0-1.0)")
    parser.add_argument("--adaptive-interval", action="store_true", help="啟用自適應間隔提取")
    parser.add_argument("--skip-processed", action="store_true", help="跳過已處理的影片")
    args = parser.parse_args()
    if args.create_config:
        save_default_config(args.config)
        return 0
    config = load_config(args.config)
    if args.interval:
        config['frame_interval'] = args.interval
    if args.format:
        config['output_format'] = args.format
    if args.width:
        config['target_width'] = args.width
    if args.height:
        config['target_height'] = args.height
    if args.no_resize:
        config['resize'] = False
    if args.recursive:
        config['recursive'] = True
    if args.workers:
        config['max_workers'] = args.workers
    if args.dedup:
        config['dedup_frames'] = True
    if args.similarity_check:
        config['similarity_check'] = True
    if args.similarity_threshold:
        config['similarity_threshold'] = args.similarity_threshold
    if args.adaptive_interval:
        config['adaptive_interval'] = True
    if args.skip_processed:
        config['skip_processed'] = True
    if not os.path.exists(args.input_dir):
        logger.error(f"輸入目錄不存在: {args.input_dir}")
        return 1
    logger.info("=" * 50)
    logger.info("智能影片幀提取工具啟動")
    logger.info(f"輸入目錄: {os.path.abspath(args.input_dir)}")
    logger.info(f"輸出目錄: {os.path.abspath(args.output_dir)}")
    logger.info(f"幀提取間隔: {config['frame_interval']}")
    logger.info(f"輸出格式: {config['output_format']}")
    logger.info(f"調整大小: {config['resize']}")
    if config['resize']:
        logger.info(f"目標尺寸: {config['target_width']}x{config['target_height']}")
    logger.info(f"工作進程數: {config['max_workers']}")
    logger.info(f"幀去重: {config['dedup_frames']}")
    logger.info(f"相似度檢查: {config['similarity_check']}")
    if config['similarity_check']:
        logger.info(f"相似度閾值: {config['similarity_threshold']}")
        logger.info(f"使用SSIM: {config['use_ssim']}")
        logger.info(f"使用特徵比較: {config['use_feature_comparison']}")
    logger.info(f"自適應間隔: {config['adaptive_interval']}")
    logger.info(f"遞歸搜尋: {config['recursive']}")
    logger.info(f"跳過已處理: {config['skip_processed']}")
    logger.info("=" * 50)
    try:
        extractor = FrameExtractor(config)
        extractor.process_videos_in_directory(args.input_dir, args.output_dir)
    except KeyboardInterrupt:
        logger.info("用戶中斷處理")
        return 1
    except Exception as e:
        logger.error(f"發生未預期的錯誤: {e}")
        import traceback
        traceback.print_exc()
        return 1
    logger.info("程式執行完畢")
    return 0

if __name__ == "__main__":
    multiprocessing.freeze_support()
    exit_code = main()
    exit(exit_code)
    