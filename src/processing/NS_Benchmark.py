import os
import time
import torch
import logging
import numpy as np
import psutil
import random
import gc
import statistics
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal
from torchvision import transforms
from pathlib import Path

from src.processing.NS_PatchProcessor import process_image_in_patches


logger = logging.getLogger(__name__)

def log_gpu_memory(device, message=""):
    """長門櫻會記錄當前設備記憶體使用情況"""
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / (1024*1024)
        reserved = torch.cuda.memory_reserved(device) / (1024*1024)
        logger.debug(f"CUDA顯存狀態 {message}: 已分配={allocated:.2f}MB, 已保留={reserved:.2f}MB")
    elif device.type == 'mps':
        try:
            if hasattr(torch.mps, 'current_allocated_memory'):
                allocated = torch.mps.current_allocated_memory() / (1024*1024)
                logger.debug(f"MPS記憶體狀態 {message}: 已分配={allocated:.2f}MB")
            elif hasattr(torch.mps, 'driver_allocated_memory'):
                allocated = torch.mps.driver_allocated_memory() / (1024*1024)
                logger.debug(f"MPS記憶體狀態 {message}: 已分配={allocated:.2f}MB")
        except Exception as e:
            logger.debug(f"無法獲取MPS記憶體狀態 {message}: {str(e)}")

def sync_device_memory(device):
    """長門櫻會同步設備記憶體並釋放暫存資料"""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        if hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()

class BenchmarkProcessor:
    """長門櫻-影像魅影基準測試"""
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.stop_flag = False
        self.test_images_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets", "preview")
        self.reference_time = {
            "inference": 0.5, 
            "real_usage": 1.2 
        }
        self.baseline_score = 10000
    
    def _get_test_images(self, count=5):
        """長門櫻會從 assets/preview 目錄為主人挑選適合的測試圖片"""
        image_files = []
        if os.path.exists(self.test_images_dir):
            all_files = [f for f in os.listdir(self.test_images_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
            if len(all_files) >= count:
                selected_files = sorted(all_files)[:min(count, len(all_files))]
                image_files = [os.path.join(self.test_images_dir, f) for f in selected_files]
            else:
                available_files = [os.path.join(self.test_images_dir, f) for f in all_files]
                while len(image_files) < count:
                    image_files.extend(available_files[:min(count - len(image_files), len(available_files))])
        if not image_files:
            logger.warning(f"長門櫻無法從 assets/preview 目錄找到測試圖片，請主人確認目錄是否存在並包含圖片")
        return image_files
    
    def _calculate_performance_score(self, avg_time, resolution, is_real_usage, times=None):
        """長門櫻會更精準地計算主人系統的性能分數，使分數差距與實際性能差距成正比"""
        if avg_time <= 0:
            return 0
        width, height = resolution
        pixels = width * height
        standard_pixels = 1280 * 640
        resolution_factor = pixels / standard_pixels
        reference_time = self.reference_time["real_usage" if is_real_usage else "inference"]
        raw_score = self.baseline_score * (reference_time / avg_time) * resolution_factor
        stability_factor = 1.0
        if times and len(times) > 2:
            cv = statistics.stdev(times) / avg_time
            if cv < 0.03: 
                stability_factor = 1.02
            elif cv < 0.05:
                stability_factor = 1.01
            elif cv > 0.15: 
                stability_factor = 0.99
            elif cv > 0.25: 
                stability_factor = 0.98
        adjusted_score = raw_score * stability_factor
        if getattr(self, 'is_cpu_test', False) and is_real_usage:
            adjusted_score *= 1.1
        return int(adjusted_score)
    
    def _remove_outliers(self, times):
        """長門櫻會去除異常值以提高測試結果的穩定性"""
        if len(times) <= 4: 
            return times
        q1 = np.percentile(times, 25)
        q3 = np.percentile(times, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_times = [t for t in times if lower_bound <= t <= upper_bound]
        if len(filtered_times) < max(3, len(times) * 0.5):
            return times 
        return filtered_times
    
    def run_model_inference_test(self, model, device, width, height, iterations, use_amp=None, progress_callback=None, step_callback=None):
        """長門櫻會用心為主人測試模型推理性能"""
        if step_callback:
            step_callback("長門櫻正在為主人初始化模型推理測試中...")
        model.eval()
        self.stop_flag = False
        self.is_cpu_test = (device.type == 'cpu')
        original_width, original_height = width, height
        gc.collect()
        if device.type in ['cuda', 'mps']:
            sync_device_memory(device)
            log_gpu_memory(device, "測試開始前") 
        block_size = self._determine_optimal_block_size(device)
        chunks_x = (width + block_size - 1) // block_size
        chunks_y = (height + block_size - 1) // block_size
        total_chunks = chunks_x * chunks_y
        if step_callback:
            step_callback(f"長門櫻會使用 {chunks_x}x{chunks_y} 的區塊進行測試 (每塊 {block_size}x{block_size})，請稍等一下")
        if progress_callback:
            progress_callback(0, iterations)
        test_tensor = torch.randn(1, 3, block_size, block_size, device=device)
        test_tensor = torch.clamp(test_tensor, -1.0, 1.0)
        if use_amp is None and device.type == 'cuda':
            use_amp = self._should_use_amp(device)
        else:
            use_amp = False if device.type != 'cuda' else (use_amp or False)
        start_memory = psutil.virtual_memory().used / (1024 * 1024)  # MB
        gpu_memory_start = 0
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            gpu_memory_start = torch.cuda.memory_allocated(device) / (1024 * 1024)  # MB
        try:
            self._warmup_model(model, test_tensor, device, use_amp, step_callback, warmup_iterations=3)
        except Exception as e:
            logger.error(f"長門櫻在預熱模型時遇到了困難: {str(e)}")
            if device.type in ['cuda', 'mps']:
                test_tensor = None
                sync_device_memory(device)
                gc.collect()
            if progress_callback:
                progress_callback(0, iterations)
            return {"error": f"模型預熱失敗了，長門櫻很抱歉: {str(e)}"}
        times = []
        warmup_iterations = min(2, iterations // 4) 
        total_start_time = time.time()
        total_test_iterations = iterations
        try:
            with torch.no_grad():
                for i in range(iterations + warmup_iterations):
                    if self.stop_flag:
                        if device.type in ['cuda', 'mps']:
                            test_tensor = None
                            sync_device_memory(device)
                            gc.collect()
                        return {"error": "長門櫻已停止了測試"}
                    if step_callback:
                        if i < warmup_iterations:
                            step_callback(f"長門櫻正在額外預熱模型 ({i+1}/{warmup_iterations})...")
                        else:
                            step_callback(f"長門櫻正在為主人執行第 {i+1-warmup_iterations}/{iterations} 次測試...")   
                    iter_start = time.time()
                    for y in range(chunks_y):
                        for x in range(chunks_x):
                            if use_amp and device.type == 'cuda':
                                with torch.amp.autocast(device_type='cuda'):
                                    _ = model(test_tensor)
                            else:
                                _ = model(test_tensor)
                            if (y * chunks_x + x + 1) % 10 == 0 and device.type in ['cuda', 'mps']:
                                if device.type == 'cuda':
                                    torch.cuda.synchronize()
                                elif device.type == 'mps' and hasattr(torch.mps, 'synchronize'):
                                    torch.mps.synchronize()
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    elif device.type == 'mps' and hasattr(torch.mps, 'synchronize'):
                        torch.mps.synchronize()
                    iter_time = time.time() - iter_start
                    scaled_time = iter_time * (width * height) / (total_chunks * block_size * block_size)
                    if i >= warmup_iterations:
                        times.append(scaled_time)
                        current_iteration = i + 1 - warmup_iterations
                        if progress_callback and current_iteration <= iterations:
                            progress_callback(current_iteration, iterations)
                            logger.debug(f"推理進度更新: {current_iteration}/{iterations} ({(current_iteration/iterations)*100:.1f}%)")
        except Exception as e:
            logger.error(f"長門櫻在推理測試時遇到了困難: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            if device.type in ['cuda', 'mps']:
                test_tensor = None
                sync_device_memory(device)
                gc.collect()
            return {"error": f"抱歉啊主人~測試失敗了: {str(e)}"}
        total_time = time.time() - total_start_time
        filtered_times = self._remove_outliers(times)
        avg_time = np.mean(filtered_times)
        min_time = min(filtered_times)
        max_time = max(filtered_times)
        median_time = np.median(filtered_times)
        std_dev = np.std(filtered_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        megapixels_per_second = (width * height) / (1024 * 1024 * avg_time) if avg_time > 0 else 0
        end_memory = psutil.virtual_memory().used / (1024 * 1024)  # MB
        memory_delta = end_memory - start_memory
        gpu_memory_delta = 0
        peak_memory_usage = 0
        if device.type == 'cuda':
            gpu_memory_end = torch.cuda.memory_allocated(device) / (1024 * 1024)  # MB
            gpu_memory_delta = gpu_memory_end - gpu_memory_start
            peak_memory_usage = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # MB
        elif device.type == 'mps':
            try:
                if hasattr(torch.mps, 'current_allocated_memory'):
                    gpu_memory_end = torch.mps.current_allocated_memory() / (1024 * 1024)  # MB
                    gpu_memory_delta = gpu_memory_end - gpu_memory_start
                peak_memory_usage = gpu_memory_end
            except Exception as e:
                logger.debug(f"無法獲取MPS記憶體結束狀態: {str(e)}")
                gpu_memory_delta = 0
                peak_memory_usage = 0
        score = self._calculate_performance_score(avg_time, (original_width, original_height), False, filtered_times)
        if step_callback:
            step_callback("長門櫻正在為主人整理測試結果...")
        if device.type in ['cuda', 'mps']:
            test_tensor = None
            sync_device_memory(device)
            gc.collect()
            log_gpu_memory(device, "測試結束後")
        if progress_callback:
            progress_callback(iterations, iterations)
            logger.debug("推理測試完成，進度設為100%")
        return {
            "test_type": "模型推理測試",
            "times": times,
            "filtered_times": filtered_times,
            "average_time": avg_time,
            "median_time": median_time,
            "min_time": min_time,
            "max_time": max_time,
            "std_dev": std_dev,
            "cv": std_dev / avg_time if avg_time > 0 else 0,
            "fps": fps,
            "megapixels_per_second": megapixels_per_second,
            "total_time": total_time,
            "memory_delta": memory_delta,
            "gpu_memory_delta": gpu_memory_delta,
            "peak_memory_usage": peak_memory_usage,
            "resolution": f"{original_width}x{original_height}",
            "actual_resolution": f"{original_width}x{original_height} (區塊處理)",
            "iterations": iterations,
            "device": device.type,
            "amp_used": use_amp,
            "score": score,
            "block_size": block_size,
            "total_blocks": total_chunks,
            "test_completed": True
        }
    
    def run_real_usage_test(self, model, device, width, height, num_images, block_size, overlap,
                        use_amp=None, progress_callback=None, step_callback=None):
        """長門櫻會用真實場景為主人測試模型性能，請耐心等待"""
        if step_callback:
            step_callback("長門櫻正在為主人初始化實際場景測試...")
        model.eval()
        self.stop_flag = False
        self.is_cpu_test = (device.type == 'cpu')
        warmup_count = 1
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            log_gpu_memory(device, "實際場景測試開始前")
        if step_callback:
            step_callback("長門櫻正在為主人準備 assets/preview 中的測試圖片...")
        total_images_needed = num_images + warmup_count
        image_files = self._get_test_images(total_images_needed)
        images = []
        if not image_files:
            if progress_callback:
                progress_callback(0, 100)
            return {"error": "長門櫻無法從 assets/preview 目錄中找到測試用圖片，請主人確認目錄是否存在並包含圖片"}
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert("RGB")
                img_array = np.array(img)
                img_array = np.clip(img_array, 0, 255)
                img = Image.fromarray(img_array.astype(np.uint8))
                img = img.resize((1280, 640), Image.Resampling.LANCZOS)
                images.append(img)
            except Exception as e:
                logger.error(f"長門櫻加載圖片 {img_path} 時遇到了困難: {str(e)}")
                if progress_callback:
                    progress_callback(0, 100)
                return {"error": f"加載測試圖片失敗了，長門櫻很抱歉: {str(e)}"}  
        if not images:
            if progress_callback:
                progress_callback(0, 100)
            return {"error": "長門櫻無法處理任何測試圖片，請主人檢查圖片格式是否正確"}
        if len(images) < total_images_needed:
            logger.warning(f"長門櫻只找到 {len(images)} 張圖片，少於需要的 {total_images_needed} 張")
            while len(images) < total_images_needed:
                images.append(images[len(images) % len(image_files)])       
        if use_amp is None and device.type == 'cuda':
            use_amp = self._should_use_amp(device)
        else:
            use_amp = False if device.type != 'cuda' else (use_amp or False) 
        start_memory = psutil.virtual_memory().used / (1024 * 1024)  # MB
        gpu_memory_start = 0
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            gpu_memory_start = torch.cuda.memory_allocated(device) / (1024 * 1024)  # MB
        log_gpu_memory(device, "預熱前")
        test_enhancer = None
        try:
            if step_callback:
                step_callback("長門櫻正在幫模型預熱中，請主人稍等一下...")
            test_enhancer = process_image_in_patches(
                model, 
                images[0], 
                device, 
                block_size=block_size, 
                overlap=overlap, 
                use_weight_mask=True, 
                blending_mode='改進型高斯分佈',
                use_amp=use_amp
            )
            _ = test_enhancer.process(lambda x, y: None)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
            if device.type == 'cuda':
                _ = test_enhancer.process(lambda x, y: None)
                torch.cuda.empty_cache()
                gc.collect()
        except Exception as e:
            logger.error(f"長門櫻在預熱模型時遇到了困難: {str(e)}")
            if test_enhancer is not None:
                if hasattr(test_enhancer, 'cleanup') and callable(test_enhancer.cleanup):
                    test_enhancer.cleanup()
                test_enhancer = None
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
            if progress_callback:
                progress_callback(0, 100)
            return {"error": f"模型預熱失敗了，長門櫻很抱歉: {str(e)}"}
        log_gpu_memory(device, "預熱後")
        times = []
        total_start_time = time.time()
        enhancers = []
        total_blocks = 0
        blocks_per_image = []
        for img in images[warmup_count:]:
            img_width, img_height = img.size
            step = block_size - overlap
            x_blocks = max(1, (img_width - overlap) // step + (1 if (img_width - overlap) % step != 0 else 0))
            y_blocks = max(1, (img_height - overlap) // step + (1 if (img_height - overlap) % step != 0 else 0))
            img_blocks = x_blocks * y_blocks
            blocks_per_image.append(img_blocks)
            total_blocks += img_blocks
        completed_blocks = 0
        if progress_callback:
            progress_callback(0, total_blocks)
            logger.debug(f"初始化進度條: 0/{total_blocks}")
        try:
            for i, img in enumerate(images):
                if self.stop_flag:
                    self._cleanup_enhancers(enhancers, device)
                    return {"error": "主人中止了測試，長門櫻已經停止了"}
                if step_callback:
                    if i < warmup_count:
                        step_callback(f"長門櫻正在額外預熱模型 ({i+1}/{warmup_count})...")
                    else:
                        step_callback(f"長門櫻正在為主人處理第 {i+1-warmup_count}/{num_images} 張圖片...")
                blocks_done_in_current_image = 0
                total_blocks_in_current_image = blocks_per_image[i-warmup_count] if i >= warmup_count else 0
                def block_progress_callback(blocks_done, total):
                    nonlocal blocks_done_in_current_image
                    if i >= warmup_count:
                        delta = blocks_done - blocks_done_in_current_image
                        if delta > 0: 
                            blocks_done_in_current_image = blocks_done
                            global_blocks_done = completed_blocks + blocks_done
                            if global_blocks_done <= total_blocks and progress_callback:
                                progress_callback(global_blocks_done, total_blocks)
                                logger.debug(f"區塊進度更新: {global_blocks_done}/{total_blocks} ({(global_blocks_done/total_blocks)*100:.1f}%)")
                iter_start = time.time()
                enhancer = process_image_in_patches(
                    model, 
                    img, 
                    device, 
                    block_size=block_size, 
                    overlap=overlap, 
                    use_weight_mask=True, 
                    blending_mode='改進型高斯分佈',
                    use_amp=use_amp
                )
                enhancers.append(enhancer)
                _ = enhancer.process(block_progress_callback)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                iter_time = time.time() - iter_start
                if i >= warmup_count:
                    times.append(iter_time)
                    completed_blocks += blocks_per_image[i-warmup_count]
                    if progress_callback:
                        progress_callback(completed_blocks, total_blocks)
                        logger.debug(f"圖片 {i+1-warmup_count} 完成，累計進度: {completed_blocks}/{total_blocks} ({(completed_blocks/total_blocks)*100:.1f}%)")
        except Exception as e:
            logger.error(f"長門櫻在實際場景測試時遇到了困難: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self._cleanup_enhancers(enhancers, device)
            return {"error": f"測試失敗了，長門櫻向主人道歉: {str(e)}"}
        total_time = time.time() - total_start_time
        filtered_times = self._remove_outliers(times)
        avg_time = np.mean(filtered_times) if filtered_times else 0
        min_time = min(filtered_times) if filtered_times else 0
        max_time = max(filtered_times) if filtered_times else 0
        median_time = np.median(filtered_times) if filtered_times else 0
        std_dev = np.std(filtered_times) if filtered_times else 0
        fps = 1.0 / avg_time if avg_time > 0 else 0
        megapixels_per_second = (1280 * 640) / (1024 * 1024 * avg_time) if avg_time > 0 else 0
        end_memory = psutil.virtual_memory().used / (1024 * 1024)  # MB
        memory_delta = end_memory - start_memory
        gpu_memory_delta = 0
        peak_memory_usage = 0
        if device.type == 'cuda':
            gpu_memory_end = torch.cuda.memory_allocated(device) / (1024 * 1024)  # MB
            gpu_memory_delta = gpu_memory_end - gpu_memory_start
            peak_memory_usage = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # MB
        score = self._calculate_performance_score(avg_time, (1280, 640), True, filtered_times)
        if step_callback:
            step_callback("長門櫻正在為主人整理測試結果...")
        self._cleanup_enhancers(enhancers, device)
        log_gpu_memory(device, "最終清理後")
        if progress_callback:
            progress_callback(total_blocks, total_blocks)
            logger.debug("實際場景測試完成，進度設為100%")
        return {
            "test_type": "實際場景測試",
            "times": times,
            "filtered_times": filtered_times,
            "average_time": avg_time,
            "median_time": median_time,
            "min_time": min_time,
            "max_time": max_time,
            "std_dev": std_dev,
            "cv": std_dev / avg_time if avg_time > 0 else 0,
            "fps": fps,
            "megapixels_per_second": megapixels_per_second,
            "total_time": total_time,
            "memory_delta": memory_delta,
            "gpu_memory_delta": gpu_memory_delta,
            "peak_memory_usage": peak_memory_usage,
            "resolution": "1280x640",
            "iterations": num_images, 
            "block_size": block_size,
            "overlap": overlap,
            "device": device.type,
            "amp_used": use_amp,
            "score": score,
            "total_blocks": total_blocks,
            "processed_blocks": completed_blocks,
            "test_completed": True 
        }
        
    def _cleanup_enhancers(self, enhancers, device):
        """長門櫻會清理所有NS-IQE模型並釋放顯存"""
        try:
            if device.type in ['cuda', 'mps']:
                logger.debug(f"開始清理 {len(enhancers)} 個NS-IQE模型")
                log_gpu_memory(device, "清理前")
                for i in range(len(enhancers)):
                    if enhancers[i] is None:
                        continue
                    if hasattr(enhancers[i], 'cleanup') and callable(enhancers[i].cleanup):
                        enhancers[i].cleanup()
                    else:
                        if hasattr(enhancers[i], 'image_tensor') and enhancers[i].image_tensor is not None:
                            enhancers[i].image_tensor = None
                        if hasattr(enhancers[i], 'processor') and enhancers[i].processor is not None:
                            if hasattr(enhancers[i].processor, 'weight_mask'):
                                enhancers[i].processor.weight_mask = None
                            enhancers[i].processor = None
                    enhancers[i] = None  
                enhancers.clear()
                sync_device_memory(device)
                gc.collect()
                time.sleep(0.1)
                sync_device_memory(device)
                log_gpu_memory(device, "清理後")
        except Exception as e:
            logger.error(f"清理NS-IQE模型時出錯: {str(e)}")
            if device.type in ['cuda', 'mps']:
                sync_device_memory(device)
                gc.collect()
    
    def _determine_optimal_block_size(self, device):
        """長門櫻會動態確定最佳區塊大小"""
        block_size = 128
        if device.type == 'cuda':
            try:
                total_memory = torch.cuda.get_device_properties(device).total_memory
                allocated_memory = torch.cuda.memory_allocated(device)
                free_memory = total_memory - allocated_memory
                if free_memory > 1 * 1024 * 1024 * 1024: 
                    block_size = 256
                logger.info(f"長門櫻根據主人的顯存情況（{free_memory / (1024**3):.2f} GB可用）選擇了 {block_size} 的區塊大小")
            except Exception as e:
                logger.warning(f"長門櫻無法檢測主人的顯存情況，會使用保守的區塊大小: {str(e)}")
        elif device.type == 'mps':
            block_size = 256
            logger.info(f"長門櫻為MPS設備選擇了 {block_size} 的區塊大小")
        else:
            block_size = 256
            logger.info(f"長門櫻為CPU設備選擇了 {block_size} 的區塊大小")
        return block_size
    
    def _warmup_model(self, model, test_tensor, device, use_amp, step_callback=None, warmup_iterations=3):
        """更全面的模型預熱"""
        if step_callback:
            step_callback("長門櫻正在幫模型預熱中，這樣測試會更準確...")
        with torch.no_grad():
            for i in range(warmup_iterations):
                if i > 0 and step_callback:
                    step_callback(f"長門櫻正在進行第 {i+1}/{warmup_iterations} 次預熱...")
                if use_amp and device.type == 'cuda':
                    with torch.amp.autocast(device_type='cuda'):
                        _ = model(test_tensor)
                else:
                    _ = model(test_tensor)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                    if i < warmup_iterations - 1:
                        torch.cuda.empty_cache()
                elif device.type == 'mps':
                    if hasattr(torch.mps, 'synchronize'):
                        torch.mps.synchronize()
                    if i < warmup_iterations - 1 and hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps' and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        gc.collect()
    
    def stop(self):
        """長門櫻會立即停止測試，隨時聽從主人的指示"""
        self.stop_flag = True
    
    def _should_use_amp(self, device):
        """長門櫻會貼心判斷主人的顯卡是否適合使用混合精度計算"""
        if device.type != 'cuda':
            if device.type == 'mps':
                logger.info("長門櫻檢測到MPS設備，MPS暫不支援混合精度計算")
            return False
        try:
            gpu_name = torch.cuda.get_device_name(device)
            excluded_gpus = ['1650', '1660', 'MX', 'P4', 'P40', 'K80', 'M4', 'GTX 16', 'GTX 10']
            for model in excluded_gpus:
                if model in gpu_name:
                    logger.info(f"長門櫻檢測到主人的GPU型號 {gpu_name} 不適合混合精度計算，為了穩定性已禁用")
                    return False
            compute_capability = torch.cuda.get_device_capability(device)
            if compute_capability[0] >= 7:
                logger.info(f"長門櫻檢測到主人的GPU計算能力為 {compute_capability[0]}.{compute_capability[1]} >= 7.0，可以啟用混合精度")
                return True
            amp_supported_gpus = ['RTX', 'A100', 'A10', 'V100', 'T4', '30', '40', 'TITAN V']
            for gpu_model in amp_supported_gpus:
                if gpu_model in gpu_name:
                    logger.info(f"長門櫻發現主人的GPU型號 {gpu_name} 支持混合精度計算，太好了！")
                    return True
            if hasattr(torch.cuda, 'get_device_properties'):
                props = torch.cuda.get_device_properties(device)
                if hasattr(props, 'major') and props.major >= 7:
                    logger.info(f"長門櫻檢測到主人的GPU支持混合精度計算 (CUDA能力: {props.major}.{props.minor})")
                    return True
        except Exception as e:
            logger.warning(f"長門櫻在檢測混合精度支持時遇到了困難: {str(e)}")
        return False

class BenchmarkWorker(QThread):
    """長門櫻的，會在背景為主人執行測試，不打擾主人使用電腦"""
    progress_signal = pyqtSignal(int, int)
    step_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)
    
    def __init__(self, processor, device, width, height, iterations, 
                is_real_usage=True, block_size=256, overlap=64, num_images=5, use_amp=None):
        super().__init__()
        self.processor = processor
        self.device = device
        self.width = width
        self.height = height
        self.iterations = iterations
        self.is_real_usage = is_real_usage
        self.block_size = block_size
        self.overlap = overlap
        self.num_images = num_images
        self.use_amp = use_amp
        self.stop_flag = False
    
    def run(self):
        """長門櫻會認真執行測試，請主人稍等片刻"""
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.progress_signal.emit(0, 100)
            self.step_signal.emit("長門櫻正在為主人載入模型中...")
            success = self.processor.model_manager.prepare_model_for_inference()
            if not success:
                self.finished_signal.emit({"error": "長門櫻無法載入模型，請主人確認是否已選擇有效模型"})
                return
            model = self.processor.model_manager.get_current_model()
            if model is None:
                self.finished_signal.emit({"error": "長門櫻找不到已選擇的模型，請主人先選擇一個模型再試試看"})
                return
            if self.is_real_usage:
                results = self.processor.run_real_usage_test(
                    model, 
                    self.device, 
                    1280, 
                    640,  
                    self.num_images,
                    self.block_size, 
                    self.overlap,
                    self.use_amp,
                    progress_callback=self.progress_signal.emit,
                    step_callback=self.step_signal.emit
                )
            else:
                results = self.processor.run_model_inference_test(
                    model, 
                    self.device, 
                    self.width, 
                    self.height, 
                    self.iterations,
                    self.use_amp,
                    progress_callback=self.progress_signal.emit,
                    step_callback=self.step_signal.emit
                )
            if not self.stop_flag and "error" not in results:
                self.progress_signal.emit(100, 100)
                logger.debug("測試完成，進度條設為100%")
            self.finished_signal.emit(results)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                log_gpu_memory(self.device, "測試完成後最終清理")
        except Exception as e:
            logger.error(f"長門櫻在執行基準測試時遇到了困難: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self.finished_signal.emit({"error": f"測試失敗了，長門櫻向主人道歉: {str(e)}"})
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def stop(self):
        """長門櫻會立即停止測試，隨時聽從主人的指示"""
        self.processor.stop()
        self.stop_flag = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()