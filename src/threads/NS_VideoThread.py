import os
import cv2
import tempfile
import shutil
import time
import threading
import gc
import re
import subprocess
import numpy as np
import torch
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal
import logging

from src.processing.NS_PatchProcessor import process_image_in_patches

logger = logging.getLogger(__name__)

class VideoEnhancerThread(QThread):
    """影片增強處理執行緒，負責逐幀處理並輸出增強後的影片"""
    progress_signal = pyqtSignal(int, int, str)  # 進度信號：目前進度、總進度、狀態訊息
    finished_signal = pyqtSignal(str, float)     # 完成信號：輸出檔案路徑、耗時
    preview_signal = pyqtSignal(object, int)     # 預覽信號：增強後的幀、幀索引

    def __init__(self, model, input_path, output_path, device, block_size, overlap, 
                use_weight_mask, blending_mode, frame_step=1, preview_interval=1, keep_audio=True,
                sync_preview=True, video_options=None, strength=1.0):
        super().__init__()
        self.model = model
        self.input_path = input_path
        self.output_path = output_path
        self.device = device
        self.block_size = block_size
        self.overlap = overlap
        self.use_weight_mask = use_weight_mask
        self.blending_mode = blending_mode
        self.frame_step = frame_step 
        self.preview_interval = preview_interval
        self.keep_audio = keep_audio
        self.sync_preview = sync_preview
        self.video_options = video_options or {}
        self.strength = strength
        self.stop_flag = threading.Event()
        self.cap = None
        self.out = None
        self.temp_dir = None
        self.current_preview_frame = None
        self.ffmpeg_process = None
        self.high_frequency_preview = self.video_options.get('high_freq_preview', True)
        self.use_amp = self.video_options.get('use_amp', None)
        if self.use_amp is None:
            self.use_amp = self._should_use_amp(device)
        if device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(device)
            logger.info(f"使用GPU: {gpu_name}")
            logger.info(f"CUDA版本: {torch.version.cuda}")
            logger.info(f"混合精度計算: {'啟用' if self.use_amp else '禁用'}")
            if self.use_amp:
                logger.info("使用自動混合精度計算模式 (自動偵測結果)")
            else:
                logger.info("使用標準精度計算模式 (自動偵測結果)")

    def _should_use_amp(self, device):
        if device.type != 'cuda':
            return False
        gpu_name = torch.cuda.get_device_name(device)
        logger.info(f"正在檢測GPU '{gpu_name}' 是否適合使用混合精度...")
        try:
            cuda_version = torch.version.cuda
            if cuda_version:
                cuda_major = int(cuda_version.split('.')[0])
                logger.info(f"CUDA主要版本: {cuda_major}")
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
            logger.info(f"GPU計算能力: {compute_capability}")
            if compute_capability >= 7.0:
                logger.info(f"GPU計算能力 {compute_capability} >= 7.0，啟用混合精度")
                return True
        try:
            test_tensor = torch.randn(1, 3, 64, 64, device=device, dtype=torch.float32)
            with torch.no_grad():
                try:
                    with torch.amp.autocast(device_type='cuda'):
                        _ = self.model(test_tensor)
                    logger.info("混合精度測試成功，啟用混合精度計算")
                    return True
                except Exception as e:
                    logger.warning(f"混合精度測試失敗: {e}")
                    return False
        except Exception as e:
            logger.warning(f"混合精度功能測試出錯: {e}")
        logger.info("無法確定GPU是否支持混合精度，為安全起見禁用")
        return False

    def stop(self):
        logger.info("正在停止影片處理...")
        self.stop_flag.set()
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.terminate()
                logger.info("已終止FFmpeg進程")
            except Exception as e:
                logger.error(f"終止FFmpeg進程時出錯: {str(e)}")
        self.safe_release_capture()
        self.safe_release_writer()
        if self.current_preview_frame:
            del self.current_preview_frame
            self.current_preview_frame = None
        gc.collect()
        logger.info("已釋放所有資源")

    def safe_release_capture(self):
        if hasattr(self, 'cap') and self.cap is not None:
            try:
                self.cap.release()
                logger.debug("已釋放視頻捕獲資源")
            except Exception as e:
                logger.error(f"釋放視頻捕獲資源時出錯: {str(e)}")
            self.cap = None

    def safe_release_writer(self):
        if hasattr(self, 'out') and self.out is not None:
            try:
                self.out.release()
                logger.debug("已釋放視頻寫入資源")
            except Exception as e:
                logger.error(f"釋放視頻寫入資源時出錯: {str(e)}")
            self.out = None

    def safe_remove_temp_dir(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                time.sleep(0.5)
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                logger.debug(f"已刪除臨時目錄: {self.temp_dir}")
                if os.path.exists(self.temp_dir):
                    logger.warning(f"無法完全刪除臨時目錄: {self.temp_dir}")
            except Exception as e:
                logger.error(f"刪除臨時目錄時出錯: {str(e)}")

    def check_ffmpeg_installed(self):
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True)
            return result.returncode == 0
        except Exception:
            return False

    def run(self):
        try:
            if not self.check_ffmpeg_installed():
                self.progress_signal.emit(0, 100, "錯誤：未找到FFmpeg。請安裝FFmpeg並確保它在系統路徑中。")
                logger.error("未找到FFmpeg，無法繼續處理")
                return
            start_time = time.time()
            self.temp_dir = tempfile.mkdtemp(prefix="ns_video_")
            input_frames_dir = os.path.join(self.temp_dir, "input_frames")
            output_frames_dir = os.path.join(self.temp_dir, "output_frames")
            os.makedirs(input_frames_dir, exist_ok=True)
            os.makedirs(output_frames_dir, exist_ok=True)
            self.progress_signal.emit(0, 100, "正在提取影片幀")
            self.cap = cv2.VideoCapture(self.input_path)
            if not self.cap.isOpened():
                raise Exception("無法打開影片")
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            orig_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width, height = orig_width, orig_height
            resolution = self.video_options.get('resolution', '原始大小')
            scale_factor = float(self.video_options.get('scale_factor', 1.0))
            custom_width = self.video_options.get('custom_width')
            custom_height = self.video_options.get('custom_height')
            if resolution == '原始大小':
                width, height = orig_width, orig_height
            elif resolution == '超分倍率':
                width = int(orig_width * scale_factor)
                height = int(orig_height * scale_factor)
                logger.info(f"使用超分倍率: {scale_factor}，輸出尺寸: {width}x{height}")
            elif resolution == '自訂':
                try:
                    width = int(custom_width)
                    height = int(custom_height)
                    logger.info(f"使用自訂分辨率: {width}x{height}")
                except Exception:
                    logger.warning("無法解析自訂分辨率，使用原始分辨率")
                    width, height = orig_width, orig_height
            width = width + (width % 2)
            height = height + (height % 2)
            frame_count = 0
            saved_count = 0
            while not self.stop_flag.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    break
                crop_mode = self.video_options.get('crop_mode', '無裁切')
                if crop_mode == '居中裁切':
                    h, w = frame.shape[:2]
                    new_w = min(w, int(h * 16/9))
                    new_h = min(h, int(w * 9/16))
                    x = (w - new_w) // 2
                    y = (h - new_h) // 2
                    frame = frame[y:y+new_h, x:x+new_w]
                elif crop_mode == '智能裁切':
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (5, 5), 0)
                    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        max_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(max_contour)
                        margin = 20
                        x = max(0, x - margin)
                        y = max(0, y - margin)
                        w = min(frame.shape[1] - x, w + 2*margin)
                        h = min(frame.shape[0] - y, h + 2*margin)
                        frame = frame[y:y+h, x:x+w]
                if frame_count % self.frame_step == 0:
                    frame_filename = f"frame_{saved_count:06d}.png"
                    frame_path = os.path.join(input_frames_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    saved_count += 1
                    progress = int((frame_count / total_frames) * 40) 
                    self.progress_signal.emit(progress, 100, f"提取幀：{frame_count}/{total_frames}")
                frame_count += 1
                if frame_count % 30 == 0 and self.stop_flag.is_set():
                    break
            self.safe_release_capture()
            if self.stop_flag.is_set():
                self.safe_remove_temp_dir()
                return
            input_frames = sorted([os.path.join(input_frames_dir, f) for f in os.listdir(input_frames_dir) if f.endswith(".png")])
            total_input_frames = len(input_frames)
            if total_input_frames == 0:
                raise Exception("未能提取任何有效的視頻幀")
            self.progress_signal.emit(40, 100, f"開始處理 {total_input_frames} 個幀")
            for i, frame_path in enumerate(input_frames):
                if self.stop_flag.is_set():
                    break
                output_filename = os.path.basename(frame_path)
                output_frame_path = os.path.join(output_frames_dir, output_filename)
                try:
                    original_image = Image.open(frame_path).convert("RGB")
                    target_width, target_height = orig_width, orig_height
                    if resolution == '超分倍率':
                        target_width = width
                        target_height = height
                    elif resolution == '自訂':
                        target_width = width
                        target_height = height
                    else:
                        target_width = orig_width
                        target_height = orig_height
                    enhanced_image = self.process_single_frame(
                        original_image,
                        target_width=target_width,
                        target_height=target_height,
                        scale_factor=scale_factor if resolution == '超分倍率' else 1.0
                    )
                    if self.strength < 1.0:
                        original_array = np.array(original_image)
                        enhanced_array = np.array(enhanced_image)
                        blended_array = cv2.addWeighted(
                            original_array, 1.0 - self.strength,
                            enhanced_array, self.strength,
                            0
                        )
                        enhanced_image = Image.fromarray(blended_array)
                    if (resolution == '原始大小' and (enhanced_image.width != orig_width or enhanced_image.height != orig_height)):
                        enhanced_image = enhanced_image.resize((orig_width, orig_height), Image.Resampling.LANCZOS)
                    elif (enhanced_image.width != width or enhanced_image.height != height):
                        enhanced_image = enhanced_image.resize((width, height), Image.Resampling.LANCZOS)
                    min_preview_interval = 0.2 if self.high_frequency_preview else 0.5
                    actual_interval = min(self.preview_interval, min_preview_interval)
                    preview_frames = max(1, int(fps * actual_interval / self.frame_step))
                    max_frames_interval = 20
                    preview_frames = min(preview_frames, max_frames_interval)
                    if i % preview_frames == 0 or i == 0 or i == total_input_frames - 1:
                        self.current_preview_frame = enhanced_image.copy()
                        current_frame_index = i * self.frame_step
                        self.preview_signal.emit(self.current_preview_frame, current_frame_index)
                        logger.debug(f"發送預覽信號，幀索引: {current_frame_index}")
                    enhanced_image.save(output_frame_path)
                    enhanced_image = None
                except Exception as e:
                    logger.error(f"處理幀 {frame_path} 時出錯: {str(e)}")
                    try:
                        original_image.save(output_frame_path)
                        logger.info(f"使用原始幀代替處理失敗的幀: {frame_path}")
                    except Exception as backup_err:
                        logger.error(f"保存原始幀作為備用時也出錯: {str(backup_err)}")
                    continue
                progress = 40 + int((i / total_input_frames) * 50) 
                self.progress_signal.emit(progress, 100, f"處理幀：{i+1}/{total_input_frames}")
                if i % 10 == 0 and self.stop_flag.is_set():
                    break
            if self.current_preview_frame:
                del self.current_preview_frame
                self.current_preview_frame = None
            if self.stop_flag.is_set():
                self.safe_remove_temp_dir()
                return
            self.progress_signal.emit(90, 100, "正在合成影片")
            output_frames = sorted([os.path.join(output_frames_dir, f) for f in os.listdir(output_frames_dir) if f.endswith(".png")])
            if not output_frames:
                raise Exception("沒有生成任何輸出幀，無法合成影片")
            video_created = self.create_video_with_ffmpeg(output_frames, fps, width, height)
            if not video_created or self.stop_flag.is_set():
                self.safe_remove_temp_dir()
                return
            audio_mode = self.video_options.get('audio_mode', 'keep')
            if self.keep_audio and not self.stop_flag.is_set() and audio_mode != 'none':
                success = self.process_audio()
                if not success:
                    logger.warning("音軌處理失敗，輸出將不包含音軌")
            if self.video_options.get('clean_temp_files', True):
                self.safe_remove_temp_dir()
            else:
                logger.info(f"保留臨時目錄: {self.temp_dir}")
            elapsed_time = time.time() - start_time
            logger.info(f"影片處理完成，耗時 {elapsed_time:.2f} 秒")
            self.finished_signal.emit(self.output_path, elapsed_time)
        except Exception as e:
            logger.error(f"影片處理過程中發生錯誤: {str(e)}")
            self.progress_signal.emit(0, 100, f"處理失敗：{str(e)}")
            self.safe_release_capture()
            self.safe_release_writer()
            self.safe_remove_temp_dir()

    def process_single_frame(self, image, target_width=None, target_height=None, scale_factor=1.0):
        enhancer = process_image_in_patches(
            self.model,
            image,
            self.device,
            block_size=self.block_size,
            overlap=self.overlap,
            use_weight_mask=self.use_weight_mask,
            blending_mode=self.blending_mode,
            use_amp=self.use_amp
        )
        enhanced_image = enhancer.process()
        if target_width and target_height:
            if enhanced_image.width != target_width or enhanced_image.height != target_height:
                enhanced_image = enhanced_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        elif scale_factor and scale_factor != 1.0:
            new_width = int(enhanced_image.width * scale_factor)
            new_height = int(enhanced_image.height * scale_factor)
            enhanced_image = enhanced_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return enhanced_image

    def get_encoder_settings(self):
        codec_type = self.video_options.get('codec_type', 'H.264')
        encoder = self.video_options.get('encoder', 'x264 (CPU)')
        video_codec = 'libx264'
        extra_params = []
        if codec_type == 'H.264':
            if 'NVENC' in encoder:
                video_codec = 'h264_nvenc'
                extra_params.extend(['-preset', 'p7'])
            elif 'QuickSync' in encoder:
                video_codec = 'h264_qsv'
            elif 'AMF' in encoder:
                video_codec = 'h264_amf'
            elif '10-bit' in encoder:
                video_codec = 'libx264'
                extra_params.extend(['-pix_fmt', 'yuv420p10le', '-preset', 'medium'])
            else:
                video_codec = 'libx264'
                extra_params.extend(['-pix_fmt', 'yuv420p', '-preset', 'medium'])
        elif codec_type == 'H.265/HEVC':
            if 'NVENC' in encoder:
                video_codec = 'hevc_nvenc'
                extra_params.extend(['-preset', 'p7'])
            elif 'QuickSync' in encoder:
                video_codec = 'hevc_qsv'
            elif 'AMF' in encoder:
                video_codec = 'hevc_amf'
            elif '10-bit' in encoder:
                video_codec = 'libx265'
                extra_params.extend(['-pix_fmt', 'yuv420p10le', '-preset', 'medium', '-x265-params', 'log-level=error'])
            else:
                video_codec = 'libx265'
                extra_params.extend(['-pix_fmt', 'yuv420p', '-preset', 'medium', '-x265-params', 'log-level=error'])
        elif codec_type == 'VP9':
            video_codec = 'libvpx-vp9'
            if '10-bit' in encoder:
                extra_params.extend(['-pix_fmt', 'yuv420p10le', '-deadline', 'good', '-cpu-used', '2'])
            else:
                extra_params.extend(['-pix_fmt', 'yuv420p', '-deadline', 'good', '-cpu-used', '2'])
        elif codec_type == 'AV1':
            if 'NVENC' in encoder:
                video_codec = 'av1_nvenc'
            elif 'SVT' in encoder:
                video_codec = 'libsvtav1'
                extra_params.extend(['-preset', '7'])
            else:
                video_codec = 'libaom-av1'
                extra_params.extend(['-cpu-used', '4', '-row-mt', '1'])
        bitrate_mode = self.video_options.get('bitrate_mode', 'abr')
        if bitrate_mode == 'crf':
            crf = self.video_options.get('crf', 23)
            if video_codec in ['libx264', 'h264_nvenc', 'h264_qsv', 'h264_amf']:
                extra_params.extend(['-crf', str(crf)])
            elif video_codec in ['libx265', 'hevc_nvenc', 'hevc_qsv', 'hevc_amf']:
                extra_params.extend(['-crf', str(crf)])
            elif 'vpx' in video_codec:
                extra_params.extend(['-crf', str(crf), '-b:v', '0'])
            elif 'aom' in video_codec or 'svt' in video_codec:
                extra_params.extend(['-crf', str(crf)])
        else: 
            bitrate = self.video_options.get('bitrate', 8000000) 
            extra_params.extend(['-b:v', f'{bitrate}'])
        return video_codec, extra_params

    def create_video_with_ffmpeg(self, output_frames, fps, width, height):
        try:
            output_dir = os.path.dirname(self.output_path)
            os.makedirs(output_dir, exist_ok=True)
            temp_output_path = self.output_path
            if os.path.exists(self.output_path):
                try:
                    with open(self.output_path, 'w+b') as f:
                        pass
                except PermissionError:
                    base_name, ext = os.path.splitext(self.output_path)
                    temp_output_path = f"{base_name}_{int(time.time())}{ext}"
                    logger.warning(f"原輸出文件被占用，使用臨時名稱: {temp_output_path}")
            frames_list_txt = os.path.join(self.temp_dir, "frames_list.txt")
            with open(frames_list_txt, 'w') as f:
                for frame_path in output_frames:
                    safe_path = frame_path.replace('\\', '/')
                    f.write(f"file '{safe_path}'\n")
                    f.write(f"duration {1.0/fps}\n")
                if output_frames:
                    safe_path = output_frames[-1].replace('\\', '/')
                    f.write(f"file '{safe_path}'\n")
            video_codec, extra_params = self.get_encoder_settings()
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', frames_list_txt,
                '-c:v', video_codec,
                '-r', str(fps),
            ]
            cmd.extend(extra_params)
            cmd.extend(['-s', f'{width}x{height}'])
            cmd.append(temp_output_path)
            cmd_str = ' '.join(cmd)
            logger.info(f"執行視頻合成命令: {cmd_str}")
            self.ffmpeg_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                errors='replace'
            )
            total_frames = len(output_frames)
            while True:
                if self.ffmpeg_process.poll() is not None:
                    break
                line = self.ffmpeg_process.stderr.readline().strip()
                if not line:
                    continue
                frame_match = re.search(r'frame=\s*(\d+)', line)
                if frame_match:
                    current_frame = int(frame_match.group(1))
                    progress = 90 + min(9, int((current_frame / total_frames) * 9))
                    self.progress_signal.emit(progress, 100, f"合成影片：{current_frame}/{total_frames}")
                if self.stop_flag.is_set():
                    self.ffmpeg_process.terminate()
                    logger.info("用戶請求停止，已終止視頻合成")
                    return False
                time.sleep(0.1)
            _, stderr = self.ffmpeg_process.communicate()
            if self.ffmpeg_process.returncode != 0:
                logger.error(f"FFmpeg 視頻合成失敗: {stderr}")
                return False
            if temp_output_path != self.output_path:
                try:
                    if os.path.exists(self.output_path):
                        os.remove(self.output_path)
                    os.rename(temp_output_path, self.output_path)
                except Exception as e:
                    logger.error(f"無法重命名輸出文件: {str(e)}")
                    self.output_path = temp_output_path
            self.progress_signal.emit(99, 100, "視頻合成完成")
            return True
        except Exception as e:
            logger.error(f"創建視頻時出錯: {str(e)}")
            return False

    def process_audio(self):
        try:
            self.progress_signal.emit(99, 100, "處理音軌中...")
            temp_video = os.path.join(self.temp_dir, "temp_video_with_audio.mp4")
            try:
                shutil.copy2(self.output_path, temp_video)
            except Exception as e:
                logger.error(f"複製臨時視頻失敗: {str(e)}")
                return False
            audio_mode = self.video_options.get('audio_mode', 'keep')
            cmd = ['ffmpeg', '-y']
            cmd.extend(['-i', temp_video])
            cmd.extend(['-i', self.input_path])
            cmd.extend(['-c:v', 'copy'])
            if audio_mode == 'keep':
                cmd.extend(['-c:a', 'copy'])
            elif audio_mode == 'reencode':
                audio_codec = self.video_options.get('audio_codec', 'AAC').lower()
                if audio_codec == 'aac':
                    cmd.extend(['-c:a', 'aac', '-strict', 'experimental'])
                elif audio_codec == 'opus':
                    cmd.extend(['-c:a', 'libopus'])
                elif audio_codec == 'vorbis':
                    cmd.extend(['-c:a', 'libvorbis'])
                elif audio_codec == 'mp3':
                    cmd.extend(['-c:a', 'libmp3lame'])
                audio_bitrate = self.video_options.get('audio_bitrate', '192k')
                cmd.extend(['-b:a', audio_bitrate])
                cmd.extend(['-ar', '48000'])
            cmd.extend([
                '-map', '0:v:0', 
                '-map', '1:a:0'  
            ])
            output_format = self.video_options.get('format', 'mp4')
            if output_format:
                cmd.extend(['-f', output_format])
            cmd.append(self.output_path)
            cmd_str = ' '.join(cmd)
            logger.info(f"執行音軌處理命令: {cmd_str}")
            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors='replace'
            )
            stdout, stderr = self.ffmpeg_process.communicate()
            if self.ffmpeg_process.returncode != 0:
                logger.error(f"FFmpeg 音軌處理失敗: {stderr}")
                shutil.copy2(temp_video, self.output_path)
                return False
            self.progress_signal.emit(100, 100, "音軌處理完成")
            return True 
        except Exception as e:
            logger.error(f"處理音軌時出錯: {str(e)}")
            return False