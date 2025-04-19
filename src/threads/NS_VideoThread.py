import os
import cv2
import tempfile
import shutil
import time
import threading
import gc
import re
import subprocess
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
                sync_preview=True, video_options=None):
        """初始化影片增強執行緒
        Args:
            model: 增強模型
            input_path: 輸入影片路徑
            output_path: 輸出影片路徑
            device: 處理裝置 (cpu/cuda)
            block_size: 分塊大小
            overlap: 重疊區域大小
            use_weight_mask: 是否使用權重遮罩
            blending_mode: 混合模式
            frame_step: 處理每幾幀 (跳幀處理)
            preview_interval: 預覽間隔 (秒)，預設為1秒
            keep_audio: 是否保留音軌
            sync_preview: 是否同步預覽
            video_options: 影片輸出選項字典
        """
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
        self.stop_flag = threading.Event()
        self.cap = None
        self.out = None
        self.temp_dir = None
        self.current_preview_frame = None
        self.ffmpeg_process = None
        self.high_frequency_preview = self.video_options.get('high_freq_preview', True)

    def stop(self):
        """停止處理並釋放資源"""
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
        """安全釋放視頻捕獲資源"""
        if hasattr(self, 'cap') and self.cap is not None:
            try:
                self.cap.release()
                logger.debug("已釋放視頻捕獲資源")
            except Exception as e:
                logger.error(f"釋放視頻捕獲資源時出錯: {str(e)}")
            self.cap = None
    
    def safe_release_writer(self):
        """安全釋放視頻寫入資源"""
        if hasattr(self, 'out') and self.out is not None:
            try:
                self.out.release()
                logger.debug("已釋放視頻寫入資源")
            except Exception as e:
                logger.error(f"釋放視頻寫入資源時出錯: {str(e)}")
            self.out = None
    
    def safe_remove_temp_dir(self):
        """安全刪除臨時目錄"""
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
        """檢查系統是否安裝了FFmpeg"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True)
            return result.returncode == 0
        except Exception:
            return False
    
    def run(self):
        """執行影片處理主程序"""
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
            resolution = self.video_options.get('resolution', '原始分辨率')
            if resolution == '720p':
                height = 720
                width = int(orig_width * (720 / orig_height))
            elif resolution == '1080p':
                height = 1080
                width = int(orig_width * (1080 / orig_height))
            elif resolution == '1440p':
                height = 1440
                width = int(orig_width * (1440 / orig_height))
            elif resolution == '4K':
                height = 2160
                width = int(orig_width * (2160 / orig_height))
            elif resolution == '自訂':
                custom_width = self.video_options.get('custom_width')
                custom_height = self.video_options.get('custom_height')
                if custom_width and custom_height:
                    try:
                        width = int(custom_width)
                        height = int(custom_height)
                        logger.info(f"使用自訂分辨率: {width}x{height}")
                    except ValueError:
                        logger.warning("無法解析自訂分辨率，使用原始分辨率")
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
                    with Image.open(frame_path).convert("RGB") as image:
                        enhanced_image = self.process_single_frame(image)
                        if width != orig_width or height != orig_height:
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
            self.safe_remove_temp_dir()
            elapsed_time = time.time() - start_time
            logger.info(f"影片處理完成，耗時 {elapsed_time:.2f} 秒")
            self.finished_signal.emit(self.output_path, elapsed_time)
        except Exception as e:
            logger.error(f"影片處理過程中發生錯誤: {str(e)}")
            self.progress_signal.emit(0, 100, f"處理失敗：{str(e)}")
            self.safe_release_capture()
            self.safe_release_writer()
            self.safe_remove_temp_dir()
    
    def process_single_frame(self, image):
        """處理單個圖像幀
        Args:
            image: PIL.Image對象，輸入圖像
            
        Returns:
            PIL.Image對象，處理後的圖像
        """
        enhancer = process_image_in_patches(
            self.model,
            image,
            self.device,
            block_size=self.block_size,
            overlap=self.overlap,
            use_weight_mask=self.use_weight_mask,
            blending_mode=self.blending_mode
        )
        enhanced_image = enhancer.process()
        return enhanced_image
        
    def get_encoder_settings(self):
        """根據選擇的編碼器返回FFmpeg參數
        Returns:
            元組 (編碼器名稱, 額外參數列表)
        """
        codec_type = self.video_options.get('codec_type', 'H.264')
        encoder = self.video_options.get('encoder', 'x264 (CPU)')
        video_codec = 'libx264'
        extra_params = []
        
        # H.264 編碼器
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
        
        # H.265/HEVC 編碼器
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
        
        # VP9 編碼器
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
        """使用FFmpeg從圖像序列創建視頻
        Args:
            output_frames: 圖像幀路徑列表
            fps: 影片幀率
            width: 影片寬度
            height: 影片高度
            
        Returns:
            bool: 是否成功創建視頻
        """
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
        """處理音軌
        Returns:
            bool: 是否成功處理音軌
        """
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