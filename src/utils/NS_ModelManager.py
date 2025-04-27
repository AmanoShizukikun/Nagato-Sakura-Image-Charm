import os
import json
import torch
import logging
import threading
import requests
from datetime import datetime
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QMessageBox

from src.models.NS_ImageEnhancer import ImageQualityEnhancer
from src.utils.NS_DownloadManager import DownloadManager
from src.utils.NS_ExtractUtility import ExtractUtility 


logger = logging.getLogger(__name__)

class ModelManager(QObject):
    """模型管理器，負責模型的加載、下載、更新和管理"""
    update_available_signal = pyqtSignal(bool, str)
    update_progress_signal = pyqtSignal(str)
    update_finished_signal = pyqtSignal(bool, str)
    model_loaded_signal = pyqtSignal(str)
    model_imported_signal = pyqtSignal(str)
    model_deleted_signal = pyqtSignal(str)
    model_downloaded_signal = pyqtSignal(str)
    download_progress_signal = pyqtSignal(int, int, float)
    download_finished_signal = pyqtSignal(bool, str)
    download_retry_signal = pyqtSignal(str)
    
    def __init__(self):
        """初始化模型管理器"""
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.config_dir = os.path.join(self.base_dir, "config")
        self.models_dir = os.path.join(self.base_dir, "models")
        self.models_file = os.path.join(self.config_dir, "models_data.json")
        self.usage_stats_file = os.path.join(self.config_dir, "model_usage_stats.json")
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        self.models_data = {}
        self.usage_stats = self._load_usage_stats()
        self.load_models_data()
        self.available_models = []
        self.model_info = {}
        self.model_statuses = {}
        self.current_model = None
        self.current_model_path = None
        self.registered_model_path = None
        self.model_cache = {}
        self.max_cache_size = 2
        self.downloader = None
        self.download_thread = None
        self.current_download_info = None
        self.observers = []
        self.scan_models_directory()
            
    def add_observer(self, observer):
        """添加觀察者，用於接收模型變更通知"""
        if observer not in self.observers:
            self.observers.append(observer)
            logger.debug(f"已添加觀察者: {observer}")
    
    def remove_observer(self, observer):
        """移除觀察者"""
        if observer in self.observers:
            self.observers.remove(observer)
            logger.debug(f"已移除觀察者: {observer}")
    
    def notify_observers(self, method_name, *args, **kwargs):
        """通知所有觀察者"""
        for observer in list(self.observers): 
            if hasattr(observer, method_name):
                try:
                    getattr(observer, method_name)(*args, **kwargs)
                except Exception as e:
                    logger.error(f"通知觀察者 {observer} 時發生錯誤: {str(e)}")
    
    # ========== 基本信息獲取 ==========
    
    def get_device(self):
        """獲取當前使用的設備 (GPU/CPU)"""
        return self.device
        
    # ========== 模型資訊管理 ==========
    
    def load_models_data(self):
        """從本地 JSON 檔案載入模型數據"""
        try:
            if not os.path.exists(self.models_file):
                self._create_default_models_file()
                return True
            with open(self.models_file, 'r', encoding='utf-8') as f:
                self.models_data = json.load(f)
            logger.info(f"已載入 {len(self.models_data.get('models', {}))} 個模型資料")
            return True
        except Exception as e:
            logger.error(f"載入模型數據失敗: {str(e)}")
            self._create_default_models_file()
            return False
    
    def _create_default_models_file(self):
        """創建默認的模型數據文件"""
        default_data = {
            "version": "1.1.0",
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "models": {
                "NS-IC-Kyouka": {
                    "name": "Kyouka -《鏡花・碎象還映》",
                    "url": "https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/releases/download/model-space/NS-IC-Kyouka-v6-314.pth",
                    "description": "碎裂之象，在鏡中重構原初之姿。鏡中之花非虛幻，碎裂之象皆可還映。",
                    "details": "泛用動畫 JPEG 壓縮錯色與鋸齒修復還原模型。融合細節還原與邊緣柔化，用以還映碎裂之象。\n\n- 版本：v6.31\n- 訓練樣本數： 10K\n- 檔案大小：約 73.5 MB\n- 適用場景：動畫圖、二次元插圖的壓縮錯色與鋸齒修復",
                    "preview": "assets/model_previews/Kyouka.png",
                    "category": "動畫",
                    "added_date": "2025-04-16",
                    "author": "天野靜樹"
                },
                "NS-IC-Kyouka-LQ": {
                    "name": "Kyouka-LQ -《鏡花・幽映深層》",
                    "url": "https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/releases/download/model-space/NS-IC-Kyouka-LQ-v6-310.pth",
                    "description": "深層碎象，綻映破碎中的微光輪廓。在低質的殘影中，亦能尋回幻境之原貌。",
                    "details": "低畫質特化動畫圖像還原。針對重度 JPEG 壓縮錯色、鋸齒塊狀與細節遺失進行修復與視覺補全。\n\n- 版本：v6.31\n- 訓練樣本數： 10K\n- 檔案大小：約 73.5 MB\n- 適用場景：低畫質動畫修復、低畫質二次元插圖、低分辨素材強化",
                    "preview": "assets/model_previews/Kyouka-LQ.png",
                    "category": "動畫",
                    "added_date": "2025-04-15",
                    "author": "天野靜樹"
                },
                "NS-IC-Kyouka-MQ": {
                    "name": "Kyouka-MQ -《鏡花・霞緲輪影》",
                    "url": "https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/releases/download/model-space/NS-IC-Kyouka-MQ-v7-349.pth",
                    "description": "層層視象，於半碎鏡花中映現原生構圖。非極損畫面，亦可升華細節與色彩。",
                    "details": "適用於中等畫質動畫圖像的細節提升與重構。在保留原圖氛圍的基礎，加強清晰度。\n\n- 版本：v7.34\n- 訓練樣本數： 10K\n- 檔案大小：約 73.5 MB\n- 適用場景：一般畫質動漫圖像、漫畫掃圖強化、插圖分享畫質提升",
                    "preview": "assets/model_previews/Kyouka-MQ.png",
                    "category": "動畫",
                    "added_date": "2025-04-21",
                    "author": "天野靜樹"
                },
                "NS-IC-Kyouka-MQ": {
                    "name": "Ritsuka-HQ -《斷律・映格輪廻》",
                    "url": "https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/releases/download/model-space/NS-IC-Ritsuka-HQ-v7-314.pth",
                    "description": "格狀的影塊，是序斷的殘響。在光與影的律動中，復甦真實之形。",
                    "details": "處理最高4格等寬動畫圖像的 pixel block 所造成的格子馬賽克效果，重構色塊邊界與失落邊緣資訊。\n\n- 版本：v7.31\n- 訓練樣本數： 10K\n- 檔案大小：約 73.5 MB\n- 適用場景：動漫馬賽克還原",
                    "preview": "assets/model_previews/Ritsuka-HQ.png",
                    "category": "馬賽克",
                    "added_date": "2025-04-27",
                    "author": "天野靜樹"
                }
            },
            "categories": ["動畫","寫實","馬賽克"],
            "remote_update_url": "https://raw.githubusercontent.com/AmanoShizukikun/Nagato-Sakura-Image-Charm/main/config/models_data.json"
        }
          
        os.makedirs(os.path.dirname(self.models_file), exist_ok=True)
        with open(self.models_file, 'w', encoding='utf-8') as f:
            json.dump(default_data, f, ensure_ascii=False, indent=2)
        self.models_data = default_data
        logger.info("已創建默認模型數據檔案")
    
    def get_models(self):
        """獲取所有模型數據"""
        return self.models_data.get('models', {})
    
    def get_categories(self):
        """獲取所有模型類別"""
        return self.models_data.get('categories', [])
    
    def get_version(self):
        """獲取模型數據版本"""
        return self.models_data.get('version', '1.0.0')
    
    def get_last_updated(self):
        """獲取模型數據最後更新時間"""
        return self.models_data.get('last_updated', datetime.now().strftime("%Y-%m-%d"))
    
    # ========== 模型更新 ==========
    
    def check_for_updates(self):
        """檢查是否有模型數據更新"""
        try:
            self.update_progress_signal.emit("正在檢查更新...")
            remote_url = self.models_data.get('remote_update_url', '')
            if not remote_url:
                self.update_available_signal.emit(False, "未設定更新連結")
                return False 
            response = requests.get(remote_url, timeout=10)
            if response.status_code == 200:
                remote_data = response.json()
                remote_version = remote_data.get('version', '0.0.0')
                local_version = self.get_version()
                if self._compare_versions(remote_version, local_version) > 0:
                    self.update_available_signal.emit(True, f"有可用更新: {local_version} → {remote_version}")
                    return True
                else:
                    self.update_available_signal.emit(False, f"已是最新版本: {local_version}")
                    return False
            else:
                self.update_available_signal.emit(False, f"檢查更新失敗: HTTP {response.status_code}")
                return False 
        except Exception as e:
            self.update_available_signal.emit(False, f"檢查更新失敗: {str(e)}")
            return False
    
    def _compare_versions(self, ver1, ver2):
        """比較兩個版本號，如果ver1 > ver2返回1，相等返回0，小於返回-1"""
        try:
            v1_parts = [int(x) for x in ver1.split('.')]
            v2_parts = [int(x) for x in ver2.split('.')]
            for i in range(max(len(v1_parts), len(v2_parts))):
                v1 = v1_parts[i] if i < len(v1_parts) else 0
                v2 = v2_parts[i] if i < len(v2_parts) else 0
                if v1 > v2:
                    return 1
                elif v1 < v2:
                    return -1
            return 0
        except (ValueError, TypeError, AttributeError):
            logger.error(f"比較版本號時出錯: {ver1} vs {ver2}")
            return 0
    
    def update_models_data(self):
        """更新模型數據"""
        try:
            self.update_progress_signal.emit("開始下載更新...")
            remote_url = self.models_data.get('remote_update_url', '')
            if not remote_url:
                self.update_finished_signal.emit(False, "未設定更新連結")
                return False
            response = requests.get(remote_url, timeout=10)
            if response.status_code == 200:
                remote_data = response.json()
                with open(self.models_file, 'w', encoding='utf-8') as f:
                    json.dump(remote_data, f, ensure_ascii=False, indent=2)
                self.models_data = remote_data
                self.update_finished_signal.emit(True, f"已更新模型數據至版本 {remote_data.get('version', '未知')}")
                return True
            else:
                self.update_finished_signal.emit(False, f"下載更新失敗: HTTP {response.status_code}")
                return False
        except Exception as e:
            self.update_finished_signal.emit(False, f"更新失敗: {str(e)}")
            return False
    
    # ========== 模型下載 ==========
    
    def download_model_from_url(self, url, save_path=None, num_threads=4, retry_count=3, auto_extract=True):
        """使用下載器下載模型"""
        if not url:
            logger.error("URL不能為空")
            self.download_finished_signal.emit(False, "URL不能為空")
            return False
        try:
            if not save_path:
                save_path = os.path.join(self.models_dir, os.path.basename(url))
            save_dir = os.path.dirname(os.path.abspath(save_path))
            os.makedirs(save_dir, exist_ok=True)
            self.downloader = DownloadManager()
            self.downloader.progress_signal.connect(self._forward_download_progress)
            self.downloader.finished_signal.connect(
                lambda success, msg: self._handle_download_finished(success, msg, save_path, auto_extract)
            )
            self.download_thread = threading.Thread(
                target=self.downloader.download_file,
                args=(url, save_path),
                kwargs={
                    'num_threads': num_threads,
                    'retry_count': retry_count
                }
            )
            self.download_thread.daemon = True
            self.download_thread.start()
            self.update_progress_signal.emit(f"開始下載: {os.path.basename(save_path)}")
            return True
        except Exception as e:
            logger.error(f"下載初始化失敗: {str(e)}")
            self.download_finished_signal.emit(False, f"下載初始化失敗: {str(e)}")
            return False
    
    def download_official_model(self, model_id, save_path=None, num_threads=4, retry_count=3, auto_extract=True):
        """下載官方模型庫中的指定模型，支援備用載點"""
        try:
            if model_id not in self.models_data.get('models', {}):
                message = f"找不到模型ID: {model_id}"
                logger.error(message)
                self.download_finished_signal.emit(False, message)
                return False
            model_info = self.models_data['models'][model_id]
            url = model_info.get('url')
            if not url:
                message = f"模型 {model_id} 沒有下載連結"
                logger.error(message)
                self.download_finished_signal.emit(False, message)
                return False
            if save_path is None:
                filename = os.path.basename(url)
                save_path = os.path.join(self.models_dir, filename)
            self.current_download_info = {
                'model_id': model_id,
                'save_path': save_path,
                'num_threads': num_threads,
                'retry_count': retry_count,
                'auto_extract': auto_extract,
                'using_backup': False
            }
            self.update_progress_signal.emit(f"開始下載模型: {model_info['name']}")
            return self.download_model_from_url(url, save_path, num_threads, retry_count, auto_extract)
        except Exception as e:
            logger.error(f"下載模型失敗: {str(e)}")
            self.download_finished_signal.emit(False, f"下載模型失敗: {str(e)}")
            return False
    
    def cancel_download(self):
        """中止當前模型下載並清理文件"""
        try:
            if self.downloader and self.downloader.is_downloading:
                success = self.downloader.cancel_download()
                if success:
                    self.update_progress_signal.emit("下載已取消並清理不完整文件")
                    return True
            return False
        except Exception as e:
            logger.error(f"取消下載時出錯: {str(e)}")
            return False
    
    def _forward_download_progress(self, current, total, speed):
        """轉發下載進度信號"""
        try:
            self.download_progress_signal.emit(current, total, speed)
        except Exception as e:
            logger.error(f"轉發下載進度信號時出錯: {str(e)}")
    
    def _handle_download_finished(self, success, message, file_path, auto_extract=True):
        """處理下載完成後的操作，包括可選的自動解壓功能和備用載點支援"""
        try:
            if success:
                self.update_progress_signal.emit("下載完成")
                if auto_extract and ExtractUtility.is_archive_file(file_path):
                    self.update_progress_signal.emit("正在解壓縮文件...")
                    extract_success, extract_msg, extracted_files = ExtractUtility.extract_file(
                        file_path, 
                        extract_dir=self.models_dir, 
                        delete_after=True
                    )
                    if extract_success:
                        self.update_progress_signal.emit("解壓縮完成")
                        model_files = ExtractUtility.get_extracted_files_of_type(
                            extracted_files, 
                            ['.pth', '.pt', '.ckpt', '.safetensors']
                        )
                        if model_files:
                            message += f" (已自動解壓 {len(model_files)} 個模型文件)"
                        else:
                            message += " (已自動解壓)"
                    else:
                        message += f" (解壓縮失敗: {extract_msg})"
                self.scan_models_directory()
                self.download_finished_signal.emit(True, message)
                self.model_downloaded_signal.emit(file_path)
            else:
                if "取消" in message and file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        self.update_progress_signal.emit("已清理取消下載的不完整文件")
                    except Exception as e:
                        logger.error(f"清理文件時出錯: {str(e)}")
                if hasattr(self, 'current_download_info') and self.current_download_info and not self.current_download_info.get('using_backup', True):
                    model_id = self.current_download_info.get('model_id')
                    if model_id in self.models_data.get('models', {}) and 'url2' in self.models_data['models'][model_id]:
                        backup_url = self.models_data['models'][model_id]['url2']
                        if backup_url:
                            self.update_progress_signal.emit("主載點下載失敗，嘗試使用備用載點...")
                            self.download_retry_signal.emit("正在嘗試備用載點...")
                            save_path = self.current_download_info['save_path']
                            num_threads = self.current_download_info['num_threads']
                            retry_count = self.current_download_info['retry_count']
                            auto_extract = self.current_download_info['auto_extract']
                            self.current_download_info['using_backup'] = True
                            self.download_model_from_url(backup_url, save_path, num_threads, retry_count, auto_extract)
                            return
                self.update_progress_signal.emit(f"下載失敗: {message}")
                self.download_finished_signal.emit(False, message)
        except Exception as e:
            logger.error(f"處理下載完成時出錯: {str(e)}")
            self.download_finished_signal.emit(False, f"處理下載完成時出錯: {str(e)}")
    
    # ========== 本地模型掃描與管理 ==========
    
    def scan_models_directory(self):
        """掃描models目錄下所有可用的模型，但不載入任何模型"""
        try:
            self.available_models = []
            self.model_info = {}
            if not os.path.exists(self.models_dir):
                os.makedirs(self.models_dir, exist_ok=True)
                logger.info(f"創建模型目錄: {self.models_dir}")
                return
            for root, _, files in os.walk(self.models_dir):
                for file in files:
                    if file.endswith(('.pth', '.pt', '.ckpt', '.safetensors')):
                        model_path = os.path.join(root, file)
                        rel_path = os.path.relpath(model_path, self.models_dir)
                        self.available_models.append(model_path)
                        self.model_statuses[model_path] = "available"
                        model_name = os.path.splitext(file)[0]
                        size_bytes = os.path.getsize(model_path)
                        size_mb = size_bytes / (1024 * 1024)
                        last_modified = os.path.getmtime(model_path)
                        last_modified_date = datetime.fromtimestamp(last_modified).strftime('%Y-%m-%d %H:%M:%S')
                        extra_info = self._get_model_extra_info(model_name)
                        self.model_info[model_path] = {
                            "name": model_name,
                            "path": model_path,
                            "rel_path": rel_path,
                            "size_bytes": size_bytes,
                            "size_mb": size_mb,
                            "last_modified": last_modified_date,
                            "last_used": self._get_last_used_time(model_path),
                            **extra_info
                        }
            logger.info(f"掃描到 {len(self.available_models)} 個模型")
            return self.available_models
        except Exception as e:
            logger.error(f"掃描模型目錄時出錯: {str(e)}")
            return []
    
    def _get_model_extra_info(self, model_name):
        """從模型資料庫中獲取額外的模型資訊"""
        try:
            extra_info = {
                "description": "",
                "category": "未分類",
                "author": "未知",
                "details": ""
            }
            for model_id, model_data in self.models_data.get('models', {}).items():
                db_model_name = os.path.splitext(os.path.basename(model_data.get('url', '')))[0]
                if model_name.lower() == db_model_name.lower() or model_name.lower() == model_id.lower():
                    extra_info.update({
                        "description": model_data.get("description", ""),
                        "category": model_data.get("category", "未分類"),
                        "author": model_data.get("author", "未知"),
                        "details": model_data.get("details", ""),
                        "preview": model_data.get("preview", ""),
                        "model_id": model_id
                    })
                    break
            return extra_info
        except Exception as e:
            logger.error(f"獲取模型額外信息時出錯: {str(e)}")
            return {}
    
    def get_available_models(self):
        """返回可用模型列表"""
        if not self.available_models:
            self.scan_models_directory()
        return self.available_models
    
    def get_local_model_info(self):
        """返回本地模型詳細資訊"""
        if not self.model_info:
            self.scan_models_directory()
        return self.model_info
    
    def register_default_model(self):
        """註冊預設模型，但不實際載入"""
        try:
            models = self.get_available_models()
            if models:
                model_usage = [(path, self._get_last_used_time(path)) for path in models]
                model_usage.sort(key=lambda x: x[1], reverse=True)
                self.register_model(model_usage[0][0])
                return True
            return False
        except Exception as e:
            logger.error(f"註冊預設模型時出錯: {str(e)}")
            return False
    
    def register_model(self, model_path):
        """註冊模型，但不載入（僅記錄為將要使用的模型）"""
        try:
            if model_path not in self.available_models:
                if not os.path.exists(model_path):
                    logger.error(f"模型文件不存在: {model_path}")
                    return False
                self.available_models.append(model_path)
            self.registered_model_path = model_path
            for path in self.model_statuses:
                self.model_statuses[path] = "available"
            self.model_statuses[model_path] = "registered"
            return True
        except Exception as e:
            logger.error(f"註冊模型時出錯: {str(e)}")
            return False
    
    def prepare_model_for_inference(self):
        """根據已註冊的模型路徑載入模型到記憶體，用於推理前的準備"""
        if not self.registered_model_path:
            logger.warning("沒有註冊的模型，無法準備推理")
            return False
        model_path = self.registered_model_path
        try:
            if model_path in self.model_cache:
                self.current_model = self.model_cache[model_path]
                self.current_model_path = model_path
                self.model_statuses[model_path] = "active"
                logger.info(f"從快取中載入模型: {os.path.basename(model_path)}")
                self._record_model_usage(model_path)
                self.model_loaded_signal.emit(os.path.basename(model_path))
                return True
            logger.info(f"載入模型: {os.path.basename(model_path)}")
            self.update_progress_signal.emit(f"正在載入模型: {os.path.basename(model_path)}...")
            model = ImageQualityEnhancer()
            model_state = torch.load(model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(model_state)
            model = model.to(self.device)
            if len(self.model_cache) >= self.max_cache_size:
                least_used = min(self.model_cache.keys(), key=lambda x: self._get_last_used_time(x))
                del self.model_cache[least_used]
                logger.debug(f"從快取中移除模型: {os.path.basename(least_used)}")
            self.model_cache[model_path] = model
            self.current_model = model
            self.current_model_path = model_path
            self.model_statuses[model_path] = "active"
            self._record_model_usage(model_path)
            self.model_loaded_signal.emit(os.path.basename(model_path))
            self.update_progress_signal.emit(f"模型已載入: {os.path.basename(model_path)}")
            return True
        except Exception as e:
            logger.error(f"載入模型失敗: {str(e)}")
            self.update_progress_signal.emit(f"載入模型失敗: {str(e)}")
            return False
    
    def get_current_model(self):
        """獲取當前載入的模型實例，如果沒有則返回None"""
        return self.current_model
        
    def get_registered_model_path(self):
        """獲取已註冊模型的路徑，如果沒有則返回None"""
        return self.registered_model_path
    
    def get_model_status(self, model_path):
        """獲取模型狀態: available, registered, active 或 unknown"""
        return self.model_statuses.get(model_path, "unknown")
    
    def has_models(self):
        """檢查是否有可用模型"""
        return len(self.get_available_models()) > 0
    
    def clear_cache(self):
        """清空模型快取，釋放記憶體和顯存"""
        try:
            logger.info("開始清理模型快取和釋放顯存...")
            for model_path, model in self.model_cache.items():
                try:
                    logger.debug(f"將模型從顯存移至 CPU: {os.path.basename(model_path)}")
                    model.cpu()
                except Exception as e:
                    logger.error(f"將模型移至 CPU 時出錯: {str(e)}")
            self.model_cache.clear()
            self.current_model = None
            self.current_model_path = None
            for path in self.model_statuses:
                if self.model_statuses[path] == "active":
                    self.model_statuses[path] = "registered" if path == self.registered_model_path else "available"
            import gc
            gc.collect()
            if torch.cuda.is_available():
                try:
                    before_free = torch.cuda.memory_reserved() / 1024**2
                    logger.info(f"CUDA 緩存清理前顯存保留: {before_free:.2f} MB")
                except Exception as e:
                    logger.warning(f"無法獲取釋放前顯存資訊: {str(e)}")
                torch.cuda.empty_cache()
                try:
                    after_free = torch.cuda.memory_reserved() / 1024**2
                    logger.info(f"CUDA 緩存清理後顯存保留: {after_free:.2f} MB")
                    logger.info(f"釋放了約 {before_free - after_free:.2f} MB 顯存")
                except Exception as e:
                    logger.warning(f"無法獲取釋放後顯存資訊: {str(e)}")
            gc.collect()
            logger.info("模型快取清理完成")
            return True
        except Exception as e:
            logger.error(f"清空模型快取時出錯: {str(e)}")
            return False
    
    # ========== 模型匯入與刪除 ==========
    
    def import_external_model(self, source_path):
        """匯入外部模型到models目錄，但不載入模型"""
        try:
            if not os.path.exists(source_path):
                logger.error(f"原始檔案不存在: {source_path}")
                return False, "原始檔案不存在"
            filename = os.path.basename(source_path)
            target_path = os.path.join(self.models_dir, filename)
            if os.path.exists(target_path):
                logger.warning(f"目標路徑已存在同名檔案: {target_path}")
                return False, "目標路徑已存在同名檔案"
            import shutil
            shutil.copy2(source_path, target_path)
            self.scan_models_directory()
            self.model_imported_signal.emit(target_path)
            logger.info(f"成功匯入模型: {filename}")
            return True, target_path 
        except Exception as e:
            logger.error(f"匯入模型失敗: {str(e)}")
            return False, f"匯入模型失敗: {str(e)}"
        
    def delete_model(self, model_path):
        """從模型目錄中刪除模型"""
        try:
            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}")
                return False, "模型文件不存在"
            if model_path == self.current_model_path:
                logger.warning("無法刪除當前正在使用的模型")
                return False, "無法刪除當前正在使用的模型"
            if model_path in self.model_cache:
                del self.model_cache[model_path]
            os.remove(model_path)
            if model_path in self.available_models:
                self.available_models.remove(model_path)
            if model_path in self.model_statuses:
                del self.model_statuses[model_path]
            if model_path in self.model_info:
                del self.model_info[model_path]
            if model_path == self.registered_model_path:
                self.registered_model_path = None
            self.model_deleted_signal.emit(model_path)
            logger.info(f"成功刪除模型: {os.path.basename(model_path)}")
            return True, f"成功刪除模型: {os.path.basename(model_path)}"
        except Exception as e:
            logger.error(f"刪除模型失敗: {str(e)}")
            return False, f"刪除模型失敗: {str(e)}"
    
    # ========== 模型使用統計功能 ==========
    
    def _load_usage_stats(self):
        """載入模型使用統計資料"""
        try:
            if os.path.exists(self.usage_stats_file):
                with open(self.usage_stats_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"載入模型使用統計資料失敗: {str(e)}")
            return {}
    
    def _save_usage_stats(self):
        """保存模型使用統計資料"""
        try:
            with open(self.usage_stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.usage_stats, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"保存模型使用統計資料失敗: {str(e)}")
            return False
    
    def _record_model_usage(self, model_path):
        """記錄模型使用情況"""
        try:
            model_key = os.path.basename(model_path)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if model_key not in self.usage_stats:
                self.usage_stats[model_key] = {
                    "use_count": 1,
                    "first_used": now,
                    "last_used": now
                }
            else:
                self.usage_stats[model_key]["use_count"] += 1
                self.usage_stats[model_key]["last_used"] = now
            self._save_usage_stats()
            return True
        except Exception as e:
            logger.error(f"記錄模型使用情況失敗: {str(e)}")
            return False
    
    def _get_last_used_time(self, model_path):
        """獲取模型最後使用時間，如果沒有記錄則返回0"""
        try:
            model_key = os.path.basename(model_path)
            if model_key in self.usage_stats:
                return self.usage_stats[model_key].get("last_used", "1970-01-01 00:00:00")
            return "1970-01-01 00:00:00"
        except Exception as e:
            logger.error(f"獲取模型最後使用時間失敗: {str(e)}")
            return "1970-01-01 00:00:00"
    
    def recommend_models(self, count=3):
        """根據使用情況推薦模型"""
        try:
            models = self.get_available_models()
            if not models:
                return []
            model_usage = [(path, self.usage_stats.get(os.path.basename(path), {}).get("use_count", 0)) for path in models]
            model_usage.sort(key=lambda x: x[1], reverse=True)
            return [path for path, _ in model_usage[:count]]
        except Exception as e:
            logger.error(f"推薦模型失敗: {str(e)}")
            return []
    
    def group_models_by_category(self):
        """根據類別分組模型"""
        try:
            result = {}
            for model_path, info in self.model_info.items():
                category = info.get("category", "未分類")
                if category not in result:
                    result[category] = []
                result[category].append(model_path)
            return result
        except Exception as e:
            logger.error(f"按類別分組模型失敗: {str(e)}")
            return {"未分類": list(self.model_info.keys())}
       
    def unregister_model(self):
        """註銷當前註冊的模型"""
        try:
            if self.registered_model_path:
                path = self.registered_model_path
                self.model_statuses[path] = "available"
                self.registered_model_path = None
                logger.info("已註銷當前模型")
                return True
            return False
        except Exception as e:
            logger.error(f"註銷模型時出錯: {str(e)}")
            return False