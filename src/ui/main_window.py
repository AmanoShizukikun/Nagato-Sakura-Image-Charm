import sys
import os
import time
import logging
from pathlib import Path
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QTabWidget, QStatusBar, 
                            QMenu, QMenuBar, QDialog, QMessageBox, QFileDialog, QInputDialog)
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtCore import Qt, QUrl, QTimer
from PyQt6.QtGui import QDesktopServices

from src.ui.image_tab import ImageProcessingTab
from src.ui.video_tab import VideoProcessingTab
from src.utils.NS_ModelManager import ModelManager
from src.ui.dialogs import DownloadModelDialog
from src.ui.training_tab import TrainingTab
from src.ui.assessment_tab import AssessmentTab

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImageEnhancerApp(QMainWindow):
    version = "1.0.0"
    def __init__(self):
        super().__init__()
        current_directory = Path(__file__).resolve().parent.parent.parent
        self.setWindowTitle("Nagato-Sakura-Image-Charm")
        self.setWindowIcon(QIcon(str(current_directory / "assets" / "icon" / f"{self.version}.ico")))
        self.setGeometry(100, 100, 1200, 800)
        self.model = None
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.init_model_manager()
        self.ffmpeg_available = self.check_ffmpeg()
        if not self.ffmpeg_available:
            self.statusBar.showMessage("警告：未偵測到ffmpeg，影片處理將不保留音軌")
        self.create_menu_bar()
        self.init_ui()
        QTimer.singleShot(100, self.check_models_after_ui_shown)
    
    def check_models_after_ui_shown(self):
        """在UI顯示後檢查模型並註冊預設模型，但不立即載入"""
        if not self.model_manager.has_models():
            self.no_models_found()
        else:
            success = self.model_manager.register_default_model()
            if success:
                model_path = self.model_manager.get_registered_model_path()
                model_name = os.path.basename(model_path) if model_path else "未知模型"
                self.statusBar.showMessage(f"使用設備: {self.device} | 已選擇模型: {model_name}")
                logger.info(f"已選擇模型: {model_name}")
            else:
                self.statusBar.showMessage(f"警告：模型註冊失敗，請從模型選單中選擇模型")
        self.reload_all_tabs_models()
    
    def init_model_manager(self):
        """初始化模型管理器並掃描可用模型"""
        self.model_manager = ModelManager()
        self.model_manager.model_loaded_signal.connect(self.on_model_loaded)
        self.model_manager.model_imported_signal.connect(self.on_model_imported)
        self.model_manager.model_deleted_signal.connect(self.on_model_deleted)
        self.model_manager.model_downloaded_signal.connect(self.on_model_downloaded)
        self.model_manager.update_available_signal.connect(self.on_update_available)
        self.model_manager.update_progress_signal.connect(self.on_update_progress)
        self.model_manager.update_finished_signal.connect(self.on_update_finished)
        self.model_manager.download_progress_signal.connect(self.on_download_progress)
        self.model_manager.download_finished_signal.connect(self.on_download_finished)
        self.device = self.model_manager.get_device()
        self.statusBar.showMessage(f"使用設備: {self.device}")
    
    def on_update_available(self, available, message):
        """處理模型更新檢查結果"""
        if available:
            reply = QMessageBox.question(
                self, "有可用更新", 
                f"{message}\n是否現在更新？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.model_manager.update_models_data()
        else:
            self.statusBar.showMessage(message, 5000)
    
    def on_update_progress(self, message):
        """更新進度信息"""
        self.statusBar.showMessage(message)
    
    def on_update_finished(self, success, message):
        """更新完成的處理"""
        if success:
            QMessageBox.information(self, "更新完成", message)
            self.refresh_models()
        else:
            QMessageBox.warning(self, "更新失敗", message)
    
    def on_download_progress(self, current, total, speed):
        """處理下載進度更新"""
        if total > 0:
            percentage = int(current / total * 100)
            self.statusBar.showMessage(
                f"下載中: {current/1024/1024:.1f} MB / {total/1024/1024:.1f} MB ({percentage}%) - {speed/1024/1024:.2f} MB/s"
            )
    
    def on_download_finished(self, success, message):
        """處理下載完成事件"""
        if success:
            self.statusBar.showMessage("下載完成", 3000)
        else:
            self.statusBar.showMessage(f"下載失敗: {message}", 5000)
    
    def on_model_downloaded(self, model_path):
        """處理模型下載完成事件"""
        if model_path:
            model_name = os.path.basename(model_path)
            QMessageBox.information(self, "下載完成", f"模型 {model_name} 已下載成功！")
            self.refresh_models()
            reply = QMessageBox.question(
                self, "使用新模型", 
                f"是否立即切換到新下載的模型 {model_name}？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.register_and_use_model(model_path)
    
    def on_model_imported(self, model_path):
        """處理模型匯入完成事件"""
        if model_path:
            model_name = os.path.basename(model_path)
            self.statusBar.showMessage(f"模型 {model_name} 已匯入成功", 3000)
            self.refresh_models()
    
    def on_model_deleted(self, model_name):
        """處理模型刪除完成事件"""
        self.statusBar.showMessage(f"模型 {model_name} 已刪除", 3000)
        self.refresh_models()
        
    def no_models_found(self):
        """當沒有找到模型時觸發"""
        reply = QMessageBox.question(
            self, "找不到模型", 
            "未在models目錄中找到可用模型。\n是否要下載模型？", 
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.download_model()
        else:
            self.statusBar.showMessage("警告：沒有可用模型，某些功能可能無法使用")
    
    def init_ui(self):
        """初始化使用者介面"""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout()
        self.central_widget.setLayout(main_layout)
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        self.image_tab = ImageProcessingTab(self)
        self.tab_widget.addTab(self.image_tab, "圖片處理")
        self.video_tab = VideoProcessingTab(self)
        self.tab_widget.addTab(self.video_tab, "影片處理")
        self.training_tab = TrainingTab(self)
        self.tab_widget.addTab(self.training_tab, "訓練模型")
        self.assessment_tab = AssessmentTab(self)
        self.tab_widget.addTab(self.assessment_tab, "圖像評估")
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        self.pass_model_manager_to_tabs()
    
    def pass_model_manager_to_tabs(self):
        """傳遞模型管理器給各分頁並註冊為觀察者"""
        tabs = [self.image_tab, self.video_tab, self.training_tab, self.assessment_tab]
        for tab in tabs:
            if hasattr(tab, "set_model_manager"):
                tab.set_model_manager(self.model_manager)
                self.model_manager.add_observer(tab)
    
    def reload_all_tabs_models(self):
        """重新載入所有分頁的模型列表"""
        tabs = [self.image_tab, self.video_tab, self.training_tab, self.assessment_tab]
        for tab in tabs:
            if hasattr(tab, "reload_models"):
                tab.reload_models()
    
    def notify_tabs_model_changed(self, model):
        """通知各分頁模型已變更"""
        tabs = [self.image_tab, self.video_tab, self.training_tab, self.assessment_tab]
        for tab in tabs:
            if hasattr(tab, "on_model_changed"):
                tab.on_model_changed(model)
    
    def check_ffmpeg(self):
        """檢查ffmpeg是否可用"""
        try:
            import subprocess
            result = subprocess.run(['ffmpeg', '-version'], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE)
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"ffmpeg檢查失敗: {str(e)}")
            return False
    
    def create_menu_bar(self):
        """創建選單列"""
        menu_bar = QMenuBar()
        self.setMenuBar(menu_bar)
        file_menu = QMenu("檔案", self)
        menu_bar.addMenu(file_menu)
        open_image_action = QAction(QIcon.fromTheme("document-open"), "開啟圖片", self)
        open_image_action.setShortcut("Ctrl+O")
        open_image_action.triggered.connect(self.open_image)
        file_menu.addAction(open_image_action)
        open_video_action = QAction(QIcon.fromTheme("camera-video"), "開啟影片", self)
        open_video_action.setShortcut("Ctrl+V")
        open_video_action.triggered.connect(self.open_video)
        file_menu.addAction(open_video_action)
        save_action = QAction(QIcon.fromTheme("document-save"), "儲存圖片", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)
        exit_action = QAction(QIcon.fromTheme("application-exit"), "離開", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 模型選單
        self.model_menu = QMenu("模型", self)
        menu_bar.addMenu(self.model_menu)
        
        # 下載模型
        download_model_action = QAction(QIcon.fromTheme("system-software-update"), "下載模型", self)
        download_model_action.setShortcut("Ctrl+D")
        download_model_action.triggered.connect(self.download_model)
        self.model_menu.addAction(download_model_action)
        
        # 匯入外部模型
        import_model_action = QAction(QIcon.fromTheme("folder-open"), "匯入模型", self)
        import_model_action.triggered.connect(self.import_external_model)
        import_model_action.setShortcut("Ctrl+I")
        self.model_menu.addAction(import_model_action)
        
        # 分隔線
        self.model_menu.addSeparator()
        
        # 清除模型快取
        clear_cache_action = QAction(QIcon.fromTheme("edit-clear"), "清除快取", self)
        clear_cache_action.triggered.connect(self.clear_model_cache)
        self.model_menu.addAction(clear_cache_action)
        
        # 重新掃描模型
        refresh_models_action = QAction(QIcon.fromTheme("view-refresh"), "掃描模型", self)
        refresh_models_action.triggered.connect(self.refresh_models)
        refresh_models_action.setShortcut("Ctrl+R")
        self.model_menu.addAction(refresh_models_action)
    
        # 刪除模型
        delete_model_action = QAction(QIcon.fromTheme("edit-delete"), "刪除模型", self)
        delete_model_action.triggered.connect(self.delete_model)
        self.model_menu.addAction(delete_model_action)
        
        # 分隔線
        self.model_menu.addSeparator()
        
        # 說明選單
        help_menu = menu_bar.addMenu("說明")
        about_action = QAction(QIcon.fromTheme("help-about"), "關於", self)
        about_action.setShortcut("Ctrl+A")
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        website_action = QAction(QIcon.fromTheme("applications-internet"), "官方網站", self)
        website_action.setShortcut("Ctrl+W")
        website_action.triggered.connect(self.open_website)
        help_menu.addAction(website_action)
    
    def open_website(self):
        """開啟專案的GitHub頁面"""
        url = QUrl("https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm")
        QDesktopServices.openUrl(url)
        self.statusBar.showMessage("正在開啟官方網站...", 3000)
    
    def on_tab_changed(self, index):
        """當切換頁籤時觸發"""
        tab_messages = {
            0: "圖片處理模式",
            1: "影片處理模式",
            2: "模型訓練模式",
            3: "圖像評估模式"
        }
        self.statusBar.showMessage(tab_messages.get(index, "未知模式"))
    
    def open_image(self):
        """開啟圖片"""
        if hasattr(self, "image_tab"):
            self.tab_widget.setCurrentIndex(0)
            self.image_tab.open_image()
    
    def open_video(self):
        """開啟影片"""
        if hasattr(self, "video_tab"):
            self.tab_widget.setCurrentIndex(1) 
            self.video_tab.open_video()
    
    def save_image(self):
        """儲存圖片"""
        if hasattr(self, "image_tab"):
            self.image_tab.save_image()
    
    def download_model(self):
        """開啟模型下載對話框"""
        try:
            dialog = DownloadModelDialog(self.model_manager, parent=self)
            dialog.exec()
        except Exception as e:
            logger.error(f"開啟下載對話框失敗: {str(e)}")
            QMessageBox.critical(self, "錯誤", f"開啟下載對話框失敗: {str(e)}")
    
    def import_external_model(self):
        """開啟模型選擇對話框並匯入外部模型"""
        model_path, _ = QFileDialog.getOpenFileName(
            self, "選擇模型文件", "", "PyTorch 模型 (*.pth);;壓縮檔 (*.zip *.tar *.tar.gz *.rar *.7z);;所有文件 (*.*)"
        )
        if model_path:
            self.statusBar.showMessage(f"正在匯入模型: {os.path.basename(model_path)}...")
            success, result = self.model_manager.import_external_model(model_path)
            if success:
                self.register_and_use_model(result)
                self.refresh_models()
                model_name = os.path.basename(result)
                self.statusBar.showMessage(f"已成功匯入並註冊模型: {model_name}", 5000)
                logger.info(f"已匯入並註冊模型: {model_name}")
            else:
                message = f"匯入模型失敗: {model_path} - {result}"
                logger.error(message)
                QMessageBox.critical(self, "錯誤", message)
    
    def delete_model(self):
        """刪除選擇的模型"""
        if not self.model_manager.has_models():
            QMessageBox.information(self, "提示", "沒有可用的模型可刪除")
            return
        models = self.model_manager.get_available_models()
        model_info = self.model_manager.get_local_model_info()
        model_dict = {}
        for model_path in models:
            name = os.path.basename(model_path)
            size = model_info.get(model_path, {}).get('size', 0)
            model_dict[f"{name} ({size:.1f} MB)"] = model_path
        model_names = list(model_dict.keys())
        if not model_names:
            QMessageBox.information(self, "提示", "沒有可刪除的模型")
            return
        item, ok = QInputDialog.getItem(
            self, "刪除模型", "請選擇要刪除的模型:", model_names, 0, False
        )
        if ok and item:
            model_path = model_dict[item]
            reply = QMessageBox.question(
                self, "確認刪除", 
                f"確定要刪除模型 {item}？\n此操作無法復原。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                success = self.model_manager.delete_model(model_path)
                if success:
                    QMessageBox.information(self, "刪除成功", f"模型 {item} 已刪除")
                    self.refresh_models()
                else:
                    QMessageBox.critical(self, "刪除失敗", f"無法刪除模型 {item}")
    
    def clear_model_cache(self):
        """清除模型快取"""
        try:
            self.model_manager.clear_cache()
            QMessageBox.information(self, "清除完成", "模型快取已清除並釋放記憶體")
            self.statusBar.showMessage("模型快取已清除", 3000)
            self.model = None
        except Exception as e:
            logger.error(f"清除模型快取失敗: {str(e)}")
            QMessageBox.critical(self, "錯誤", f"清除模型快取失敗: {str(e)}")
    
    def register_and_use_model(self, model_path):
        """註冊模型並在需要時實際載入"""
        try:
            self.statusBar.showMessage("正在註冊模型...")
            success = self.model_manager.register_model(model_path)
            if not success:
                self.statusBar.showMessage("模型註冊失敗")
                return
            model_name = os.path.basename(model_path)
            self.statusBar.showMessage(f"使用設備: {self.device} | 已註冊模型: {model_name} (需要時才會載入)")
            self.notify_tabs_registered_model_changed(model_path)
            
        except Exception as e:
            logger.error(f"註冊模型失敗: {str(e)}")
            QMessageBox.critical(self, "錯誤", f"無法註冊模型: {str(e)}")
            self.statusBar.showMessage(f"模型註冊失敗: {str(e)}")
    
    def refresh_models(self):
        """重新掃描模型目錄並更新UI"""
        try:
            self.statusBar.showMessage("正在重新掃描模型目錄...")
            self.model_manager.scan_models_directory()
            self.reload_all_tabs_models()
            self.statusBar.showMessage("已重新掃描模型目錄", 3000)
        except Exception as e:
            logger.error(f"重新掃描模型失敗: {str(e)}")
            QMessageBox.critical(self, "錯誤", f"重新掃描模型失敗: {str(e)}")
    
    def update_status_with_model(self, model_name):
        """更新狀態欄顯示模型資訊"""
        self.statusBar.showMessage(f"使用設備: {self.device} | 使用模型: {model_name}")
        logger.info(f"使用模型: {model_name}")
    
    def notify_tabs_registered_model_changed(self, model_path):
        """通知各分頁已註冊的模型已變更"""
        model_name = os.path.basename(model_path)
        
        tabs = [self.image_tab, self.video_tab, self.training_tab, self.assessment_tab]
        for tab in tabs:
            if hasattr(tab, "on_registered_model_changed"):
                tab.on_registered_model_changed(model_path)
            elif hasattr(tab, "update_model_info"):
                tab.update_model_info(model_name)
    
    def on_model_loaded(self, model_name):
        """處理模型載入成功事件"""
        self.statusBar.showMessage(f"已載入模型: {model_name}")
        self.reload_all_tabs_models()
        self.model = self.model_manager.get_current_model()
        self.notify_tabs_model_changed(self.model)
    
    def show_about(self):
        """顯示關於對話框"""
        try:
            current_directory = Path(__file__).resolve().parent.parent.parent
            about_text = f"""
<html>
<body style='font-family: "微軟正黑體"; text-align: center;'>
    <img src="{str(current_directory / 'assets' / 'icon' / f'{self.version}.ico')}" style="width: 64px; height: 64px; margin-bottom: 10px;">
    <h3 style='font-size: 8pt; margin: 5px;'>Nagato-Sakura-Image-Charm</h3>
    <p style='font-size: 8pt; margin: 5px;'>Version {self.version}</p>
    <p style='font-size: 9pt; margin: 5px;'>Copyright ©2025 Nagato-Sakura-Image-Charm</p>
    <p style='font-size: 9pt; margin: 5px;'>Developed by 天野靜樹</p>
    <p style='font-size: 9pt; margin: 5px;'>Licensed under the Apache License, Version 2.0</p>
</body>
</html>
            """
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("關於")
            msg_box.setTextFormat(Qt.TextFormat.RichText)
            msg_box.setText(about_text)
            msg_box.exec()
        except Exception as e:
            logger.error(f"顯示關於對話框失敗: {str(e)}")
            QMessageBox.critical(self, "錯誤", f"顯示關於對話框失敗: {str(e)}")