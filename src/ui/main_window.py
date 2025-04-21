import sys
import os
import time
import logging
from pathlib import Path
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QTabWidget, QStatusBar,
                            QMenu, QMenuBar, QDialog, QMessageBox, QFileDialog, QInputDialog, QLabel, QApplication)
from PyQt6.QtGui import QAction, QIcon, QPixmap
from PyQt6.QtCore import Qt, QUrl, QTimer, pyqtSignal
from PyQt6.QtGui import QDesktopServices

from src.ui.image_tab import ImageProcessingTab
from src.ui.video_tab import VideoProcessingTab
from src.utils.NS_ModelManager import ModelManager
from src.ui.dialogs import DownloadModelDialog
from src.ui.training_tab import TrainingTab
from src.ui.assessment_tab import AssessmentTab
from src.ui.easter_egg import NagatoSakuraEasterEggDialog 


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 主應用程式類別 ---
class ImageEnhancerApp(QMainWindow):
    version = "1.0.0"
    def __init__(self):
        super().__init__()
        self.about_clicks = 0 
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
                if hasattr(self.model_manager, "add_observer"):
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
                                stderr=subprocess.PIPE,
                                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0) 
            return result.returncode == 0
        except FileNotFoundError:
             logger.warning("找不到 ffmpeg 命令，請確保已安裝並添加到系統 PATH。")
             return False
        except Exception as e:
            logger.warning(f"ffmpeg檢查失敗: {str(e)}")
            return False

    def create_menu_bar(self):
        """創建選單列"""
        menu_bar = QMenuBar()
        self.setMenuBar(menu_bar)
        file_menu = QMenu("檔案", self)
        menu_bar.addMenu(file_menu)
        icon_dir = Path(__file__).resolve().parent.parent.parent / "assets/icons"
        open_image_action = QAction(QIcon.fromTheme("document-open", QIcon(str(icon_dir / "image.png"))), "開啟圖片", self)
        open_image_action.setShortcut("Ctrl+O")
        open_image_action.triggered.connect(self.open_image)
        file_menu.addAction(open_image_action)
        open_video_action = QAction(QIcon.fromTheme("camera-video", QIcon(str(icon_dir / "video.png"))), "開啟影片", self)
        open_video_action.setShortcut("Ctrl+V")
        open_video_action.triggered.connect(self.open_video)
        file_menu.addAction(open_video_action)
        save_action = QAction(QIcon.fromTheme("document-save", QIcon(str(icon_dir / "save.png"))), "儲存圖片", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)
        exit_action = QAction(QIcon.fromTheme("application-exit", QIcon(str(icon_dir / "exit.png"))), "離開", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 模型選單
        self.model_menu = QMenu("模型", self)
        menu_bar.addMenu(self.model_menu)
        download_model_action = QAction(QIcon.fromTheme("system-software-update", QIcon(str(icon_dir / "download.png"))), "下載模型", self)
        download_model_action.setShortcut("Ctrl+D")
        download_model_action.triggered.connect(self.download_model)
        self.model_menu.addAction(download_model_action)
        import_model_action = QAction(QIcon.fromTheme("folder-open", QIcon(str(icon_dir / "import.png"))), "匯入模型", self)
        import_model_action.triggered.connect(self.import_external_model)
        import_model_action.setShortcut("Ctrl+I")
        self.model_menu.addAction(import_model_action)
        self.model_menu.addSeparator()
        clear_cache_action = QAction(QIcon.fromTheme("edit-clear", QIcon(str(icon_dir / "clear.png"))), "清除快取", self)
        clear_cache_action.triggered.connect(self.clear_model_cache)
        self.model_menu.addAction(clear_cache_action)
        refresh_models_action = QAction(QIcon.fromTheme("view-refresh", QIcon(str(icon_dir / "refresh.png"))), "掃描模型", self)
        refresh_models_action.triggered.connect(self.refresh_models)
        refresh_models_action.setShortcut("Ctrl+R")
        self.model_menu.addAction(refresh_models_action)
        delete_model_action = QAction(QIcon.fromTheme("edit-delete", QIcon(str(icon_dir / "delete.png"))), "刪除模型", self)
        delete_model_action.triggered.connect(self.delete_model)
        self.model_menu.addAction(delete_model_action)
        self.model_menu.addSeparator()

        # 說明選單
        help_menu = menu_bar.addMenu("說明")
        about_action = QAction(QIcon.fromTheme("help-about", QIcon(str(icon_dir / "about.png"))), "關於", self)
        about_action.setShortcut("Ctrl+A")
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        website_action = QAction(QIcon.fromTheme("applications-internet", QIcon(str(icon_dir / "web.png"))), "官方網站", self)
        website_action.setShortcut("Ctrl+W")
        website_action.triggered.connect(self.open_website)
        help_menu.addAction(website_action)
        benchmark_action = QAction(QIcon.fromTheme("utilities-system-monitor", QIcon(str(icon_dir / "benchmark.png"))), "基準測試", self)
        benchmark_action.triggered.connect(self.run_benchmark)
        benchmark_action.setShortcut("Ctrl+B")
        help_menu.addAction(benchmark_action)

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
        if hasattr(self, "image_tab") and self.tab_widget.currentIndex() == 0:
            self.image_tab.save_image()
        else:
            self.statusBar.showMessage("請先切換到圖片處理分頁以儲存圖片", 3000)


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
                self.statusBar.showMessage(f"匯入模型失敗: {result}", 5000)

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
            size = model_info.get(model_path, {}).get('size_mb', 0)
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
                success, message = self.model_manager.delete_model(model_path)
                if success:
                    QMessageBox.information(self, "刪除成功", message)
                    self.refresh_models() 
                else:
                    QMessageBox.critical(self, "刪除失敗", message)

    def clear_model_cache(self):
        """清除模型快取"""
        try:
            self.model_manager.clear_cache()
            QMessageBox.information(self, "清除完成", "模型快取已清除並釋放記憶體")
            self.statusBar.showMessage("模型快取已清除", 3000)
            self.model = None 
            self.notify_tabs_model_changed(None)
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
                QMessageBox.warning(self, "註冊失敗", f"無法註冊模型: {model_path}")
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
            current_registered_path = self.model_manager.get_registered_model_path()
            if current_registered_path and not os.path.exists(current_registered_path):
                logger.warning(f"先前註冊的模型 {os.path.basename(current_registered_path)} 不再存在。")
                self.model_manager.clear_registered_model() 
                self.statusBar.showMessage("已重新掃描模型目錄，請重新選擇模型", 5000)
                self.notify_tabs_registered_model_changed(None)
            elif current_registered_path:
                model_name = os.path.basename(current_registered_path)
                self.statusBar.showMessage(f"已重新掃描模型目錄 | 當前註冊: {model_name}", 3000)
            else:
                self.statusBar.showMessage("已重新掃描模型目錄", 3000)
        except Exception as e:
            logger.error(f"重新掃描模型失敗: {str(e)}")
            QMessageBox.critical(self, "錯誤", f"重新掃描模型失敗: {str(e)}")
            self.statusBar.showMessage("重新掃描模型失敗", 5000)

    def update_status_with_model(self, model_name):
        """更新狀態欄顯示模型資訊"""
        if model_name:
            self.statusBar.showMessage(f"使用設備: {self.device} | 使用模型: {model_name}")
            logger.info(f"使用模型: {model_name}")
        else:
            self.statusBar.showMessage(f"使用設備: {self.device} | 未選擇模型")

    def notify_tabs_registered_model_changed(self, model_path):
        """通知各分頁已註冊的模型已變更"""
        model_name = os.path.basename(model_path) if model_path else None
        tabs = [self.image_tab, self.video_tab, self.training_tab, self.assessment_tab]
        for tab in tabs:
            if hasattr(tab, "on_registered_model_changed"):
                tab.on_registered_model_changed(model_path)
            elif hasattr(tab, "update_model_info"):
                tab.update_model_info(model_name if model_name else "未選擇")

    def on_model_loaded(self, model_name):
        """處理模型載入成功事件"""
        self.statusBar.showMessage(f"已載入模型: {model_name}")
        self.update_status_with_model(model_name)
        self.model = self.model_manager.get_current_model()
        self.notify_tabs_model_changed(self.model)

    def show_about(self):
        """顯示關於對話框"""
        try:
            current_directory = Path(__file__).resolve().parent.parent.parent
            icon_path = str(current_directory / 'assets' / 'icon' / f'{self.version}.ico')
            dialog = AboutDialog(self.version, icon_path, self)
            dialog.iconClicked.connect(self.handle_about_icon_click)
            dialog.exec()
        except Exception as e:
            logger.error(f"顯示關於對話框失敗: {str(e)}")
            QMessageBox.critical(self, "錯誤", f"顯示關於對話框失敗: {str(e)}")

    def handle_about_icon_click(self):
        """處理關於對話框圖標的點擊事件"""
        self.about_clicks += 1
        logger.debug(f"About icon clicked {self.about_clicks} times.")  # 除錯用
        if self.about_clicks == 9:
            self.about_clicks = 0 
            for widget in QApplication.topLevelWidgets():
                if isinstance(widget, AboutDialog):
                    widget.close()
                    break
            try:
                easter_egg = NagatoSakuraEasterEggDialog(self)
                easter_egg.exec()
            except Exception as e:
                logger.error(f"彩蛋顯示失敗: {str(e)}")
                QMessageBox.information(self, "彩蛋！", 
                    "噹啷! 主人找到長門櫻了！\n\n"
                    "嘻嘻，被您發現這個小秘密了呢！\n"
                    "謝謝主人一直使用這個程式，能幫上主人的忙，是長門櫻最開心的事情了！\n\n"
                    "請繼續讓長門櫻陪伴在您身邊吧！🌸")

    def run_benchmark(self):
        """執行基準測試"""
        from src.ui.benchmark import BenchmarkDialog
        dialog = BenchmarkDialog(self.model_manager, self)
        dialog.exec()
        
# --- 自訂關於對話框 ---
class AboutDialog(QDialog):
    iconClicked = pyqtSignal() 
    def __init__(self, version, icon_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("關於")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label = QLabel(self)
        pixmap = QPixmap(icon_path)
        if not pixmap.isNull():
             self.icon_label.setPixmap(pixmap.scaled(256, 256, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
             logger.warning(f"無法載入關於對話框圖標: {icon_path}")
             self.icon_label.setText("Icon")  
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setToolTip("點我試試？") 
        self.icon_label.setCursor(Qt.CursorShape.PointingHandCursor)  
        self.icon_label.mousePressEvent = self.icon_mouse_press 
        layout.addWidget(self.icon_label)
        about_text = f"""
<div style='font-family: "微軟正黑體"; text-align: center;'>
    <h3 style='font-size: 8pt; margin: 5px;'>Nagato-Sakura-Image-Charm</h3>
    <p style='font-size: 8pt; margin: 5px;'>Version {version}</p>
    <p style='font-size: 9pt; margin: 5px;'>Copyright ©2025 Nagato-Sakura-Image-Charm</p>
    <p style='font-size: 9pt; margin: 5px;'>Developed by 天野靜樹</p>
    <p style='font-size: 9pt; margin: 5px;'>Licensed under the Apache License, Version 2.0</p>
</div>
        """
        text_label = QLabel(about_text)
        text_label.setTextFormat(Qt.TextFormat.RichText)
        text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text_label.setWordWrap(True) 
        layout.addWidget(text_label)
        self.setFixedSize(350, 400)

    def icon_mouse_press(self, event):
        """處理圖標點擊事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.iconClicked.emit() 
        super(QLabel, self.icon_label).mousePressEvent(event)

# --- 主程式進入點 ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = ImageEnhancerApp()
    main_win.show()
    sys.exit(app.exec())