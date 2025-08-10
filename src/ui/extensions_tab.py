import os
import json
import logging
import shutil
import time
import stat
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, 
                             QPushButton, QLabel, QFrame, QProgressBar, QTextEdit, 
                             QMessageBox, QGroupBox, QGridLayout, QLineEdit, 
                             QComboBox, QDialog, QFormLayout, QTextEdit as QTextEditDialog,
                             QDialogButtonBox)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QFont, QKeySequence, QShortcut

from src.utils.NS_GitManager import GitManager


logger = logging.getLogger(__name__)

class GitOperationThread(QThread):
    """Git操作線程"""
    progress = pyqtSignal(str)
    progress_value = pyqtSignal(int)
    finished = pyqtSignal(bool, str)
    error = pyqtSignal(str)
    
    def __init__(self, operation, repo_url, target_path):
        super().__init__()
        self.operation = operation
        self.repo_url = repo_url
        self.target_path = target_path
        self.git_manager = GitManager()
        
    def run(self):
        try:
            if self.operation == 'clone':
                self.progress.emit("正在檢查儲存庫...")
                self.progress_value.emit(5)
                if os.path.exists(self.target_path):
                    self.error.emit(f"目標目錄已存在: {self.target_path}")
                    return
                self.progress.emit("正在建立目標目錄...")
                self.progress_value.emit(10)
                from pathlib import Path
                Path(self.target_path).parent.mkdir(parents=True, exist_ok=True)
                self.progress.emit("正在連接到遠端儲存庫...")
                self.progress_value.emit(15)
                self._clone_with_progress()
                
                if os.path.exists(self.target_path):
                    self.progress.emit("驗證下載內容...")
                    self.progress_value.emit(95)
                    self.progress.emit("安裝完成")
                    self.progress_value.emit(100)
                    self.finished.emit(True, "擴充功能安裝成功")
                else:
                    self.error.emit("克隆失敗: 目標目錄未建立")
            elif self.operation == 'update':
                self._update_with_progress()
        except subprocess.TimeoutExpired:
            self.error.emit("操作超時，請檢查網路連線")
        except Exception as e:
            self.error.emit(f"操作失敗: {str(e)}")
    
    def _clone_with_progress(self):
        """執行 Git clone 操作並監控進度"""
        import subprocess
        import threading
        import queue
        import re
        cmd = ['git', 'clone', '--depth', '1', '--progress', self.repo_url, self.target_path]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            universal_newlines=True,
            bufsize=1
        )
        def monitor_progress():
            current_progress = 20
            while True:
                line = process.stderr.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                if "Cloning into" in line:
                    self.progress.emit("開始克隆儲存庫...")
                    current_progress = 25
                    self.progress_value.emit(current_progress)
                elif "remote: Enumerating objects" in line:
                    self.progress.emit("正在枚舉遠端物件...")
                    current_progress = 30
                    self.progress_value.emit(current_progress)
                elif "remote: Counting objects" in line:
                    match = re.search(r'(\d+)%', line)
                    if match:
                        remote_progress = int(match.group(1))
                        current_progress = 35 + int(remote_progress * 0.15)
                        self.progress.emit(f"正在計算物件... {remote_progress}%")
                        self.progress_value.emit(min(current_progress, 50))
                    else:
                        self.progress.emit("正在計算物件...")
                        current_progress = 35
                        self.progress_value.emit(current_progress)
                elif "remote: Compressing objects" in line:
                    match = re.search(r'(\d+)%', line)
                    if match:
                        compress_progress = int(match.group(1))
                        current_progress = 50 + int(compress_progress * 0.15)
                        self.progress.emit(f"正在壓縮物件... {compress_progress}%")
                        self.progress_value.emit(min(current_progress, 65))
                    else:
                        self.progress.emit("正在壓縮物件...")
                        current_progress = 50
                        self.progress_value.emit(current_progress)
                elif "Receiving objects" in line:
                    match = re.search(r'(\d+)%.*\((\d+/\d+)\)', line)
                    if match:
                        receive_progress = int(match.group(1))
                        objects_info = match.group(2)
                        current_progress = 65 + int(receive_progress * 0.25)
                        self.progress.emit(f"正在接收物件... {receive_progress}% ({objects_info})")
                        self.progress_value.emit(min(current_progress, 90))
                    else:
                        match = re.search(r'(\d+)%', line)
                        if match:
                            receive_progress = int(match.group(1))
                            current_progress = 65 + int(receive_progress * 0.25)
                            self.progress.emit(f"正在接收物件... {receive_progress}%")
                            self.progress_value.emit(min(current_progress, 90))
                elif "Resolving deltas" in line:
                    match = re.search(r'(\d+)%.*\((\d+/\d+)\)', line)
                    if match:
                        delta_progress = int(match.group(1))
                        delta_info = match.group(2)
                        current_progress = 90 + int(delta_progress * 0.05)
                        self.progress.emit(f"正在解析增量... {delta_progress}% ({delta_info})")
                        self.progress_value.emit(min(current_progress, 95))
                    else:
                        self.progress.emit("正在解析增量...")
                        current_progress = 90
                        self.progress_value.emit(current_progress)
                elif "Checking out files" in line:
                    match = re.search(r'(\d+)%.*\((\d+/\d+)\)', line)
                    if match:
                        checkout_progress = int(match.group(1))
                        files_info = match.group(2)
                        self.progress.emit(f"正在檢出檔案... {checkout_progress}% ({files_info})")
                        self.progress_value.emit(95)
                    else:
                        self.progress.emit("正在檢出檔案...")
                        self.progress_value.emit(95)
        progress_thread = threading.Thread(target=monitor_progress)
        progress_thread.daemon = True
        progress_thread.start()
        try:
            stdout, stderr = process.communicate(timeout=300)
            progress_thread.join(timeout=5)
            if process.returncode == 0:
                return True
            else:
                error_msg = stderr or stdout or "未知錯誤"
                self.error.emit(f"克隆失敗: {error_msg}")
                return False  
        except subprocess.TimeoutExpired:
            process.kill()
            self.error.emit("克隆超時，請檢查網路連線")
            return False
    
    def _update_with_progress(self):
        """執行 Git update 操作並監控進度"""
        import subprocess
        import threading
        self.progress.emit("正在檢查本地儲存庫...")
        self.progress_value.emit(15)
        if not self.git_manager.is_git_repository(self.target_path):
            self.error.emit("不是有效的 Git 儲存庫")
            return False
        self.progress.emit("正在獲取遠端更新信息...")
        self.progress_value.emit(25)
        try:
            fetch_process = subprocess.run(
                ['git', 'fetch', 'origin'],
                cwd=self.target_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            if fetch_process.returncode != 0:
                self.error.emit(f"獲取更新信息失敗: {fetch_process.stderr}")
                return False
            self.progress.emit("正在檢查是否有更新...")
            self.progress_value.emit(35)
            status_process = subprocess.run(
                ['git', 'status', '-uno', '--porcelain'],
                cwd=self.target_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            diff_process = subprocess.run(
                ['git', 'rev-list', '--count', 'HEAD..origin/main'],
                cwd=self.target_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            if diff_process.returncode == 0 and diff_process.stdout.strip() == '0':
                self.progress.emit("已是最新版本")
                self.progress_value.emit(100)
                self.finished.emit(True, "擴充功能已是最新版本")
                return True
            self.progress.emit("正在下載更新...")
            self.progress_value.emit(50)
            pull_process = subprocess.Popen(
                ['git', 'pull', 'origin'],
                cwd=self.target_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                universal_newlines=True
            )
            current_progress = 50
            while True:
                line = pull_process.stderr.readline()
                if not line and pull_process.poll() is not None:
                    break
                
                if line:
                    line = line.strip()
                    if "Updating" in line:
                        self.progress.emit("正在更新檔案...")
                        current_progress = 70
                        self.progress_value.emit(current_progress)
                    elif "Fast-forward" in line:
                        self.progress.emit("正在快進更新...")
                        current_progress = 85
                        self.progress_value.emit(current_progress)
            stdout, stderr = pull_process.communicate()
            if pull_process.returncode == 0:
                self.progress.emit("更新完成")
                self.progress_value.emit(100)
                self.finished.emit(True, "擴充功能更新成功")
                return True
            else:
                error_msg = stderr or stdout or "未知錯誤"
                self.error.emit(f"更新失敗: {error_msg}")
                return False
        except subprocess.TimeoutExpired:
            self.error.emit("更新超時")
            return False
        except Exception as e:
            self.error.emit(f"更新過程發生錯誤: {str(e)}")
            return False


class ExtensionCard(QFrame):
    """擴充功能卡片組件"""
    def __init__(self, extension_data, extensions_tab):
        super().__init__()
        self.extension_data = extension_data
        self.extensions_tab = extensions_tab
        self.is_installing = False
        self.git_manager = GitManager()
        self.setup_ui()
        
    def setup_ui(self):
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        title_layout = QHBoxLayout()
        name_label = QLabel(self.extension_data['name'])
        name_font = QFont()
        name_font.setPointSize(12)
        name_font.setBold(True)
        name_label.setFont(name_font)
        title_layout.addWidget(name_label)
        title_layout.addStretch()
        if 'added_date' in self.extension_data:
            date_label = QLabel(f"添加日期: {self.extension_data['added_date']}")
            title_layout.addWidget(date_label)
        layout.addLayout(title_layout)
        desc_label = QLabel(self.extension_data.get('description', '無描述'))
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        if 'author' in self.extension_data:
            author_layout = QHBoxLayout()
            author_layout.addStretch()
            author_label = QLabel(f"作者: {self.extension_data['author']}")
            author_layout.addWidget(author_label)
            layout.addLayout(author_layout)
        tags_buttons_layout = QHBoxLayout()
        if 'tags' in self.extension_data and self.extension_data['tags']:
            tags_label = QLabel("標籤:")
            tags_buttons_layout.addWidget(tags_label)
            for tag in self.extension_data['tags']:
                if tag.strip():
                    tag_button = QPushButton(tag)
                    tag_button.setMaximumHeight(25)
                    tag_button.clicked.connect(lambda checked, t=tag: self.extensions_tab.filter_by_tag(t))
                    tags_buttons_layout.addWidget(tag_button)
        tags_buttons_layout.addStretch()
        repo_name = self.extract_repo_name(self.extension_data['url'])
        extension_path = os.path.join("extensions", repo_name)
        if os.path.exists(extension_path):
            self.uninstall_button = QPushButton("卸載")
            self.uninstall_button.clicked.connect(self.uninstall_extension)
            tags_buttons_layout.addWidget(self.uninstall_button)
            self.check_and_show_update_button(extension_path, tags_buttons_layout)
        else:
            self.action_button = QPushButton("安裝")
            self.action_button.clicked.connect(self.install_extension)
            tags_buttons_layout.addWidget(self.action_button)
        layout.addLayout(tags_buttons_layout)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel()
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)
    
    def extract_repo_name(self, url):
        """從Git URL提取倉庫名稱"""
        if url.endswith('.git'):
            url = url[:-4]
        repo_name = url.split('/')[-1]
        return repo_name
    
    def check_update_available(self, extension_path):
        """檢查是否有更新可用"""
        try:
            if not self.git_manager.is_git_repository(extension_path):
                return False
            import subprocess
            fetch_result = subprocess.run(
                ['git', 'fetch', 'origin'],
                cwd=extension_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            if fetch_result.returncode != 0:
                return False
            status_result = subprocess.run(
                ['git', 'rev-list', '--count', 'HEAD..origin/main'],
                cwd=extension_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            if status_result.returncode == 0:
                commits_behind = status_result.stdout.strip()
                return int(commits_behind) > 0 if commits_behind.isdigit() else False
            status_result = subprocess.run(
                ['git', 'rev-list', '--count', 'HEAD..origin/master'],
                cwd=extension_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            if status_result.returncode == 0:
                commits_behind = status_result.stdout.strip()
                return int(commits_behind) > 0 if commits_behind.isdigit() else False
        except Exception as e:
            logger.warning(f"檢查更新時發生錯誤: {str(e)}")
        return False
    
    def check_and_show_update_button(self, extension_path, layout):
        """異步檢查更新並顯示更新按鈕"""
        import threading
        def check_update():
            try:
                has_update = self.check_update_available(extension_path)
                if has_update:
                    QTimer.singleShot(0, lambda: self.add_update_button(layout))
            except Exception as e:
                logger.warning(f"異步檢查更新時發生錯誤: {str(e)}")
        update_thread = threading.Thread(target=check_update)
        update_thread.daemon = True
        update_thread.start()
    
    def add_update_button(self, layout):
        """添加更新按鈕到佈局"""
        try:
            self.action_button = QPushButton("更新")
            self.action_button.clicked.connect(self.update_extension)
            layout.insertWidget(layout.count() - 1, self.action_button)
        except Exception as e:
            logger.warning(f"添加更新按鈕時發生錯誤: {str(e)}")
    
    def install_extension(self):
        """安裝擴充功能"""
        if self.is_installing:
            return
        self.is_installing = True
        self.action_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("準備中... 0%")
        self.status_label.setVisible(True)
        self.status_label.setText("準備安裝...")
        repo_name = self.extract_repo_name(self.extension_data['url'])
        target_path = os.path.join("extensions", repo_name)
        self.git_thread = GitOperationThread(
            'clone', 
            self.extension_data['url'], 
            target_path
        )
        self.git_thread.progress.connect(self.update_status)
        self.git_thread.progress_value.connect(self.update_progress)
        self.git_thread.finished.connect(self.on_operation_finished)
        self.git_thread.error.connect(self.on_operation_error)
        self.git_thread.start()
    
    def update_extension(self):
        """更新擴充功能"""
        if self.is_installing:
            return
        self.is_installing = True
        self.action_button.setEnabled(False)
        if hasattr(self, 'uninstall_button'):
            self.uninstall_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("準備中... 0%")
        self.status_label.setVisible(True)
        self.status_label.setText("準備更新...")
        repo_name = self.extract_repo_name(self.extension_data['url'])
        target_path = os.path.join("extensions", repo_name)
        self.git_thread = GitOperationThread(
            'update', 
            self.extension_data['url'], 
            target_path
        )
        self.git_thread.progress.connect(self.update_status)
        self.git_thread.progress_value.connect(self.update_progress)
        self.git_thread.finished.connect(self.on_operation_finished)
        self.git_thread.error.connect(self.on_operation_error)
        self.git_thread.start()
    
    def uninstall_extension(self):
        """卸載擴充功能"""
        repo_name = self.extract_repo_name(self.extension_data['url'])
        extension_path = os.path.join("extensions", repo_name)
        reply = QMessageBox.question(
            self,
            "確認卸載",
            f"確定要卸載擴充功能 '{self.extension_data['name']}' 嗎？\n\n這將刪除整個擴充功能資料夾：\n{extension_path}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                if os.path.exists(extension_path):
                    success, error_msg = self.safe_remove_directory(extension_path)
                    if success:
                        QMessageBox.information(self, "卸載成功", f"擴充功能 '{self.extension_data['name']}' 已成功卸載。")
                        self.extensions_tab.refresh_extensions()
                    else:
                        reply = QMessageBox.question(
                            self, 
                            "卸載部分失敗", 
                            f"無法完全刪除擴充功能資料夾：\n{error_msg}\n\n是否要打開資料夾位置，讓您手動刪除？",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.Yes
                        )
                        if reply == QMessageBox.StandardButton.Yes:
                            try:
                                import subprocess
                                subprocess.run(['explorer', os.path.dirname(extension_path)])
                            except Exception as e:
                                logger.warning(f"無法打開檔案總管: {str(e)}")
                                QMessageBox.information(
                                    self, 
                                    "手動刪除", 
                                    f"請手動刪除以下資料夾：\n{extension_path}"
                                )
                else:
                    QMessageBox.warning(self, "卸載失敗", "擴充功能資料夾不存在。")
            except Exception as e:
                logger.error(f"卸載擴充功能時發生未預期錯誤: {str(e)}")
                QMessageBox.critical(self, "卸載失敗", f"卸載過程中發生錯誤：\n{str(e)}")
    
    def safe_remove_directory(self, path, max_retries=3, retry_delay=1):
        """
        安全地刪除目錄，處理Windows上的權限和檔案佔用問題
        Args:
            path: 要刪除的目錄路徑
            max_retries: 最大重試次數
            retry_delay: 重試間隔（秒）
        Returns:
            (bool, str): (是否成功, 錯誤訊息)
        """
        def handle_remove_readonly(func, path, exc):
            """處理唯讀檔案的回調函數"""
            try:
                os.chmod(path, stat.S_IWRITE)
                func(path)
            except:
                pass
        for attempt in range(max_retries):
            try:
                shutil.rmtree(path, onerror=handle_remove_readonly)
                if not os.path.exists(path):
                    return True, "成功刪除"
                self.force_remove_files(path)
                if not os.path.exists(path):
                    return True, "成功刪除"
            except PermissionError as e:
                logger.warning(f"權限錯誤，嘗試 {attempt + 1}/{max_retries}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    return False, f"權限被拒絕，無法刪除檔案。請確保沒有程序正在使用這些檔案，或以管理員身份運行。\n錯誤: {str(e)}"
            except OSError as e:
                logger.warning(f"作業系統錯誤，嘗試 {attempt + 1}/{max_retries}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    return False, f"檔案系統錯誤。\n錯誤: {str(e)}"
            except Exception as e:
                logger.error(f"刪除目錄時發生未知錯誤: {str(e)}")
                return False, f"刪除過程中發生未知錯誤: {str(e)}"
        return False, "超過最大重試次數，無法完全刪除目錄"
    
    def force_remove_files(self, path):
        """強制刪除目錄中的所有檔案"""
        try:
            for root, dirs, files in os.walk(path, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.chmod(file_path, stat.S_IWRITE)
                        os.unlink(file_path)
                    except:
                        pass
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    try:
                        os.chmod(dir_path, stat.S_IWRITE)
                        os.rmdir(dir_path)
                    except:
                        pass
            try:
                os.chmod(path, stat.S_IWRITE)
                os.rmdir(path)
            except:
                pass
        except Exception as e:
            logger.warning(f"強制刪除檔案時出錯: {str(e)}")
    
    def get_locked_files(self, path):
        """檢查目錄中哪些檔案被佔用"""
        locked_files = []
        try:
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r+b'):
                            pass
                    except PermissionError:
                        locked_files.append(file_path)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"檢查檔案佔用狀態時出錯: {str(e)}")
        
        return locked_files
    
    def update_status(self, message):
        """更新狀態消息"""
        self.status_label.setText(message)
        current_value = self.progress_bar.value()
        if current_value < 100:
            self.progress_bar.setFormat(f"{message} {current_value}%")
        else:
            self.progress_bar.setFormat("完成！ 100%")
    
    def update_progress(self, value):
        """更新進度條數值"""
        self.progress_bar.setValue(value)
        if value <= 5:
            self.progress_bar.setFormat("檢查中...")
        elif value <= 15:
            self.progress_bar.setFormat(f"初始化... {value}%")
        elif value <= 25:
            self.progress_bar.setFormat(f"連接中... {value}%")
        elif value <= 35:
            self.progress_bar.setFormat(f"準備下載... {value}%")
        elif value <= 50:
            self.progress_bar.setFormat(f"處理遠端... {value}%")
        elif value <= 65:
            self.progress_bar.setFormat(f"壓縮物件... {value}%")
        elif value <= 90:
            self.progress_bar.setFormat(f"下載中... {value}%")
        elif value < 100:
            self.progress_bar.setFormat(f"完成中... {value}%")
        else:
            self.progress_bar.setFormat("完成！ 100%")
    
    def on_operation_finished(self, success, message):
        """Git操作完成"""
        self.is_installing = False
        self.action_button.setEnabled(True)
        if hasattr(self, 'uninstall_button'):
            self.uninstall_button.setEnabled(True)
        if success:
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("完成！")
            self.status_label.setText(message)
            QTimer.singleShot(2000, self.hide_progress_bar_safely)
            QTimer.singleShot(3000, self.hide_status_label_safely)
            self.extensions_tab.refresh_extensions()
        else:
            self.progress_bar.setVisible(False)
            self.status_label.setText(f"失敗: {message}")
            QTimer.singleShot(5000, self.hide_status_label_safely)
    
    def hide_progress_bar_safely(self):
        """安全地隱藏進度條"""
        try:
            if hasattr(self, 'progress_bar') and self.progress_bar is not None:
                self.progress_bar.setVisible(False)
        except RuntimeError:
            pass
    
    def hide_status_label_safely(self):
        """安全地隱藏狀態標籤"""
        try:
            if hasattr(self, 'status_label') and self.status_label is not None:
                self.status_label.setVisible(False)
        except RuntimeError:
            pass
    
    def on_operation_error(self, error_message):
        """Git操作出錯"""
        self.is_installing = False
        self.action_button.setEnabled(True)
        if hasattr(self, 'uninstall_button'):
            self.uninstall_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"錯誤: {error_message}")
        QTimer.singleShot(5000, self.hide_status_label_safely)
    
    def closeEvent(self, event):
        """處理窗口關閉事件"""
        if hasattr(self, 'git_thread') and self.git_thread.isRunning():
            self.git_thread.terminate()
            self.git_thread.wait()
        super().closeEvent(event)
    
    def deleteLater(self):
        """延遲刪除物件前的清理"""
        if hasattr(self, 'git_thread') and self.git_thread.isRunning():
            self.git_thread.terminate()
            self.git_thread.wait()
        super().deleteLater()


class AddCustomExtensionDialog(QDialog):
    """添加自訂擴充功能對話框"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("添加自訂擴充功能")
        self.setFixedSize(500, 400)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QFormLayout(self)
        self.name_input = QLineEdit()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://github.com/用戶名/倉庫名.git")
        self.description_input = QTextEditDialog()
        self.description_input.setMaximumHeight(80)
        self.author_input = QLineEdit()
        self.tags_input = QLineEdit()
        self.tags_input.setPlaceholderText("用逗號分隔多個標籤")
        layout.addRow("名稱*:", self.name_input)
        layout.addRow("Git URL*:", self.url_input)
        layout.addRow("描述:", self.description_input)
        layout.addRow("作者:", self.author_input)
        layout.addRow("標籤:", self.tags_input)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def get_extension_data(self):
        """獲取輸入的擴充功能數據"""
        return {
            "name": self.name_input.text().strip(),
            "url": self.url_input.text().strip(),
            "description": self.description_input.toPlainText().strip(),
            "author": self.author_input.text().strip(),
            "tags": [tag.strip() for tag in self.tags_input.text().split(",") if tag.strip()],
            "added_date": datetime.now().strftime("%Y-%m-%d")
        }


class ExtensionsTab(QWidget):
    """擴充功能管理頁面"""
    def __init__(self):
        super().__init__()
        self.extensions_data = []
        self.extension_cards = []
        self.setup_ui()
        self.load_extensions_index()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        title_label = QLabel("擴充功能")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        self.result_label = QLabel("正在載入擴充功能...")
        self.result_label.setContentsMargins(5, 0, 0, 0)
        layout.addWidget(self.result_label)
        control_layout = QHBoxLayout()
        self.refresh_button = QPushButton("重新整理")
        self.refresh_button.clicked.connect(self.refresh_extensions)
        control_layout.addWidget(self.refresh_button)
        search_label = QLabel("搜尋:")
        control_layout.addWidget(search_label)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("請輸入擴充功能名稱、作者或標籤...")
        self.search_input.textChanged.connect(self.search_extensions)
        self.search_input.setMinimumWidth(200)
        control_layout.addWidget(self.search_input)
        self.clear_search_button = QPushButton("清除搜尋")
        self.clear_search_button.clicked.connect(self.clear_search)
        control_layout.addWidget(self.clear_search_button)
        self.clear_all_button = QPushButton("清除所有過濾")
        self.clear_all_button.clicked.connect(self.clear_all_filters)
        control_layout.addWidget(self.clear_all_button)
        filter_label = QLabel("狀態:")
        control_layout.addWidget(filter_label)
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("全部", "all")
        self.filter_combo.addItem("已安裝", "installed")
        self.filter_combo.addItem("未安裝", "not_installed")
        self.filter_combo.currentTextChanged.connect(self.apply_filters)
        control_layout.addWidget(self.filter_combo)
        tags_label = QLabel("標籤:")
        control_layout.addWidget(tags_label)
        self.tags_combo = QComboBox()
        self.tags_combo.addItem("所有標籤", "all_tags")
        self.tags_combo.currentTextChanged.connect(self.apply_filters)
        control_layout.addWidget(self.tags_combo)
        sort_label = QLabel("排序:")
        control_layout.addWidget(sort_label)
        self.sort_combo = QComboBox()
        self.sort_combo.addItem("名稱 (A-Z)", "name_asc")
        self.sort_combo.addItem("名稱 (Z-A)", "name_desc")
        self.sort_combo.addItem("作者 (A-Z)", "author_asc")
        self.sort_combo.addItem("作者 (Z-A)", "author_desc")
        self.sort_combo.addItem("日期 (最新)", "date_desc")
        self.sort_combo.addItem("日期 (最舊)", "date_asc")
        self.sort_combo.addItem("安裝狀態", "install_status")
        self.sort_combo.currentTextChanged.connect(self.sort_extensions)
        control_layout.addWidget(self.sort_combo)
        control_layout.addStretch()
        self.add_custom_button = QPushButton("添加自訂擴充功能")
        self.add_custom_button.clicked.connect(self.show_add_custom_dialog)
        control_layout.addWidget(self.add_custom_button)
        layout.addLayout(control_layout)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setSpacing(10)
        self.scroll_layout.addStretch()
        self.scroll_area.setWidget(self.scroll_content)
        layout.addWidget(self.scroll_area)
        self.setup_shortcuts()
    
    def setup_shortcuts(self):
        """設置鍵盤快捷鍵"""
        search_shortcut = QShortcut(QKeySequence("Ctrl+F"), self)
        search_shortcut.activated.connect(self.focus_search)
        clear_shortcut = QShortcut(QKeySequence("Escape"), self)
        clear_shortcut.activated.connect(self.clear_search_if_focused)
        refresh_shortcut = QShortcut(QKeySequence("F5"), self)
        refresh_shortcut.activated.connect(self.refresh_extensions)
        reset_shortcut = QShortcut(QKeySequence("Ctrl+R"), self)
        reset_shortcut.activated.connect(self.clear_all_filters)
    
    def focus_search(self):
        """聚焦到搜尋框"""
        self.search_input.setFocus()
        self.search_input.selectAll()
    
    def clear_search_if_focused(self):
        """如果搜尋框有焦點，則清除搜尋"""
        if self.search_input.hasFocus():
            self.clear_search()
    
    def load_extensions_index(self):
        """載入擴充功能索引"""
        try:
            index_path = os.path.join("config", "extensions_index.json")
            if os.path.exists(index_path):
                with open(index_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.extensions_data = data.get('extensions', [])
            else:
                logger.warning(f"擴充功能索引檔案不存在: {index_path}")
                self.create_default_extensions_index(index_path)
                if os.path.exists(index_path):
                    with open(index_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.extensions_data = data.get('extensions', [])
                else:
                    self.extensions_data = []
            self.create_extension_cards()
            self.update_tags_filter()
            self.sort_extensions()
            self.result_label.setText(f"共 {len(self.extensions_data)} 個擴充功能")
        except Exception as e:
            logger.error(f"載入擴充功能索引失敗: {str(e)}")
            self.result_label.setText("載入擴充功能失敗")
            QMessageBox.critical(self, "錯誤", f"載入擴充功能索引失敗：\n{str(e)}")
    
    def create_default_extensions_index(self, index_path):
        """創建預設的擴充功能索引檔案"""
        default_data = {
            "version": "1.0.0",
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "extensions": [
                {
                    "name": "Nagato-Sakura-Image-Classification (推薦)",
                    "url": "https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Classification",
                    "description": "「長門櫻-圖像分類」 - 採用先進的 EfficientNet-B0 深度學習架構，能自動識別圖像內容類型，並推薦最適合的增強模型，讓您的每一張圖片都能獲得最佳的處理結果。",
                    "author": "天野靜樹",
                    "added_date": "2025-08-10",
                    "tags": ["圖像分類", "模型推薦"]
                },
                {
                    "name": "Nagato-Sakura-Image-Quality-Assessment (推薦)",
                    "url": "https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Quality-Assessment",
                    "description": "「長門櫻-圖像品質評分」 - 基於輕量化深度可分離卷積 CNN 模型設計，提供快速的圖像品質評估服務幫助用戶快速且客觀地評估圖像質量。",
                    "author": "天野靜樹",
                    "added_date": "2025-08-10",
                    "tags": ["圖像評分", "品質評估"]
                },
                {
                    "name": "Nagato-Sakura-Bounce-py",
                    "url": "https://github.com/AmanoShizukikun/Nagato-Sakura-Bounce-py",
                    "description": "「長門櫻-彈跳球」 - 完整版資源包，經典的彈跳球玩法結合彈幕射擊遊戲，融合現代遊戲設計理念。具有流暢的操作手感、精美的視覺效果和動態音效。",
                    "author": "天野靜樹",
                    "added_date": "2025-08-10",
                    "tags": ["彩蛋遊戲", "彩蛋資源"]
                }
            ]
        }
        try:
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(default_data, f, indent=2, ensure_ascii=False)
            logger.info(f"已創建預設擴充功能索引檔案: {index_path}")
        except Exception as e:
            logger.error(f"創建預設索引檔案失敗: {str(e)}")
    
    def create_extension_cards(self):
        """創建擴充功能卡片"""
        for card in self.extension_cards:
            card.setParent(None)
            card.deleteLater()
        self.extension_cards.clear()
        for ext_data in self.extensions_data:
            card = ExtensionCard(ext_data, self)
            self.extension_cards.append(card)
            self.scroll_layout.insertWidget(self.scroll_layout.count() - 1, card)
    
    def update_tags_filter(self):
        """更新標籤過濾器選項"""
        all_tags = set()
        for ext_data in self.extensions_data:
            tags = ext_data.get('tags', [])
            if isinstance(tags, list):
                all_tags.update(tags)
        current_selection = self.tags_combo.currentData()
        self.tags_combo.clear()
        self.tags_combo.addItem("所有標籤", "all_tags")
        for tag in sorted(all_tags):
            if tag:
                self.tags_combo.addItem(tag, tag)
        if current_selection and current_selection != "all_tags":
            index = self.tags_combo.findData(current_selection)
            if index >= 0:
                self.tags_combo.setCurrentIndex(index)
    
    def refresh_extensions(self):
        """重新載入擴充功能"""
        self.load_extensions_index()
    
    def search_extensions(self):
        """搜尋擴充功能"""
        self.apply_filters()
    
    def clear_search(self):
        """清除搜尋內容"""
        self.search_input.clear()
        self.apply_filters()
    
    def filter_by_tag(self, tag):
        """根據標籤快速篩選"""
        index = self.tags_combo.findData(tag)
        if index >= 0:
            self.tags_combo.setCurrentIndex(index)
    
    def clear_all_filters(self):
        """清除所有過濾條件"""
        self.search_input.clear()
        self.filter_combo.setCurrentIndex(0)
        self.tags_combo.setCurrentIndex(0)
        self.sort_combo.setCurrentIndex(0)
        self.apply_filters()
    
    def sort_extensions(self):
        """排序擴充功能並重新創建卡片"""
        sort_type = self.sort_combo.currentData()
        if not sort_type or not self.extensions_data:
            return
        sorted_extensions = self.extensions_data.copy()
        try:
            if sort_type == "name_asc":
                sorted_extensions.sort(key=lambda x: x.get('name', '').lower())
            elif sort_type == "name_desc":
                sorted_extensions.sort(key=lambda x: x.get('name', '').lower(), reverse=True)
            elif sort_type == "author_asc":
                sorted_extensions.sort(key=lambda x: x.get('author', '').lower())
            elif sort_type == "author_desc":
                sorted_extensions.sort(key=lambda x: x.get('author', '').lower(), reverse=True)
            elif sort_type == "date_desc":
                sorted_extensions.sort(key=lambda x: x.get('added_date', ''), reverse=True)
            elif sort_type == "date_asc":
                sorted_extensions.sort(key=lambda x: x.get('added_date', ''))
            elif sort_type == "install_status":
                def get_install_status(ext):
                    repo_name = self.extract_repo_name_from_url(ext.get('url', ''))
                    extension_path = os.path.join("extensions", repo_name)
                    return 0 if os.path.exists(extension_path) else 1
                sorted_extensions.sort(key=get_install_status)
            self.create_sorted_extension_cards(sorted_extensions)
        except Exception as e:
            logger.error(f"排序擴充功能時出錯: {str(e)}")
    
    def extract_repo_name_from_url(self, url):
        """從Git URL提取倉庫名稱（用於排序）"""
        if url.endswith('.git'):
            url = url[:-4]
        return url.split('/')[-1] if url else "unknown"
    
    def create_sorted_extension_cards(self, sorted_extensions):
        """根據排序後的數據創建擴充功能卡片"""
        for card in self.extension_cards:
            card.setParent(None)
            card.deleteLater()
        self.extension_cards.clear()
        for ext_data in sorted_extensions:
            card = ExtensionCard(ext_data, self)
            self.extension_cards.append(card)
            self.scroll_layout.insertWidget(self.scroll_layout.count() - 1, card)
        self.apply_filters()
    
    def apply_filters(self):
        """應用搜尋和過濾條件"""
        search_text = self.search_input.text().lower().strip()
        filter_type = self.filter_combo.currentData()
        selected_tag = self.tags_combo.currentData()
        visible_count = 0
        for card in self.extension_cards:
            show_by_filter = True
            if filter_type == "installed":
                repo_name = card.extract_repo_name(card.extension_data['url'])
                extension_path = os.path.join("extensions", repo_name)
                show_by_filter = os.path.exists(extension_path)
            elif filter_type == "not_installed":
                repo_name = card.extract_repo_name(card.extension_data['url'])
                extension_path = os.path.join("extensions", repo_name)
                show_by_filter = not os.path.exists(extension_path)
            show_by_tag = True
            if selected_tag and selected_tag != "all_tags":
                extension_tags = card.extension_data.get('tags', [])
                if isinstance(extension_tags, list):
                    show_by_tag = selected_tag in extension_tags
                else:
                    show_by_tag = False
            show_by_search = True
            if search_text:
                extension_data = card.extension_data
                searchable_text = []
                searchable_text.append(extension_data.get('name', '').lower())
                searchable_text.append(extension_data.get('description', '').lower())
                searchable_text.append(extension_data.get('author', '').lower())
                tags = extension_data.get('tags', [])
                if isinstance(tags, list):
                    searchable_text.extend([tag.lower() for tag in tags])
                show_by_search = any(search_text in text for text in searchable_text)
            should_show = show_by_filter and show_by_tag and show_by_search
            card.setVisible(should_show)
            if should_show:
                visible_count += 1
        filter_info = []
        if search_text:
            filter_info.append(f"搜尋「{search_text}」")
        if filter_type != "all":
            filter_info.append(f"狀態: {self.filter_combo.currentText()}")
        if selected_tag and selected_tag != "all_tags":
            filter_info.append(f"標籤: {selected_tag}")
        sort_info = f"排序: {self.sort_combo.currentText()}"
        if filter_info:
            self.result_label.setText(f"顯示 {visible_count} 個擴充功能（{', '.join(filter_info)}，{sort_info}）")
        else:
            self.result_label.setText(f"顯示 {visible_count} 個擴充功能（{sort_info}）")
    
    def filter_extensions(self):
        """保留向後兼容性的過濾方法"""
        self.apply_filters()
    
    def show_add_custom_dialog(self):
        """顯示添加自訂擴充功能對話框"""
        dialog = AddCustomExtensionDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            extension_data = dialog.get_extension_data()
            if not extension_data['name'] or not extension_data['url']:
                QMessageBox.warning(self, "輸入錯誤", "名稱和Git URL是必填欄位。")
                return
            try:
                index_path = os.path.join("config", "extensions_index.json")
                if os.path.exists(index_path):
                    with open(index_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                else:
                    data = {"extensions": []}
                data["extensions"].append(extension_data)
                os.makedirs(os.path.dirname(index_path), exist_ok=True)
                with open(index_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                self.load_extensions_index()
                QMessageBox.information(self, "成功", "自訂擴充功能已添加。")
            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"添加擴充功能時發生錯誤：\n{str(e)}")
