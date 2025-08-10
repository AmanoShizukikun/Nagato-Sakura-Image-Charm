import os
import logging
from pathlib import Path
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, 
                            QPushButton, QLabel, QCheckBox, QComboBox, QFrame)
from PyQt6.QtCore import QTimer, QFileSystemWatcher, pyqtSignal, Qt
from PyQt6.QtGui import QIcon, QFont, QTextCursor

logger = logging.getLogger(__name__)

class LogViewerDialog(QDialog):
    """日誌檔案監視器對話框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("日誌記錄監視器")
        try:
            current_directory = Path(__file__).resolve().parent.parent.parent
            icon_path = current_directory / "assets" / "icon" / "1.3.0.ico"
            if icon_path.exists():
                self.setWindowIcon(QIcon(str(icon_path)))
            else:
                self.setWindowIcon(QIcon.fromTheme("text-x-generic", QIcon.fromTheme("document-properties")))
        except Exception:
            self.setWindowIcon(QIcon())
        self.resize(800, 600)
        self.setModal(False)
        self.setWindowFlags(
            Qt.WindowType.Window | 
            Qt.WindowType.WindowMinMaxButtonsHint | 
            Qt.WindowType.WindowCloseButtonHint |
            Qt.WindowType.WindowTitleHint
        )
        self.setMinimumSize(400, 300)
        self.log_file_path = self.get_log_file_path()
        self.file_watcher = QFileSystemWatcher()
        if os.path.exists(self.log_file_path):
            self.file_watcher.addPath(self.log_file_path)
        self.file_watcher.fileChanged.connect(self.on_file_changed)
        self.auto_scroll = True
        self.font_size = 10
        self.setup_ui()
        self.load_log_content()
        self.rewatch_timer = QTimer()
        self.rewatch_timer.setSingleShot(True)
        self.rewatch_timer.timeout.connect(self.rewatch_file)
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.check_file_update)
        self.refresh_timer.start(1000)
        self.last_modified = 0
        if os.path.exists(self.log_file_path):
            self.last_modified = os.path.getmtime(self.log_file_path)
    
    def get_log_file_path(self):
        """獲取日誌檔案路徑"""
        current_directory = Path(__file__).resolve().parent.parent.parent
        return str(current_directory / "logs" / "app.log")
    
    def setup_ui(self):
        """設定用戶介面"""
        layout = QVBoxLayout()
        control_frame = QFrame()
        control_layout = QHBoxLayout(control_frame)
        path_label = QLabel(f"日誌檔案：{self.log_file_path}")
        path_label.setStyleSheet("font-size: 10px; color: #666;")
        control_layout.addWidget(path_label)
        control_layout.addStretch()
        level_label = QLabel("等級篩選：")
        control_layout.addWidget(level_label)
        self.level_combo = QComboBox()
        self.level_combo.addItems(["全部", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.level_combo.setCurrentText("全部")
        self.level_combo.currentTextChanged.connect(self.filter_logs)
        control_layout.addWidget(self.level_combo)
        self.auto_scroll_checkbox = QCheckBox("自動滾動")
        self.auto_scroll_checkbox.setChecked(True)
        self.auto_scroll_checkbox.toggled.connect(self.toggle_auto_scroll)
        control_layout.addWidget(self.auto_scroll_checkbox)
        font_label = QLabel("字體：")
        control_layout.addWidget(font_label)
        font_smaller_btn = QPushButton("A-")
        font_smaller_btn.setMaximumWidth(30)
        font_smaller_btn.clicked.connect(self.decrease_font_size)
        control_layout.addWidget(font_smaller_btn)
        font_larger_btn = QPushButton("A+")
        font_larger_btn.setMaximumWidth(30)
        font_larger_btn.clicked.connect(self.increase_font_size)
        control_layout.addWidget(font_larger_btn)
        clear_btn = QPushButton("清除")
        clear_btn.clicked.connect(self.clear_display)
        control_layout.addWidget(clear_btn)
        reload_btn = QPushButton("重載")
        reload_btn.clicked.connect(self.load_log_content)
        control_layout.addWidget(reload_btn)
        layout.addWidget(control_frame)
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.update_font()
        self.log_display.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3e3e3e;
            }
        """)
        layout.addWidget(self.log_display)
        self.status_label = QLabel("準備就緒")
        self.status_label.setStyleSheet("font-size: 10px; color: #888; padding: 5px;")
        layout.addWidget(self.status_label)
        self.setLayout(layout)
    
    def load_log_content(self):
        """載入日誌檔案內容"""
        try:
            if os.path.exists(self.log_file_path):
                with open(self.log_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.log_display.setPlainText(content)
                    if self.auto_scroll:
                        self.scroll_to_bottom()
                    self.status_label.setText(f"已載入日誌檔案 - {os.path.getsize(self.log_file_path)} 位元組")
                self.last_modified = os.path.getmtime(self.log_file_path)
            else:
                self.log_display.setPlainText("日誌檔案尚未建立")
                self.status_label.setText("日誌檔案不存在")
                self.last_modified = 0
        except Exception as e:
            logger.error(f"載入日誌檔案失敗: {str(e)}")
            self.log_display.setPlainText(f"載入日誌檔案失敗: {str(e)}")
            self.status_label.setText("載入失敗")
    
    def on_file_changed(self, path):
        """檔案變更時的處理"""
        QTimer.singleShot(100, self.load_log_content)
        if not os.path.exists(path):
            self.rewatch_timer.start(1000)
    
    def rewatch_file(self):
        """重新監視檔案"""
        if os.path.exists(self.log_file_path):
            paths = self.file_watcher.files()
            if paths:
                self.file_watcher.removePaths(paths)
            self.file_watcher.addPath(self.log_file_path)
            self.load_log_content()
    
    def check_file_update(self):
        """定期檢查檔案是否有更新（備用機制）"""
        try:
            if os.path.exists(self.log_file_path):
                current_modified = os.path.getmtime(self.log_file_path)
                if current_modified > self.last_modified:
                    self.last_modified = current_modified
                    self.load_log_content()
        except Exception as e:
            pass
    
    def scroll_to_bottom(self):
        """滾動到底部"""
        cursor = self.log_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_display.setTextCursor(cursor)
    
    def toggle_auto_scroll(self, checked):
        """切換自動滾動"""
        self.auto_scroll = checked
        if checked:
            self.scroll_to_bottom()
    
    def filter_logs(self, level):
        """根據等級篩選日誌"""
        if level == "全部":
            self.load_log_content()
            return
        try:
            if os.path.exists(self.log_file_path):
                with open(self.log_file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                filtered_lines = []
                for line in lines:
                    if f" - {level} - " in line:
                        filtered_lines.append(line)
                self.log_display.setPlainText(''.join(filtered_lines))
                if self.auto_scroll:
                    self.scroll_to_bottom()
                self.status_label.setText(f"已篩選 {level} 等級日誌 - {len(filtered_lines)} 條記錄")
        except Exception as e:
            logger.error(f"篩選日誌失敗: {str(e)}")
            self.status_label.setText("篩選失敗")
    
    def clear_display(self):
        """清除顯示內容"""
        self.log_display.clear()
        self.status_label.setText("顯示已清除")
    
    def update_font(self):
        """更新字體"""
        font = QFont("Consolas", self.font_size)
        self.log_display.setFont(font)
    
    def increase_font_size(self):
        """增加字體大小"""
        if self.font_size < 20:
            self.font_size += 1
            self.update_font()
    
    def decrease_font_size(self):
        """減少字體大小"""
        if self.font_size > 6:
            self.font_size -= 1
            self.update_font()
    
    def closeEvent(self, event):
        """關閉事件"""
        if hasattr(self, 'refresh_timer'):
            self.refresh_timer.stop()
        if hasattr(self, 'rewatch_timer'):
            self.rewatch_timer.stop()
        self.file_watcher.removePaths(self.file_watcher.files())
        event.accept()
