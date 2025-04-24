import os
import platform
import time
import torch
import psutil
import numpy as np
import cpuinfo
import logging
import winreg
import pyqtgraph as pg
from datetime import datetime
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QWidget,
    QPushButton, QTextEdit, QProgressBar, QStackedWidget, QFormLayout,
    QSpinBox, QMessageBox, QFrame, QTabWidget, QSplitter, QSizePolicy,
    QSpacerItem, QGridLayout, QStyle
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt6.QtGui import QGuiApplication, QFont, QIcon

from src.ui.main_window import ImageEnhancerApp
from src.processing.NS_Benchmark import BenchmarkProcessor, BenchmarkWorker


logger = logging.getLogger(__name__)

# --- 主介面 ---
class CardFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CardFrame")
        self.setStyleSheet("""
            #CardFrame {
                background-color: #23272e;
                border-radius: 12px;
                border: 1.5px solid #3a3f4b;
            }
        """)

class InfoCard(CardFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.items = {}
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        title_label = QLabel(self.title)
        title_label.setStyleSheet("color: #00bfff; font-size: 13pt; font-weight: bold;")
        layout.addWidget(title_label)
        layout.addSpacing(8)
        self.form_layout = QFormLayout()
        self.form_layout.setVerticalSpacing(10)
        self.form_layout.setHorizontalSpacing(18)
        self.form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addLayout(self.form_layout)
        layout.addStretch()

    def addItem(self, key, label):
        value_label = QLabel("長門櫻正在讀取中...")
        value_label.setStyleSheet("color: #e0e0e0;")
        value_label.setWordWrap(True)
        self.form_layout.addRow(f"{label}：", value_label)
        self.items[key] = value_label

    def setValue(self, key, value):
        if key in self.items:
            self.items[key].setText(value)
            self.items[key].setToolTip(value)

class MetricCard(CardFrame):
    def __init__(self, title, value="--", icon=None, parent=None):
        super().__init__(parent)
        self.title = title
        self.value = value
        self.icon = icon
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        title_layout = QHBoxLayout()
        if self.icon:
            icon_label = QLabel()
            icon_label.setPixmap(self.icon.pixmap(QSize(18, 18)))
            title_layout.addWidget(icon_label)
        title_label = QLabel(self.title)
        title_label.setStyleSheet("color: #b0b8c1; font-size: 10.5pt;")
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        self.value_label = QLabel(self.value)
        self.value_label.setStyleSheet("color: #00bfff; font-size: 20pt; font-weight: bold;")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addLayout(title_layout)
        layout.addWidget(self.value_label)

    def setValue(self, value):
        self.value = value
        self.value_label.setText(value)

# --- 分數動畫 ---
class ScoreDisplay(CardFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._score = 0
        self.targetScore = 0
        self.animation = None
        self.setMinimumHeight(180)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label = QLabel("長門櫻測量的性能分數")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 15pt; color: #b0b8c1; font-weight: bold;")
        self.score_label = QLabel("等待測試")
        font = QFont()
        font.setPointSize(60)
        font.setBold(True)
        self.score_label.setFont(font)
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.score_label.setStyleSheet("color: #00bfff;")
        layout.addWidget(title_label)
        layout.addWidget(self.score_label)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("background-color: #444;")
        self.desc_label = QLabel("長門櫻會為主人量測設備性能，分數越高越好")
        self.desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.desc_label.setStyleSheet("font-size: 10pt; color: #888;")
        layout.addWidget(line)
        layout.addWidget(self.desc_label)

    @pyqtProperty(int)
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        self._score = value
        self.score_label.setText(f"{int(value)}")

    def setScore(self, score):
        self.targetScore = score
        if self.animation is not None:
            self.animation.stop()
        self._score = 0
        self.animation = QPropertyAnimation(self, b"score")
        self.animation.setDuration(1200)
        self.animation.setStartValue(0)
        self.animation.setEndValue(score)
        self.animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.animation.start()

# --- 結果圖表 ---
class ResultChart(CardFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.setMinimumHeight(220)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#23272e')
        self.plot_widget.setLabel('left', '處理時間 (秒)', color='#b0b8c1')
        self.plot_widget.setLabel('bottom', '處理次數', color='#b0b8c1')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.getAxis('left').setTextPen('#b0b8c1')
        self.plot_widget.getAxis('bottom').setTextPen('#b0b8c1')
        self.plot_widget.getAxis('left').setPen(pg.mkPen(color='#444'))
        self.plot_widget.getAxis('bottom').setPen(pg.mkPen(color='#444'))
        layout.addWidget(self.plot_widget)

    def updateChart(self, data, avg_time, is_real_usage=False):
        self.plot_widget.clear()
        if not data:
            return
        x_label = "圖片編號" if is_real_usage else "迭代次數"
        num_points = len(data)
        x_axis = list(range(1, num_points + 1))
        self.plot_widget.setLabel('bottom', x_label)
        bargraph = pg.BarGraphItem(x=x_axis, height=data, width=0.7, brush='#00bfff')
        self.plot_widget.addItem(bargraph)
        avg_line = pg.InfiniteLine(pos=avg_time, angle=0, movable=False,
                                   pen=pg.mkPen(color='#50c878', width=2, style=Qt.PenStyle.DashLine))
        self.plot_widget.addItem(avg_line)
        avg_text = pg.TextItem(f"平均值: {avg_time:.4f}秒", color='#50c878', anchor=(0, 1))
        avg_text.setPos(x_axis[-1] * 0.85, avg_time)
        self.plot_widget.addItem(avg_text)
        self.plot_widget.setXRange(0, num_points + 1)
        min_y = min(data) * 0.9 if data else 0
        max_y = max(data) * 1.1 if data else 1
        self.plot_widget.setYRange(min_y, max_y)

# --- 主對話框 ---
class BenchmarkDialog(QDialog):
    """長門櫻會為主人精心設計的基準測試對話框"""
    def __init__(self, model_manager, parent=None):
        super().__init__(parent)
        self.model_manager = model_manager
        self.benchmark_processor = BenchmarkProcessor(self.model_manager)
        self.benchmark_worker = None
        self.results = {}
        self.setWindowTitle("基準測試")
        self.setMinimumWidth(980)
        self.setMinimumHeight(720)
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1d23;
                color: #e0e0e0;
            }
            QLabel {
                color: #e0e0e0;
            }
            QPushButton {
                background-color: #23272e;
                color: #e0e0e0;
                border: 1.5px solid #3a3f4b;
                border-radius: 6px;
                padding: 7px 18px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background-color: #2a2e36;
                border: 1.5px solid #00bfff;
                color: #00bfff;
            }
            QPushButton:pressed {
                background-color: #181b20;
            }
            QProgressBar {
                border: 1.5px solid #3a3f4b;
                border-radius: 5px;
                background-color: #23272e;
                text-align: center;
                color: #e0e0e0;
            }
            QProgressBar::chunk {
                background-color: #00bfff;
                width: 1px;
            }
            QTabWidget::pane {
                border: none;
            }
            QTabBar::tab {
                background-color: #23272e;
                color: #b0b8c1;
                border: 1.5px solid #3a3f4b;
                border-bottom: none;
                border-top-left-radius: 7px;
                border-top-right-radius: 7px;
                padding: 7px 18px;
                font-size: 11pt;
            }
            QTabBar::tab:selected {
                background-color: #1a1d23;
                color: #00bfff;
                border-bottom: 1.5px solid #1a1d23;
            }
            QTabBar::tab:hover:!selected {
                background-color: #2a2e36;
            }
        """)
        self.initUI()
        self.populateSystemInfo()
        self.checkMemoryAvailability()

    def initUI(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)
        title = QLabel("長門櫻的基準測試中心")
        title.setStyleSheet("font-size: 20pt; font-weight: bold; color: #00bfff;")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        subtitle = QLabel("長門櫻會幫主人評估系統處理圖像時的性能，並提供詳細的分析報告")
        subtitle.setStyleSheet("color: #b0b8c1; font-size: 11.5pt;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignLeft)
        version = QLabel(f"版本: {ImageEnhancerApp.version}")
        version.setStyleSheet("color: #888; font-size: 10pt;")
        version.setAlignment(Qt.AlignmentFlag.AlignLeft)
        main_layout.addWidget(title)
        main_layout.addWidget(subtitle)
        main_layout.addWidget(version)
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self.createSetupPage(), "測試設定")
        self.tabs.addTab(self.createResultPage(), "測試結果")
        self.tabs.setCurrentIndex(0)
        main_layout.addWidget(self.tabs, 1)
        status_layout = QHBoxLayout()
        self.status_label = QLabel("請設定參數後點擊「開始測試」按鈕，長門櫻會立即為主人的系統進行性能評分")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        status_layout.addWidget(self.status_label, 2)
        status_layout.addWidget(self.progress_bar, 1)
        main_layout.addLayout(status_layout)
        btn_layout = QHBoxLayout()
        self.copy_button = QPushButton("複製結果")
        self.copy_button.setIcon(QIcon.fromTheme("edit-copy"))
        self.copy_button.setEnabled(False)
        self.copy_button.clicked.connect(self.copyResults)
        self.stop_button = QPushButton("停止測試")
        self.stop_button.setIcon(QIcon.fromTheme("process-stop"))
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("background-color: #d9534f; color: white;")
        self.stop_button.clicked.connect(self.stopBenchmark)
        self.start_button = QPushButton("開始測試")
        self.start_button.setIcon(QIcon.fromTheme("media-playback-start"))
        self.start_button.setStyleSheet("background-color: #00bfff; color: white; font-weight: bold;")
        self.start_button.setMinimumHeight(36)
        self.start_button.clicked.connect(self.startBenchmark)
        self.close_button = QPushButton("關閉")
        self.close_button.setIcon(QIcon.fromTheme("window-close"))
        self.close_button.clicked.connect(self.close)
        btn_layout.addWidget(self.copy_button)
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.stop_button)
        btn_layout.addWidget(self.start_button)
        btn_layout.addWidget(self.close_button)
        main_layout.addLayout(btn_layout)

    def createSetupPage(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setSpacing(18)
        self.system_card = InfoCard("主人的系統資訊")
        for key, label in [
            ("os", "作業系統"), ("cpu", "處理器"), ("memory", "記憶體"),
            ("gpu", "顯示卡"), ("gpu_memory", "顯示記憶體"),
            ("version", "程式版本"), ("pytorch", "PyTorch版本")
        ]:
            self.system_card.addItem(key, label)
        layout.addWidget(self.system_card, 2)
        settings_card = CardFrame()
        settings_layout = QVBoxLayout(settings_card)
        settings_layout.setContentsMargins(18, 14, 18, 14)
        settings_title = QLabel("測試參數設定")
        settings_title.setStyleSheet("color: #00bfff; font-size: 13pt; font-weight: bold;")
        settings_layout.addWidget(settings_title)
        form_layout = QFormLayout()
        form_layout.setVerticalSpacing(12)
        form_layout.setHorizontalSpacing(18)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.device_combo = QComboBox()
        self.device_combo.addItem("自動選擇", "auto")
        if torch.cuda.is_available():
            self.device_combo.addItem("CUDA (GPU)", "cuda")
        self.device_combo.addItem("CPU", "cpu")
        form_layout.addRow("計算設備：", self.device_combo)
        self.amp_combo = QComboBox()
        self.amp_combo.addItems(["自動偵測", "強制啟用", "強制禁用"])
        self.amp_combo.setToolTip("自動偵測: 長門櫻會根據主人的GPU決定\n強制啟用: 較快但精度較低\n強制禁用: 較慢但精度最高")
        form_layout.addRow("混合精度：", self.amp_combo)
        self.test_mode_combo = QComboBox()
        self.test_mode_combo.addItems(["實際場景測試", "模型推理測試"])
        self.test_mode_combo.currentIndexChanged.connect(self.updateSettingsVisibility)
        form_layout.addRow("測試模式：", self.test_mode_combo)
        self.iter_widget = QWidget()
        iter_layout = QHBoxLayout(self.iter_widget)
        iter_layout.setContentsMargins(0, 0, 0, 0)
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(5, 100)
        self.iterations_spin.setValue(10)
        iter_layout.addWidget(self.iterations_spin)
        self.image_widget = QWidget()
        image_layout = QHBoxLayout(self.image_widget)
        image_layout.setContentsMargins(0, 0, 0, 0)
        self.num_images_spin = QSpinBox()
        self.num_images_spin.setRange(1, 50)
        self.num_images_spin.setValue(10)
        image_layout.addWidget(self.num_images_spin)
        form_layout.addRow("測試次數：", self.iter_widget)
        form_layout.addRow("處理圖片數：", self.image_widget)
        self.block_widget = QWidget()
        block_layout = QGridLayout(self.block_widget)
        block_layout.setContentsMargins(0, 0, 0, 0)
        self.block_size_spin = QSpinBox()
        self.block_size_spin.setRange(128, 512)
        self.block_size_spin.setValue(256)
        self.block_size_spin.setSingleStep(32)
        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(16, 256)
        self.overlap_spin.setValue(128)
        self.overlap_spin.setSingleStep(16)
        block_layout.addWidget(QLabel("區塊大小:"), 0, 0)
        block_layout.addWidget(self.block_size_spin, 0, 1)
        block_layout.addWidget(QLabel("重疊大小:"), 1, 0)
        block_layout.addWidget(self.overlap_spin, 1, 1)
        form_layout.addRow("區塊處理設定：", self.block_widget)
        self.memory_warning = QLabel("")
        self.memory_warning.setStyleSheet("color: #ff9f43; font-style: italic;")
        self.memory_warning.setWordWrap(True)
        settings_layout.addLayout(form_layout)
        settings_layout.addSpacing(8)
        settings_layout.addWidget(self.memory_warning)
        settings_layout.addStretch()
        layout.addWidget(settings_card, 3)
        return page

    def createResultPage(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(14)
        self.score_display = ScoreDisplay()
        layout.addWidget(self.score_display)
        metrics_layout = QHBoxLayout()
        self.fps_card = MetricCard("平均處理幀率", "--")
        self.time_card = MetricCard("平均處理時間", "--")
        self.mpps_card = MetricCard("像素處理速度", "--")
        self.total_card = MetricCard("總測試時間", "--")
        metrics_layout.addWidget(self.fps_card)
        metrics_layout.addWidget(self.time_card)
        metrics_layout.addWidget(self.mpps_card)
        metrics_layout.addWidget(self.total_card)
        layout.addLayout(metrics_layout)
        tabs = QTabWidget()
        chart_tab = QWidget()
        chart_layout = QVBoxLayout(chart_tab)
        self.result_chart = ResultChart()
        chart_layout.addWidget(self.result_chart)
        details_tab = QWidget()
        details_layout = QVBoxLayout(details_tab)
        self.details_info = QTextEdit()
        self.details_info.setReadOnly(True)
        self.details_info.setStyleSheet("""
            QTextEdit {
                background-color: #23272e;
                color: #e0e0e0;
                border: 1.5px solid #3a3f4b;
                font-family: monospace;
                font-size: 10.5pt;
            }
        """)
        details_layout.addWidget(self.details_info)
        tabs.addTab(chart_tab, "性能圖表")
        tabs.addTab(details_tab, "詳細資訊")
        layout.addWidget(tabs)
        return page

    # --- UI 行為 ---
    def updateSettingsVisibility(self):
        is_real_usage = self.test_mode_combo.currentText() == "實際場景測試"
        self.iter_widget.setVisible(not is_real_usage)
        self.image_widget.setVisible(is_real_usage)
        self.block_widget.setEnabled(is_real_usage)
        if hasattr(self, 'is_low_memory_gpu') and self.is_low_memory_gpu:
            if not is_real_usage:
                self.memory_warning.setText("長門櫻提醒主人：低顯存環境下，模型推理測試可能會導致失敗，建議使用實際場景測試")
            else:
                self.memory_warning.setText(f"長門櫻已為主人啟動低顯存模式 ({self.total_gpu_memory:.2f} GB)，並自動調整了參數")
        else:
            self.memory_warning.setText("")

    def populateSystemInfo(self):
        try:
            system = platform.system()
            architecture = platform.architecture()[0]
            os_info = f"{system} {platform.release()} {architecture}"
            if system == "Windows":
                try:
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion") as key:
                        build_number = int(winreg.QueryValueEx(key, "CurrentBuildNumber")[0])
                        product_name = winreg.QueryValueEx(key, "ProductName")[0]
                        windows_version = "11" if build_number >= 22000 else "10"
                        os_info = f"{product_name} ({windows_version} Build {build_number}) {architecture}"
                except Exception as e:
                    logger.warning(f"長門櫻無法從註冊表獲取 Windows 版本: {str(e)}")
            self.system_card.setValue("os", os_info)
            try:
                cpu_info_dict = cpuinfo.get_cpu_info()
                cpu_brand = cpu_info_dict.get('brand_raw', platform.processor() or "未知 CPU")
            except Exception:
                cpu_brand = platform.processor() or "未知 CPU"
            cpu_count = psutil.cpu_count(logical=True)
            physical_cpu_count = psutil.cpu_count(logical=False)
            cpu_info = f"{cpu_brand} ({physical_cpu_count} 核心, {cpu_count} 執行緒)"
            self.system_card.setValue("cpu", cpu_info)
            total_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)
            available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)
            memory_info = f"{total_memory:.2f} GB"
            self.system_card.setValue("memory", memory_info)
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                compute_capability = torch.cuda.get_device_capability(0)
                gpu_info = f"{gpu_name} (CC {compute_capability[0]}.{compute_capability[1]})"
                self.system_card.setValue("gpu", gpu_info)
                total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                allocated = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_info = f"{total_gpu_mem:.2f} GB (已分配 {allocated:.2f} GB)"
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    total_gpu_mem_nv = mem_info.total / (1024**3)
                    used_gpu_mem = mem_info.used / (1024**3)
                    gpu_memory_info = f"{total_gpu_mem_nv:.2f} GB"
                    pynvml.nvmlShutdown()
                except (ImportError, Exception):
                    pass
                self.system_card.setValue("gpu_memory", gpu_memory_info)
            else:
                self.system_card.setValue("gpu", "長門櫻未檢測到支援的顯示卡")
                self.system_card.setValue("gpu_memory", "不適用")
            self.system_card.setValue("version", ImageEnhancerApp.version)
            self.system_card.setValue("pytorch", torch.__version__)
        except Exception as e:
            logger.error(f"長門櫻在獲取系統資訊時遇到困難: {str(e)}")
            for key in ["os", "cpu", "memory", "gpu", "gpu_memory"]:
                self.system_card.setValue(key, "長門櫻讀取失敗")

    def checkMemoryAvailability(self):
        if not torch.cuda.is_available():
            return
        try:
            total_memory = 0
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_memory = gpu_memory.total / (1024 * 1024 * 1024)
                pynvml.nvmlShutdown()
            except (ImportError, Exception) as e:
                logger.warning(f"長門櫻無法使用 pynvml 獲取顯存資訊: {str(e)}")
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)
            self.total_gpu_memory = total_memory
            if total_memory < 3:
                self.is_low_memory_gpu = True
                warning_text = f"長門櫻發現主人的顯卡只有 {total_memory:.2f} GB 顯存，已自動為主人您調整參數以適應您的設備"
                self.memory_warning.setText(warning_text)
                self.test_mode_combo.setCurrentText("實際場景測試")
                if self.block_size_spin.value() > 192:
                    self.block_size_spin.setValue(192)
                if self.overlap_spin.value() > 32:
                    self.overlap_spin.setValue(32)
                if self.num_images_spin.value() > 5:
                    self.num_images_spin.setValue(5)
            else:
                self.is_low_memory_gpu = False
        except Exception as e:
            logger.error(f"長門櫻在檢查顯存時出錯: {str(e)}")
        finally:
            self.updateSettingsVisibility()

    # --- 執行測試 ---
    def startBenchmark(self):
        try:
            device_selection = self.device_combo.currentData()
            device = self.model_manager.get_device() if device_selection == "auto" else torch.device(device_selection)
            if not self.model_manager.get_registered_model_path():
                QMessageBox.warning(self, "長門櫻提醒", "主人~請先在主界面選擇一個模型，長門櫻才能為您執行基準測試")
                return
            amp_setting = self.amp_combo.currentText()
            use_amp = None
            if amp_setting == "強制啟用":
                use_amp = True
            elif amp_setting == "強制禁用":
                use_amp = False
            is_real_usage = self.test_mode_combo.currentText() == "實際場景測試"
            iterations = self.iterations_spin.value()
            num_images = self.num_images_spin.value()
            block_size = self.block_size_spin.value()
            overlap = self.overlap_spin.value()
            if device.type == 'cuda' and hasattr(self, 'is_low_memory_gpu') and self.is_low_memory_gpu and not is_real_usage:
                reply = QMessageBox.warning(
                    self, "長門櫻的顯存警告",
                    f"主人~您的顯卡只有 {self.total_gpu_memory:.2f} GB 顯存，在「模型推理測試」模式下可能會失敗\n\n長門櫻建議主人使用「實際場景測試」\n\n主人確定要繼續嗎？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    return
            self.start_button.setEnabled(False)
            self.start_button.setText("測試進行中...")
            self.stop_button.setEnabled(True)
            self.copy_button.setEnabled(False)
            self.close_button.setEnabled(False)
            self.tabs.setCurrentIndex(0)
            self.status_label.setText("長門櫻正在主人的系統準備測試環境...")
            self.progress_bar.setValue(0)
            self.benchmark_worker = BenchmarkWorker(
                self.benchmark_processor,
                device,
                1280,
                640, 
                iterations,
                is_real_usage,
                block_size,
                overlap,
                num_images,
                use_amp
            )
            self.benchmark_worker.progress_signal.connect(self.updateProgress)
            self.benchmark_worker.step_signal.connect(self.updateStatus)
            self.benchmark_worker.finished_signal.connect(self.onBenchmarkFinished)
            self.benchmark_worker.start()
        except Exception as e:
            QMessageBox.critical(self, "長門櫻報告錯誤", f"長門櫻在執行基準測試時遇到困難: {str(e)}")
            self.start_button.setEnabled(True)
            self.start_button.setText("開始測試")
            self.stop_button.setEnabled(False)
            self.close_button.setEnabled(True)

    def stopBenchmark(self):
        if self.benchmark_worker and self.benchmark_worker.isRunning():
            self.status_label.setText("長門櫻正在停止測試...")
            self.benchmark_worker.stop()
            self.stop_button.setEnabled(False)

    def updateProgress(self, current, total):
        if total > 0:
            progress_percent = int(current / total * 100)
            self.progress_bar.setValue(progress_percent)
        else:
            self.progress_bar.setValue(0)

    def updateStatus(self, message):
        self.status_label.setText(message)

    def onBenchmarkFinished(self, results):
        self.start_button.setEnabled(True)
        self.start_button.setText("開始測試")
        self.stop_button.setEnabled(False)
        self.close_button.setEnabled(True)
        self.progress_bar.setValue(100 if "error" not in results else 0)
        if "error" in results:
            error_msg = f"長門櫻測試失敗: {results['error']}"
            QMessageBox.critical(self, "長門櫻報告錯誤", error_msg)
            self.status_label.setText(error_msg)
            return
        self.results = results
        self.displayResults()
        self.copy_button.setEnabled(True)
        self.tabs.setCurrentIndex(1)
        self.status_label.setText("長門櫻已完成基準測試，請主人查看結果")

    def displayResults(self):
        if not self.results:
            return
        score = self.results.get('score', 0)
        avg_time = self.results.get('average_time', 0)
        fps = self.results.get('fps', 0)
        mpps = self.results.get('megapixels_per_second', 0)
        total_time = self.results.get('total_time', 0)
        self.score_display.setScore(score)
        self.fps_card.setValue(f"{fps:.2f} FPS")
        self.time_card.setValue(f"{avg_time:.4f} 秒")
        self.mpps_card.setValue(f"{mpps:.2f} MP/s")
        self.total_card.setValue(f"{total_time:.2f} 秒")
        times = self.results.get("times", [])
        is_real_usage = self.results.get("test_type", "") == "實際場景測試"
        self.result_chart.updateChart(times, avg_time, is_real_usage)
        self.generateTextSummary()

    def generateTextSummary(self):
        if not self.results:
            self.details_info.setText("抱歉啊主人~長門櫻還沒有可以呈現的結果")
            return
        device_name = "CPU"
        if str(self.results.get("device", "")) == "cuda":
            try:
                device_name = f"GPU ({torch.cuda.get_device_name(0)})"
            except:
                device_name = "GPU (未知型號)"
        times = self.results.get("times", [])
        min_time = min(times) if times else 0
        max_time = max(times) if times else 0
        std_dev = np.std(times) if times else 0
        test_type = self.results.get("test_type", "未知類型測試")
        is_real_usage = test_type == "實際場景測試"
        summary = f"""=== 長門櫻的基準測試結果報告 ===

測試類型: {test_type}
測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

建構資訊:
  程式版本: {ImageEnhancerApp.version}
  PyTorch 版本: {torch.__version__}

主人的系統資訊:
  作業系統: {self.system_card.items["os"].text()}
  處理器: {self.system_card.items["cpu"].text()}
  記憶體: {self.system_card.items["memory"].text()}
  顯示卡: {self.system_card.items["gpu"].text()}
  顯示記憶體: {self.system_card.items["gpu_memory"].text()}

處理設定:
  計算設備: {device_name}
  輸入解析度: {self.results.get('resolution', 'N/A')}
  {'處理圖片數' if is_real_usage else '測試次數'}: {self.results.get('iterations', 'N/A')}
"""
        if is_real_usage:
            summary += f"""  區塊大小: {self.results.get('block_size', 'N/A')}
  重疊大小: {self.results.get('overlap', 'N/A')}
  混合精度計算: {'是' if self.results.get('amp_used', False) else '否'}
"""
        summary += f"""
長門櫻的基準測試結果:
  性能分數: {self.results.get('score', 'N/A')}
  平均處理時間: {self.results.get('average_time', 0):.4f} 秒
  最快處理時間: {min_time:.4f} 秒
  最慢處理時間: {max_time:.4f} 秒
  處理時間標準差: {std_dev:.6f}
  每秒處理幀數: {self.results.get('fps', 0):.2f} FPS
  每秒處理像素: {self.results.get('megapixels_per_second', 0):.2f} MP/s
  總測試時間: {self.results.get('total_time', 0):.2f} 秒
"""
        if is_real_usage:
            summary += f"""
記憶體使用統計:
  記憶體變化: {self.results.get('memory_delta', 0):.2f} MB
  顯存變化: {self.results.get('gpu_memory_delta', 0):.2f} MB
"""
        self.details_info.setText(summary.strip())

    def copyResults(self):
        if not self.results:
            QMessageBox.information(self, "長門櫻提醒", "主人~您目前尚未執行任何基準測試")
            return
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(self.details_info.toPlainText())
        QMessageBox.information(self, "長門櫻", "主人~長門櫻已經把測試結果複製到剪貼板囉")

    def closeEvent(self, event):
        if self.benchmark_worker and self.benchmark_worker.isRunning():
            reply = QMessageBox.question(self, "長門櫻請求確認",
                "主人~長門櫻還在為您進行基準測試中，確定要中止測試並關閉嗎？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.stopBenchmark()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()