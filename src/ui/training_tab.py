import os
import time
import logging
import numpy as np
from datetime import datetime, timedelta
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                          QFileDialog, QProgressBar, QSpinBox, QDoubleSpinBox, 
                          QTabWidget, QTextEdit, QGroupBox, QFormLayout, QLineEdit,
                          QCheckBox, QComboBox, QStyle, QScrollArea, QFrame, QSlider, QGridLayout,
                          QSplitter, QStackedWidget, QListWidget, QListWidgetItem, QSizePolicy, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QThread, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QPalette, QColor, QIcon

from src.processing.NS_DataProcessor import DataProcessor, ProcessingMode, BlurType, NoiseType
from src.training.NS_Trainer import Trainer
from src.utils.NS_VideoToImage import FrameExtractor


class TrainingWorker(QThread):
    progress_updated = pyqtSignal(int, int, int, float, float)
    epoch_completed = pyqtSignal(int, float, float, float)
    training_completed = pyqtSignal()
    training_error = pyqtSignal(str)
    
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer
        self.start_time = None
        self.is_running = True
        
    def run(self):
        try:
            self.start_time = time.time()
            self.trainer.train(
                progress_callback=self.update_progress,
                epoch_callback=self.epoch_complete,
                stop_check_callback=self.check_stop
            )
            self.training_completed.emit()
        except Exception as e:
            self.training_error.emit(str(e))
    
    def update_progress(self, epoch, batch, total_batches, g_loss, d_loss):
        self.progress_updated.emit(epoch, batch, total_batches, g_loss, d_loss)
        
    def epoch_complete(self, epoch, g_loss, d_loss, psnr):
        self.epoch_completed.emit(epoch, g_loss, d_loss, psnr)
    
    def check_stop(self):
        """檢查是否中止訓練的回調函數"""
        return not self.is_running
        
    def stop(self):
        """中止訓練"""
        self.is_running = False


class EnhancedDataProcessingWorker(QThread):
    progress_updated = pyqtSignal(int, int, float)
    processing_completed = pyqtSignal(dict)
    processing_error = pyqtSignal(str)
    stats_updated = pyqtSignal(dict)
    
    def __init__(self, processor, input_dir, output_dir):
        super().__init__()
        self.processor = processor
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.is_running = True
        self.start_time = None
        
    def run(self):
        try:
            self.start_time = time.time()
            
            def progress_callback(current, total):
                if not self.is_running:
                    return False
                elapsed = time.time() - self.start_time
                speed = current / elapsed if elapsed > 0 else 0
                self.progress_updated.emit(current, total, speed)
                if current % max(1, total // 20) == 0:
                    stats = self.processor.get_processing_stats()
                    self.stats_updated.emit(stats)
                return True
            processed_count = self.processor.process_images(
                self.input_dir, 
                self.output_dir, 
                progress_callback
            )
            if self.is_running:
                final_stats = self.processor.get_processing_stats()
                final_stats['total_processed'] = processed_count
                self.processing_completed.emit(final_stats)  
        except Exception as e:
            if self.is_running:
                self.processing_error.emit(str(e))
    
    def stop(self):
        """停止處理"""
        self.is_running = False


class DataProcessingWorker(QThread):
    progress_updated = pyqtSignal(int, int)
    processing_completed = pyqtSignal(int)
    processing_error = pyqtSignal(str)
    
    def __init__(self, processor, input_dir, output_dir):
        super().__init__()
        self.processor = processor
        self.input_dir = input_dir
        self.output_dir = output_dir
        
    def run(self):
        try:
            result = self.processor.process_images(
                self.input_dir, 
                self.output_dir,
                progress_callback=self.update_progress
            )
            self.processing_completed.emit(result)
        except Exception as e:
            self.processing_error.emit(str(e))
    
    def update_progress(self, current, total):
        self.progress_updated.emit(current, total)


class VideoExtractionWorker(QThread):
    progress_updated = pyqtSignal(int, int, float)
    processing_completed = pyqtSignal(dict)
    processing_error = pyqtSignal(str)
    log_message = pyqtSignal(str, str)
    
    def __init__(self, config, input_dir, output_dir):
        super().__init__()
        self.config = config
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.is_running = True
        
    def run(self):
        try:
            self.log_message.emit("開始影片幀擷取...", "info")
            extractor = FrameExtractor(self.config)
            extractor.process_videos_in_directory(self.input_dir, self.output_dir)
            final_stats = {
                'total_videos': extractor.processed_stats['total_videos'],
                'success_videos': extractor.processed_stats['success_videos'],
                'failed_videos': extractor.processed_stats['failed_videos'],
                'total_frames': extractor.processed_stats['total_frames'],
                'processing_time': extractor.processed_stats['processing_time'],
                'skipped_duplicate': extractor.processed_stats['skipped_duplicate'],
                'skipped_similar': extractor.processed_stats['skipped_similar']
            }
            self.processing_completed.emit(final_stats)
        except Exception as e:
            self.processing_error.emit(str(e))
    
    def stop(self):
        self.is_running = False


class TrainingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger("TrainingTab")
        self.training_start_time = None
        self.training_worker = None
        self.data_worker = None
        self.processing_timer = QTimer()
        self.processing_timer.timeout.connect(self.update_processing_timer)
        self.processing_start_time = None
        self.init_styles()
        self.init_ui()
        
    def init_styles(self):
        """初始化樣式"""
        pass
        
    def init_ui(self):
        """初始化使用者介面"""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        tab_widget = QTabWidget()
        data_tab = self.create_data_processing_tab()
        tab_widget.addTab(data_tab, "資料處理")
        training_tab = self.create_training_tab()
        tab_widget.addTab(training_tab, "模型訓練")
        main_layout.addWidget(tab_widget)
        self.setLayout(main_layout)
        
    def create_training_tab(self):
        """創建模型訓練標籤頁"""
        tab = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(15, 15, 10, 15)
        scroll_area = QScrollArea()
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_layout.addWidget(self.create_basic_settings_group())
        scroll_layout.addWidget(self.create_training_parameters_group())
        scroll_layout.addWidget(self.create_optimizer_settings_group())
        scroll_layout.addWidget(self.create_data_settings_group())
        scroll_layout.addStretch()
        scroll_content.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_content)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setMinimumWidth(400)
        left_layout.addWidget(scroll_area)
        left_panel.setLayout(left_layout)
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(10, 15, 15, 15)
        right_layout.addWidget(self.create_training_progress_group())
        button_group = QGroupBox("訓練控制")
        button_layout = QVBoxLayout()
        button_layout.setSpacing(10)
        self.start_training_btn = QPushButton("開始訓練")
        self.start_training_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.start_training_btn.clicked.connect(self.start_model_training)
        self.start_training_btn.setMinimumHeight(40)
        self.stop_training_btn = QPushButton("停止訓練")
        self.stop_training_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.stop_training_btn.clicked.connect(self.stop_model_training)
        self.stop_training_btn.setEnabled(False)
        self.stop_training_btn.setMinimumHeight(40)
        button_layout.addWidget(self.start_training_btn)
        button_layout.addWidget(self.stop_training_btn)
        button_group.setLayout(button_layout)
        right_layout.addWidget(button_group)
        right_layout.addStretch()
        right_panel.setLayout(right_layout)
        right_panel.setMinimumWidth(250)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([700, 300])
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        main_layout.addWidget(splitter)
        tab.setLayout(main_layout)
        return tab
        
    def create_data_processing_tab(self):
        """建立資料處理標籤頁"""
        tab = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)
        mode_group = QGroupBox("處理模式")
        mode_layout = QVBoxLayout()
        self.processing_mode_combo = QComboBox()
        self.processing_mode_combo.addItems([
            "影片擷取 - 從影片檔案中提取圖片幀",
            "JPEG壓縮 - 模擬圖片品質劣化",
            "雜訊添加 - 模擬網路攝影機雜訊", 
            "像素化效果 - 模擬低解析度影像",
            "模糊效果 - 模擬運動和散焦模糊",
            "色彩失真 - 模擬色彩偏移和飽和度變化",
            "混合劣化 - 綜合多種劣化效果",
            "自訂流程 - 使用自訂處理管線"
        ])
        self.processing_mode_combo.currentTextChanged.connect(self.on_processing_mode_changed)
        mode_layout.addWidget(self.processing_mode_combo)
        self.mode_description = QLabel("影片擷取模式：從影片檔案中提取圖片幀，支援智能去重和相似度過濾，適合從影片中快速生成訓練數據。")
        self.mode_description.setWordWrap(True)
        self.mode_description.setStyleSheet("color: #666666; font-size: 12px; margin: 5px;")
        mode_layout.addWidget(self.mode_description)
        mode_group.setLayout(mode_layout)
        left_layout.addWidget(mode_group)
        self.mode_params_group = QGroupBox("模式參數設定")
        self.mode_params_layout = QVBoxLayout()
        self.mode_params_group.setLayout(self.mode_params_layout)
        left_layout.addWidget(self.mode_params_group)
        self.create_video_extraction_params()
        settings_group = QGroupBox("基本設定")
        form_layout = QFormLayout()
        form_layout.setVerticalSpacing(12)
        input_layout = QHBoxLayout()
        self.input_dir_edit = QLineEdit("./data/input_images")
        self.input_dir_edit.setMinimumWidth(300)
        self.browse_input_btn = QPushButton("瀏覽...")
        self.browse_input_btn.clicked.connect(self.browse_input_directory)
        input_layout.addWidget(self.input_dir_edit)
        input_layout.addWidget(self.browse_input_btn)
        form_layout.addRow("輸入目錄:", input_layout)
        output_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit("./data/quality_dataset")
        self.output_dir_edit.setMinimumWidth(300)
        self.browse_output_btn = QPushButton("瀏覽...")
        self.browse_output_btn.clicked.connect(self.browse_output_directory)
        output_layout.addWidget(self.output_dir_edit)
        output_layout.addWidget(self.browse_output_btn)
        form_layout.addRow("輸出目錄:", output_layout)
        quality_layout = QHBoxLayout()
        self.min_quality_spin = QSpinBox()
        self.min_quality_spin.setRange(1, 100)
        self.min_quality_spin.setValue(10)
        self.min_quality_spin.setSuffix(" (最差品質)")
        self.max_quality_spin = QSpinBox()
        self.max_quality_spin.setRange(2, 101)
        self.max_quality_spin.setValue(101)
        self.max_quality_spin.setSuffix(" (最高品質)")
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(5, 30)
        self.interval_spin.setValue(10)
        self.interval_spin.setSuffix(" (間隔)")
        quality_layout.addWidget(QLabel("最小:"))
        quality_layout.addWidget(self.min_quality_spin)
        quality_layout.addWidget(QLabel("最大:"))
        quality_layout.addWidget(self.max_quality_spin)
        quality_layout.addWidget(QLabel("間隔:"))
        quality_layout.addWidget(self.interval_spin)
        quality_layout.addStretch()
        form_layout.addRow("品質範圍:", quality_layout)
        workers_layout = QHBoxLayout()
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, os.cpu_count())
        self.workers_spin.setValue(max(1, os.cpu_count() // 2))
        self.workers_spin.setSuffix(" 個程序")
        workers_layout.addWidget(self.workers_spin)
        cpu_preset_layout = QHBoxLayout()
        self.cpu_low_btn = QPushButton("低 (25%)")
        self.cpu_med_btn = QPushButton("中 (50%)")
        self.cpu_high_btn = QPushButton("高 (75%)")
        self.cpu_max_btn = QPushButton("最大")
        self.cpu_low_btn.clicked.connect(lambda: self.workers_spin.setValue(max(1, os.cpu_count() // 4)))
        self.cpu_med_btn.clicked.connect(lambda: self.workers_spin.setValue(max(1, os.cpu_count() // 2)))
        self.cpu_high_btn.clicked.connect(lambda: self.workers_spin.setValue(max(1, int(os.cpu_count() * 0.75))))
        self.cpu_max_btn.clicked.connect(lambda: self.workers_spin.setValue(os.cpu_count()))
        cpu_preset_layout.addWidget(self.cpu_low_btn)
        cpu_preset_layout.addWidget(self.cpu_med_btn)
        cpu_preset_layout.addWidget(self.cpu_high_btn)
        cpu_preset_layout.addWidget(self.cpu_max_btn)
        cpu_preset_layout.addStretch()
        workers_layout.addLayout(cpu_preset_layout)
        form_layout.addRow("工作程序:", workers_layout)
        settings_group.setLayout(form_layout)
        left_layout.addWidget(settings_group)
        advanced_group = QGroupBox("進階設定")
        advanced_group.setCheckable(True)
        advanced_group.setChecked(False)
        advanced_layout = QFormLayout()
        format_layout = QHBoxLayout()
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(["jpg", "png", "webp", "tiff"])
        format_layout.addWidget(self.output_format_combo)
        format_layout.addWidget(QLabel("輸入格式過濾:"))
        self.input_filter_edit = QLineEdit("jpg,jpeg,png,bmp,tiff,webp")
        self.input_filter_edit.setPlaceholderText("支援的輸入格式，用逗號分隔")
        format_layout.addWidget(self.input_filter_edit)
        advanced_layout.addRow("檔案格式:", format_layout)
        resize_layout = QHBoxLayout()
        self.resize_images_check = QCheckBox("啟用尺寸調整")
        self.target_width_spin = QSpinBox()
        self.target_width_spin.setRange(64, 4096)
        self.target_width_spin.setValue(512)
        self.target_width_spin.setEnabled(False)
        self.target_height_spin = QSpinBox()
        self.target_height_spin.setRange(64, 4096)
        self.target_height_spin.setValue(512)
        self.target_height_spin.setEnabled(False)
        self.maintain_aspect_check = QCheckBox("保持縱橫比")
        self.maintain_aspect_check.setChecked(True)
        self.maintain_aspect_check.setEnabled(False)
        self.resize_images_check.toggled.connect(self.target_width_spin.setEnabled)
        self.resize_images_check.toggled.connect(self.target_height_spin.setEnabled)
        self.resize_images_check.toggled.connect(self.maintain_aspect_check.setEnabled)
        resize_layout.addWidget(self.resize_images_check)
        resize_layout.addWidget(QLabel("寬度:"))
        resize_layout.addWidget(self.target_width_spin)
        resize_layout.addWidget(QLabel("高度:"))
        resize_layout.addWidget(self.target_height_spin)
        resize_layout.addWidget(self.maintain_aspect_check)
        resize_layout.addStretch()
        advanced_layout.addRow("尺寸調整:", resize_layout)
        output_quality_layout = QHBoxLayout()
        self.output_quality_spin = QSpinBox()
        self.output_quality_spin.setRange(1, 100)
        self.output_quality_spin.setValue(95)
        self.output_quality_spin.setSuffix("%")
        output_quality_layout.addWidget(self.output_quality_spin)
        output_quality_layout.addWidget(QLabel("(儲存時的品質)"))
        output_quality_layout.addStretch()
        advanced_layout.addRow("輸出品質:", output_quality_layout)
        processing_options_layout = QHBoxLayout()
        self.skip_existing_check = QCheckBox("跳過已存在檔案")
        self.generate_previews_check = QCheckBox("生成預覽圖")
        self.generate_previews_check.setChecked(True)
        self.preserve_metadata_check = QCheckBox("保留元數據")
        self.preserve_metadata_check.setChecked(True)
        self.create_backup_check = QCheckBox("創建備份")
        processing_options_layout.addWidget(self.skip_existing_check)
        processing_options_layout.addWidget(self.generate_previews_check)
        processing_options_layout.addWidget(self.preserve_metadata_check)
        processing_options_layout.addWidget(self.create_backup_check)
        processing_options_layout.addStretch()
        advanced_layout.addRow("處理選項:", processing_options_layout)
        memory_layout = QHBoxLayout()
        self.memory_limit_check = QCheckBox("啟用記憶體限制")
        self.memory_limit_spin = QSpinBox()
        self.memory_limit_spin.setRange(512, 16384)
        self.memory_limit_spin.setValue(4096)
        self.memory_limit_spin.setSuffix(" MB")
        self.memory_limit_spin.setEnabled(False)
        self.memory_limit_check.toggled.connect(self.memory_limit_spin.setEnabled)
        memory_layout.addWidget(self.memory_limit_check)
        memory_layout.addWidget(self.memory_limit_spin)
        memory_layout.addStretch()
        advanced_layout.addRow("記憶體管理:", memory_layout)
        advanced_group.setLayout(advanced_layout)
        left_layout.addWidget(advanced_group)
        left_layout.addStretch()
        left_panel.setLayout(left_layout)
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(10, 15, 15, 15)
        progress_group = QGroupBox("處理進度")
        progress_layout = QVBoxLayout()
        self.data_progress_bar = QProgressBar()
        self.data_progress_bar.setRange(0, 100)
        self.data_progress_bar.setTextVisible(True)
        self.data_progress_bar.setFormat("%p% (%v/%m)")
        self.data_progress_bar.setMinimumHeight(25)
        progress_status_layout = QHBoxLayout()
        progress_status_layout.addWidget(QLabel("狀態:"))
        self.data_status_label = QLabel("準備就緒")
        self.data_status_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        progress_status_layout.addWidget(self.data_status_label)
        progress_status_layout.addStretch()
        stats_layout = QGridLayout()
        stats_layout.addWidget(QLabel("已處理:"), 0, 0)
        self.processed_count_label = QLabel("0")
        self.processed_count_label.setStyleSheet("font-weight: bold;")
        stats_layout.addWidget(self.processed_count_label, 0, 1)
        stats_layout.addWidget(QLabel("已生成:"), 0, 2)
        self.generated_count_label = QLabel("0")
        self.generated_count_label.setStyleSheet("font-weight: bold;")
        stats_layout.addWidget(self.generated_count_label, 0, 3)
        stats_layout.addWidget(QLabel("處理時間:"), 1, 0)
        self.processing_time_label = QLabel("00:00:00")
        self.processing_time_label.setStyleSheet("font-weight: bold;")
        stats_layout.addWidget(self.processing_time_label, 1, 1)
        stats_layout.addWidget(QLabel("處理速度:"), 1, 2)
        self.processing_speed_label = QLabel("0.0 img/s")
        self.processing_speed_label.setStyleSheet("font-weight: bold;")
        stats_layout.addWidget(self.processing_speed_label, 1, 3)
        progress_layout.addWidget(self.data_progress_bar)
        progress_layout.addLayout(progress_status_layout)
        progress_layout.addLayout(stats_layout)
        progress_group.setLayout(progress_layout)
        right_layout.addWidget(progress_group)
        quality_group = QGroupBox("品質統計")
        quality_layout = QGridLayout()
        quality_layout.addWidget(QLabel("平均PSNR:"), 0, 0)
        self.avg_psnr_label = QLabel("N/A")
        self.avg_psnr_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        quality_layout.addWidget(self.avg_psnr_label, 0, 1)
        quality_layout.addWidget(QLabel("PSNR範圍:"), 0, 2)
        self.psnr_range_label = QLabel("N/A ~ N/A dB")
        self.psnr_range_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        quality_layout.addWidget(self.psnr_range_label, 0, 3)
        quality_layout.addWidget(QLabel("平均SSIM:"), 1, 0)
        self.avg_ssim_label = QLabel("N/A")
        self.avg_ssim_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        quality_layout.addWidget(self.avg_ssim_label, 1, 1)
        quality_layout.addWidget(QLabel("SSIM範圍:"), 1, 2)
        self.ssim_range_label = QLabel("N/A ~ N/A")
        self.ssim_range_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        quality_layout.addWidget(self.ssim_range_label, 1, 3)
        quality_layout.addWidget(QLabel("平均檔案大小:"), 2, 0)
        self.avg_filesize_label = QLabel("N/A")
        self.avg_filesize_label.setStyleSheet("font-weight: bold; color: #FF9800;")
        quality_layout.addWidget(self.avg_filesize_label, 2, 1)
        quality_layout.addWidget(QLabel("壓縮比:"), 2, 2)
        self.compression_ratio_label = QLabel("N/A")
        self.compression_ratio_label.setStyleSheet("font-weight: bold; color: #FF9800;")
        quality_layout.addWidget(self.compression_ratio_label, 2, 3)
        quality_group.setLayout(quality_layout)
        right_layout.addWidget(quality_group)
        monitoring_group = QGroupBox("處理監控")
        monitoring_layout = QGridLayout()
        monitoring_layout.addWidget(QLabel("已用時間:"), 0, 0)
        self.elapsed_time_label = QLabel("00:00:00")
        self.elapsed_time_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        monitoring_layout.addWidget(self.elapsed_time_label, 0, 1)
        monitoring_layout.addWidget(QLabel("預估剩餘:"), 1, 0)
        self.remaining_time_label = QLabel("計算中...")
        self.remaining_time_label.setStyleSheet("font-weight: bold; color: #FF9800;")
        monitoring_layout.addWidget(self.remaining_time_label, 1, 1)
        monitoring_layout.addWidget(QLabel("預計完成:"), 2, 0)
        self.estimated_finish_label = QLabel("N/A")
        self.estimated_finish_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        monitoring_layout.addWidget(self.estimated_finish_label, 2, 1)
        monitoring_layout.addWidget(QLabel("預計輸出:"), 3, 0)
        self.estimated_output_label = QLabel("N/A")
        self.estimated_output_label.setStyleSheet("font-weight: bold;")
        monitoring_layout.addWidget(self.estimated_output_label, 3, 1)
        monitoring_group.setLayout(monitoring_layout)
        right_layout.addWidget(monitoring_group)
        log_group = QGroupBox("處理日誌")
        log_layout = QVBoxLayout()
        self.processing_log = QTextEdit()
        self.processing_log.setMaximumHeight(150)
        self.processing_log.setReadOnly(True)
        self.processing_log.setStyleSheet("font-family: 'Consolas', monospace; font-size: 12px;")
        log_control_layout = QHBoxLayout()
        self.clear_log_btn = QPushButton("清除日誌")
        self.clear_log_btn.clicked.connect(self.processing_log.clear)
        self.save_log_btn = QPushButton("保存日誌")
        self.save_log_btn.clicked.connect(self.save_processing_log)
        log_control_layout.addWidget(self.clear_log_btn)
        log_control_layout.addWidget(self.save_log_btn)
        log_control_layout.addStretch()
        log_layout.addWidget(self.processing_log)
        log_layout.addLayout(log_control_layout)
        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group)
        button_group = QGroupBox("處理控制")
        button_layout = QVBoxLayout()
        button_layout.setSpacing(10)
        self.start_processing_btn = QPushButton("開始處理")
        self.start_processing_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.start_processing_btn.clicked.connect(self.start_data_processing)
        self.start_processing_btn.setMinimumHeight(40)
        self.start_processing_btn.setStyleSheet("font-weight: bold;")
        self.stop_processing_btn = QPushButton("停止處理")
        self.stop_processing_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.stop_processing_btn.clicked.connect(self.stop_data_processing)
        self.stop_processing_btn.setEnabled(False)
        self.stop_processing_btn.setMinimumHeight(40)
        self.reset_processing_btn = QPushButton("重置設定")
        self.reset_processing_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.reset_processing_btn.clicked.connect(self.reset_processing_settings)
        self.reset_processing_btn.setMinimumHeight(40)
        button_layout.addWidget(self.start_processing_btn)
        button_layout.addWidget(self.stop_processing_btn)
        button_layout.addWidget(self.reset_processing_btn)
        button_group.setLayout(button_layout)
        right_layout.addWidget(button_group)
        right_layout.addStretch()
        right_panel.setLayout(right_layout)
        right_panel.setMinimumWidth(250)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([700, 300])
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        main_layout.addWidget(splitter)
        tab.setLayout(main_layout)
        return tab
        
    def on_processing_mode_changed(self, mode_text):
        """處理模式變更時更新參數設定和說明"""
        descriptions = {
            "影片擷取 - 從影片檔案中提取圖片幀": "影片擷取模式：從影片檔案中提取圖片幀，支援智能去重和相似度過濾，適合從影片中快速生成訓練數據。",
            "JPEG壓縮 - 模擬圖片品質劣化": "JPEG壓縮模式：透過不同的JPEG品質設定生成訓練數據，適合訓練圖片品質提升模型。",
            "雜訊添加 - 模擬網路攝影機雜訊": "雜訊添加模式：添加高斯、椒鹽、斑點雜訊等，模擬真實環境中的圖像劣化。",
            "像素化效果 - 模擬低解析度影像": "像素化模式：透過下採樣和上採樣創建像素化效果，適合超解析度重建訓練。",
            "模糊效果 - 模擬運動和散焦模糊": "模糊效果模式：應用高斯模糊、運動模糊、散焦模糊等，適合圖像去模糊訓練。",
            "色彩失真 - 模擬色彩偏移和飽和度變化": "色彩失真模式：調整亮度、對比度、飽和度、色調等，適合色彩校正訓練。",
            "混合劣化 - 綜合多種劣化效果": "混合劣化模式：隨機組合多種劣化效果，創建更複雜的訓練樣本。",
            "自訂流程 - 使用自訂處理管線": "自訂流程模式：可以手動配置處理步驟和參數，實現個性化的圖像處理流程。"
        }
        self.mode_description.setText(descriptions.get(mode_text, "未知模式"))
        for i in reversed(range(self.mode_params_layout.count())):
            item = self.mode_params_layout.itemAt(i)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                else:
                    layout = item.layout()
                    if layout is not None:
                        self.clear_layout(layout)
                    self.mode_params_layout.removeItem(item)
        if "影片擷取" in mode_text:
            self.create_video_extraction_params()
        elif "JPEG壓縮" in mode_text:
            self.create_jpeg_params()
        elif "雜訊添加" in mode_text:
            self.create_noise_params()
        elif "像素化" in mode_text:
            self.create_pixel_params()
        elif "模糊效果" in mode_text:
            self.create_blur_params()
        elif "色彩失真" in mode_text:
            self.create_color_params()
        elif "混合劣化" in mode_text:
            self.create_mixed_params()
        elif "自訂流程" in mode_text:
            self.create_custom_params()
    
    def create_video_extraction_params(self):
        """創建影片擷取參數設定"""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setMaximumHeight(400)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout()
        basic_group = QGroupBox("基本設定")
        basic_layout = QFormLayout()
        self.frame_interval = QSpinBox()
        self.frame_interval.setRange(1, 500)
        self.frame_interval.setValue(72)
        self.frame_interval.setToolTip("每隔多少幀提取一張圖片")
        basic_layout.addRow("提取間隔(幀):", self.frame_interval)
        format_layout = QHBoxLayout()
        self.video_output_format = QComboBox()
        self.video_output_format.addItems(["jpg", "png"])
        self.video_output_format.setCurrentText("jpg")
        self.video_quality_spin = QSpinBox()
        self.video_quality_spin.setRange(80, 100)
        self.video_quality_spin.setValue(95)
        self.video_quality_spin.setSuffix("%")
        format_layout.addWidget(self.video_output_format)
        format_layout.addWidget(QLabel("品質:"))
        format_layout.addWidget(self.video_quality_spin)
        format_layout.addStretch()
        basic_layout.addRow("輸出格式:", format_layout)
        self.max_workers = QSpinBox()
        self.max_workers.setRange(1, 8)
        self.max_workers.setValue(min(4, os.cpu_count()))
        basic_layout.addRow("並行進程數:", self.max_workers)
        basic_group.setLayout(basic_layout)
        scroll_layout.addWidget(basic_group)
        image_group = QGroupBox("圖像設定")
        image_layout = QVBoxLayout()
        self.resize_frames = QCheckBox("調整圖片大小")
        self.resize_frames.setChecked(False)
        self.resize_frames.toggled.connect(self.on_resize_frames_toggled)
        image_layout.addWidget(self.resize_frames)
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("尺寸:"))
        self.target_width = QSpinBox()
        self.target_width.setRange(480, 4096)
        self.target_width.setValue(1280)
        self.target_width.setEnabled(False)
        size_layout.addWidget(self.target_width)
        size_layout.addWidget(QLabel("×"))
        self.target_height = QSpinBox()
        self.target_height.setRange(360, 2160)
        self.target_height.setValue(720)
        self.target_height.setEnabled(False)
        size_layout.addWidget(self.target_height)
        self.keep_aspect_ratio = QCheckBox("保持比例")
        self.keep_aspect_ratio.setChecked(True)
        self.keep_aspect_ratio.setEnabled(False)
        size_layout.addWidget(self.keep_aspect_ratio)
        size_layout.addStretch()
        image_layout.addLayout(size_layout)
        image_group.setLayout(image_layout)
        scroll_layout.addWidget(image_group)
        filter_group = QGroupBox("智能過濾")
        filter_layout = QVBoxLayout()
        self.dedup_frames = QCheckBox("去除重複幀")
        self.dedup_frames.setChecked(True)
        filter_layout.addWidget(self.dedup_frames)
        self.similarity_check = QCheckBox("相似度過濾")
        self.similarity_check.setChecked(True)
        self.similarity_check.toggled.connect(self.on_similarity_check_toggled)
        filter_layout.addWidget(self.similarity_check)
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("閾值:"))
        self.similarity_threshold = QDoubleSpinBox()
        self.similarity_threshold.setRange(0.60, 0.95)
        self.similarity_threshold.setValue(0.85)
        self.similarity_threshold.setSingleStep(0.05)
        self.similarity_threshold.setDecimals(2)
        threshold_layout.addWidget(self.similarity_threshold)
        threshold_layout.addWidget(QLabel("比較幀數:"))
        self.max_compare_frames = QSpinBox()
        self.max_compare_frames.setRange(3, 10)
        self.max_compare_frames.setValue(5)
        threshold_layout.addWidget(self.max_compare_frames)
        threshold_layout.addStretch()
        filter_layout.addLayout(threshold_layout)
        filter_group.setLayout(filter_layout)
        scroll_layout.addWidget(filter_group)
        other_group = QGroupBox("其他選項")
        other_layout = QVBoxLayout()
        self.recursive_search = QCheckBox("搜尋子目錄")
        self.recursive_search.setChecked(True)
        other_layout.addWidget(self.recursive_search)
        self.skip_processed = QCheckBox("跳過已處理影片")
        self.skip_processed.setChecked(True)
        other_layout.addWidget(self.skip_processed)
        other_group.setLayout(other_layout)
        scroll_layout.addWidget(other_group)
        scroll_layout.addStretch()
        scroll_content.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_content)
        self.mode_params_layout.addWidget(scroll_area)
    
    def on_resize_frames_toggled(self, checked):
        """當調整大小選項改變時"""
        self.target_width.setEnabled(checked)
        self.target_height.setEnabled(checked)
        self.keep_aspect_ratio.setEnabled(checked)
    
    def on_similarity_check_toggled(self, checked):
        """當相似度檢查選項改變時"""
        pass
    
    def create_jpeg_params(self):
        """創建JPEG壓縮參數設定"""
        form_layout = QFormLayout()
        quality_layout = QHBoxLayout()
        self.jpeg_min_quality = QSpinBox()
        self.jpeg_min_quality.setRange(1, 100)
        self.jpeg_min_quality.setValue(10)
        self.jpeg_min_quality.setSuffix("%")
        self.jpeg_max_quality = QSpinBox()
        self.jpeg_max_quality.setRange(2, 101)
        self.jpeg_max_quality.setValue(95)
        self.jpeg_max_quality.setSuffix("%")
        self.jpeg_step = QSpinBox()
        self.jpeg_step.setRange(5, 50)
        self.jpeg_step.setValue(15)
        quality_layout.addWidget(QLabel("最小:"))
        quality_layout.addWidget(self.jpeg_min_quality)
        quality_layout.addWidget(QLabel("最大:"))
        quality_layout.addWidget(self.jpeg_max_quality)
        quality_layout.addWidget(QLabel("步長:"))
        quality_layout.addWidget(self.jpeg_step)
        quality_layout.addStretch()
        form_layout.addRow("品質範圍:", quality_layout)
        compression_layout = QHBoxLayout()
        self.jpeg_progressive = QCheckBox("漸進式JPEG")
        self.jpeg_optimize = QCheckBox("最佳化")
        self.jpeg_optimize.setChecked(True)
        compression_layout.addWidget(self.jpeg_progressive)
        compression_layout.addWidget(self.jpeg_optimize)
        compression_layout.addStretch()
        form_layout.addRow("壓縮選項:", compression_layout)
        self.mode_params_layout.addLayout(form_layout)
    
    def create_noise_params(self):
        """創建雜訊添加參數設定"""
        form_layout = QFormLayout()
        noise_type_layout = QVBoxLayout()
        self.noise_gaussian = QCheckBox("高斯雜訊")
        self.noise_gaussian.setChecked(True)
        self.noise_salt_pepper = QCheckBox("椒鹽雜訊")
        self.noise_speckle = QCheckBox("斑點雜訊")
        self.noise_poisson = QCheckBox("泊松雜訊")
        noise_type_layout.addWidget(self.noise_gaussian)
        noise_type_layout.addWidget(self.noise_salt_pepper)
        noise_type_layout.addWidget(self.noise_speckle)
        noise_type_layout.addWidget(self.noise_poisson)
        form_layout.addRow("雜訊類型:", noise_type_layout)
        intensity_layout = QHBoxLayout()
        self.noise_min_intensity = QDoubleSpinBox()
        self.noise_min_intensity.setRange(0.01, 1.0)
        self.noise_min_intensity.setValue(0.05)
        self.noise_min_intensity.setSingleStep(0.01)
        self.noise_max_intensity = QDoubleSpinBox()
        self.noise_max_intensity.setRange(0.02, 1.0)
        self.noise_max_intensity.setValue(0.25)
        self.noise_max_intensity.setSingleStep(0.01)
        intensity_layout.addWidget(QLabel("最小強度:"))
        intensity_layout.addWidget(self.noise_min_intensity)
        intensity_layout.addWidget(QLabel("最大強度:"))
        intensity_layout.addWidget(self.noise_max_intensity)
        intensity_layout.addStretch()
        form_layout.addRow("雜訊強度:", intensity_layout)
        self.mode_params_layout.addLayout(form_layout)
    
    def create_pixel_params(self):
        """創建像素化參數設定"""
        form_layout = QFormLayout()
        pixel_layout = QHBoxLayout()
        self.pixel_min_scale = QDoubleSpinBox()
        self.pixel_min_scale.setRange(0.1, 0.9)
        self.pixel_min_scale.setValue(0.2)
        self.pixel_min_scale.setSingleStep(0.1)
        self.pixel_max_scale = QDoubleSpinBox()
        self.pixel_max_scale.setRange(0.2, 1.0)
        self.pixel_max_scale.setValue(0.8)
        self.pixel_max_scale.setSingleStep(0.1)
        pixel_layout.addWidget(QLabel("最小縮放:"))
        pixel_layout.addWidget(self.pixel_min_scale)
        pixel_layout.addWidget(QLabel("最大縮放:"))
        pixel_layout.addWidget(self.pixel_max_scale)
        pixel_layout.addStretch()
        form_layout.addRow("像素化程度:", pixel_layout)
        interp_layout = QHBoxLayout()
        self.pixel_interpolation = QComboBox()
        self.pixel_interpolation.addItems(["最近鄰", "雙線性", "雙三次"])
        self.pixel_interpolation.setCurrentText("最近鄰")
        interp_layout.addWidget(self.pixel_interpolation)
        interp_layout.addStretch()
        form_layout.addRow("插值方法:", interp_layout)
        self.mode_params_layout.addLayout(form_layout)
    
    def create_blur_params(self):
        """創建模糊效果參數設定"""
        form_layout = QFormLayout()
        blur_type_layout = QVBoxLayout()
        self.blur_gaussian = QCheckBox("高斯模糊")
        self.blur_gaussian.setChecked(True)
        self.blur_motion = QCheckBox("運動模糊")
        self.blur_defocus = QCheckBox("散焦模糊")
        blur_type_layout.addWidget(self.blur_gaussian)
        blur_type_layout.addWidget(self.blur_motion)
        blur_type_layout.addWidget(self.blur_defocus)
        form_layout.addRow("模糊類型:", blur_type_layout)
        blur_strength_layout = QHBoxLayout()
        self.blur_min_strength = QDoubleSpinBox()
        self.blur_min_strength.setRange(0.5, 10.0)
        self.blur_min_strength.setValue(1.0)
        self.blur_min_strength.setSingleStep(0.5)
        self.blur_max_strength = QDoubleSpinBox()
        self.blur_max_strength.setRange(1.0, 20.0)
        self.blur_max_strength.setValue(5.0)
        self.blur_max_strength.setSingleStep(0.5)
        blur_strength_layout.addWidget(QLabel("最小強度:"))
        blur_strength_layout.addWidget(self.blur_min_strength)
        blur_strength_layout.addWidget(QLabel("最大強度:"))
        blur_strength_layout.addWidget(self.blur_max_strength)
        blur_strength_layout.addStretch()
        form_layout.addRow("模糊強度:", blur_strength_layout)
        motion_layout = QHBoxLayout()
        self.motion_angle_min = QSpinBox()
        self.motion_angle_min.setRange(0, 360)
        self.motion_angle_min.setValue(0)
        self.motion_angle_min.setSuffix("°")
        self.motion_angle_max = QSpinBox()
        self.motion_angle_max.setRange(0, 360)
        self.motion_angle_max.setValue(180)
        self.motion_angle_max.setSuffix("°")
        motion_layout.addWidget(QLabel("最小角度:"))
        motion_layout.addWidget(self.motion_angle_min)
        motion_layout.addWidget(QLabel("最大角度:"))
        motion_layout.addWidget(self.motion_angle_max)
        motion_layout.addStretch()
        form_layout.addRow("運動方向:", motion_layout)
        self.mode_params_layout.addLayout(form_layout)
    
    def create_color_params(self):
        """創建色彩失真參數設定"""
        form_layout = QFormLayout()
        brightness_layout = QHBoxLayout()
        self.brightness_min = QDoubleSpinBox()
        self.brightness_min.setRange(-0.5, 0.5)
        self.brightness_min.setValue(-0.2)
        self.brightness_min.setSingleStep(0.1)
        self.brightness_max = QDoubleSpinBox()
        self.brightness_max.setRange(-0.5, 0.5)
        self.brightness_max.setValue(0.2)
        self.brightness_max.setSingleStep(0.1)
        brightness_layout.addWidget(QLabel("最小:"))
        brightness_layout.addWidget(self.brightness_min)
        brightness_layout.addWidget(QLabel("最大:"))
        brightness_layout.addWidget(self.brightness_max)
        brightness_layout.addStretch()
        form_layout.addRow("亮度調整:", brightness_layout)
        contrast_layout = QHBoxLayout()
        self.contrast_min = QDoubleSpinBox()
        self.contrast_min.setRange(0.5, 2.0)
        self.contrast_min.setValue(0.7)
        self.contrast_min.setSingleStep(0.1)
        self.contrast_max = QDoubleSpinBox()
        self.contrast_max.setRange(0.5, 2.0)
        self.contrast_max.setValue(1.3)
        self.contrast_max.setSingleStep(0.1)
        contrast_layout.addWidget(QLabel("最小:"))
        contrast_layout.addWidget(self.contrast_min)
        contrast_layout.addWidget(QLabel("最大:"))
        contrast_layout.addWidget(self.contrast_max)
        contrast_layout.addStretch()
        form_layout.addRow("對比度:", contrast_layout)
        saturation_layout = QHBoxLayout()
        self.saturation_min = QDoubleSpinBox()
        self.saturation_min.setRange(0.0, 2.0)
        self.saturation_min.setValue(0.5)
        self.saturation_min.setSingleStep(0.1)
        self.saturation_max = QDoubleSpinBox()
        self.saturation_max.setRange(0.0, 2.0)
        self.saturation_max.setValue(1.5)
        self.saturation_max.setSingleStep(0.1)
        saturation_layout.addWidget(QLabel("最小:"))
        saturation_layout.addWidget(self.saturation_min)
        saturation_layout.addWidget(QLabel("最大:"))
        saturation_layout.addWidget(self.saturation_max)
        saturation_layout.addStretch()
        form_layout.addRow("飽和度:", saturation_layout)
        hue_layout = QHBoxLayout()
        self.hue_shift_max = QSpinBox()
        self.hue_shift_max.setRange(0, 180)
        self.hue_shift_max.setValue(30)
        self.hue_shift_max.setSuffix("°")
        hue_layout.addWidget(QLabel("最大偏移:"))
        hue_layout.addWidget(self.hue_shift_max)
        hue_layout.addStretch()
        form_layout.addRow("色調偏移:", hue_layout)
        self.mode_params_layout.addLayout(form_layout)
    
    def create_mixed_params(self):
        """創建混合劣化參數設定"""
        form_layout = QFormLayout()
        effects_layout = QVBoxLayout()
        self.mix_jpeg = QCheckBox("包含JPEG壓縮")
        self.mix_jpeg.setChecked(True)
        self.mix_noise = QCheckBox("包含雜訊")
        self.mix_noise.setChecked(True)
        self.mix_blur = QCheckBox("包含模糊")
        self.mix_color = QCheckBox("包含色彩失真")
        effects_layout.addWidget(self.mix_jpeg)
        effects_layout.addWidget(self.mix_noise)
        effects_layout.addWidget(self.mix_blur)
        effects_layout.addWidget(self.mix_color)
        form_layout.addRow("混合效果:", effects_layout)
        intensity_layout = QHBoxLayout()
        self.mix_intensity_min = QDoubleSpinBox()
        self.mix_intensity_min.setRange(0.1, 1.0)
        self.mix_intensity_min.setValue(0.3)
        self.mix_intensity_min.setSingleStep(0.1)
        self.mix_intensity_max = QDoubleSpinBox()
        self.mix_intensity_max.setRange(0.2, 1.0)
        self.mix_intensity_max.setValue(0.8)
        self.mix_intensity_max.setSingleStep(0.1)
        intensity_layout.addWidget(QLabel("最小強度:"))
        intensity_layout.addWidget(self.mix_intensity_min)
        intensity_layout.addWidget(QLabel("最大強度:"))
        intensity_layout.addWidget(self.mix_intensity_max)
        intensity_layout.addStretch()
        form_layout.addRow("混合強度:", intensity_layout)
        count_layout = QHBoxLayout()
        self.mix_effects_count = QSpinBox()
        self.mix_effects_count.setRange(1, 4)
        self.mix_effects_count.setValue(2)
        count_layout.addWidget(self.mix_effects_count)
        count_layout.addWidget(QLabel("(每張圖像同時應用的效果數量)"))
        count_layout.addStretch()
        form_layout.addRow("同時效果數:", count_layout)
        self.mode_params_layout.addLayout(form_layout)
    
    def create_custom_params(self):
        """創建自訂流程參數設定"""
        form_layout = QFormLayout()
        info_label = QLabel("自訂流程允許您手動配置處理步驟和參數。請按順序添加處理步驟：")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666666; margin-bottom: 10px;")
        form_layout.addRow(info_label)
        self.custom_steps_list = QListWidget()
        self.custom_steps_list.setMaximumHeight(120)
        form_layout.addRow("處理步驟:", self.custom_steps_list)
        buttons_layout = QHBoxLayout()
        self.add_jpeg_step_btn = QPushButton("+ JPEG壓縮")
        self.add_noise_step_btn = QPushButton("+ 添加雜訊")
        self.add_blur_step_btn = QPushButton("+ 模糊效果")
        self.add_color_step_btn = QPushButton("+ 色彩調整")
        self.add_jpeg_step_btn.clicked.connect(lambda: self.add_custom_step("JPEG壓縮"))
        self.add_noise_step_btn.clicked.connect(lambda: self.add_custom_step("添加雜訊"))
        self.add_blur_step_btn.clicked.connect(lambda: self.add_custom_step("模糊效果"))
        self.add_color_step_btn.clicked.connect(lambda: self.add_custom_step("色彩調整"))
        buttons_layout.addWidget(self.add_jpeg_step_btn)
        buttons_layout.addWidget(self.add_noise_step_btn)
        buttons_layout.addWidget(self.add_blur_step_btn)
        buttons_layout.addWidget(self.add_color_step_btn)
        buttons_layout.addStretch()
        form_layout.addRow("添加步驟:", buttons_layout)
        control_layout = QHBoxLayout()
        self.remove_step_btn = QPushButton("移除選中步驟")
        self.clear_steps_btn = QPushButton("清空所有步驟")
        self.save_pipeline_btn = QPushButton("保存流程")
        self.load_pipeline_btn = QPushButton("載入流程")
        self.remove_step_btn.clicked.connect(self.remove_custom_step)
        self.clear_steps_btn.clicked.connect(self.clear_custom_steps)
        self.save_pipeline_btn.clicked.connect(self.save_custom_pipeline)
        self.load_pipeline_btn.clicked.connect(self.load_custom_pipeline)
        control_layout.addWidget(self.remove_step_btn)
        control_layout.addWidget(self.clear_steps_btn)
        control_layout.addWidget(self.save_pipeline_btn)
        control_layout.addWidget(self.load_pipeline_btn)
        form_layout.addRow("管理流程:", control_layout)
        config_layout = QHBoxLayout()
        self.pipeline_config_edit = QLineEdit("./config/custom_pipeline.json")
        self.browse_config_btn = QPushButton("瀏覽...")
        self.browse_config_btn.clicked.connect(self.browse_pipeline_config)
        config_layout.addWidget(self.pipeline_config_edit)
        config_layout.addWidget(self.browse_config_btn)
        form_layout.addRow("配置檔案:", config_layout)
        self.mode_params_layout.addLayout(form_layout)
    
    def add_custom_step(self, step_type):
        """添加自訂處理步驟"""
        from PyQt6.QtWidgets import QInputDialog
        default_params = {
            "JPEG壓縮": "品質=75",
            "添加雜訊": "類型=高斯,強度=0.1",
            "模糊效果": "類型=高斯,強度=2.0",
            "色彩調整": "亮度=0,對比度=1.0,飽和度=1.0"
        }
        param_text, ok = QInputDialog.getText(
            self, f"設定{step_type}參數", 
            f"請輸入{step_type}的參數（格式：參數名=值,參數名=值）：", 
            text=default_params.get(step_type, "")
        )
        if ok and param_text:
            step_item = f"{step_type}: {param_text}"
            self.custom_steps_list.addItem(step_item)
    
    def remove_custom_step(self):
        """移除選中的自訂處理步驟"""
        current_row = self.custom_steps_list.currentRow()
        if current_row >= 0:
            self.custom_steps_list.takeItem(current_row)
    
    def clear_custom_steps(self):
        """清空所有自訂處理步驟"""
        self.custom_steps_list.clear()
    
    def save_custom_pipeline(self):
        """保存自訂處理流程"""
        import json
        steps = []
        for i in range(self.custom_steps_list.count()):
            item_text = self.custom_steps_list.item(i).text()
            steps.append(item_text)
        config_path = self.pipeline_config_edit.text()
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump({"pipeline_steps": steps}, f, ensure_ascii=False, indent=2)
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "成功", f"流程已保存到 {config_path}")
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "錯誤", f"保存失敗：{e}")
    
    def load_custom_pipeline(self):
        """載入自訂處理流程"""
        import json
        config_path = self.pipeline_config_edit.text()
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.custom_steps_list.clear()
            for step in config.get("pipeline_steps", []):
                self.custom_steps_list.addItem(step)
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "成功", f"流程已從 {config_path} 載入")
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "錯誤", f"載入失敗：{e}")
    
    def browse_pipeline_config(self):
        """瀏覽流程配置檔案"""
        from PyQt6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "選擇流程配置檔案", self.pipeline_config_edit.text(), 
            "JSON檔案 (*.json);;所有檔案 (*.*)"
        )
        if file_path:
            self.pipeline_config_edit.setText(file_path)
    
    def clear_layout(self, layout):
        """遞歸清空佈局"""
        if layout is not None:
            for i in reversed(range(layout.count())):
                item = layout.itemAt(i)
                if item is not None:
                    widget = item.widget()
                    if widget is not None:
                        widget.setParent(None)
                    else:
                        child_layout = item.layout()
                        if child_layout is not None:
                            self.clear_layout(child_layout)
                        layout.removeItem(item)
        
    def estimate_processing_time(self):
        """估算處理時間"""
        input_dir = self.input_dir_edit.text()
        if not os.path.exists(input_dir):
            QMessageBox.warning(self, "警告", "輸入目錄不存在！")
            return
        mode_map = {
            0: ProcessingMode.JPEG_COMPRESSION,
            1: ProcessingMode.NOISE_ADDITION,
            2: ProcessingMode.PIXELATION,
            3: ProcessingMode.BLUR_EFFECTS,
            4: ProcessingMode.COLOR_DISTORTION,
            5: ProcessingMode.MIXED_DEGRADATION,
            6: ProcessingMode.CUSTOM_PIPELINE
        }
        processing_mode = mode_map.get(self.processing_mode_combo.currentIndex(), ProcessingMode.JPEG_COMPRESSION)
        temp_processor = DataProcessor(
            min_quality=self.min_quality_spin.value(),
            max_quality=self.max_quality_spin.value(),
            quality_interval=self.interval_spin.value(),
            processing_mode=processing_mode,
            num_workers=self.workers_spin.value()
        )
        estimation = temp_processor.estimate_processing_time(input_dir)
        if estimation:
            minutes = int(estimation['estimated_minutes'])
            seconds = int(estimation['estimated_seconds'] % 60)
            self.estimated_time_label.setText(f"{minutes}分{seconds}秒")
            self.estimated_output_label.setText(f"{estimation['total_outputs']} 張圖片")
            self.add_processing_log(f"估算處理時間: {minutes}分{seconds}秒", "info")
            self.add_processing_log(f"輸入檔案: {estimation['total_files']} 張", "info")
            self.add_processing_log(f"預計輸出: {estimation['total_outputs']} 張", "info")
        else:
            self.estimated_time_label.setText("估算失敗")
            self.estimated_output_label.setText("N/A")
        
    def update_processing_monitor(self):
        """更新處理監控資訊"""
        if not hasattr(self, 'processing_start_time') or not self.processing_start_time:
            return
        try:
            elapsed_time = datetime.datetime.now() - self.processing_start_time
            elapsed_seconds = elapsed_time.total_seconds()
            elapsed_hours = int(elapsed_seconds // 3600)
            elapsed_minutes = int((elapsed_seconds % 3600) // 60)
            elapsed_secs = int(elapsed_seconds % 60)
            elapsed_str = f"{elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_secs:02d}"
            self.elapsed_time_label.setText(f"已過時間: {elapsed_str}")
            if hasattr(self, 'total_files') and hasattr(self, 'processed_files'):
                processed = getattr(self, 'processed_files', 0)
                total = getattr(self, 'total_files', 1)
                if processed > 0 and elapsed_seconds > 0:
                    processing_speed = processed / elapsed_seconds
                    remaining_files = total - processed
                    if processing_speed > 0:
                        remaining_seconds = remaining_files / processing_speed
                        remaining_hours = int(remaining_seconds // 3600)
                        remaining_minutes = int((remaining_seconds % 3600) // 60)
                        remaining_secs = int(remaining_seconds % 60)
                        remaining_str = f"{remaining_hours:02d}:{remaining_minutes:02d}:{remaining_secs:02d}"
                        estimated_finish = datetime.datetime.now() + datetime.timedelta(seconds=remaining_seconds)
                        finish_str = estimated_finish.strftime("%H:%M:%S")
                        self.remaining_time_label.setText(f"剩餘時間: {remaining_str}")
                        self.estimated_finish_label.setText(f"預估完成: {finish_str}")
                    else:
                        self.remaining_time_label.setText("剩餘時間: 計算中...")
                        self.estimated_finish_label.setText("預估完成: 計算中...")
                else:
                    self.remaining_time_label.setText("剩餘時間: 計算中...")
                    self.estimated_finish_label.setText("預估完成: 計算中...")
        except Exception as e:
            self.add_processing_log(f"監控更新錯誤: {str(e)}", "error")
        
    def start_video_extraction(self):
        """開始影片擷取處理"""
        input_dir = self.input_dir_edit.text()
        output_dir = self.output_dir_edit.text()
        if not os.path.exists(input_dir):
            self.add_processing_log(f"輸入目錄不存在: {input_dir}", "error")
            QMessageBox.warning(self, "錯誤", f"輸入目錄不存在: {input_dir}")
            return
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            self.add_processing_log(f"無法創建輸出目錄: {e}", "error")
            QMessageBox.warning(self, "錯誤", f"無法創建輸出目錄: {e}")
            return
        from datetime import datetime
        self.processing_start_time = datetime.now()
        if not hasattr(self, 'monitoring_timer'):
            from PyQt6.QtCore import QTimer
            self.monitoring_timer = QTimer()
            self.monitoring_timer.timeout.connect(self.update_processing_monitor)
        self.monitoring_timer.start(1000)
        config = {
            'frame_interval': self.frame_interval.value(),
            'output_format': self.video_output_format.currentText(),
            'quality': self.video_quality_spin.value(),
            'resize': self.resize_frames.isChecked(),
            'target_width': self.target_width.value(),
            'target_height': self.target_height.value(),
            'keep_aspect_ratio': self.keep_aspect_ratio.isChecked(),
            'max_workers': self.max_workers.value(),
            'dedup_frames': self.dedup_frames.isChecked(),
            'similarity_check': self.similarity_check.isChecked(),
            'similarity_threshold': self.similarity_threshold.value(),
            'use_ssim': True,
            'use_histogram': True,
            'use_edge_density': True,
            'use_feature_comparison': True,
            'histogram_threshold': self.histogram_threshold.value(),
            'edge_density_threshold': 0.02,
            'solid_color_threshold': 150.0,
            'perceptual_hash_threshold': 8,
            'feature_similarity_threshold': 0.88,
            'adaptive_interval': False,
            'min_scene_change': 0.3,
            'max_compare_frames': self.max_compare_frames.value(),
            'max_reference_frames': 15,
            'recursive': self.recursive_search.isChecked(),
            'skip_processed': self.skip_processed.isChecked(),
            'sort_by_size': False
        }
        self.data_status_label.setText("影片擷取中...")
        self.data_status_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        self.start_processing_btn.setEnabled(False)
        self.stop_processing_btn.setEnabled(True)
        self.data_progress_bar.setValue(0)
        self.processed_count_label.setText("0")
        self.generated_count_label.setText("0")
        self.processing_speed_label.setText("0.0 video/s")
        self.add_processing_log(f"開始影片擷取處理", "info")
        self.add_processing_log(f"輸入目錄: {input_dir}", "info")
        self.add_processing_log(f"輸出目錄: {output_dir}", "info")
        self.add_processing_log(f"擷取間隔: {config['frame_interval']} 幀", "info")
        self.add_processing_log(f"輸出格式: {config['output_format']}", "info")
        self.data_worker = VideoExtractionWorker(config, input_dir, output_dir)
        self.data_worker.progress_updated.connect(self.update_video_extraction_progress)
        self.data_worker.processing_completed.connect(self.video_extraction_completed)
        self.data_worker.processing_error.connect(self.enhanced_data_processing_error)
        self.data_worker.log_message.connect(self.handle_video_extraction_log)
        self.data_worker.start()
        self.logger.info(f"影片擷取開始 - {config}")

    def update_video_extraction_progress(self, current, total, speed):
        """更新影片擷取進度"""
        if total > 0:
            progress = int((current / total) * 100)
            self.data_progress_bar.setMaximum(total)
            self.data_progress_bar.setValue(current)
            self.processed_count_label.setText(str(current))
            self.processing_speed_label.setText(f"{speed:.1f} video/s")
            self.processed_files = current
            if not hasattr(self, 'total_files'):
                self.total_files = total

    def video_extraction_completed(self, final_stats):
        """影片擷取完成"""
        self.data_status_label.setText("擷取完成")
        self.data_status_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        self.data_progress_bar.setValue(self.data_progress_bar.maximum())
        self.start_processing_btn.setEnabled(True)
        self.stop_processing_btn.setEnabled(False)
        if hasattr(self, 'monitoring_timer') and self.monitoring_timer.isActive():
            self.monitoring_timer.stop()
        if hasattr(self, 'processing_start_time'):
            self.processing_start_time = None
        self.generated_count_label.setText(str(final_stats.get('total_frames', 0)))
        self.add_processing_log(f"影片擷取完成！", "success")
        self.add_processing_log(f"處理影片: {final_stats.get('success_videos', 0)} 個", "success")
        self.add_processing_log(f"擷取幀數: {final_stats.get('total_frames', 0)} 張", "success")
        self.add_processing_log(f"跳過重複: {final_stats.get('skipped_duplicate', 0)} 張", "success")
        self.add_processing_log(f"跳過相似: {final_stats.get('skipped_similar', 0)} 張", "success")
        self.add_processing_log(f"處理時間: {final_stats.get('processing_time', 0):.1f} 秒", "success")
        self.logger.info(f"影片擷取完成 - {final_stats}")
    
    def handle_video_extraction_log(self, message, level):
        """處理影片擷取的日誌消息"""
        self.add_processing_log(message, level)
    
    def stop_data_processing(self):
        """停止資料處理"""
        if self.data_worker and self.data_worker.isRunning():
            self.data_worker.stop()
            self.data_worker.wait()
            self.data_status_label.setText("已停止")
            self.start_processing_btn.setEnabled(True)
            self.stop_processing_btn.setEnabled(False)
            if hasattr(self, 'processing_timer') and self.processing_timer.isActive():
                self.processing_timer.stop()
            if hasattr(self, 'processing_start_time'):
                self.processing_start_time = None
            self.add_processing_log("處理已停止", "warning")
        
    def reset_processing_settings(self):
        """重置處理設定"""
        reply = QMessageBox.question(self, "重置設定", "確定要重置所有處理設定嗎？", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.input_dir_edit.setText("./data/input_images")
            self.output_dir_edit.setText("./data/quality_dataset_t")
            self.min_quality_spin.setValue(10)
            self.max_quality_spin.setValue(101)
            self.interval_spin.setValue(10)
            self.workers_spin.setValue(max(1, os.cpu_count() // 2))
            self.processing_mode_combo.setCurrentIndex(0)
            self.file_filter_edit.setText("jpg,jpeg,png,bmp,tiff,webp")
            self.output_quality_spin.setValue(95)
            self.memory_limit_check.setChecked(False)
            self.memory_limit_spin.setValue(4096)
            self.processed_count_label.setText("0")
            self.generated_count_label.setText("0")
            self.processing_time_label.setText("00:00:00")
            self.processing_speed_label.setText("0.0 img/s")
            self.avg_psnr_label.setText("N/A")
            self.psnr_range_label.setText("N/A ~ N/A dB")
            self.avg_ssim_label.setText("N/A")
            self.ssim_range_label.setText("N/A ~ N/A")
            self.avg_filesize_label.setText("N/A")
            self.compression_ratio_label.setText("N/A")
            self.processing_log.clear()
            self.add_processing_log("設定已重置", "info")
        
    def save_processing_log(self):
        """保存處理日誌"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存處理日誌", 
            f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "文字檔案 (*.txt);;所有檔案 (*)"
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.processing_log.toPlainText())
                self.add_processing_log(f"日誌已保存到: {file_path}", "info")
            except Exception as e:
                QMessageBox.warning(self, "錯誤", f"保存日誌失敗: {e}")
        
    def add_processing_log(self, message, level="info"):
        """添加處理日誌"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        level_prefix = {
            "info": "[資訊]",
            "warning": "[警告]", 
            "error": "[錯誤]",
            "success": "[成功]"
        }.get(level, "[資訊]")
        log_entry = f"{timestamp} {level_prefix} {message}"
        self.processing_log.append(log_entry)
        cursor = self.processing_log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.processing_log.setTextCursor(cursor)
        
    def update_processing_timer(self):
        """更新處理計時器"""
        if self.processing_start_time:
            elapsed = datetime.now() - self.processing_start_time
            self.processing_time_label.setText(str(elapsed).split('.')[0])
        
    def create_basic_settings_group(self):
        """創建基本設定群組"""
        group = QGroupBox("基本設定")
        form_layout = QFormLayout()
        form_layout.setVerticalSpacing(10)
        dataset_layout = QHBoxLayout()
        self.dataset_dir_edit = QLineEdit("./data/quality_dataset")
        self.dataset_dir_edit.setReadOnly(True)
        self.browse_dataset_btn = QPushButton("瀏覽...")
        self.browse_dataset_btn.clicked.connect(self.browse_dataset_directory)
        dataset_layout.addWidget(self.dataset_dir_edit)
        dataset_layout.addWidget(self.browse_dataset_btn)
        form_layout.addRow("資料集目錄:", dataset_layout)
        model_dir_layout = QHBoxLayout()
        self.model_dir_edit = QLineEdit("./models")
        self.browse_model_dir_btn = QPushButton("瀏覽...")
        self.browse_model_dir_btn.clicked.connect(self.browse_model_directory)
        model_dir_layout.addWidget(self.model_dir_edit)
        model_dir_layout.addWidget(self.browse_model_dir_btn)
        form_layout.addRow("模型儲存目錄:", model_dir_layout)
        model_name_layout = QHBoxLayout()
        self.model_name_edit = QLineEdit("NS-IC-Custom")
        model_name_layout.addWidget(self.model_name_edit)
        model_name_layout.addStretch()
        form_layout.addRow("模型名稱:", model_name_layout)
        form_layout.addRow("", QLabel(""))
        form_layout.addRow("模型保存選項:", QLabel(""))
        save_options_grid = QGridLayout()
        save_options_grid.setHorizontalSpacing(20)
        save_options_grid.setVerticalSpacing(5)
        self.save_best_loss_cb = QCheckBox("保存最佳損失模型")
        self.save_best_loss_cb.setChecked(True)
        save_options_grid.addWidget(self.save_best_loss_cb, 0, 0)
        self.save_best_psnr_cb = QCheckBox("保存最佳PSNR模型")
        self.save_best_psnr_cb.setChecked(True)
        save_options_grid.addWidget(self.save_best_psnr_cb, 0, 1)
        self.save_final_cb = QCheckBox("保存最終模型")
        self.save_final_cb.setChecked(True)
        save_options_grid.addWidget(self.save_final_cb, 1, 0)
        self.save_checkpoint_cb = QCheckBox("保存週期檢查點")
        save_options_grid.addWidget(self.save_checkpoint_cb, 1, 1)
        save_options_widget = QWidget()
        save_options_widget.setLayout(save_options_grid)
        form_layout.addRow("", save_options_widget)
        checkpoint_interval_layout = QHBoxLayout()
        checkpoint_interval_layout.addWidget(QLabel("檢查點間隔:"))
        self.checkpoint_interval_spin = QSpinBox()
        self.checkpoint_interval_spin.setRange(1, 100)
        self.checkpoint_interval_spin.setValue(50)
        self.checkpoint_interval_spin.setEnabled(False)
        self.checkpoint_interval_spin.setSuffix(" epochs")
        checkpoint_interval_layout.addWidget(self.checkpoint_interval_spin)
        checkpoint_interval_layout.addStretch()
        self.save_checkpoint_cb.toggled.connect(self.checkpoint_interval_spin.setEnabled)
        form_layout.addRow("", checkpoint_interval_layout)
        group.setLayout(form_layout)
        return group
    
    def create_training_parameters_group(self):
        """創建訓練參數群組"""
        group = QGroupBox("訓練參數")
        main_layout = QVBoxLayout()
        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(20)
        grid_layout.setVerticalSpacing(10)
        grid_layout.addWidget(QLabel("訓練週期數:"), 0, 0)
        self.num_epochs_spin = QSpinBox()
        self.num_epochs_spin.setRange(1, 10000)
        self.num_epochs_spin.setValue(1000)
        self.num_epochs_spin.setSuffix(" epochs")
        grid_layout.addWidget(self.num_epochs_spin, 0, 1)
        grid_layout.addWidget(QLabel("批次大小:"), 1, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 32)
        self.batch_size_spin.setValue(6)
        grid_layout.addWidget(self.batch_size_spin, 1, 1)
        grid_layout.addWidget(QLabel("裁剪大小:"), 2, 0)
        self.crop_size_spin = QSpinBox()
        self.crop_size_spin.setRange(64, 1024)
        self.crop_size_spin.setValue(256)
        self.crop_size_spin.setSuffix(" px")
        grid_layout.addWidget(self.crop_size_spin, 2, 1)
        grid_layout.addWidget(QLabel("最小品質:"), 0, 2)
        self.train_min_quality_spin = QSpinBox()
        self.train_min_quality_spin.setRange(1, 100)
        self.train_min_quality_spin.setValue(10)
        self.train_min_quality_spin.setPrefix("q")
        grid_layout.addWidget(self.train_min_quality_spin, 0, 3)
        grid_layout.addWidget(QLabel("最大品質:"), 1, 2)
        self.train_max_quality_spin = QSpinBox()
        self.train_max_quality_spin.setRange(1, 100)
        self.train_max_quality_spin.setValue(90)
        self.train_max_quality_spin.setPrefix("q")
        grid_layout.addWidget(self.train_max_quality_spin, 1, 3)
        grid_layout.addWidget(QLabel("驗證間隔:"), 2, 2)
        self.validation_interval_spin = QSpinBox()
        self.validation_interval_spin.setRange(1, 100)
        self.validation_interval_spin.setValue(1)
        self.validation_interval_spin.setSuffix(" epochs")
        grid_layout.addWidget(self.validation_interval_spin, 2, 3)
        main_layout.addLayout(grid_layout)
        validation_options_layout = QHBoxLayout()
        self.fast_validation_cb = QCheckBox("啟用快速驗證")
        validation_options_layout.addWidget(self.fast_validation_cb)
        validation_options_layout.addWidget(QLabel("驗證批次數:"))
        self.validate_batches_spin = QSpinBox()
        self.validate_batches_spin.setRange(1, 500)
        self.validate_batches_spin.setValue(50)
        self.validate_batches_spin.setEnabled(False)
        self.validate_batches_spin.setSuffix(" batches")
        self.fast_validation_cb.toggled.connect(self.validate_batches_spin.setEnabled)
        validation_options_layout.addWidget(self.validate_batches_spin)
        validation_options_layout.addStretch()
        main_layout.addLayout(validation_options_layout)
        group.setLayout(main_layout)
        return group
    
    def create_optimizer_settings_group(self):
        """創建優化器設定群組"""
        group = QGroupBox("優化器設定")
        main_layout = QVBoxLayout()
        lr_grid = QGridLayout()
        lr_grid.setHorizontalSpacing(20)
        lr_grid.setVerticalSpacing(10)
        lr_grid.addWidget(QLabel("生成器學習率:"), 0, 0)
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(1e-8, 1e-2)
        self.learning_rate_spin.setValue(5e-6)
        self.learning_rate_spin.setDecimals(8)
        self.learning_rate_spin.setSingleStep(1e-6)
        lr_grid.addWidget(self.learning_rate_spin, 0, 1)
        lr_grid.addWidget(QLabel("判別器係數:"), 0, 2)
        self.d_lr_factor_spin = QDoubleSpinBox()
        self.d_lr_factor_spin.setRange(0.1, 2.0)
        self.d_lr_factor_spin.setValue(0.5)
        self.d_lr_factor_spin.setDecimals(2)
        self.d_lr_factor_spin.setSingleStep(0.1)
        lr_grid.addWidget(self.d_lr_factor_spin, 0, 3)
        lr_grid.addWidget(QLabel("優化器類型:"), 1, 0)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["AdamW", "Adam"])
        self.optimizer_combo.setCurrentText("AdamW")
        lr_grid.addWidget(self.optimizer_combo, 1, 1)
        lr_grid.addWidget(QLabel("權重衰減:"), 1, 2)
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0, 1e-2)
        self.weight_decay_spin.setValue(1e-6)
        self.weight_decay_spin.setDecimals(8)
        self.weight_decay_spin.setSingleStep(1e-6)
        lr_grid.addWidget(self.weight_decay_spin, 1, 3)
        lr_grid.addWidget(QLabel("最小學習率:"), 2, 0)
        self.min_lr_spin = QDoubleSpinBox()
        self.min_lr_spin.setRange(1e-10, 1e-4)
        self.min_lr_spin.setValue(1e-7)
        self.min_lr_spin.setDecimals(10)
        self.min_lr_spin.setSingleStep(1e-7)
        lr_grid.addWidget(self.min_lr_spin, 2, 1)
        main_layout.addLayout(lr_grid)
        scheduler_group = QGroupBox("學習率調度器")
        scheduler_layout = QVBoxLayout()
        scheduler_type_layout = QHBoxLayout()
        scheduler_type_layout.addWidget(QLabel("調度器類型:"))
        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems(["cosine", "plateau", "step"])
        self.scheduler_combo.setCurrentText("cosine")
        self.scheduler_combo.currentTextChanged.connect(self.on_scheduler_changed)
        scheduler_type_layout.addWidget(self.scheduler_combo)
        scheduler_type_layout.addStretch()
        scheduler_layout.addLayout(scheduler_type_layout)
        self.scheduler_params_frame = QFrame()
        self.scheduler_params_layout = QGridLayout()
        self.scheduler_params_frame.setLayout(self.scheduler_params_layout)
        scheduler_layout.addWidget(self.scheduler_params_frame)
        scheduler_group.setLayout(scheduler_layout)
        main_layout.addWidget(scheduler_group)
        grad_grid = QGridLayout()
        grad_grid.setHorizontalSpacing(20)
        grad_grid.setVerticalSpacing(10)
        grad_grid.addWidget(QLabel("累積步數:"), 0, 0)
        self.grad_accum_spin = QSpinBox()
        self.grad_accum_spin.setRange(1, 16)
        self.grad_accum_spin.setValue(4)
        self.grad_accum_spin.setSuffix(" steps")
        grad_grid.addWidget(self.grad_accum_spin, 0, 1)
        grad_grid.addWidget(QLabel("梯度裁剪:"), 0, 2)
        self.max_grad_norm_spin = QDoubleSpinBox()
        self.max_grad_norm_spin.setRange(0.1, 10.0)
        self.max_grad_norm_spin.setValue(0.5)
        self.max_grad_norm_spin.setDecimals(1)
        grad_grid.addWidget(self.max_grad_norm_spin, 0, 3)
        main_layout.addLayout(grad_grid)
        self.on_scheduler_changed("cosine")
        group.setLayout(main_layout)
        return group
    
    def create_data_settings_group(self):
        """創建數據設定群組"""
        group = QGroupBox("數據設定")
        main_layout = QVBoxLayout()
        perf_grid = QGridLayout()
        perf_grid.setHorizontalSpacing(20)
        perf_grid.setVerticalSpacing(10)
        perf_grid.addWidget(QLabel("工作線程數:"), 0, 0)
        self.num_workers_spin = QSpinBox()
        self.num_workers_spin.setRange(0, 16)
        self.num_workers_spin.setValue(4)
        self.num_workers_spin.setSuffix(" 線程")
        perf_grid.addWidget(self.num_workers_spin, 0, 1)
        perf_grid.addWidget(QLabel("隨機種子:"), 0, 2)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        perf_grid.addWidget(self.seed_spin, 0, 3)
        main_layout.addLayout(perf_grid)
        memory_layout = QHBoxLayout()
        self.cache_images_cb = QCheckBox("緩存圖像到記憶體")
        self.pin_memory_cb = QCheckBox("啟用 Pin Memory")
        self.pin_memory_cb.setChecked(True)
        memory_layout.addWidget(self.cache_images_cb)
        memory_layout.addWidget(self.pin_memory_cb)
        memory_layout.addStretch()
        main_layout.addLayout(memory_layout)
        group.setLayout(main_layout)
        return group
    
    def create_training_progress_group(self):
        """創建訓練進度群組"""
        group = QGroupBox("訓練監控")
        layout = QVBoxLayout()
        layout.setSpacing(15)
        status_group = QGroupBox("訓練狀態")
        status_layout = QGridLayout()
        status_layout.setHorizontalSpacing(10)
        status_layout.setVerticalSpacing(8)
        status_layout.addWidget(QLabel("狀態:"), 0, 0)
        self.training_status_label = QLabel("準備就緒")
        self.training_status_label.setStyleSheet("font-weight: bold; color: blue;")
        status_layout.addWidget(self.training_status_label, 0, 1)
        status_layout.addWidget(QLabel("週期:"), 1, 0)
        self.current_epoch_label = QLabel("0/0")
        self.current_epoch_label.setStyleSheet("font-weight: bold;")
        status_layout.addWidget(self.current_epoch_label, 1, 1)
        status_layout.addWidget(QLabel("已用時間:"), 2, 0)
        self.elapsed_time_label = QLabel("00:00:00")
        self.elapsed_time_label.setStyleSheet("font-weight: bold;")
        status_layout.addWidget(self.elapsed_time_label, 2, 1)
        status_layout.addWidget(QLabel("預估剩餘:"), 3, 0)
        self.eta_label = QLabel("00:00:00")
        self.eta_label.setStyleSheet("font-weight: bold;")
        status_layout.addWidget(self.eta_label, 3, 1)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        progress_group = QGroupBox("訓練進度")
        progress_layout = QVBoxLayout()
        self.training_progress_bar = QProgressBar()
        self.training_progress_bar.setRange(0, 100)
        self.training_progress_bar.setTextVisible(True)
        self.training_progress_bar.setMinimumHeight(25)
        progress_layout.addWidget(self.training_progress_bar)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        metrics_group = QGroupBox("訓練指標")
        metrics_layout = QGridLayout()
        metrics_layout.setHorizontalSpacing(10)
        metrics_layout.setVerticalSpacing(8)
        metrics_layout.addWidget(QLabel("生成器損失:"), 0, 0)
        self.g_loss_label = QLabel("0.0000")
        self.g_loss_label.setStyleSheet("font-weight: bold; color: red;")
        metrics_layout.addWidget(self.g_loss_label, 0, 1)
        metrics_layout.addWidget(QLabel("判別器損失:"), 1, 0)
        self.d_loss_label = QLabel("0.0000")
        self.d_loss_label.setStyleSheet("font-weight: bold; color: red;")
        metrics_layout.addWidget(self.d_loss_label, 1, 1)
        metrics_layout.addWidget(QLabel("PSNR:"), 2, 0)
        self.psnr_label = QLabel("0.00 dB")
        self.psnr_label.setStyleSheet("font-weight: bold; color: green;")
        metrics_layout.addWidget(self.psnr_label, 2, 1)
        metrics_layout.addWidget(QLabel("SSIM:"), 3, 0)
        self.ssim_label = QLabel("0.0000")
        self.ssim_label.setStyleSheet("font-weight: bold; color: green;")
        metrics_layout.addWidget(self.ssim_label, 3, 1)
        metrics_layout.addWidget(QLabel("學習率:"), 4, 0)
        self.lr_label = QLabel("0.000000")
        self.lr_label.setStyleSheet("font-weight: bold; color: purple;")
        metrics_layout.addWidget(self.lr_label, 4, 1)
        metrics_layout.addWidget(QLabel("GPU記憶體:"), 5, 0)
        self.gpu_memory_label = QLabel("0 MB")
        self.gpu_memory_label.setStyleSheet("font-weight: bold; color: orange;")
        metrics_layout.addWidget(self.gpu_memory_label, 5, 1)
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        log_group = QGroupBox("訓練日誌")
        log_layout = QVBoxLayout()
        self.training_log = QTextEdit()
        self.training_log.setMaximumHeight(200)
        self.training_log.setReadOnly(True)
        self.training_log.setPlainText("等待訓練開始...")
        self.training_log.setStyleSheet("""
            QTextEdit {
                background-color: #2c3e50;
                color: #ecf0f1;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                border: 1px solid #34495e;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        log_layout.addWidget(self.training_log)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        group.setLayout(layout)
        return group
    
    def on_scheduler_changed(self, scheduler_type):
        """當學習率調度器類型改變時更新參數"""
        for i in reversed(range(self.scheduler_params_layout.count())): 
            child = self.scheduler_params_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        if scheduler_type == "cosine":
            t_max_label = QLabel("T_max:")
            self.cosine_t_max_spin = QSpinBox()
            self.cosine_t_max_spin.setRange(1, 10000)
            self.cosine_t_max_spin.setValue(1000)
            self.cosine_t_max_spin.setSuffix(" epochs")
            self.scheduler_params_layout.addWidget(t_max_label, 0, 0)
            self.scheduler_params_layout.addWidget(self.cosine_t_max_spin, 0, 1)
        elif scheduler_type == "plateau":
            patience_label = QLabel("耐心值:")
            self.plateau_patience_spin = QSpinBox()
            self.plateau_patience_spin.setRange(1, 100)
            self.plateau_patience_spin.setValue(15)
            self.plateau_patience_spin.setSuffix(" epochs")
            factor_label = QLabel("衰減因子:")
            self.plateau_factor_spin = QDoubleSpinBox()
            self.plateau_factor_spin.setRange(0.01, 0.9)
            self.plateau_factor_spin.setValue(0.2)
            self.plateau_factor_spin.setDecimals(2)
            self.scheduler_params_layout.addWidget(patience_label, 0, 0)
            self.scheduler_params_layout.addWidget(self.plateau_patience_spin, 0, 1)
            self.scheduler_params_layout.addWidget(factor_label, 1, 0)
            self.scheduler_params_layout.addWidget(self.plateau_factor_spin, 1, 1)
        elif scheduler_type == "step":
            step_label = QLabel("步長:")
            self.step_size_spin = QSpinBox()
            self.step_size_spin.setRange(1, 1000)
            self.step_size_spin.setValue(100)
            self.step_size_spin.setSuffix(" epochs")
            gamma_label = QLabel("衰減因子:")
            self.step_gamma_spin = QDoubleSpinBox()
            self.step_gamma_spin.setRange(0.01, 0.9)
            self.step_gamma_spin.setValue(0.5)
            self.step_gamma_spin.setDecimals(2)
            self.scheduler_params_layout.addWidget(step_label, 0, 0)
            self.scheduler_params_layout.addWidget(self.step_size_spin, 0, 1)
            self.scheduler_params_layout.addWidget(gamma_label, 1, 0)
            self.scheduler_params_layout.addWidget(self.step_gamma_spin, 1, 1)
    
    def browse_input_directory(self):
        """瀏覽輸入目錄"""
        directory = QFileDialog.getExistingDirectory(self, "選擇輸入圖像目錄")
        if directory:
            self.input_dir_edit.setText(directory)
    
    def browse_output_directory(self):
        """瀏覽輸出目錄"""
        directory = QFileDialog.getExistingDirectory(self, "選擇輸出目錄")
        if directory:
            self.output_dir_edit.setText(directory)
    
    def browse_dataset_directory(self):
        """瀏覽資料集目錄"""
        directory = QFileDialog.getExistingDirectory(self, "選擇訓練資料集目錄")
        if directory:
            self.dataset_dir_edit.setText(directory)
    
    def browse_model_directory(self):
        """瀏覽模型保存目錄"""
        directory = QFileDialog.getExistingDirectory(self, "選擇模型保存目錄")
        if directory:
            self.model_dir_edit.setText(directory)
    
    def start_data_processing(self):
        """開始資料處理"""
        input_dir = self.input_dir_edit.text()
        output_dir = self.output_dir_edit.text()
        if not os.path.exists(input_dir):
            self.add_processing_log(f"輸入目錄不存在: {input_dir}", "error")
            QMessageBox.warning(self, "錯誤", f"輸入目錄不存在: {input_dir}")
            return
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            self.add_processing_log(f"無法創建輸出目錄: {e}", "error")
            QMessageBox.warning(self, "錯誤", f"無法創建輸出目錄: {e}")
            return
        selected_mode = self.processing_mode_combo.currentText()
        if "影片擷取" in selected_mode:
            self.start_video_extraction()
            return
        mode_map = {
            1: ProcessingMode.JPEG_COMPRESSION,
            2: ProcessingMode.NOISE_ADDITION,
            3: ProcessingMode.PIXELATION,
            4: ProcessingMode.BLUR_EFFECTS,
            5: ProcessingMode.COLOR_DISTORTION,
            6: ProcessingMode.MIXED_DEGRADATION,
            7: ProcessingMode.CUSTOM_PIPELINE
        }
        processing_mode = mode_map.get(self.processing_mode_combo.currentIndex(), ProcessingMode.JPEG_COMPRESSION)
        from datetime import datetime
        self.processing_start_time = datetime.now()
        self.total_files_to_process = 0
        try:
            supported_formats = self.input_filter_edit.text().split(',')
            file_count = 0
            for fmt in supported_formats:
                fmt = fmt.strip()
                pattern = os.path.join(input_dir, f"*.{fmt}")
                import glob
                file_count += len(glob.glob(pattern))
            self.total_files_to_process = file_count
            if file_count > 0:
                quality_levels = len(range(self.min_quality_spin.value(), 
                                         self.max_quality_spin.value() + 1, 
                                         self.interval_spin.value()))
                total_outputs = file_count * quality_levels
                self.estimated_output_label.setText(str(total_outputs))
            else:
                self.estimated_output_label.setText("0")
        except Exception as e:
            self.add_processing_log(f"預計算檔案數失敗: {e}", "warning")
            self.estimated_output_label.setText("未知")
        if not hasattr(self, 'monitoring_timer'):
            from PyQt6.QtCore import QTimer
            self.monitoring_timer = QTimer()
            self.monitoring_timer.timeout.connect(self.update_processing_monitor)
        self.monitoring_timer.start(1000)
        processor = DataProcessor(
            min_quality=self.min_quality_spin.value(),
            max_quality=self.max_quality_spin.value(),
            quality_interval=self.interval_spin.value(),
            processing_mode=processing_mode,
            num_workers=self.workers_spin.value(),
            preserve_metadata=self.preserve_metadata_check.isChecked(),
            generate_previews=self.generate_previews_check.isChecked(),
            enable_validation=True
        )
        processor.update_config(
            output_format=self.output_format_combo.currentText(),
            output_quality=self.output_quality_spin.value(),
            resize_images=self.resize_images_check.isChecked(),
            target_size=(self.target_width_spin.value(), self.target_height_spin.value()),
            maintain_aspect_ratio=self.maintain_aspect_check.isChecked(),
            skip_existing=self.skip_existing_check.isChecked(),
            create_backup=self.create_backup_check.isChecked(),
            validate_output=True
        )
        processor.reset_stats()
        self.data_worker = EnhancedDataProcessingWorker(processor, input_dir, output_dir)
        self.data_worker.progress_updated.connect(self.update_enhanced_data_progress)
        self.data_worker.processing_completed.connect(self.enhanced_data_processing_completed)
        self.data_worker.processing_error.connect(self.enhanced_data_processing_error)
        self.data_worker.stats_updated.connect(self.update_processing_stats)
        self.start_processing_btn.setEnabled(False)
        self.stop_processing_btn.setEnabled(True)
        self.data_status_label.setText("正在處理...")
        self.data_status_label.setStyleSheet("font-weight: bold; color: #FF9800;")
        self.data_progress_bar.setValue(0)
        self.processing_start_time = datetime.now()
        self.processing_timer.start(1000)
        self.add_processing_log(f"開始處理模式: {processing_mode.value}", "info")
        self.add_processing_log(f"輸入目錄: {input_dir}", "info")
        self.add_processing_log(f"輸出目錄: {output_dir}", "info")
        self.add_processing_log(f"品質範圍: {self.min_quality_spin.value()}-{self.max_quality_spin.value()}", "info")
        self.add_processing_log(f"工作程序數: {self.workers_spin.value()}", "info")
        self.data_worker.start()
    
    @pyqtSlot(int, int, float)
    def update_enhanced_data_progress(self, current, total, speed):
        """更新增強的資料處理進度"""
        if total > 0:
            progress = int((current / total) * 100)
            self.data_progress_bar.setMaximum(total)
            self.data_progress_bar.setValue(current)
            self.processed_count_label.setText(str(current))
            self.processing_speed_label.setText(f"{speed:.1f} img/s")
            self.processed_files = current
            if not hasattr(self, 'total_files'):
                self.total_files = total
    
    @pyqtSlot(dict)
    def update_processing_stats(self, stats):
        """更新處理統計資訊"""
        self.generated_count_label.setText(str(stats.get('total_generated', 0)))
        if stats.get('average_psnr', 0) > 0:
            self.avg_psnr_label.setText(f"{stats['average_psnr']:.2f} dB")
            min_psnr = stats.get('min_psnr', 0)
            max_psnr = stats.get('max_psnr', 0)
            self.psnr_range_label.setText(f"{min_psnr:.2f} ~ {max_psnr:.2f} dB")
        if stats.get('average_ssim', 0) > 0:
            self.avg_ssim_label.setText(f"{stats['average_ssim']:.4f}")
            min_ssim = stats.get('min_ssim', 0)
            max_ssim = stats.get('max_ssim', 0)
            self.ssim_range_label.setText(f"{min_ssim:.4f} ~ {max_ssim:.4f}")
        if stats.get('file_sizes'):
            avg_size = np.mean(stats['file_sizes'])
            total_size = sum(stats['file_sizes'])
            self.avg_filesize_label.setText(self.format_file_size(avg_size))
            if hasattr(self, 'original_total_size'):
                compression_ratio = total_size / self.original_total_size
                self.compression_ratio_label.setText(f"{compression_ratio:.2f}x")
    
    def format_file_size(self, size_bytes):
        """格式化檔案大小"""
        if size_bytes < 1024:
            return f"{size_bytes:.0f} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes/1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes/(1024*1024):.1f} MB"
        else:
            return f"{size_bytes/(1024*1024*1024):.1f} GB"
    
    @pyqtSlot(dict)
    def enhanced_data_processing_completed(self, final_stats):
        """增強的資料處理完成"""
        self.data_status_label.setText("處理完成")
        self.data_status_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        self.data_progress_bar.setValue(self.data_progress_bar.maximum())
        self.start_processing_btn.setEnabled(True)
        self.stop_processing_btn.setEnabled(False)
        self.processing_timer.stop()
        if hasattr(self, 'monitoring_timer') and self.monitoring_timer.isActive():
            self.monitoring_timer.stop()
        if hasattr(self, 'processing_start_time'):
            self.processing_start_time = None
        self.update_processing_stats(final_stats)
        self.add_processing_log(f"處理完成！", "success")
        self.add_processing_log(f"總共處理: {final_stats.get('total_processed', 0)} 張圖片", "success")
        self.add_processing_log(f"總共生成: {final_stats.get('total_generated', 0)} 張圖片", "success")
        self.add_processing_log(f"處理時間: {final_stats.get('processing_time', 0):.1f} 秒", "success")
        if final_stats.get('average_psnr', 0) > 0:
            self.add_processing_log(f"平均PSNR: {final_stats['average_psnr']:.2f} dB", "success")
        self.logger.info(f"資料處理完成 - {final_stats}")
    
    @pyqtSlot(str)
    def enhanced_data_processing_error(self, error_message):
        """增強的資料處理錯誤"""
        self.data_status_label.setText("處理錯誤")
        self.data_status_label.setStyleSheet("font-weight: bold; color: #F44336;")
        self.start_processing_btn.setEnabled(True)
        self.stop_processing_btn.setEnabled(False)
        self.processing_timer.stop()
        if hasattr(self, 'monitoring_timer') and self.monitoring_timer.isActive():
            self.monitoring_timer.stop()
        if hasattr(self, 'processing_start_time'):
            self.processing_start_time = None
        self.add_processing_log(f"處理錯誤: {error_message}", "error")
        self.logger.error(f"資料處理錯誤: {error_message}")
        QMessageBox.critical(self, "處理錯誤", f"資料處理時發生錯誤:\n{error_message}")
    
    @pyqtSlot(int, int)
    def update_data_progress(self, current, total):
        """更新資料處理進度（兼容性方法）"""
        self.update_enhanced_data_progress(current, total, 0.0)
    
    @pyqtSlot(int)
    def data_processing_completed(self, total_processed):
        """資料處理完成（兼容性方法）"""
        stats = {'total_processed': total_processed, 'total_generated': 0, 'processing_time': 0}
        self.enhanced_data_processing_completed(stats)
    
    @pyqtSlot(str)
    def data_processing_error(self, error_message):
        """資料處理錯誤（兼容性方法）"""
        self.enhanced_data_processing_error(error_message)
    
    def start_model_training(self):
        """開始模型訓練"""
        training_args = self.get_training_args()
        save_options = self.get_save_options()
        trainer = Trainer(
            data_dir=self.dataset_dir_edit.text(),
            save_dir=self.model_dir_edit.text(),
            log_dir="./logs",
            batch_size=self.batch_size_spin.value(),
            learning_rate=self.learning_rate_spin.value(),
            num_epochs=self.num_epochs_spin.value(),
            save_options=save_options,
            training_args=training_args
        )
        self.training_worker = TrainingWorker(trainer)
        self.training_worker.progress_updated.connect(self.update_training_progress)
        self.training_worker.epoch_completed.connect(self.training_epoch_completed)
        self.training_worker.training_completed.connect(self.training_completed)
        self.training_worker.training_error.connect(self.training_error)
        self.start_training_btn.setEnabled(False)
        self.stop_training_btn.setEnabled(True)
        self.training_status_label.setText("訓練中...")
        self.training_start_time = time.time()
        self.training_worker.start()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time_display)
        self.timer.start(1000)
    
    def stop_model_training(self):
        """停止模型訓練"""
        if self.training_worker and self.training_worker.isRunning():
            self.training_worker.stop()
            self.training_status_label.setText("正在停止...")
            self.stop_training_btn.setEnabled(False)
    
    def get_training_args(self):
        """獲取訓練參數"""
        args = {
            'crop_size': self.crop_size_spin.value(),
            'min_quality': self.train_min_quality_spin.value(),
            'max_quality': self.train_max_quality_spin.value(),
            'optimizer': self.optimizer_combo.currentText(),
            'weight_decay': self.weight_decay_spin.value(),
            'scheduler': self.scheduler_combo.currentText(),
            'd_lr_factor': self.d_lr_factor_spin.value(),
            'grad_accum': self.grad_accum_spin.value(),
            'max_grad_norm': self.max_grad_norm_spin.value(),
            'min_lr': self.min_lr_spin.value(),
            'cache_images': self.cache_images_cb.isChecked(),
            'validation_interval': self.validation_interval_spin.value(),
            'fast_validation': self.fast_validation_cb.isChecked(),
            'validate_batches': self.validate_batches_spin.value(),
            'checkpoint_interval': self.checkpoint_interval_spin.value(),
            'num_workers': self.num_workers_spin.value(),
            'pin_memory': self.pin_memory_cb.isChecked()
        }
        scheduler_type = self.scheduler_combo.currentText()
        if scheduler_type == "cosine":
            args['cosine_t_max'] = self.cosine_t_max_spin.value()
        elif scheduler_type == "plateau":
            args['plateau_patience'] = self.plateau_patience_spin.value()
            args['plateau_factor'] = self.plateau_factor_spin.value()
        elif scheduler_type == "step":
            args['step_size'] = self.step_size_spin.value()
            args['step_gamma'] = self.step_gamma_spin.value()
        return args
    
    def get_save_options(self):
        """獲取保存選項"""
        return {
            'save_best_loss': self.save_best_loss_cb.isChecked(),
            'save_best_psnr': self.save_best_psnr_cb.isChecked(),
            'save_final': self.save_final_cb.isChecked(),
            'save_checkpoint': self.save_checkpoint_cb.isChecked(),
            'checkpoint_interval': self.checkpoint_interval_spin.value()
        }
    
    @pyqtSlot(int, int, int, float, float)
    def update_training_progress(self, epoch, batch, total_batches, g_loss, d_loss):
        """更新訓練進度"""
        epoch_progress = int((batch / total_batches) * 100)
        self.training_progress_bar.setValue(epoch_progress)
        self.current_epoch_label.setText(f"{epoch + 1}/{self.num_epochs_spin.value()}")
        self.g_loss_label.setText(f"{g_loss:.4f}")
        self.d_loss_label.setText(f"{d_loss:.4f}")
        log_message = f"Epoch {epoch + 1}, Batch {batch + 1}/{total_batches}: G={g_loss:.4f}, D={d_loss:.4f}\n"
        self.training_log.append(log_message.strip())
        cursor = self.training_log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.training_log.setTextCursor(cursor)
    
    @pyqtSlot(int, float, float, float)
    def training_epoch_completed(self, epoch, g_loss, d_loss, psnr):
        """訓練週期完成"""
        self.psnr_label.setText(f"{psnr:.4f} dB")
        total_progress = int(((epoch + 1) / self.num_epochs_spin.value()) * 100)
        log_message = f"Epoch {epoch + 1} 完成 - G: {g_loss:.4f}, D: {d_loss:.4f}, PSNR: {psnr:.4f} dB\n"
        self.training_log.append(log_message.strip())
    
    @pyqtSlot()
    def training_completed(self):
        """訓練完成"""
        self.training_status_label.setText("訓練完成")
        self.start_training_btn.setEnabled(True)
        self.stop_training_btn.setEnabled(False)
        self.training_progress_bar.setValue(100)
        if hasattr(self, 'timer'):
            self.timer.stop()
        self.training_log.append("=== 訓練完成 ===")
        self.logger.info("模型訓練完成")
    
    @pyqtSlot(str)
    def training_error(self, error_message):
        """訓練錯誤"""
        self.training_status_label.setText(f"錯誤: {error_message}")
        self.start_training_btn.setEnabled(True)
        self.stop_training_btn.setEnabled(False)
        if hasattr(self, 'timer'):
            self.timer.stop()
        self.training_log.append(f"錯誤: {error_message}")
        self.logger.error(f"訓練錯誤: {error_message}")
    
    def update_time_display(self):
        """更新時間顯示"""
        if self.training_start_time:
            elapsed_seconds = time.time() - self.training_start_time
            elapsed_time = self.format_time(elapsed_seconds)
            self.elapsed_time_label.setText(elapsed_time)
            current_epoch = int(self.current_epoch_label.text().split('/')[0]) if '/' in self.current_epoch_label.text() else 0
            total_epochs = self.num_epochs_spin.value()
            if current_epoch > 0:
                avg_time_per_epoch = elapsed_seconds / current_epoch
                remaining_epochs = total_epochs - current_epoch
                eta_seconds = avg_time_per_epoch * remaining_epochs
                eta_time = self.format_time(eta_seconds)
                self.eta_label.setText(eta_time)
    
    def format_time(self, seconds):
        """格式化時間顯示"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def browse_dataset_folder(self):
        """瀏覽選擇資料集資料夾"""
        folder = QFileDialog.getExistingDirectory(
            self, "選擇資料集資料夾", 
            self.dataset_path_edit.text() or "."
        )
        if folder:
            self.dataset_path_edit.setText(folder)
    
    def browse_model_save_folder(self):
        """瀏覽選擇模型保存資料夾"""
        folder = QFileDialog.getExistingDirectory(
            self, "選擇模型保存資料夾", 
            self.model_save_path_edit.text() or "./models"
        )
        if folder:
            self.model_save_path_edit.setText(folder)
    
    def start_training(self):
        """開始訓練"""
        if self.is_training:
            return
        try:
            config = self.collect_training_config()
            from src.training.NS_Trainer import Trainer
            self.trainer = Trainer(config)
            self.trainer.set_callback(self.on_training_update)
            self.is_training = True
            self.start_button.setText("訓練中...")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.progress_bar.setValue(0)
            self.status_label.setText("準備開始訓練...")
            import threading
            self.training_thread = threading.Thread(target=self.trainer.train)
            self.training_thread.start()
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"啟動訓練失敗：{str(e)}")
            self.is_training = False
            self.start_button.setText("開始訓練")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
    
    def stop_training(self):
        """停止訓練"""
        if not self.is_training:
            return
        reply = QMessageBox.question(
            self, "確認", "確定要停止訓練嗎？", 
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            if hasattr(self, 'trainer'):
                self.trainer.stop_training()
            self.is_training = False
            self.start_button.setText("開始訓練")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.status_label.setText("訓練已停止")
    
    def on_training_update(self, info):
        """訓練進度更新回調"""
        if 'epoch' in info and 'total_epochs' in info:
            progress = int((info['epoch'] / info['total_epochs']) * 100)
            self.progress_bar.setValue(progress)
        if 'status' in info:
            self.status_label.setText(info['status'])
        if 'loss' in info:
            self.loss_value_label.setText(f"{info['loss']:.6f}")
        if 'psnr' in info:
            self.psnr_value_label.setText(f"{info['psnr']:.2f} dB")
        if 'ssim' in info:
            self.ssim_value_label.setText(f"{info['ssim']:.4f}")
        if 'lr' in info:
            self.lr_value_label.setText(f"{info['lr']:.2e}")
        if info.get('completed', False):
            self.is_training = False
            self.start_button.setText("開始訓練")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.status_label.setText("訓練完成")
    
    def collect_training_config(self):
        """收集所有訓練配置參數"""
        config = {
            'dataset_path': self.dataset_path_edit.text(),
            'model_save_path': self.model_save_path_edit.text(),
            'model_name': self.model_name_edit.text(),
            'epochs': self.epochs_spin.value(),
            'batch_size': self.batch_size_spin.value(),
            'learning_rate': self.learning_rate_spin.value(),
            'optimizer': self.optimizer_combo.currentText(),
            'weight_decay': self.weight_decay_spin.value(),
            'beta1': self.beta1_spin.value(),
            'beta2': self.beta2_spin.value(),
            'scheduler': self.scheduler_combo.currentText(),
            'mse_weight': self.mse_weight_spin.value(),
            'perceptual_weight': self.perceptual_weight_spin.value(),
            'ssim_weight': self.ssim_weight_spin.value(),
            'adversarial_weight': self.adversarial_weight_spin.value(),
            'save_frequency': self.save_frequency_spin.value(),
            'validation_frequency': self.validation_frequency_spin.value(),
            'mixed_precision': self.mixed_precision_check.isChecked(),
            'gradient_clipping': self.gradient_clipping_check.isChecked(),
            'max_norm': self.max_norm_spin.value() if self.gradient_clipping_check.isChecked() else None,
        }
        scheduler_type = self.scheduler_combo.currentText()
        if scheduler_type == "cosine" and hasattr(self, 'cosine_t_max_spin'):
            config['scheduler_params'] = {'T_max': self.cosine_t_max_spin.value()}
        elif scheduler_type == "plateau" and hasattr(self, 'plateau_patience_spin'):
            config['scheduler_params'] = {
                'patience': self.plateau_patience_spin.value(),
                'factor': self.plateau_factor_spin.value()
            }
        elif scheduler_type == "step" and hasattr(self, 'step_size_spin'):
            config['scheduler_params'] = {
                'step_size': self.step_size_spin.value(),
                'gamma': self.step_gamma_spin.value()
            }
        return config
