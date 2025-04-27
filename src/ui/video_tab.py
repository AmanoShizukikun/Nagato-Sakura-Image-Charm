import os
import sys
import time
import cv2
import logging
import torch
from PIL import Image
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                            QFileDialog, QProgressBar, QComboBox, QSpinBox, QCheckBox, 
                            QGroupBox, QMessageBox, QDoubleSpinBox, QRadioButton, QButtonGroup,
                            QSplitter, QFrame, QToolButton, QScrollArea, QLineEdit, QSlider,
                            QStackedWidget)
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QIcon, QFont, QIntValidator

from src.ui.views import MultiViewWidget
from src.threads.NS_VideoThread import VideoEnhancerThread

from src.utils.NS_DeviceInfo import get_system_info, get_device_options, get_device_name


logger = logging.getLogger(__name__)

class CollapsibleBox(QWidget):
    """可折疊的參數區塊"""
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.toggle_button = QToolButton(self)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)
        self.toggle_button.setArrowType(Qt.ArrowType.DownArrow)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_button.setText(title)
        self.toggle_button.setStyleSheet("QToolButton { font-weight: bold; }")
        self.toggle_button.setIconSize(QSize(14, 14))
        self.content_area = QScrollArea()
        self.content_area.setFrameShape(QFrame.Shape.NoFrame)
        self.content_area.setWidgetResizable(True)
        lay = QVBoxLayout(self)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toggle_button)
        lay.addWidget(self.content_area)
        self.toggle_button.clicked.connect(self.on_toggle)
        self.content_frame = QFrame()
        self.content_layout = QVBoxLayout()
        self.content_frame.setLayout(self.content_layout)
        self.content_area.setWidget(self.content_frame)
    
    def on_toggle(self, checked):
        """處理折疊/展開事件"""
        if checked:
            self.toggle_button.setArrowType(Qt.ArrowType.DownArrow)
            self.content_area.show()
        else:
            self.toggle_button.setArrowType(Qt.ArrowType.RightArrow)
            self.content_area.hide()
    
    def setContentLayout(self, layout):
        """設置內容區域的布局"""
        self.content_layout.addLayout(layout)
        self.on_toggle(self.toggle_button.isChecked())

class VideoProcessingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.input_video_path = None
        self.original_video_size = (0, 0)
        self.system_info = get_system_info()
        self.setup_ui()
        self.current_frame_index = 0
        self.frame_cache = {}  
        self.model_manager = None
    
    def set_model_manager(self, model_manager):
        """設置模型管理器"""
        self.model_manager = model_manager
        self.reload_models()
    
    def reload_models(self):
        """重新載入模型列表"""
        if not hasattr(self, 'model_selector'):
            return
        if not self.model_manager:
            return
        try:
            self.model_selector.blockSignals(True)
            self.model_selector.clear()
            available_models = self.model_manager.get_available_models()
            for model_path in available_models:
                model_name = os.path.basename(model_path)
                display_text = f"{model_name}"
                self.model_selector.addItem(display_text, model_path)
            registered_path = self.model_manager.get_registered_model_path()
            if registered_path:
                for i in range(self.model_selector.count()):
                    if self.model_selector.itemData(i) == registered_path:
                        self.model_selector.setCurrentIndex(i)
                        break
        finally:
            self.model_selector.blockSignals(False)
    
    def on_registered_model_changed(self, model_path):
        """當已選擇的模型變更時調用"""
        self.reload_models()
    
    def on_model_selected(self, index):
        """當用戶選擇不同模型時觸發"""
        if index < 0 or not self.model_manager:
            return
        model_path = self.model_selector.itemData(index)
        if model_path:
            success = self.model_manager.register_model(model_path)
            if success:
                model_name = os.path.basename(model_path)
                if self.parent:
                    self.parent.statusBar.showMessage(f"已選擇模型: {model_name}")
                logger.info(f"已選擇模型: {model_name}")
                self.reload_models()
    
    def update_vid_strength_label(self, value):
        """更新影片處理強度標籤顯示"""
        actual_strength = (value + 1) * 10
        self.vid_strength_value_label.setText(f"{actual_strength}%")
    
    def update_scale_factor_info(self, value):
        """更新超分倍率資訊"""
        factor = value
        if self.original_video_size != (0, 0):
            new_width = int(self.original_video_size[0] * factor)
            new_height = int(self.original_video_size[1] * factor)
            self.scale_size_info_label.setText(f"輸出尺寸: {new_width} x {new_height} 像素")
    
    def update_custom_size_info(self):
        """更新自訂尺寸資訊"""
        try:
            width = int(self.width_input.text() or "0")
            height = int(self.height_input.text() or "0")
            if width <= 0 or height <= 0:
                self.custom_size_info_label.setText("請輸入有效的寬度和高度")
                return
            self.custom_size_info_label.setText(f"輸出尺寸: {width} x {height} 像素")
        except ValueError:
            self.custom_size_info_label.setText("請輸入有效的寬度和高度")
    
    def on_size_option_changed(self):
        self.scale_factor_spinbox.setEnabled(self.upscale_radio.isChecked())
        self.scale_size_info_label.setEnabled(self.upscale_radio.isChecked())
        self.width_input.setEnabled(self.custom_size_radio.isChecked())
        self.height_input.setEnabled(self.custom_size_radio.isChecked())
        self.custom_size_info_label.setEnabled(self.custom_size_radio.isChecked())
        if self.upscale_radio.isChecked():
            self.update_scale_factor_info(self.scale_factor_spinbox.value())
        if self.custom_size_radio.isChecked():
            self.update_custom_size_info()
    
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        self.multi_view = MultiViewWidget(self)
        self.multi_view.image_a_name = "原始幀預覽"
        self.multi_view.image_b_name = "增強幀預覽"
        self.multi_view.image_a_group.setTitle("原始幀預覽")
        self.multi_view.image_b_group.setTitle("增強幀預覽")
        preview_layout.addWidget(self.multi_view)
        info_control_layout = QHBoxLayout()
        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel("當前影片:"))
        self.vid_path_label = QLabel("未選擇影片")
        info_layout.addWidget(self.vid_path_label)
        info_layout.addStretch()
        info_control_layout.addLayout(info_layout, 1)
        controls_layout = QHBoxLayout()
        self.toggle_params_btn = QPushButton("顯示參數面板")
        self.toggle_params_btn.setCheckable(True)
        self.toggle_params_btn.setChecked(False)
        self.toggle_params_btn.clicked.connect(self.toggle_params_panel)
        controls_layout.addWidget(self.toggle_params_btn)
        self.open_video_button = QPushButton("開啟影片")
        self.open_video_button.clicked.connect(self.open_video)
        controls_layout.addWidget(self.open_video_button)
        self.enhance_video_button = QPushButton("優化影片")
        self.enhance_video_button.setEnabled(False)
        self.enhance_video_button.clicked.connect(self.enhance_video)
        controls_layout.addWidget(self.enhance_video_button)
        self.stop_processing_button = QPushButton("停止處理")
        self.stop_processing_button.setEnabled(False)
        self.stop_processing_button.clicked.connect(self.stop_video_processing)
        controls_layout.addWidget(self.stop_processing_button)
        info_control_layout.addLayout(controls_layout)
        preview_layout.addLayout(info_control_layout)
        progress_layout = QHBoxLayout()
        self.vid_progress_bar = QProgressBar()
        self.vid_progress_bar.setValue(0)
        progress_layout.addWidget(self.vid_progress_bar, 3)
        self.vid_status_label = QLabel("等待處理...")
        progress_layout.addWidget(self.vid_status_label, 2)
        self.vid_remaining_label = QLabel("預計剩餘時間: --:--:--")
        progress_layout.addWidget(self.vid_remaining_label, 1)
        self.model_status_label = QLabel("模型狀態: 未載入")
        progress_layout.addWidget(self.model_status_label, 1)
        preview_layout.addLayout(progress_layout)
        params_widget = QWidget()
        params_layout = QVBoxLayout(params_widget)
        params_layout.setContentsMargins(0, 0, 0, 0)
        params_header = QLabel("參數設置")
        params_header.setStyleSheet("font-size: 14px; font-weight: bold;")
        params_layout.addWidget(params_header)
        params_content_layout = QHBoxLayout()
        # === 模型參數區塊 ===
        model_param_box = CollapsibleBox("模型參數設定")
        model_param_layout = QVBoxLayout()
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("選擇模型:"))
        self.model_selector = QComboBox()
        self.model_selector.setMinimumWidth(300)
        self.model_selector.currentIndexChanged.connect(self.on_model_selected)
        model_layout.addWidget(self.model_selector)
        model_param_layout.addLayout(model_layout)
        block_size_layout = QHBoxLayout()
        block_size_layout.addWidget(QLabel("區塊大小:"))
        self.vid_block_size_spin = QSpinBox()
        self.vid_block_size_spin.setRange(128, 512)
        self.vid_block_size_spin.setValue(256)
        self.vid_block_size_spin.setSingleStep(32)
        block_size_layout.addWidget(self.vid_block_size_spin)
        model_param_layout.addLayout(block_size_layout)
        overlap_layout = QHBoxLayout()
        overlap_layout.addWidget(QLabel("重疊大小:"))
        self.vid_overlap_spin = QSpinBox()
        self.vid_overlap_spin.setRange(16, 256)
        self.vid_overlap_spin.setValue(128)
        self.vid_overlap_spin.setSingleStep(16)
        overlap_layout.addWidget(self.vid_overlap_spin)
        model_param_layout.addLayout(overlap_layout)
        weight_mask_layout = QHBoxLayout()
        self.vid_weight_mask_check = QCheckBox("使用權重遮罩")
        self.vid_weight_mask_check.setChecked(True)
        weight_mask_layout.addWidget(self.vid_weight_mask_check)
        model_param_layout.addLayout(weight_mask_layout)
        blending_layout = QHBoxLayout()
        blending_layout.addWidget(QLabel("混合模式:"))
        self.vid_blending_combo = QComboBox()
        self.vid_blending_combo.addItems(['高斯分佈', '改進型高斯分佈', '線性分佈', '餘弦分佈', '泊松分佈'])
        self.vid_blending_combo.setCurrentText('改進型高斯分佈')
        blending_layout.addWidget(self.vid_blending_combo)
        model_param_layout.addLayout(blending_layout)
        strength_layout = QHBoxLayout()
        strength_layout.addWidget(QLabel("處理強度:"))
        self.vid_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.vid_strength_slider.setRange(0, 9)
        self.vid_strength_slider.setValue(9)
        self.vid_strength_slider.setTickInterval(1)
        self.vid_strength_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.vid_strength_value_label = QLabel("100%")
        self.vid_strength_value_label.setMinimumWidth(50)
        self.vid_strength_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.vid_strength_slider.valueChanged.connect(self.update_vid_strength_label)
        strength_layout.addWidget(self.vid_strength_slider)
        strength_layout.addWidget(self.vid_strength_value_label)
        model_param_layout.addLayout(strength_layout)
        model_param_box.setContentLayout(model_param_layout)
        params_content_layout.addWidget(model_param_box, 1)
        # === 影片參數區塊 ===
        video_param_box = CollapsibleBox("影片尺寸與編碼")
        video_param_layout = QVBoxLayout()
        output_size_box = QGroupBox("輸出尺寸設定")
        output_size_layout = QVBoxLayout(output_size_box)
        output_size_layout.setContentsMargins(6, 6, 6, 6)
        self.size_option_group = QButtonGroup(self)
        size_radio_layout = QHBoxLayout()
        self.original_size_radio = QRadioButton("原始大小")
        self.original_size_radio.setChecked(True)
        self.size_option_group.addButton(self.original_size_radio, 0)
        size_radio_layout.addWidget(self.original_size_radio)
        self.upscale_radio = QRadioButton("倍率超分")
        self.size_option_group.addButton(self.upscale_radio, 1)
        size_radio_layout.addWidget(self.upscale_radio)
        self.custom_size_radio = QRadioButton("指定影片大小")
        self.size_option_group.addButton(self.custom_size_radio, 2)
        size_radio_layout.addWidget(self.custom_size_radio)
        output_size_layout.addLayout(size_radio_layout)
        upscale_option_layout = QHBoxLayout()
        upscale_option_layout.setContentsMargins(20, 0, 0, 0)
        upscale_option_layout.addWidget(QLabel("超分倍率:"))
        self.scale_factor_spinbox = QDoubleSpinBox()
        self.scale_factor_spinbox.setRange(1.0, 4.0)
        self.scale_factor_spinbox.setValue(2.0)
        self.scale_factor_spinbox.setSingleStep(0.05)
        self.scale_factor_spinbox.setDecimals(2)
        self.scale_factor_spinbox.setSuffix("x")
        self.scale_factor_spinbox.setMinimumWidth(80)
        upscale_option_layout.addWidget(self.scale_factor_spinbox)
        self.scale_size_info_label = QLabel("輸出尺寸: -- x -- 像素")
        upscale_option_layout.addWidget(self.scale_size_info_label)
        output_size_layout.addLayout(upscale_option_layout)
        custom_size_option_layout = QHBoxLayout()
        custom_size_option_layout.setContentsMargins(20, 0, 0, 0)
        custom_size_option_layout.addWidget(QLabel("寬度:"))
        self.width_input = QLineEdit()
        self.width_input.setValidator(QIntValidator(32, 7680))
        self.width_input.setFixedWidth(80)
        custom_size_option_layout.addWidget(self.width_input)
        custom_size_option_layout.addWidget(QLabel("高度:"))
        self.height_input = QLineEdit()
        self.height_input.setValidator(QIntValidator(32, 4320))
        self.height_input.setFixedWidth(80)
        custom_size_option_layout.addWidget(self.height_input)
        self.custom_size_info_label = QLabel("請輸入有效的寬度和高度")
        custom_size_option_layout.addWidget(self.custom_size_info_label)
        output_size_layout.addLayout(custom_size_option_layout)
        video_param_layout.addWidget(output_size_box)
        crop_layout = QHBoxLayout()
        crop_layout.addWidget(QLabel("裁切方式:"))
        self.crop_combo = QComboBox()
        self.crop_combo.addItems(['無裁切', '居中裁切', '智能裁切'])
        crop_layout.addWidget(self.crop_combo)
        video_param_layout.addLayout(crop_layout)
        codec_type_layout = QHBoxLayout()
        codec_type_layout.addWidget(QLabel("編碼器類型:"))
        self.codec_type_combo = QComboBox()
        self.codec_type_combo.addItems(['H.264', 'H.265/HEVC', 'VP9', 'AV1'])
        self.codec_type_combo.currentTextChanged.connect(self.update_encoder_options)
        codec_type_layout.addWidget(self.codec_type_combo)
        video_param_layout.addLayout(codec_type_layout)
        encoder_layout = QHBoxLayout()
        encoder_layout.addWidget(QLabel("編碼器:"))
        self.encoder_combo = QComboBox()
        encoder_layout.addWidget(self.encoder_combo)
        video_param_layout.addLayout(encoder_layout)
        self.update_encoder_options("H.264")
        rate_control_layout = QVBoxLayout()
        rate_control_group_layout = QHBoxLayout()
        rate_control_label = QLabel("碼率控制:")
        rate_control_label.setMinimumWidth(70)
        rate_control_group_layout.addWidget(rate_control_label)
        self.rate_control_group = QButtonGroup(self)
        self.abr_radio = QRadioButton("平均碼率")
        self.abr_radio.setChecked(True)
        self.rate_control_group.addButton(self.abr_radio, 0)
        rate_control_group_layout.addWidget(self.abr_radio)
        self.crf_radio = QRadioButton("恆定品質")
        self.rate_control_group.addButton(self.crf_radio, 1)
        rate_control_group_layout.addWidget(self.crf_radio)
        rate_control_layout.addLayout(rate_control_group_layout)
        self.rate_control_stack = QStackedWidget()
        abr_widget = QWidget()
        abr_layout = QHBoxLayout(abr_widget)
        abr_layout.setContentsMargins(70, 0, 0, 0)
        abr_layout.addWidget(QLabel("碼率(Mbps):"))
        self.bitrate_spin = QSpinBox()
        self.bitrate_spin.setRange(1, 100)
        self.bitrate_spin.setValue(8)
        abr_layout.addWidget(self.bitrate_spin)
        abr_layout.addStretch()
        crf_widget = QWidget()
        crf_layout = QHBoxLayout(crf_widget)
        crf_layout.setContentsMargins(70, 0, 0, 0)
        crf_layout.addWidget(QLabel("品質(0-51):"))
        self.crf_slider = QSlider(Qt.Orientation.Horizontal)
        self.crf_slider.setRange(0, 51)
        self.crf_slider.setValue(23)
        self.crf_slider.setMinimumWidth(100)
        crf_layout.addWidget(self.crf_slider)
        self.crf_value_label = QLabel("23")
        crf_layout.addWidget(self.crf_value_label)
        self.crf_slider.valueChanged.connect(
            lambda v: self.crf_value_label.setText(str(v))
        )
        self.rate_control_stack.addWidget(abr_widget)
        self.rate_control_stack.addWidget(crf_widget)
        self.abr_radio.toggled.connect(
            lambda: self.rate_control_stack.setCurrentIndex(0 if self.abr_radio.isChecked() else 1)
        )
        rate_control_layout.addWidget(self.rate_control_stack)
        video_param_layout.addLayout(rate_control_layout)
        audio_layout = QVBoxLayout()
        audio_label = QLabel("聲音設定:")
        audio_label.setMinimumWidth(70)
        audio_layout.addWidget(audio_label)
        self.audio_mode_group = QButtonGroup(self)
        audio_options_layout = QHBoxLayout()
        audio_options_layout.setContentsMargins(70, 0, 0, 0)
        self.keep_audio_radio = QRadioButton("保留原始音軌")
        self.keep_audio_radio.setChecked(True)
        self.audio_mode_group.addButton(self.keep_audio_radio, 0)
        audio_options_layout.addWidget(self.keep_audio_radio)
        self.reencode_audio_radio = QRadioButton("重新編碼音軌")
        self.audio_mode_group.addButton(self.reencode_audio_radio, 1)
        audio_options_layout.addWidget(self.reencode_audio_radio)
        self.no_audio_radio = QRadioButton("無聲音")
        self.audio_mode_group.addButton(self.no_audio_radio, 2)
        audio_options_layout.addWidget(self.no_audio_radio)
        audio_layout.addLayout(audio_options_layout)
        self.audio_settings_widget = QWidget()
        audio_settings_layout = QHBoxLayout(self.audio_settings_widget)
        audio_settings_layout.setContentsMargins(70, 0, 0, 0)
        audio_settings_layout.addWidget(QLabel("音訊格式:"))
        self.audio_codec_combo = QComboBox()
        self.audio_codec_combo.addItems(['AAC', 'Opus', 'Vorbis', 'MP3'])
        audio_settings_layout.addWidget(self.audio_codec_combo)
        audio_settings_layout.addWidget(QLabel("音訊碼率:"))
        self.audio_bitrate_combo = QComboBox()
        self.audio_bitrate_combo.addItems(['128k', '192k', '256k', '320k'])
        self.audio_bitrate_combo.setCurrentText('192k')
        audio_settings_layout.addWidget(self.audio_bitrate_combo)
        self.audio_settings_widget.setVisible(False)
        self.reencode_audio_radio.toggled.connect(
            lambda checked: self.audio_settings_widget.setVisible(checked)
        )
        audio_layout.addWidget(self.audio_settings_widget)
        video_param_layout.addLayout(audio_layout)
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("輸出格式:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(['MP4', 'MKV', 'MOV', 'WebM'])
        format_layout.addWidget(self.format_combo)
        video_param_layout.addLayout(format_layout)
        video_param_box.setContentLayout(video_param_layout)
        params_content_layout.addWidget(video_param_box, 1)
        # === 處理選項區塊 ===
        processing_options_box = CollapsibleBox("處理選項")
        processing_options_layout = QVBoxLayout()
        frame_step_layout = QHBoxLayout()
        frame_step_layout.addWidget(QLabel("處理每幀:"))
        self.frame_step_spin = QSpinBox()
        self.frame_step_spin.setRange(1, 10)
        self.frame_step_spin.setValue(1)
        self.frame_step_spin.setSingleStep(1)
        self.frame_step_spin.setSuffix("幀")
        frame_step_layout.addWidget(self.frame_step_spin)
        processing_options_layout.addLayout(frame_step_layout)
        sync_preview_layout = QHBoxLayout()
        self.sync_preview_check = QCheckBox("同步預覽幀")
        self.sync_preview_check.setChecked(True)
        self.sync_preview_check.setToolTip("啟用後將同時更新原始幀和增強幀預覽")
        sync_preview_layout.addWidget(self.sync_preview_check)
        processing_options_layout.addLayout(sync_preview_layout)
        preview_interval_layout = QHBoxLayout()
        preview_interval_layout.addWidget(QLabel("預覽間隔:"))
        self.preview_interval_spin = QSpinBox()
        self.preview_interval_spin.setRange(1, 30)
        self.preview_interval_spin.setValue(5)
        self.preview_interval_spin.setSingleStep(1)
        self.preview_interval_spin.setSuffix("幀")
        self.preview_interval_spin.setToolTip("設置每隔多少幀更新一次預覽畫面")
        preview_interval_layout.addWidget(self.preview_interval_spin)
        processing_options_layout.addLayout(preview_interval_layout)
        priority_layout = QHBoxLayout()
        priority_layout.addWidget(QLabel("處理優先級:"))
        self.priority_combo = QComboBox()
        self.priority_combo.addItems(['低', '普通', '高'])
        self.priority_combo.setCurrentText('普通')
        self.priority_combo.setToolTip("設置處理線程的優先級")
        priority_layout.addWidget(self.priority_combo)
        processing_options_layout.addLayout(priority_layout)
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("計算設備:"))
        self.device_combo = QComboBox()
        device_options = get_device_options(self.system_info)
        for display_text, device_value in device_options:
            self.device_combo.addItem(display_text, device_value)
        self.device_combo.setToolTip("自動選擇: 使用系統默認設備\nCUDA: 使用GPU處理（較快）\nCPU: 使用CPU處理（較穩定但較慢）")
        device_layout.addWidget(self.device_combo)
        processing_options_layout.addLayout(device_layout)
        amp_layout = QHBoxLayout()
        amp_layout.addWidget(QLabel("混合精度:"))
        self.amp_combo = QComboBox()
        self.amp_combo.addItems(['自動偵測', '強制啟用', '強制禁用'])
        self.amp_combo.setCurrentText('自動偵測')
        self.amp_combo.setToolTip("自動偵測: 根據GPU類型自動決定\n強制啟用: 使用混合精度計算(較快但可能有問題)\n強制禁用: 使用完整精度計算(較穩定但較慢)")
        amp_layout.addWidget(self.amp_combo)
        processing_options_layout.addLayout(amp_layout)
        temp_file_layout = QHBoxLayout()
        self.clean_temp_check = QCheckBox("處理完成後清理暫存檔案")
        self.clean_temp_check.setChecked(True)
        self.clean_temp_check.setToolTip("處理完成或中止後自動清除臨時文件")
        temp_file_layout.addWidget(self.clean_temp_check)
        processing_options_layout.addLayout(temp_file_layout)
        processing_options_box.setContentLayout(processing_options_layout)
        params_content_layout.addWidget(processing_options_box, 1)
        params_layout.addLayout(params_content_layout)
        self.main_splitter.addWidget(preview_widget)
        self.main_splitter.addWidget(params_widget)
        self.main_splitter.setSizes([700, 300])
        self.main_splitter.widget(1).hide()
        main_layout.addWidget(self.main_splitter)
        self.size_option_group.buttonClicked.connect(self.on_size_option_changed)
        self.scale_factor_spinbox.valueChanged.connect(self.update_scale_factor_info)
        self.width_input.textChanged.connect(self.update_custom_size_info)
        self.height_input.textChanged.connect(self.update_custom_size_info)
        self.on_size_option_changed()
    
    def update_encoder_options(self, codec_type):
        self.encoder_combo.clear()
        if codec_type == 'H.264':
            self.encoder_combo.addItems([
                'x264 (CPU)', 
                'x264 10-bit (CPU)', 
                'NVENC (NVIDIA GPU)',
            ])
        elif codec_type == 'H.265/HEVC':
            self.encoder_combo.addItems([
                'x265 (CPU)', 
                'x265 10-bit (CPU)', 
                'NVENC HEVC (NVIDIA GPU)',
            ])
        elif codec_type == 'VP9':
            self.encoder_combo.addItems([
                'libvpx-vp9 (CPU)', 
                'libvpx-vp9 10-bit (CPU)'
            ])
        elif codec_type == 'AV1':
            self.encoder_combo.addItems([
                'libaom-av1 (CPU)', 
                'SVT-AV1 (CPU)', 
                'NVENC AV1 (NVIDIA GPU)'
            ])
    
    def toggle_params_panel(self, checked):
        if checked:
            self.main_splitter.widget(1).show()
            self.toggle_params_btn.setText("隱藏參數面板")
        else:
            self.main_splitter.widget(1).hide()
            self.toggle_params_btn.setText("顯示參數面板")
    
    def open_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "開啟影片", "", "影片文件 (*.mp4 *.avi *.mkv *.mov *.webm)"
        )
        if file_path:
            self.input_video_path = file_path
            self.vid_path_label.setText(os.path.basename(file_path))
            self.frame_cache = {}
            self.current_frame_index = 0
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                original_frame = Image.fromarray(frame_rgb)
                self.original_video_size = (original_frame.width, original_frame.height)
                self.multi_view.set_images(
                    image_a=original_frame,
                    image_b=None,
                    image_a_name=f"原始幀 - {os.path.basename(file_path)}",
                    image_b_name="增強幀"
                )
                self.frame_cache[0] = original_frame
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                info_text = f"影片資訊: {width}x{height}, {fps:.2f} FPS, {frame_count} 幀, {duration:.2f} 秒"
                self.vid_status_label.setText(info_text)
                self.update_scale_factor_info(self.scale_factor_spinbox.value())
                if not self.width_input.text() and not self.height_input.text():
                    self.width_input.setText(str(width))
                    self.height_input.setText(str(height))
                self.update_custom_size_info()
            cap.release()
            self.enhance_video_button.setEnabled(True)
            if self.parent:
                self.parent.tab_widget.setCurrentIndex(1)
    
    def enhance_video(self):
        if not self.input_video_path:
            QMessageBox.warning(self, "警告", "請先開啟影片。")
            return
        if not self.model_manager or not self.model_manager.get_registered_model_path():
            QMessageBox.warning(self, "警告", "請先選擇要使用的模型。")
            return
        selected_format = self.format_combo.currentText().lower()
        extension = f".{selected_format}"
        output_path, _ = QFileDialog.getSaveFileName(
            self, "設定輸出影片位置", "", f"{self.format_combo.currentText()} 文件 (*{extension})"
        )
        if not output_path:
            return
        if not output_path.lower().endswith(extension):
            output_path += extension
        self.enhance_video_button.setEnabled(False)
        self.open_video_button.setEnabled(False)
        self.stop_processing_button.setEnabled(True)
        self.vid_progress_bar.setValue(0)
        self.vid_status_label.setText("準備處理影片...")
        self.vid_remaining_label.setText("預計剩餘時間: 計算中...")
        try:
            self.model_status_label.setText("模型狀態: 載入中...")
            if self.parent:
                self.parent.statusBar.showMessage("正在載入模型到記憶體...")
            success = self.model_manager.prepare_model_for_inference()
            if not success:
                QMessageBox.warning(self, "錯誤", "模型載入失敗。")
                self.enhance_video_button.setEnabled(True)
                self.open_video_button.setEnabled(True)
                self.stop_processing_button.setEnabled(False)
                self.model_status_label.setText("模型狀態: 載入失敗")
                return
            model = self.model_manager.get_current_model()
            model_path = self.model_manager.get_registered_model_path()
            model_name = os.path.basename(model_path) if model_path else "未知模型"
            self.model_status_label.setText(f"模型狀態: 已載入 {model_name}")
            if self.parent:
                self.parent.statusBar.showMessage(f"模型已載入: {model_name}，開始處理影片...")
            device_selection = self.device_combo.currentData()
            if device_selection == "auto":
                device = self.model_manager.get_device()
                self.used_device_name = "自動選擇"
                if device.type == "cuda":
                    self.used_device_name = f"自動選擇 - {get_device_name(device, self.system_info)}"
                else:
                    self.used_device_name = f"自動選擇 - {self.system_info.get('cpu_brand_model', 'CPU')} (CPU)"
                logger.info(f"使用自動選擇的設備: {device}, {self.used_device_name}")
            else:
                device = torch.device(device_selection)
                self.used_device_name = self.device_combo.currentText()
                logger.info(f"使用手動選擇的設備: {device}, {self.used_device_name}")
            block_size = self.vid_block_size_spin.value()
            overlap = self.vid_overlap_spin.value()
            use_weight_mask = self.vid_weight_mask_check.isChecked()
            blending_mode = self.vid_blending_combo.currentText()
            strength_percent = (self.vid_strength_slider.value() + 1) * 10
            strength = strength_percent / 100.0
            frame_step = self.frame_step_spin.value()
            sync_preview = self.sync_preview_check.isChecked()
            preview_interval = self.preview_interval_spin.value()
            clean_temp_files = self.clean_temp_check.isChecked()
            if self.original_size_radio.isChecked():
                resolution = '原始大小'
                scale_factor = 1.0
                custom_width = None
                custom_height = None
            elif self.upscale_radio.isChecked():
                resolution = '超分倍率'
                scale_factor = self.scale_factor_spinbox.value()
                custom_width = None
                custom_height = None
            else:
                resolution = '自訂'
                scale_factor = 1.0
                custom_width = self.width_input.text()
                custom_height = self.height_input.text()
            bitrate_mode = "crf" if self.crf_radio.isChecked() else "abr"
            bitrate_value = self.bitrate_spin.value() * 1000000 if bitrate_mode == "abr" else None
            crf_value = self.crf_slider.value() if bitrate_mode == "crf" else None
            codec_type = self.codec_type_combo.currentText()
            encoder = self.encoder_combo.currentText()
            if self.keep_audio_radio.isChecked():
                audio_mode = "keep"
            elif self.reencode_audio_radio.isChecked():
                audio_mode = "reencode"
            else:
                audio_mode = "none"
            audio_codec = self.audio_codec_combo.currentText() if audio_mode == "reencode" else None
            audio_bitrate = self.audio_bitrate_combo.currentText() if audio_mode == "reencode" else None
            amp_setting = self.amp_combo.currentText()
            use_amp = None
            if amp_setting == '強制啟用':
                use_amp = True
                logger.info("使用強制啟用的混合精度計算模式")
            elif amp_setting == '強制禁用':
                use_amp = False
                logger.info("使用強制禁用的混合精度計算模式")
            else:
                logger.info("使用自動偵測的混合精度計算模式")
            logger.info(f"使用設備: {get_device_name(device, self.system_info)}")
            video_options = {
                'resolution': resolution,
                'scale_factor': scale_factor,
                'custom_width': custom_width,
                'custom_height': custom_height,
                'crop_mode': self.crop_combo.currentText(),
                'codec_type': codec_type,
                'encoder': encoder,
                'bitrate_mode': bitrate_mode,
                'bitrate': bitrate_value,
                'crf': crf_value,
                'audio_mode': audio_mode,
                'audio_codec': audio_codec,
                'audio_bitrate': audio_bitrate,
                'format': self.format_combo.currentText().lower(),
                'clean_temp_files': clean_temp_files,
                'use_amp': use_amp
            }
            self.video_enhancer_thread = VideoEnhancerThread(
                model,
                self.input_video_path,
                output_path,
                device, 
                block_size,
                overlap,
                use_weight_mask,
                blending_mode,
                frame_step,
                preview_interval=preview_interval,
                keep_audio=(audio_mode != "none"),
                sync_preview=sync_preview,
                video_options=video_options,
                strength=strength
            )
            self.video_enhancer_thread.progress_signal.connect(self.update_video_progress)
            self.video_enhancer_thread.finished_signal.connect(self.video_process_finished)
            self.video_enhancer_thread.preview_signal.connect(self.display_enhanced_frame)
            self.video_start_time = time.time()
            self.frame_cache = {}  
            self.current_frame_index = 0
            self.video_enhancer_thread.start()
        except Exception as e:
            logger.error(f"影片處理過程中發生錯誤: {str(e)}")
            QMessageBox.critical(self, "錯誤", f"處理時出錯: {str(e)}")
            self.enhance_video_button.setEnabled(True)
            self.open_video_button.setEnabled(True)
            self.stop_processing_button.setEnabled(False)
            self.model_status_label.setText("模型狀態: 錯誤")
            self.model_manager.clear_cache()
    
    def stop_video_processing(self):
        if hasattr(self, 'video_enhancer_thread') and self.video_enhancer_thread.isRunning():
            reply = QMessageBox.question(
                self, "確認", 
                "確定要停止影片處理嗎？當前進度將會丟失。", 
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.video_enhancer_thread.stop()
                self.enable_video_ui()
                self.vid_status_label.setText("處理已取消。")
                self.model_manager.clear_cache()
                self.model_status_label.setText("模型狀態: 已卸載")
    
    def enable_video_ui(self):
        self.enhance_video_button.setEnabled(True)
        self.open_video_button.setEnabled(True)
        self.stop_processing_button.setEnabled(False)
    
    def update_video_progress(self, current, total, status):
        self.vid_progress_bar.setValue(current)
        self.vid_status_label.setText(status)
        elapsed = time.time() - self.video_start_time
        if current > 0:
            estimated_total = elapsed * total / current
            remaining = estimated_total - elapsed
            hours, remainder = divmod(int(remaining), 3600)
            minutes, seconds = divmod(remainder, 60)
            self.vid_remaining_label.setText(f"預計剩餘時間: {hours:02d}:{minutes:02d}:{seconds:02d}")
    
    def video_process_finished(self, output_path, elapsed_time):
        self.enable_video_ui()
        try:
            self.model_manager.clear_cache()
            self.model_status_label.setText("模型狀態: 已卸載")
            logger.info(f"處理完成後已卸載模型，耗時: {elapsed_time:.2f} 秒")
        except Exception as e:
            logger.error(f"卸載模型時出錯: {str(e)}")
            self.model_status_label.setText("模型狀態: 卸載失敗")
        if output_path:
            input_cap = cv2.VideoCapture(self.input_video_path)
            frame_count = int(input_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(input_cap.get(cv2.CAP_PROP_FPS))
            input_cap.release()
            duration = frame_count / fps if fps > 0 else 0
            processing_speed = duration / elapsed_time if elapsed_time > 0 else 0
            device_info = ""
            if hasattr(self, 'used_device_name') and self.used_device_name:
                device_info = f", 設備: {self.used_device_name}"
            amp_info = ""
            if self.amp_combo.currentText() != '自動偵測':
                amp_info = f", 混合精度: {self.amp_combo.currentText()}"
            self.vid_status_label.setText(f"影片處理完成！耗時: {elapsed_time:.2f} 秒{device_info}{amp_info}")
            self.vid_remaining_label.setText(f"處理速度: {processing_speed:.2f}x")
            if self.parent:
                self.parent.statusBar.showMessage(f"影片處理完成。耗時: {elapsed_time:.2f} 秒，處理速度: {processing_speed:.2f}x")
            reply = QMessageBox.question(
                self, "處理完成", 
                f"影片處理已完成！\n"
                f"總耗時: {elapsed_time:.2f} 秒\n"
                f"處理速度: {processing_speed:.2f}x{device_info}{amp_info}\n"
                f"是否開啟輸出位置？", 
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                output_dir = os.path.dirname(os.path.abspath(output_path))
                os.startfile(output_dir)
        else:
            self.vid_status_label.setText("影片處理失敗。")
    
    def get_frame_at_index(self, frame_index):
        if frame_index in self.frame_cache:
            return self.frame_cache[frame_index]
        if not self.input_video_path or not os.path.exists(self.input_video_path):
            return None
        try:
            cap = cv2.VideoCapture(self.input_video_path)
            if not cap.isOpened():
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            cap.release()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                self.frame_cache[frame_index] = pil_image
                return pil_image
            else:
                return None
        except Exception as e:
            logger.error(f"獲取視頻幀時出錯: {str(e)}")
            return None

    def display_enhanced_frame(self, enhanced_frame, frame_index=None):
        if frame_index is not None and self.sync_preview_check.isChecked():
            self.current_frame_index = frame_index
            original_frame = self.get_frame_at_index(frame_index)
        else:
            original_frame = self.get_frame_at_index(0)
        if original_frame:
            frame_text = f"原始幀預覽 (幀: {frame_index})" if frame_index is not None else "原始幀預覽"
            enhanced_text = f"增強幀預覽 (幀: {frame_index})" if frame_index is not None else f"增強幀預覽"
            self.multi_view.set_images(
                image_a=original_frame,
                image_b=enhanced_frame,
                image_a_name=frame_text,
                image_b_name=enhanced_text
            )