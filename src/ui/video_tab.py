import os
import sys
import time
import cv2
import logging
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
        
        # 影片信息顯示與控制按鈕區域
        info_control_layout = QHBoxLayout()
        
        # 影片信息
        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel("當前影片:"))
        self.vid_path_label = QLabel("未選擇影片")
        info_layout.addWidget(self.vid_path_label)
        info_layout.addStretch()
        info_control_layout.addLayout(info_layout, 1)
        
        # 控制按鈕
        controls_layout = QHBoxLayout()
        self.toggle_params_btn = QPushButton("顯示參數面板")
        self.toggle_params_btn.setCheckable(True)
        self.toggle_params_btn.setChecked(False)  # 默認隱藏參數面板
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
        
        # 進度顯示區
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
        
        # 添加模型選擇下拉選單
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("選擇模型:"))
        self.model_selector = QComboBox()
        self.model_selector.setMinimumWidth(300)
        self.model_selector.currentIndexChanged.connect(self.on_model_selected)
        model_layout.addWidget(self.model_selector)
        model_param_layout.addLayout(model_layout)
        
        # 區塊大小設定
        block_size_layout = QHBoxLayout()
        block_size_layout.addWidget(QLabel("區塊大小:"))
        self.vid_block_size_spin = QSpinBox()
        self.vid_block_size_spin.setRange(128, 512)
        self.vid_block_size_spin.setValue(256)
        self.vid_block_size_spin.setSingleStep(32)
        block_size_layout.addWidget(self.vid_block_size_spin)
        model_param_layout.addLayout(block_size_layout)
        
        # 重疊大小設定
        overlap_layout = QHBoxLayout()
        overlap_layout.addWidget(QLabel("重疊大小:"))
        self.vid_overlap_spin = QSpinBox()
        self.vid_overlap_spin.setRange(16, 256)
        self.vid_overlap_spin.setValue(128)
        self.vid_overlap_spin.setSingleStep(16)
        overlap_layout.addWidget(self.vid_overlap_spin)
        model_param_layout.addLayout(overlap_layout)
        
        # 權重遮罩設定
        weight_mask_layout = QHBoxLayout()
        self.vid_weight_mask_check = QCheckBox("使用權重遮罩")
        self.vid_weight_mask_check.setChecked(True)
        weight_mask_layout.addWidget(self.vid_weight_mask_check)
        model_param_layout.addLayout(weight_mask_layout)
        
        # 混合模式設定
        blending_layout = QHBoxLayout()
        blending_layout.addWidget(QLabel("混合模式:"))
        self.vid_blending_combo = QComboBox()
        self.vid_blending_combo.addItems(['高斯分佈', '改進型高斯分佈', '線性分佈', '餘弦分佈', '泊松分佈'])
        self.vid_blending_combo.setCurrentText('改進型高斯分佈')
        blending_layout.addWidget(self.vid_blending_combo)
        model_param_layout.addLayout(blending_layout)
        model_param_box.setContentLayout(model_param_layout)
        params_content_layout.addWidget(model_param_box, 1)
        
        # === 影片參數區塊 ===
        video_param_box = CollapsibleBox("影片參數設定")
        video_param_layout = QVBoxLayout()
        
        # -- 輸出影片分辨率 --
        resolution_layout = QHBoxLayout()
        resolution_layout.addWidget(QLabel("輸出分辨率:"))
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(['原始分辨率', '720p', '1080p', '1440p', '4K', '自訂'])
        self.resolution_combo.currentTextChanged.connect(self.on_resolution_changed)
        resolution_layout.addWidget(self.resolution_combo)
        
        # 自訂分辨率
        self.custom_resolution_widget = QWidget()
        custom_resolution_layout = QHBoxLayout(self.custom_resolution_widget)
        custom_resolution_layout.setContentsMargins(0, 0, 0, 0)
        self.width_input = QLineEdit()
        self.width_input.setValidator(QIntValidator(32, 7680))
        self.width_input.setPlaceholderText("寬度")
        self.width_input.setMaximumWidth(80)
        self.height_input = QLineEdit()
        self.height_input.setValidator(QIntValidator(32, 4320))
        self.height_input.setPlaceholderText("高度")
        self.height_input.setMaximumWidth(80)
        custom_resolution_layout.addWidget(self.width_input)
        custom_resolution_layout.addWidget(QLabel("x"))
        custom_resolution_layout.addWidget(self.height_input)
        resolution_layout.addWidget(self.custom_resolution_widget)
        self.custom_resolution_widget.hide()
        video_param_layout.addLayout(resolution_layout)
        
        # -- 影片裁切方式 --
        crop_layout = QHBoxLayout()
        crop_layout.addWidget(QLabel("裁切方式:"))
        self.crop_combo = QComboBox()
        self.crop_combo.addItems(['無裁切', '居中裁切', '智能裁切'])
        crop_layout.addWidget(self.crop_combo)
        video_param_layout.addLayout(crop_layout)
        
        # -- 編碼器設定 --
        codec_type_layout = QHBoxLayout()
        codec_type_layout.addWidget(QLabel("編碼器類型:"))
        self.codec_type_combo = QComboBox()
        self.codec_type_combo.addItems(['H.264', 'H.265/HEVC', 'VP9', 'AV1'])
        self.codec_type_combo.currentTextChanged.connect(self.update_encoder_options)
        codec_type_layout.addWidget(self.codec_type_combo)
        video_param_layout.addLayout(codec_type_layout)
        
        # 編碼器具體選項
        encoder_layout = QHBoxLayout()
        encoder_layout.addWidget(QLabel("編碼器:"))
        self.encoder_combo = QComboBox()
        encoder_layout.addWidget(self.encoder_combo)
        video_param_layout.addLayout(encoder_layout)
        
        # 初始化編碼器選項
        self.update_encoder_options("H.264")
        
        # -- 碼率控制 --
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
        
        # 碼率控制堆疊部件，根據選擇切換顯示
        self.rate_control_stack = QStackedWidget()
        
        # 平均碼率控件
        abr_widget = QWidget()
        abr_layout = QHBoxLayout(abr_widget)
        abr_layout.setContentsMargins(70, 0, 0, 0)
        abr_layout.addWidget(QLabel("碼率(Mbps):"))
        self.bitrate_spin = QSpinBox()
        self.bitrate_spin.setRange(1, 100)
        self.bitrate_spin.setValue(8)
        abr_layout.addWidget(self.bitrate_spin)
        abr_layout.addStretch()
        
        # 恆定品質控件
        crf_widget = QWidget()
        crf_layout = QHBoxLayout(crf_widget)
        crf_layout.setContentsMargins(70, 0, 0, 0)
        crf_layout.addWidget(QLabel("品質(0-51):"))
        self.crf_slider = QSlider(Qt.Orientation.Horizontal)
        self.crf_slider.setRange(0, 51)
        self.crf_slider.setValue(23)  # 默認品質
        self.crf_slider.setMinimumWidth(100)
        crf_layout.addWidget(self.crf_slider)
        self.crf_value_label = QLabel("23")
        crf_layout.addWidget(self.crf_value_label)
        self.crf_slider.valueChanged.connect(
            lambda v: self.crf_value_label.setText(str(v))
        )
        
        self.rate_control_stack.addWidget(abr_widget)
        self.rate_control_stack.addWidget(crf_widget)
        
        # 連接信號
        self.abr_radio.toggled.connect(
            lambda: self.rate_control_stack.setCurrentIndex(0 if self.abr_radio.isChecked() else 1)
        )
        
        rate_control_layout.addWidget(self.rate_control_stack)
        video_param_layout.addLayout(rate_control_layout)
        
        # -- 聲音設定 --
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
        
        # 音訊格式和品質（當選擇重新編碼時顯示）
        self.audio_settings_widget = QWidget()
        audio_settings_layout = QHBoxLayout(self.audio_settings_widget)
        audio_settings_layout.setContentsMargins(70, 0, 0, 0)
        
        # 音訊編碼器
        audio_settings_layout.addWidget(QLabel("音訊格式:"))
        self.audio_codec_combo = QComboBox()
        self.audio_codec_combo.addItems(['AAC', 'Opus', 'Vorbis', 'MP3'])
        audio_settings_layout.addWidget(self.audio_codec_combo)
        
        # 音訊碼率
        audio_settings_layout.addWidget(QLabel("音訊碼率:"))
        self.audio_bitrate_combo = QComboBox()
        self.audio_bitrate_combo.addItems(['128k', '192k', '256k', '320k'])
        self.audio_bitrate_combo.setCurrentText('192k')
        audio_settings_layout.addWidget(self.audio_bitrate_combo)
        
        # 初始隱藏
        self.audio_settings_widget.setVisible(False)
        
        # 當選擇重新編碼時顯示音訊設定
        self.reencode_audio_radio.toggled.connect(
            lambda checked: self.audio_settings_widget.setVisible(checked)
        )
        audio_layout.addWidget(self.audio_settings_widget)
        video_param_layout.addLayout(audio_layout)
        
        # -- 影片格式 --
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
        
        # 幀處理設定
        frame_step_layout = QHBoxLayout()
        frame_step_layout.addWidget(QLabel("處理每幀:"))
        self.frame_step_spin = QSpinBox()
        self.frame_step_spin.setRange(1, 10)
        self.frame_step_spin.setValue(1)
        self.frame_step_spin.setSingleStep(1)
        self.frame_step_spin.setSuffix("幀")
        frame_step_layout.addWidget(self.frame_step_spin)
        processing_options_layout.addLayout(frame_step_layout)
        
        # 同步預覽幀設定
        sync_preview_layout = QHBoxLayout()
        self.sync_preview_check = QCheckBox("同步預覽幀")
        self.sync_preview_check.setChecked(True)
        self.sync_preview_check.setToolTip("啟用後將同時更新原始幀和增強幀預覽")
        sync_preview_layout.addWidget(self.sync_preview_check)
        processing_options_layout.addLayout(sync_preview_layout)
        
        # 預覽間隔設定
        preview_interval_layout = QHBoxLayout()
        preview_interval_layout.addWidget(QLabel("預覽間隔:"))
        self.preview_interval_spin = QSpinBox()
        self.preview_interval_spin.setRange(1, 30)
        self.preview_interval_spin.setValue(5)
        self.preview_interval_spin.setSingleStep(1)
        self.preview_interval_spin.setSuffix("秒")
        self.preview_interval_spin.setToolTip("設置每隔多少秒更新一次預覽畫面")
        preview_interval_layout.addWidget(self.preview_interval_spin)
        processing_options_layout.addLayout(preview_interval_layout)
        
        # 處理優先級
        priority_layout = QHBoxLayout()
        priority_layout.addWidget(QLabel("處理優先級:"))
        self.priority_combo = QComboBox()
        self.priority_combo.addItems(['低', '普通', '高'])
        self.priority_combo.setCurrentText('普通')
        self.priority_combo.setToolTip("設置處理線程的優先級")
        priority_layout.addWidget(self.priority_combo)
        processing_options_layout.addLayout(priority_layout)
        
        # 暫存檔案管理
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
    
    def on_resolution_changed(self, text):
        """處理分辨率選擇變更事件"""
        if text == '自訂':
            self.custom_resolution_widget.show()
        else:
            self.custom_resolution_widget.hide()
    
    def update_encoder_options(self, codec_type):
        """根據選擇的編碼器類型更新具體編碼器選項"""
        self.encoder_combo.clear()
        
        if codec_type == 'H.264':
            self.encoder_combo.addItems([
                'x264 (CPU)', 
                'x264 10-bit (CPU)', 
                'NVENC (NVIDIA GPU)',
                'QuickSync (Intel GPU)', 
                'AMF (AMD GPU)'
            ])
        elif codec_type == 'H.265/HEVC':
            self.encoder_combo.addItems([
                'x265 (CPU)', 
                'x265 10-bit (CPU)', 
                'NVENC HEVC (NVIDIA GPU)',
                'QuickSync HEVC (Intel GPU)', 
                'AMF HEVC (AMD GPU)'
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
        """顯示/隱藏參數面板"""
        if checked:
            self.main_splitter.widget(1).show()
            self.toggle_params_btn.setText("隱藏參數面板")
        else:
            self.main_splitter.widget(1).hide()
            self.toggle_params_btn.setText("顯示參數面板")
    
    def open_video(self):
        """開啟影片文件"""
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
            cap.release()
            self.enhance_video_button.setEnabled(True)
            if self.parent:
                self.parent.tab_widget.setCurrentIndex(1)
    
    def enhance_video(self):
        """開始增強處理影片"""
        if not self.input_video_path:
            QMessageBox.warning(self, "警告", "請先開啟影片。")
            return

        # 檢查是否有選擇模型
        if not self.model_manager or not self.model_manager.get_registered_model_path():
            QMessageBox.warning(self, "警告", "請先選擇要使用的模型。")
            return
        
        # 根據所選格式獲取檔案副檔名
        selected_format = self.format_combo.currentText().lower()
        extension = f".{selected_format}"
        output_path, _ = QFileDialog.getSaveFileName(
            self, "設定輸出影片位置", "", f"{self.format_combo.currentText()} 文件 (*{extension})"
        )
        if not output_path:
            return
        
        # 確保輸出文件有正確的擴展名
        if not output_path.lower().endswith(extension):
            output_path += extension
        
        # 禁用控制按鈕並重置進度顯示
        self.enhance_video_button.setEnabled(False)
        self.open_video_button.setEnabled(False)
        self.stop_processing_button.setEnabled(True)
        self.vid_progress_bar.setValue(0)
        self.vid_status_label.setText("準備處理影片...")
        self.vid_remaining_label.setText("預計剩餘時間: 計算中...")
        
        try:
            # 載入模型到記憶體
            self.model_status_label.setText("模型狀態: 載入中...")
            if self.parent:
                self.parent.statusBar.showMessage("正在載入模型到記憶體...")
            # 步驟1: 先載入模型，獲取成功/失敗結果
            success = self.model_manager.prepare_model_for_inference()
            if not success:
                QMessageBox.warning(self, "錯誤", "模型載入失敗。")
                self.enhance_video_button.setEnabled(True)
                self.open_video_button.setEnabled(True)
                self.stop_processing_button.setEnabled(False)
                self.model_status_label.setText("模型狀態: 載入失敗")
                return
            
            # 步驟2: 從模型管理器獲取實際模型物件
            model = self.model_manager.get_current_model()
                
            # 更新模型狀態
            model_path = self.model_manager.get_registered_model_path()
            model_name = os.path.basename(model_path) if model_path else "未知模型"
            self.model_status_label.setText(f"模型狀態: 已載入 {model_name}")
            if self.parent:
                self.parent.statusBar.showMessage(f"模型已載入: {model_name}，開始處理影片...")
            
            # 獲取模型參數
            block_size = self.vid_block_size_spin.value()
            overlap = self.vid_overlap_spin.value()
            use_weight_mask = self.vid_weight_mask_check.isChecked()
            blending_mode = self.vid_blending_combo.currentText()
            
            # 獲取影片處理參數
            frame_step = self.frame_step_spin.value()
            sync_preview = self.sync_preview_check.isChecked()
            preview_interval = self.preview_interval_spin.value()
            clean_temp_files = self.clean_temp_check.isChecked()
            
            # 獲取分辨率
            resolution = self.resolution_combo.currentText()
            custom_width = self.width_input.text() if resolution == '自訂' else None
            custom_height = self.height_input.text() if resolution == '自訂' else None
            
            # 獲取碼率控制參數
            bitrate_mode = "crf" if self.crf_radio.isChecked() else "abr"
            bitrate_value = self.bitrate_spin.value() * 1000000 if bitrate_mode == "abr" else None
            crf_value = self.crf_slider.value() if bitrate_mode == "crf" else None
            
            # 獲取編碼器資訊
            codec_type = self.codec_type_combo.currentText()
            encoder = self.encoder_combo.currentText()
            
            # 獲取是否保留音軌
            if self.keep_audio_radio.isChecked():
                audio_mode = "keep"
            elif self.reencode_audio_radio.isChecked():
                audio_mode = "reencode"
            else:
                audio_mode = "none"
            audio_codec = self.audio_codec_combo.currentText() if audio_mode == "reencode" else None
            audio_bitrate = self.audio_bitrate_combo.currentText() if audio_mode == "reencode" else None
            video_options = {
                'resolution': resolution, 
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
                'clean_temp_files': clean_temp_files
            }
            self.video_enhancer_thread = VideoEnhancerThread(
                model,
                self.input_video_path,
                output_path,
                self.model_manager.get_device(),
                block_size,
                overlap,
                use_weight_mask,
                blending_mode,
                frame_step,
                preview_interval=preview_interval,
                keep_audio=(audio_mode != "none"),
                sync_preview=sync_preview,
                video_options=video_options
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
        """停止影片處理"""
        if hasattr(self, 'video_enhancer_thread') and self.video_enhancer_thread.isRunning():
            reply = QMessageBox.question(
                self, "確認", 
                "確定要停止影片處理嗎？當前進度將會丟失。", 
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.vid_status_label.setText("正在停止處理，請稍候...")
                self.video_enhancer_thread.stop()
                self.vid_status_label.setText("使用者已中止處理。")
                self.enable_video_ui()
                try:
                    self.model_manager.clear_cache()
                    self.model_status_label.setText("模型狀態: 已卸載")
                    logger.info("處理終止後已卸載模型")
                except Exception as e:
                    logger.error(f"卸載模型時出錯: {str(e)}")
                    self.model_status_label.setText("模型狀態: 卸載失敗")
    
    def enable_video_ui(self):
        """啟用影片處理UI控制項"""
        self.enhance_video_button.setEnabled(True)
        self.open_video_button.setEnabled(True)
        self.stop_processing_button.setEnabled(False)
    
    def update_video_progress(self, current, total, status):
        """更新影片處理進度"""
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
        """處理完成回調"""
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
            self.vid_status_label.setText(f"影片處理完成！耗時: {elapsed_time:.2f} 秒")
            self.vid_remaining_label.setText(f"處理速度: {processing_speed:.2f}x")
            if self.parent:
                self.parent.statusBar.showMessage(f"影片增強完成。輸出到: {output_path}")
            reply = QMessageBox.question(
                self, "處理完成", 
                f"影片處理已完成！\n"
                f"總耗時: {elapsed_time:.2f} 秒\n"
                f"處理速度: {processing_speed:.2f}x\n"
                f"是否開啟輸出位置？", 
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                output_dir = os.path.dirname(output_path)
                if sys.platform == 'win32':
                    os.startfile(output_dir)
        else:
            self.vid_status_label.setText("影片處理失敗。")
    
    def get_frame_at_index(self, frame_index):
        """根據幀索引獲取原始幀"""
        if frame_index in self.frame_cache:
            return self.frame_cache[frame_index]
        if not self.input_video_path or not os.path.exists(self.input_video_path):
            return None
        try:
            cap = cv2.VideoCapture(self.input_video_path)
            if not cap.isOpened():
                logger.error(f"無法開啟視頻: {self.input_video_path}")
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            cap.release()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                original_frame = Image.fromarray(frame_rgb)
                if len(self.frame_cache) > 10:
                    oldest_key = list(self.frame_cache.keys())[0]
                    del self.frame_cache[oldest_key]
                self.frame_cache[frame_index] = original_frame
                return original_frame
            else:
                logger.warning(f"無法讀取視頻幀: {frame_index}")
        except Exception as e:
            logger.error(f"獲取視頻幀時出錯: {str(e)}")
        return None

    def display_enhanced_frame(self, enhanced_frame, frame_index=None):
        """顯示增強後的幀，支持同步顯示原始幀"""
        if frame_index is not None and self.sync_preview_check.isChecked():
            self.current_frame_index = frame_index
            original_frame = self.get_frame_at_index(frame_index)
        else:
            original_frame = self.get_frame_at_index(0)
        if original_frame:
            frame_text = f"原始幀預覽 (幀: {frame_index})" if frame_index is not None else "原始幀預覽"
            enhanced_text = f"增強幀預覽 (幀: {frame_index})" if frame_index is not None else "增強幀預覽"
            self.multi_view.set_images(
                image_a=original_frame,
                image_b=enhanced_frame,
                image_a_name=frame_text,
                image_b_name=enhanced_text
            )