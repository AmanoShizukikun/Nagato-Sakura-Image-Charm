import os
import logging
import torch
from PIL import Image
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                            QFileDialog, QProgressBar, QComboBox, QSpinBox, QCheckBox, 
                            QGroupBox, QMessageBox, QSlider, QSplitter, QFrame, QToolButton,
                            QScrollArea, QRadioButton, QLineEdit, QButtonGroup, QGridLayout,
                            QDoubleSpinBox)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIntValidator, QDoubleValidator

from src.ui.views import MultiViewWidget
from src.processing.NS_ImageProcessor import ImageProcessor
from src.threads.NS_EnhancerThread import EnhancerThread

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

class ImageProcessingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.input_image_path = None
        self.enhanced_image = None
        self.model_manager = None
        self.original_image_size = (0, 0) 
        self.system_info = get_system_info()
        self.setup_ui()
    
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
    
    def update_strength_label(self, value):
        """更新強度標籤顯示"""
        actual_strength = (value + 1) * 10
        self.strength_value_label.setText(f"{actual_strength}%")
    
    def update_scale_factor_info(self, value):
        """更新超分倍率資訊"""
        factor = value
        if self.original_image_size != (0, 0):
            new_width = int(self.original_image_size[0] * factor)
            new_height = int(self.original_image_size[1] * factor)
            self.scale_size_info_label.setText(f"輸出尺寸: {new_width} x {new_height} 像素")
    
    def update_custom_size_info(self):
        """更新自訂尺寸資訊"""
        try:
            width = int(self.width_input.text() or "0")
            height = int(self.height_input.text() or "0")
            if width <= 0 or height <= 0:
                self.custom_size_info_label.setText("請輸入有效的寬度和高度")
                return
            if self.maintain_aspect_ratio_check.isChecked() and self.original_image_size != (0, 0):
                orig_width, orig_height = self.original_image_size
                orig_ratio = orig_width / orig_height
                if self.custom_size_mode_combo.currentText() == "延伸至適合大小":
                    target_ratio = width / height
                    if orig_ratio > target_ratio:
                        adjusted_height = int(width / orig_ratio)
                        self.custom_size_info_label.setText(f"實際輸出: {width} x {adjusted_height} (延伸)")
                    else:
                        adjusted_width = int(height * orig_ratio)
                        self.custom_size_info_label.setText(f"實際輸出: {adjusted_width} x {height} (延伸)")
                else:
                    target_ratio = width / height
                    if orig_ratio > target_ratio:
                        adjusted_width = int(height * orig_ratio)
                        self.custom_size_info_label.setText(f"實際輸出: {adjusted_width} x {height} (裁切)")
                    else:
                        adjusted_height = int(width / orig_ratio)
                        self.custom_size_info_label.setText(f"實際輸出: {width} x {adjusted_height} (裁切)")
            else:
                self.custom_size_info_label.setText(f"輸出尺寸: {width} x {height} 像素")
        except ValueError:
            self.custom_size_info_label.setText("請輸入有效的寬度和高度")
    
    def on_size_option_changed(self):
        """處理尺寸選項變更"""
        self.scale_factor_spinbox.setEnabled(self.upscale_radio.isChecked())
        self.scale_size_info_label.setEnabled(self.upscale_radio.isChecked())
        self.width_input.setEnabled(self.custom_size_radio.isChecked())
        self.height_input.setEnabled(self.custom_size_radio.isChecked())
        self.maintain_aspect_ratio_check.setEnabled(self.custom_size_radio.isChecked())
        self.custom_size_mode_combo.setEnabled(self.custom_size_radio.isChecked() and self.maintain_aspect_ratio_check.isChecked())
        self.custom_size_info_label.setEnabled(self.custom_size_radio.isChecked())
        if self.upscale_radio.isChecked():
            self.update_scale_factor_info(self.scale_factor_spinbox.value())
        if self.custom_size_radio.isChecked():
            self.update_custom_size_info()
    
    def on_maintain_aspect_ratio_changed(self, state):
        """處理維持比例選項變更"""
        self.custom_size_mode_combo.setEnabled(state and self.custom_size_radio.isChecked())
        self.update_custom_size_info()
    
    def toggle_params_panel(self, checked):
        """顯示/隱藏參數面板"""
        if checked:
            self.main_splitter.widget(1).show()
            self.toggle_params_btn.setText("隱藏參數面板")
        else:
            self.main_splitter.widget(1).hide()
            self.toggle_params_btn.setText("顯示參數面板")
    
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        self.multi_view = MultiViewWidget(self)
        self.multi_view.set_images(
            image_a=None,
            image_b=None,
            image_a_name="原始圖片", 
            image_b_name="增強圖片"
        )
        preview_layout.addWidget(self.multi_view)
        info_control_layout = QHBoxLayout()
        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel("當前圖片:"))
        self.img_path_label = QLabel("未選擇圖片")
        info_layout.addWidget(self.img_path_label)
        info_layout.addStretch()
        info_control_layout.addLayout(info_layout, 1)
        controls_layout = QHBoxLayout()
        self.toggle_params_btn = QPushButton("顯示參數面板")
        self.toggle_params_btn.setCheckable(True)
        self.toggle_params_btn.setChecked(False) 
        self.toggle_params_btn.clicked.connect(self.toggle_params_panel)
        controls_layout.addWidget(self.toggle_params_btn)
        self.img_open_button = QPushButton("開啟圖片")
        self.img_open_button.clicked.connect(self.open_image)
        controls_layout.addWidget(self.img_open_button)
        self.enhance_button = QPushButton("優化圖片")
        self.enhance_button.setEnabled(False)
        self.enhance_button.clicked.connect(self.enhance_image)
        controls_layout.addWidget(self.enhance_button)
        self.save_button = QPushButton("保存結果")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_image)
        controls_layout.addWidget(self.save_button)
        info_control_layout.addLayout(controls_layout)
        preview_layout.addLayout(info_control_layout)
        progress_layout = QHBoxLayout()
        self.img_progress_bar = QProgressBar()
        self.img_progress_bar.setValue(0)
        progress_layout.addWidget(self.img_progress_bar, 3)
        self.img_status_label = QLabel("等待處理...")
        progress_layout.addWidget(self.img_status_label, 2)
        self.model_status_label = QLabel("模型狀態: 未載入")
        progress_layout.addWidget(self.model_status_label, 1)
        preview_layout.addLayout(progress_layout)
        
        # ==================== 參數面板優化區域 ====================
        params_widget = QWidget()
        params_layout = QVBoxLayout(params_widget)
        params_layout.setContentsMargins(0, 0, 0, 0)
        params_header = QLabel("參數設置")
        params_header.setStyleSheet("font-size: 14px; font-weight: bold;")
        params_layout.addWidget(params_header)
        params_content_layout = QHBoxLayout()
        
        # ==================== 第一列：模型參數 ====================
        column1_layout = QVBoxLayout()
        column1_layout.setSpacing(10)
        
        # === 模型選擇區塊 ===
        model_selection_box = CollapsibleBox("模型選擇")
        model_selection_layout = QGridLayout()
        model_selection_layout.setColumnStretch(1, 1) 
        model_selection_layout.addWidget(QLabel("選擇模型:"), 0, 0)
        self.model_selector = QComboBox()
        self.model_selector.setMinimumWidth(200)
        self.model_selector.currentIndexChanged.connect(self.on_model_selected)
        model_selection_layout.addWidget(self.model_selector, 0, 1)
        model_selection_layout.addWidget(QLabel("處理強度:"), 1, 0)
        strength_layout = QHBoxLayout()
        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 9)
        self.strength_slider.setValue(9)
        self.strength_slider.setTickInterval(1)
        self.strength_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.strength_slider.valueChanged.connect(self.update_strength_label)
        self.strength_value_label = QLabel("100%")
        self.strength_value_label.setMinimumWidth(45)
        self.strength_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        strength_layout.addWidget(self.strength_slider)
        strength_layout.addWidget(self.strength_value_label)
        model_selection_layout.addLayout(strength_layout, 1, 1)
        model_selection_box.setContentLayout(model_selection_layout)
        column1_layout.addWidget(model_selection_box)
        
        # === 處理選項區塊 ===
        processing_options_box = CollapsibleBox("處理選項")
        processing_options_grid = QGridLayout()
        processing_options_grid.setColumnStretch(1, 1) 
        processing_options_grid.addWidget(QLabel("保存格式:"), 0, 0)
        self.save_format_combo = QComboBox()
        self.save_format_combo.addItems(['PNG', 'JPEG', 'WebP'])
        self.save_format_combo.setCurrentText('PNG')
        self.save_format_combo.setMinimumWidth(100)
        processing_options_grid.addWidget(self.save_format_combo, 0, 1)
        processing_options_grid.addWidget(QLabel("計算設備:"), 1, 0)
        self.device_combo = QComboBox()
        device_options = get_device_options(self.system_info)
        for display_text, device_value in device_options:
            self.device_combo.addItem(display_text, device_value)
        
        self.device_combo.setToolTip("自動選擇: 使用系統默認設備\nCUDA: 使用GPU處理（較快）\nCPU: 使用CPU處理（較穩定但較慢）")
        processing_options_grid.addWidget(self.device_combo, 1, 1)
        
        processing_options_grid.addWidget(QLabel("混合精度:"), 2, 0)
        self.amp_combo = QComboBox()
        self.amp_combo.addItems(['自動偵測', '強制啟用', '強制禁用'])
        self.amp_combo.setCurrentText('自動偵測')
        self.amp_combo.setMinimumWidth(100)
        self.amp_combo.setToolTip("自動偵測: 根據GPU類型自動決定\n強制啟用: 使用混合精度計算(較快但可能有問題)\n強制禁用: 使用完整精度計算(較穩定但較慢)")
        processing_options_grid.addWidget(self.amp_combo, 2, 1)
        processing_options_box.setContentLayout(processing_options_grid)
        column1_layout.addWidget(processing_options_box)
        column1_layout.addStretch(1)
        
        # ==================== 第二列：區塊處理參數 ====================
        column2_layout = QVBoxLayout()
        column2_layout.setSpacing(10)
        
        # === 區塊處理參數區塊 ===
        block_param_box = CollapsibleBox("區塊處理參數")
        block_param_grid = QGridLayout()
        block_param_grid.setColumnStretch(1, 1)
        block_param_grid.addWidget(QLabel("區塊大小:"), 0, 0)
        self.block_size_spin = QSpinBox()
        self.block_size_spin.setRange(128, 512)
        self.block_size_spin.setValue(256)
        self.block_size_spin.setSingleStep(32)
        self.block_size_spin.setMinimumWidth(80)
        block_param_grid.addWidget(self.block_size_spin, 0, 1)
        block_param_grid.addWidget(QLabel("重疊大小:"), 1, 0)
        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(16, 256)
        self.overlap_spin.setValue(128)
        self.overlap_spin.setSingleStep(16)
        self.overlap_spin.setMinimumWidth(80)
        block_param_grid.addWidget(self.overlap_spin, 1, 1)
        self.weight_mask_check = QCheckBox("使用權重遮罩")
        self.weight_mask_check.setChecked(True)
        block_param_grid.addWidget(self.weight_mask_check, 2, 0, 1, 2)
        block_param_grid.addWidget(QLabel("混合模式:"), 3, 0)
        self.blending_combo = QComboBox()
        self.blending_combo.addItems(['高斯分佈', '改進型高斯分佈', '線性分佈', '餘弦分佈', '泊松分佈'])
        self.blending_combo.setCurrentText('改進型高斯分佈')
        self.blending_combo.setMinimumWidth(100)
        block_param_grid.addWidget(self.blending_combo, 3, 1)
        block_param_box.setContentLayout(block_param_grid)
        column2_layout.addWidget(block_param_box)
        column2_layout.addStretch(1)
        
        # ==================== 第三列：輸出尺寸設定 ====================
        column3_layout = QVBoxLayout()
        column3_layout.setSpacing(10)
        
        # === 輸出尺寸設定區塊 ===
        upscale_box = CollapsibleBox("輸出尺寸設定")
        upscale_layout = QVBoxLayout()
        upscale_layout.setSpacing(10)
        self.size_option_group = QButtonGroup(self)
        self.original_size_radio = QRadioButton("原始大小")
        self.original_size_radio.setChecked(True) 
        self.size_option_group.addButton(self.original_size_radio, 0)
        upscale_layout.addWidget(self.original_size_radio)
        upscale_option_layout = QVBoxLayout()
        self.upscale_radio = QRadioButton("倍率超分")
        self.size_option_group.addButton(self.upscale_radio, 1)
        upscale_option_layout.addWidget(self.upscale_radio)
        scale_factor_container = QWidget()
        scale_factor_grid = QGridLayout(scale_factor_container)
        scale_factor_grid.setContentsMargins(20, 0, 0, 0) 
        scale_factor_grid.addWidget(QLabel("超分倍率:"), 0, 0)
        self.scale_factor_spinbox = QDoubleSpinBox()
        self.scale_factor_spinbox.setRange(1.0, 4.0) 
        self.scale_factor_spinbox.setValue(2.0)  
        self.scale_factor_spinbox.setSingleStep(0.05)
        self.scale_factor_spinbox.setDecimals(2) 
        self.scale_factor_spinbox.setSuffix("x") 
        self.scale_factor_spinbox.setMinimumWidth(80)
        self.scale_factor_spinbox.valueChanged.connect(self.update_scale_factor_info)
        scale_factor_grid.addWidget(self.scale_factor_spinbox, 0, 1)
        self.scale_size_info_label = QLabel("輸出尺寸: -- x -- 像素")
        scale_factor_grid.addWidget(self.scale_size_info_label, 1, 0, 1, 2)
        upscale_option_layout.addWidget(scale_factor_container)
        upscale_layout.addLayout(upscale_option_layout)
        custom_size_option_layout = QVBoxLayout()
        self.custom_size_radio = QRadioButton("指定圖像大小")
        self.size_option_group.addButton(self.custom_size_radio, 2)
        custom_size_option_layout.addWidget(self.custom_size_radio)
        custom_size_container = QWidget()
        custom_size_grid = QGridLayout(custom_size_container)
        custom_size_grid.setContentsMargins(20, 0, 0, 0) 
        custom_size_grid.addWidget(QLabel("寬度:"), 0, 0)
        self.width_input = QLineEdit()
        self.width_input.setValidator(QIntValidator(32, 7680))
        self.width_input.setFixedWidth(80)
        custom_size_grid.addWidget(self.width_input, 0, 1)
        custom_size_grid.addWidget(QLabel("高度:"), 0, 2)
        self.height_input = QLineEdit()
        self.height_input.setValidator(QIntValidator(32, 4320))
        self.height_input.setFixedWidth(80)
        custom_size_grid.addWidget(self.height_input, 0, 3)
        self.maintain_aspect_ratio_check = QCheckBox("維持原始比例")
        self.maintain_aspect_ratio_check.setChecked(True)
        custom_size_grid.addWidget(self.maintain_aspect_ratio_check, 1, 0, 1, 2)
        custom_size_grid.addWidget(QLabel("處理模式:"), 1, 2)
        self.custom_size_mode_combo = QComboBox()
        self.custom_size_mode_combo.addItems(["延伸至適合大小", "裁切至適合大小"])
        custom_size_grid.addWidget(self.custom_size_mode_combo, 1, 3)
        self.custom_size_info_label = QLabel("請輸入有效的寬度和高度")
        custom_size_grid.addWidget(self.custom_size_info_label, 2, 0, 1, 4)
        custom_size_option_layout.addWidget(custom_size_container)
        upscale_layout.addLayout(custom_size_option_layout)
        upscale_box.setContentLayout(upscale_layout)
        column3_layout.addWidget(upscale_box)
        column3_layout.addStretch(1)
        self.size_option_group.buttonClicked.connect(self.on_size_option_changed)
        self.width_input.textChanged.connect(self.update_custom_size_info)
        self.height_input.textChanged.connect(self.update_custom_size_info)
        self.maintain_aspect_ratio_check.stateChanged.connect(self.on_maintain_aspect_ratio_changed)
        self.custom_size_mode_combo.currentIndexChanged.connect(self.update_custom_size_info)
        params_content_layout.addLayout(column1_layout, 1)
        params_content_layout.addLayout(column2_layout, 1)
        params_content_layout.addLayout(column3_layout, 1)
        params_layout.addLayout(params_content_layout)
        self.on_size_option_changed()
        self.main_splitter.addWidget(preview_widget)
        self.main_splitter.addWidget(params_widget)
        self.main_splitter.setSizes([700, 300])
        self.main_splitter.widget(1).hide()
        main_layout.addWidget(self.main_splitter)
    
    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "開啟圖片", "", "圖片文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if file_path:
            self.input_image_path = file_path
            image = Image.open(file_path).convert("RGB")
            self.original_image_size = image.size
            self.update_scale_factor_info(self.scale_factor_spinbox.value())
            if not self.width_input.text() and not self.height_input.text():
                self.width_input.setText(str(image.width))
                self.height_input.setText(str(image.height))
            self.update_custom_size_info()
            self.multi_view.set_images(
                image_a=image, 
                image_b=None, 
                image_a_name=f"原始圖片 - {os.path.basename(file_path)} ({image.width}x{image.height})",
                image_b_name="增強圖片"
            )
            self.img_path_label.setText(f"{os.path.basename(file_path)} ({image.width}x{image.height})")
            if self.parent:
                self.parent.statusBar.showMessage(f"已載入圖片: {file_path} ({image.width}x{image.height})")
            self.enhance_button.setEnabled(True)
            self.save_button.setEnabled(False)
            self.enhanced_image = None
            if self.parent:
                self.parent.tab_widget.setCurrentIndex(0)
    
    def enhance_image(self):
        if not self.input_image_path:
            QMessageBox.warning(self, "警告", "請先開啟圖片。")
            return
        if not self.model_manager or not self.model_manager.get_registered_model_path():
            QMessageBox.warning(self, "警告", "請先選擇要使用的模型。")
            return
        self.enhance_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.img_status_label.setText("載入模型中...")
        self.img_progress_bar.setValue(0)
        try:
            self.model_status_label.setText("模型狀態: 載入中...")
            if self.parent:
                self.parent.statusBar.showMessage("正在載入模型到記憶體...")
            success = self.model_manager.prepare_model_for_inference()
            if not success:
                QMessageBox.warning(self, "錯誤", "模型載入失敗。")
                self.enhance_button.setEnabled(True)
                self.model_status_label.setText("模型狀態: 載入失敗")
                return
            model = self.model_manager.get_current_model()
            model_path = self.model_manager.get_registered_model_path()
            model_name = os.path.basename(model_path) if model_path else "未知模型"
            self.model_status_label.setText(f"模型狀態: 已載入 {model_name}")
            if self.parent:
                self.parent.statusBar.showMessage(f"模型已載入: {model_name}，開始處理圖片...")
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
            
            block_size = self.block_size_spin.value()
            overlap = self.overlap_spin.value()
            use_weight_mask = self.weight_mask_check.isChecked()
            blending_mode = self.blending_combo.currentText()
            strength_percent = (self.strength_slider.value() + 1) * 10
            strength = strength_percent / 100.0
            upscale_factor = 1.0 
            target_width = 0
            target_height = 0
            maintain_aspect_ratio = False
            resize_mode = ""
            if self.upscale_radio.isChecked():
                upscale_factor = self.scale_factor_spinbox.value()
            elif self.custom_size_radio.isChecked():
                try:
                    target_width = int(self.width_input.text() or "0")
                    target_height = int(self.height_input.text() or "0")
                    if target_width <= 0 or target_height <= 0:
                        QMessageBox.warning(self, "錯誤", "請輸入有效的寬度和高度。")
                        self.enhance_button.setEnabled(True)
                        return
                    maintain_aspect_ratio = self.maintain_aspect_ratio_check.isChecked()
                    resize_mode = self.custom_size_mode_combo.currentText()
                except ValueError:
                    QMessageBox.warning(self, "錯誤", "請輸入有效的寬度和高度。")
                    self.enhance_button.setEnabled(True)
                    return
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
            self.img_status_label.setText("處理中...")
            logger.info(f"使用設備: {get_device_name(device, self.system_info)}")
            self.enhancer_thread = EnhancerThread(
                model, 
                self.input_image_path, 
                device, 
                block_size, 
                overlap, 
                use_weight_mask, 
                blending_mode,
                strength,
                upscale_factor,
                target_width,
                target_height,
                maintain_aspect_ratio,
                resize_mode,
                use_amp=use_amp 
            )
            self.enhancer_thread.progress_signal.connect(self.update_progress)
            self.enhancer_thread.finished_signal.connect(self.process_finished)
            self.enhancer_thread.start()
            
        except Exception as e:
            logger.error(f"圖片處理過程中發生錯誤: {str(e)}")
            QMessageBox.critical(self, "錯誤", f"處理時出錯: {str(e)}")
            self.enhance_button.setEnabled(True)
            self.model_status_label.setText("模型狀態: 錯誤")
            self.model_manager.clear_cache()
    
    def update_progress(self, current, total):
        progress = int(current / total * 100)
        self.img_progress_bar.setValue(progress)
        self.img_status_label.setText(f"處理中... {current}/{total} 區塊 ({progress}%)")
    
    def process_finished(self, enhanced_image, elapsed_time):
        self.enhanced_image = enhanced_image
        original_image = Image.open(self.input_image_path).convert("RGB")
        strength_percent = (self.strength_slider.value() + 1) * 10
        size_info = f"({enhanced_image.width}x{enhanced_image.height})"
        resize_info = ""
        if self.upscale_radio.isChecked():
            factor = self.scale_factor_spinbox.value()
            resize_info = f", 超分: {factor:.2f}x"
        elif self.custom_size_radio.isChecked():
            resize_info = f", 自訂尺寸"
        device_info = ""
        amp_info = ""
        if self.amp_combo.currentText() != '自動偵測':
            amp_info = f", 混合精度: {self.amp_combo.currentText()}"
        self.multi_view.set_images(
            image_a=original_image,
            image_b=enhanced_image,
            image_a_name=f"原始圖片 - {os.path.basename(self.input_image_path)} ({original_image.width}x{original_image.height})",
            image_b_name=f"增強圖片 {size_info} (強度: {strength_percent}%{resize_info}{device_info}{amp_info})"
        )
        self.img_status_label.setText(f"處理完成！耗時: {elapsed_time:.2f} 秒")
        self.enhance_button.setEnabled(True)
        self.save_button.setEnabled(True)
        try:
            self.model_manager.clear_cache()
            self.model_status_label.setText("模型狀態: 已卸載")
            if self.parent:
                model_path = self.model_manager.get_registered_model_path()
                model_name = os.path.basename(model_path) if model_path else "未知模型"
                self.parent.statusBar.showMessage(f"圖片增強完成，已卸載模型。耗時: {elapsed_time:.2f} 秒")
            logger.info(f"處理完成後已卸載模型，耗時: {elapsed_time:.2f} 秒")
        except Exception as e:
            logger.error(f"卸載模型時出錯: {str(e)}")
            self.model_status_label.setText("模型狀態: 卸載失敗")
            if self.parent:
                self.parent.statusBar.showMessage(f"圖片增強完成。耗時: {elapsed_time:.2f} 秒")
    
    def save_image(self):
        if not self.enhanced_image:
            QMessageBox.warning(self, "警告", "沒有增強後的圖片可以保存。")
            return
        selected_format = self.save_format_combo.currentText().lower()
        file_filter = f"{selected_format.upper()} 圖片 (*.{selected_format})"
        
        output_path, _ = QFileDialog.getSaveFileName(
            self, "保存圖片", "", file_filter
        )
        if output_path:
            if not output_path.lower().endswith(f".{selected_format}"):
                output_path += f".{selected_format}"
            self.enhanced_image.save(output_path)
            if self.parent:
                self.parent.statusBar.showMessage(f"圖片已保存至: {output_path}")
            logger.info(f"已保存增強圖片至: {output_path}")