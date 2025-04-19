import os
import logging
from PIL import Image
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                            QFileDialog, QProgressBar, QComboBox, QSpinBox, QCheckBox, 
                            QGroupBox, QMessageBox)
from PyQt6.QtCore import Qt

from src.ui.views import MultiViewWidget
from src.processing.NS_ImageProcessor import ImageProcessor
from src.threads.NS_EnhancerThread import EnhancerThread


logger = logging.getLogger(__name__)

class ImageProcessingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.input_image_path = None
        self.enhanced_image = None
        self.model_manager = None
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
    
    def setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.multi_view = MultiViewWidget(self)
        self.multi_view.set_images(
                image_a=None, 
                image_b=None, 
                image_a_name=f"原始圖片",
                image_b_name="增強圖片"
            )
        layout.addWidget(self.multi_view)
        
        # 參數控制區域
        control_layout = QHBoxLayout()
        param_group = QGroupBox("參數設定")
        param_layout = QVBoxLayout()
        
        # 模型下拉選單
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("選擇模型:"))
        self.model_selector = QComboBox()
        self.model_selector.setMinimumWidth(300)
        self.model_selector.currentIndexChanged.connect(self.on_model_selected)
        model_layout.addWidget(self.model_selector)
        param_layout.addLayout(model_layout)
        
        # 區塊大小設定
        block_size_layout = QHBoxLayout()
        block_size_layout.addWidget(QLabel("區塊大小:"))
        self.block_size_spin = QSpinBox()
        self.block_size_spin.setRange(128, 512)
        self.block_size_spin.setValue(256)
        self.block_size_spin.setSingleStep(32)
        block_size_layout.addWidget(self.block_size_spin)
        param_layout.addLayout(block_size_layout)
        
        # 重疊大小設定
        overlap_layout = QHBoxLayout()
        overlap_layout.addWidget(QLabel("重疊大小:"))
        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(16, 256)
        self.overlap_spin.setValue(128)
        self.overlap_spin.setSingleStep(16)
        overlap_layout.addWidget(self.overlap_spin)
        param_layout.addLayout(overlap_layout)
        
        # 權重遮罩設定
        weight_mask_layout = QHBoxLayout()
        self.weight_mask_check = QCheckBox("使用權重遮罩")
        self.weight_mask_check.setChecked(True)
        weight_mask_layout.addWidget(self.weight_mask_check)
        param_layout.addLayout(weight_mask_layout)
        
        # 混合模式設定
        blending_layout = QHBoxLayout()
        blending_layout.addWidget(QLabel("混合模式:"))
        self.blending_combo = QComboBox()
        self.blending_combo.addItems(['高斯分佈', '改進型高斯分佈', '線性分佈', '餘弦分佈', '泊松分佈'])
        self.blending_combo.setCurrentText('改進型高斯分佈')
        blending_layout.addWidget(self.blending_combo)
        param_layout.addLayout(blending_layout)
        param_group.setLayout(param_layout)
        control_layout.addWidget(param_group)
        
        # 進度區域
        progress_group = QGroupBox("處理進度")
        progress_layout = QVBoxLayout()
        self.img_progress_bar = QProgressBar()
        self.img_progress_bar.setValue(0)
        progress_layout.addWidget(self.img_progress_bar)
        self.img_status_label = QLabel("等待處理...")
        progress_layout.addWidget(self.img_status_label)
        
        # 模型狀態標籤
        self.model_status_label = QLabel("模型狀態: 未載入")
        progress_layout.addWidget(self.model_status_label)
        progress_group.setLayout(progress_layout)
        control_layout.addWidget(progress_group)
        
        # 按鈕區域
        button_layout = QVBoxLayout()
        self.img_open_button = QPushButton("開啟圖片")
        self.img_open_button.clicked.connect(self.open_image)
        button_layout.addWidget(self.img_open_button)
        self.enhance_button = QPushButton("優化圖片")
        self.enhance_button.setEnabled(False)
        self.enhance_button.clicked.connect(self.enhance_image)
        button_layout.addWidget(self.enhance_button)
        self.save_button = QPushButton("保存結果")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_image)
        button_layout.addWidget(self.save_button)
        control_layout.addLayout(button_layout)
        layout.addLayout(control_layout)
    
    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "開啟圖片", "", "圖片文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if file_path:
            self.input_image_path = file_path
            image = Image.open(file_path).convert("RGB")
            self.multi_view.set_images(
                image_a=image, 
                image_b=None, 
                image_a_name=f"原始圖片 - {os.path.basename(file_path)}",
                image_b_name="增強圖片"
            )
            if self.parent:
                self.parent.statusBar.showMessage(f"已載入圖片: {file_path}")
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
            block_size = self.block_size_spin.value()
            overlap = self.overlap_spin.value()
            use_weight_mask = self.weight_mask_check.isChecked()
            blending_mode = self.blending_combo.currentText()
            self.img_status_label.setText("處理中...")
            device = self.model_manager.get_device()
            self.enhancer_thread = EnhancerThread(
                model, 
                self.input_image_path, 
                device, 
                block_size, 
                overlap, 
                use_weight_mask, 
                blending_mode
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
        self.multi_view.set_images(
            image_a=original_image,
            image_b=enhanced_image,
            image_a_name=f"原始圖片 - {os.path.basename(self.input_image_path)}",
            image_b_name=f"增強圖片"
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
        output_path, _ = QFileDialog.getSaveFileName(
            self, "保存圖片", "", "圖片文件 (*.png *.jpg *.jpeg)"
        )
        if output_path:
            self.enhanced_image.save(output_path)
            if self.parent:
                self.parent.statusBar.showMessage(f"圖片已保存至: {output_path}")