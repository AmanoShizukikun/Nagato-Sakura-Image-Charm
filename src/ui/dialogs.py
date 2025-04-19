import os
import re
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                           QFileDialog, QProgressBar, QMessageBox, QComboBox, QTabWidget, 
                           QSpinBox, QCheckBox, QWidget, QGroupBox, QGridLayout, QSlider,
                           QTextEdit, QListWidget, QStackedWidget, QRadioButton, QButtonGroup,)
from PyQt6.QtCore import Qt, QTimer, QRect
from PyQt6.QtGui import QPixmap, QCursor


class DownloadModelDialog(QDialog):
    """模型下載對話框，支援官方模型和自訂URL下載"""
    def __init__(self, model_manager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("下載模型")
        self.setMinimumWidth(720) 
        self.setMinimumHeight(600)
        self.model_manager = model_manager
        self.init_model_data()
        self.current_model_index = 0
        self.sliding_in_progress = False
        self.is_downloading = False
        self.setup_ui()
        
    def init_model_data(self):
        """初始化官方模型數據"""
        self.official_models = self.model_manager.get_models()
        self.categories = self.model_manager.get_categories()
        self.version = self.model_manager.get_version()
        self.last_updated = self.model_manager.get_last_updated()
        
    def setup_ui(self):
        """設置主要UI佈局"""
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        version_layout = QHBoxLayout()
        self.version_label = QLabel(f"模型庫版本: {self.version} (更新日期: {self.last_updated})")
        self.version_label.setStyleSheet("color: #666; font-size: 9pt;")
        version_layout.addWidget(self.version_label)
        self.update_status_label = QLabel("")
        self.update_status_label.setStyleSheet("color: #0066cc; font-size: 9pt;")
        version_layout.addWidget(self.update_status_label)
        version_layout.addStretch()
        self.check_update_button = QPushButton("檢查更新")
        self.check_update_button.setFixedWidth(80)
        self.check_update_button.clicked.connect(self.check_for_updates)
        version_layout.addWidget(self.check_update_button)
        self.update_button = QPushButton("更新")
        self.update_button.setFixedWidth(60)
        self.update_button.setEnabled(False)
        self.update_button.clicked.connect(self.update_models_data)
        version_layout.addWidget(self.update_button)
        main_layout.addLayout(version_layout)
        self.tabs = QTabWidget()
        self.official_tab = QWidget()
        self.custom_tab = QWidget()
        self.tabs.addTab(self.official_tab, "官方模型")
        self.tabs.addTab(self.custom_tab, "自訂URL")
        self.tabs.currentChanged.connect(self.on_tab_changed)
        main_layout.addWidget(self.tabs)
        self.setup_official_tab()
        self.setup_custom_tab()
        self.setup_progress_section(main_layout)
        self.setup_options_section(main_layout)
        self.setup_buttons(main_layout)
        self.model_manager.update_available_signal.connect(self.on_update_available)
        self.model_manager.update_progress_signal.connect(self.on_update_progress)
        self.model_manager.update_finished_signal.connect(self.on_update_finished)
        self.model_manager.download_progress_signal.connect(self.update_progress)
        self.model_manager.download_finished_signal.connect(self.download_finished)
        self.model_manager.download_retry_signal.connect(self.on_download_retry)  # 備用載點信號連接
        
    def setup_progress_section(self, parent_layout):
        """設置下載進度區域"""
        progress_group = QGroupBox("下載進度")
        progress_layout = QVBoxLayout()
        progress_group.setLayout(progress_layout)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        self.status_label = QLabel("準備下載")
        progress_layout.addWidget(self.status_label)
        parent_layout.addWidget(progress_group)
    
    def setup_options_section(self, parent_layout):
        """設置下載選項區域"""
        options_group = QGroupBox("下載選項")
        options_layout = QGridLayout()
        options_group.setLayout(options_layout)
        options_layout.addWidget(QLabel("下載線程數:"), 0, 0)
        thread_layout = QVBoxLayout()
        slider_layout = QHBoxLayout()
        self.thread_slider = QSlider(Qt.Orientation.Horizontal)
        self.thread_slider.setMinimum(1)
        self.thread_slider.setMaximum(8)
        self.thread_slider.setValue(4)
        self.thread_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.thread_slider.setTickInterval(1)
        self.thread_slider.valueChanged.connect(self.update_thread_value)
        self.thread_value_label = QLabel("4")
        slider_layout.addWidget(self.thread_slider)
        slider_layout.addWidget(self.thread_value_label)
        thread_layout.addLayout(slider_layout)
        thread_recommendation = QLabel("建議：安全1-2線程，穩定3-4線程，最佳5-6線程，風險7-8線程。")
        thread_recommendation.setStyleSheet("color: gray; font-size: 9pt;")
        thread_recommendation.setWordWrap(True)
        thread_layout.addWidget(thread_recommendation)
        options_layout.addLayout(thread_layout, 0, 1)
        options_layout.addWidget(QLabel("最大重試次數:"), 1, 0)
        self.retry_spinner = QSpinBox()
        self.retry_spinner.setMinimum(1)
        self.retry_spinner.setMaximum(10)
        self.retry_spinner.setValue(3)
        options_layout.addWidget(self.retry_spinner, 1, 1)
        self.auto_extract = QCheckBox("下載後自動解壓（如為壓縮包）")
        self.auto_extract.setChecked(True)
        options_layout.addWidget(self.auto_extract, 2, 0, 1, 2)
        parent_layout.addWidget(options_group)
    
    def setup_buttons(self, parent_layout):
        """設置底部按鈕區域"""
        button_layout = QHBoxLayout()
        self.download_button = QPushButton("下載")
        self.download_button.clicked.connect(self.download)
        button_layout.addWidget(self.download_button)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.on_cancel_button_clicked)
        button_layout.addWidget(self.cancel_button)
        parent_layout.addLayout(button_layout)

    def on_cancel_button_clicked(self):
        """處理取消按鈕點擊事件"""
        if self.is_downloading:
            self.status_label.setText("正在取消下載並清理不完整檔案...")
            success = self.model_manager.cancel_download()
            if not success:
                self.reject()
        else:
            self.reject()

    def setup_official_tab(self):
        """設置官方模型標籤頁"""
        layout = QVBoxLayout()
        self.official_tab.setLayout(layout)
        if len(self.categories) > 1: 
            category_layout = QHBoxLayout()
            category_layout.addWidget(QLabel("分類過濾:"))
            self.category_combo = QComboBox()
            self.category_combo.addItem("全部分類")
            for category in self.categories:
                self.category_combo.addItem(category)
            self.category_combo.currentIndexChanged.connect(self.filter_by_category)
            category_layout.addWidget(self.category_combo)
            category_layout.addStretch()
            layout.addLayout(category_layout)
        self.setup_view_options(layout)
        self.model_view_stack = QStackedWidget()
        layout.addWidget(self.model_view_stack)
        self.setup_gallery_view()
        self.setup_list_view()
        self.setup_model_details(layout)
        if self.official_models:
            self.update_model_preview(0)
    
    def setup_view_options(self, parent_layout):
        """設置視圖切換選項"""
        view_options_layout = QHBoxLayout()
        self.view_group = QButtonGroup()
        self.gallery_view_radio = QRadioButton("圖像瀏覽")
        self.gallery_view_radio.setChecked(True)
        self.view_group.addButton(self.gallery_view_radio)
        view_options_layout.addWidget(self.gallery_view_radio)
        self.list_view_radio = QRadioButton("清單瀏覽")
        self.view_group.addButton(self.list_view_radio)
        view_options_layout.addWidget(self.list_view_radio)
        view_options_layout.addStretch()
        parent_layout.addLayout(view_options_layout)
        self.gallery_view_radio.toggled.connect(self.toggle_view_mode)
        
    def setup_gallery_view(self):
        """設置圖像瀏覽視圖"""
        self.gallery_view = QWidget()
        gallery_layout = QVBoxLayout()
        self.gallery_view.setLayout(gallery_layout)
        preview_group = QGroupBox("模型預覽")
        preview_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        preview_layout = QVBoxLayout(preview_group)
        preview_container = QWidget()
        preview_container_layout = QHBoxLayout(preview_container)
        preview_container_layout.setContentsMargins(0, 0, 0, 0)
        preview_container_layout.setSpacing(5) 
        
        # 左預覽圖
        self.left_preview = QLabel()
        self.left_preview.setFixedSize(180, 240) 
        self.left_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_preview.setStyleSheet("""
            border: 1px solid lightgray; 
            background-color: #f8f8f8; 
            border-radius: 3px;
            padding: 0px;
        """)
        self.left_preview.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.left_preview.mousePressEvent = lambda e: self.show_prev_model()
        preview_container_layout.addWidget(self.left_preview)
        
        # 中預覽圖
        self.model_preview_label = QLabel()
        self.model_preview_label.setFixedSize(192, 256)
        self.model_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.model_preview_label.setStyleSheet("""
            border: 2px solid #3498db; 
            background-color: white; 
            border-radius: 3px;
            padding: 0px;
        """)
        preview_container_layout.addWidget(self.model_preview_label)
        
        # 右側預覽縮略圖
        self.right_preview = QLabel()
        self.right_preview.setFixedSize(180, 240)
        self.right_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_preview.setStyleSheet("""
            border: 1px solid lightgray; 
            background-color: #f8f8f8; 
            border-radius: 3px;
            padding: 0px;
        """)
        self.right_preview.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.right_preview.mousePressEvent = lambda e: self.show_next_model()
        preview_container_layout.addWidget(self.right_preview)
        preview_layout.addWidget(preview_container)
        
        # 導航與資訊區
        nav_info_layout = QHBoxLayout()
        self.prev_button = QPushButton("< 上一個")
        self.prev_button.clicked.connect(self.show_prev_model)
        nav_info_layout.addWidget(self.prev_button)
        info_layout = QVBoxLayout()
        self.current_model_name = QLabel()
        self.current_model_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_model_name.setStyleSheet("font-weight: bold; font-size: 10pt;")
        info_layout.addWidget(self.current_model_name)
        self.model_count_label = QLabel()
        self.model_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.model_count_label.setStyleSheet("color: gray; font-size: 9pt;")
        info_layout.addWidget(self.model_count_label)
        nav_info_layout.addLayout(info_layout)
        self.next_button = QPushButton("下一個 >")
        self.next_button.clicked.connect(self.show_next_model)
        nav_info_layout.addWidget(self.next_button)
        preview_layout.addLayout(nav_info_layout)
        self.model_usage_label = QLabel()
        self.model_usage_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.model_usage_label.setStyleSheet("font-size: 10pt; margin: 8px 0;")
        self.model_usage_label.setWordWrap(True)
        preview_layout.addWidget(self.model_usage_label)
        model_meta_layout = QHBoxLayout()
        self.model_category_label = QLabel()
        self.model_category_label.setStyleSheet("color: #666; font-size: 9pt;")
        model_meta_layout.addWidget(self.model_category_label)
        model_meta_layout.addStretch()
        self.model_date_label = QLabel()
        self.model_date_label.setStyleSheet("color: #666; font-size: 9pt;")
        model_meta_layout.addWidget(self.model_date_label)
        preview_layout.addLayout(model_meta_layout)
        gallery_layout.addWidget(preview_group)
        self.model_view_stack.addWidget(self.gallery_view)
        
    def setup_list_view(self):
        """設置列表瀏覽視圖"""
        self.list_view = QWidget()
        list_layout = QVBoxLayout()
        self.list_view.setLayout(list_layout)
        self.model_list = QListWidget()
        self.model_list.setStyleSheet("""
            QListWidget {
                font-size: 10pt;
            }
            QListWidget::item {
                padding: 6px;
                border-bottom: 1px solid #e0e0e0;
            }
            QListWidget::item:selected {
                background-color: #d0e8f2;
                color: black;
            }
        """)
        self.refresh_model_list()
        self.model_list.currentRowChanged.connect(self.on_list_selection_changed)
        list_layout.addWidget(self.model_list)
        self.model_view_stack.addWidget(self.list_view)
    
    def refresh_model_list(self):
        """重新載入模型列表"""
        self.model_list.clear()
        for model_id, info in self.official_models.items():
            self.model_list.addItem(info["name"])
    
    def setup_model_details(self, parent_layout):
        """設置模型詳情區域（列表視圖的詳情區域）"""
        self.model_info_group = QGroupBox("模型詳情")
        model_info_layout = QGridLayout()
        self.model_info_group.setLayout(model_info_layout)
        desc_label = QLabel("描述:")
        desc_label.setStyleSheet("font-weight: bold;")
        model_info_layout.addWidget(desc_label, 0, 0)
        self.model_description_label = QLabel()
        self.model_description_label.setWordWrap(True)
        self.model_description_label.setStyleSheet("font-size: 10pt;")
        model_info_layout.addWidget(self.model_description_label, 0, 1)
        details_label = QLabel("詳細資訊:")
        details_label.setStyleSheet("font-weight: bold;")
        model_info_layout.addWidget(details_label, 1, 0, Qt.AlignmentFlag.AlignTop)
        self.model_details_text = QTextEdit()
        self.model_details_text.setReadOnly(True)
        self.model_details_text.setMaximumHeight(120)
        self.model_details_text.setStyleSheet("font-size: 10pt;")
        model_info_layout.addWidget(self.model_details_text, 1, 1)
        category_label = QLabel("類別:")
        category_label.setStyleSheet("font-weight: bold;")
        model_info_layout.addWidget(category_label, 2, 0)
        self.model_category_text = QLabel()
        model_info_layout.addWidget(self.model_category_text, 2, 1)
        date_label = QLabel("發佈日期:")
        date_label.setStyleSheet("font-weight: bold;")
        model_info_layout.addWidget(date_label, 3, 0)
        self.model_date_text = QLabel()
        model_info_layout.addWidget(self.model_date_text, 3, 1)
        save_label = QLabel("保存位置:")
        save_label.setStyleSheet("font-weight: bold;")
        model_info_layout.addWidget(save_label, 4, 0)
        save_layout = QHBoxLayout()
        self.official_save_path = QLineEdit()
        save_layout.addWidget(self.official_save_path)
        self.official_browse_button = QPushButton("瀏覽...")
        self.official_browse_button.clicked.connect(lambda: self.browse_file(self.official_save_path))
        save_layout.addWidget(self.official_browse_button)
        model_info_layout.addLayout(save_layout, 4, 1)
        parent_layout.addWidget(self.model_info_group)

    def setup_custom_tab(self):
        """設置自訂URL標籤頁"""
        layout = QVBoxLayout()
        self.custom_tab.setLayout(layout)
        url_layout = QHBoxLayout()
        url_layout.addWidget(QLabel("模型URL:"))
        self.url_edit = QLineEdit()
        url_layout.addWidget(self.url_edit)
        layout.addLayout(url_layout)
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("保存位置:"))
        self.file_edit = QLineEdit()
        self.file_edit.setText(os.path.join("models", "custom_model.pth"))
        file_layout.addWidget(self.file_edit)
        self.browse_button = QPushButton("瀏覽...")
        self.browse_button.clicked.connect(lambda: self.browse_file(self.file_edit))
        file_layout.addWidget(self.browse_button)
        layout.addLayout(file_layout)
        tip_label = QLabel("提示: 請確保下載的模型格式正確，支持 .pth, .pt, .ckpt 和 .safetensors 格式")
        tip_label.setStyleSheet("color: gray;")
        layout.addWidget(tip_label)
        layout.addStretch()

    def check_for_updates(self):
        """檢查模型數據更新"""
        self.check_update_button.setEnabled(False)
        self.update_status_label.setText("正在檢查更新...")
        self.model_manager.check_for_updates()
    
    def on_update_available(self, available, message):
        """處理更新檢查結果"""
        self.update_status_label.setText(message)
        self.check_update_button.setEnabled(True)
        self.update_button.setEnabled(available)
    
    def on_update_progress(self, message):
        """處理更新進度"""
        self.update_status_label.setText(message)
    
    def on_update_finished(self, success, message):
        """處理更新完成"""
        self.update_status_label.setText(message)
        self.check_update_button.setEnabled(True)
        self.update_button.setEnabled(False)
        if success:
            self.init_model_data()
            self.refresh_model_list()
            self.update_model_preview(0)
            self.version_label.setText(f"模型庫版本: {self.version} (更新日期: {self.last_updated})")
    
    def update_models_data(self):
        """更新模型數據"""
        self.update_button.setEnabled(False)
        self.check_update_button.setEnabled(False)
        self.update_status_label.setText("正在更新模型數據...")
        self.model_manager.update_models_data()
    
    def filter_by_category(self, index):
        """根據分類過濾模型列表"""
        if index == 0:
            self.official_models = self.model_manager.get_models()
        else:
            selected_category = self.categories[index-1]
            all_models = self.model_manager.get_models()
            self.official_models = {k: v for k, v in all_models.items() 
                                  if v.get('category', '') == selected_category}
        self.refresh_model_list()
        if self.official_models:
            self.update_model_preview(0)

    def toggle_view_mode(self, checked):
        """切換視圖模式"""
        if checked: 
            self.model_view_stack.setCurrentWidget(self.gallery_view)
            self.model_info_group.setVisible(False)
        else: 
            self.model_view_stack.setCurrentWidget(self.list_view)
            self.model_info_group.setVisible(True)
            if self.model_list.count() > 0:
                self.model_list.setCurrentRow(self.current_model_index)

    def _scale_and_align_pixmap(self, pixmap, target_width, target_height, alignment):
        """縮放圖像並根據對齊方式裁切"""
        source_ratio = pixmap.width() / pixmap.height()
        target_ratio = target_width / target_height
        if source_ratio > target_ratio:
            scaled_height = target_height
            scaled_width = int(scaled_height * source_ratio)
            scaled_pixmap = pixmap.scaled(scaled_width, scaled_height, Qt.AspectRatioMode.KeepAspectRatio)
            if alignment == Qt.AlignmentFlag.AlignLeft:
                crop_rect = QRect(0, 0, target_width, target_height)
            else:  
                crop_rect = QRect(scaled_width - target_width, 0, target_width, target_height)
        else:
            scaled_width = target_width
            scaled_height = int(scaled_width / source_ratio)
            scaled_pixmap = pixmap.scaled(scaled_width, scaled_height, Qt.AspectRatioMode.KeepAspectRatio)
            crop_y = (scaled_height - target_height) // 2
            crop_rect = QRect(0, crop_y, target_width, target_height)
        final_pixmap = scaled_pixmap.copy(crop_rect)
        return final_pixmap

    def update_model_preview(self, index, slide_direction=None):
        """更新模型預覽信息"""
        if self.sliding_in_progress or not self.official_models:
            return
        model_ids = list(self.official_models.keys())
        total_models = len(model_ids)
        if index < 0:
            index = total_models - 1
        elif index >= total_models:
            index = 0
        old_index = self.current_model_index
        if slide_direction is None:
            if index > old_index or (old_index == total_models - 1 and index == 0):
                slide_direction = "right"
            elif index < old_index or (old_index == 0 and index == total_models - 1):
                slide_direction = "left"
        self.current_model_index = index
        if total_models > 0:
            self.model_count_label.setText(f"模型 {index + 1} / {total_models}")
            left_index = (index - 1) % total_models
            right_index = (index + 1) % total_models
            current_model_id = model_ids[index]
            left_model_id = model_ids[left_index]
            right_model_id = model_ids[right_index]
            current_model = self.official_models[current_model_id]
            left_model = self.official_models[left_model_id]
            right_model = self.official_models[right_model_id]
            category = current_model.get("category", "未分類")
            added_date = current_model.get("added_date", "未知")
            self.model_category_label.setText(f"類別: {category}")
            self.model_date_label.setText(f"發布日期: {added_date}")
            self.model_category_text.setText(category)
            self.model_date_text.setText(added_date)
            if "適用場景" in current_model["details"]:
                usage_match = re.search(r'適用場景：(.+?)($|\n)', current_model["details"])
                if usage_match:
                    usage_info = usage_match.group(1).strip()
                    self.model_usage_label.setText(f"適用場景：{usage_info}")
                else:
                    self.model_usage_label.setText(current_model["description"])
            else:
                self.model_usage_label.setText(current_model["description"])
            app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            preview_path = os.path.join(app_dir, current_model.get("preview", ""))
            if os.path.exists(preview_path):
                pixmap = QPixmap(preview_path)
                self.model_preview_label.setPixmap(pixmap.scaled(
                    self.model_preview_label.width(),
                    self.model_preview_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio
                ))
                if slide_direction:
                    self.animate_slide(self.model_preview_label, slide_direction)
            else:
                self.model_preview_label.setText(f"無預覽圖\n{current_model['name']}")
            left_preview_path = os.path.join(app_dir, left_model.get("preview", ""))
            if os.path.exists(left_preview_path):
                left_pixmap = QPixmap(left_preview_path)
                left_scaled_pixmap = self._scale_and_align_pixmap(
                    left_pixmap, 
                    self.left_preview.width(), 
                    self.left_preview.height(), 
                    Qt.AlignmentFlag.AlignRight
                )
                self.left_preview.setPixmap(left_scaled_pixmap)
                self.left_preview.setToolTip(f"點擊切換至: {left_model['name']}")
            else:
                self.left_preview.setText("無預覽")
                self.left_preview.setToolTip(f"點擊切換至: {left_model['name']}")
            right_preview_path = os.path.join(app_dir, right_model.get("preview", ""))
            if os.path.exists(right_preview_path):
                right_pixmap = QPixmap(right_preview_path)
                right_scaled_pixmap = self._scale_and_align_pixmap(
                    right_pixmap, 
                    self.right_preview.width(), 
                    self.right_preview.height(), 
                    Qt.AlignmentFlag.AlignLeft
                )
                self.right_preview.setPixmap(right_scaled_pixmap)
                self.right_preview.setToolTip(f"點擊切換至: {right_model['name']}")
            else:
                self.right_preview.setText("無預覽")
                self.right_preview.setToolTip(f"點擊切換至: {right_model['name']}")
            self.current_model_name.setText(current_model["name"])
            self.model_description_label.setText(current_model["description"])
            self.model_details_text.setText(current_model["details"])
            filename = os.path.basename(current_model["url"])
            self.official_save_path.setText(os.path.join("models", filename))
            if self.model_view_stack.currentWidget() == self.list_view and self.model_list.count() > 0:
                self.model_list.setCurrentRow(index)
            self.model_info_group.setVisible(self.model_view_stack.currentWidget() == self.list_view)

    def animate_slide(self, label, direction):
        """為預覽圖添加滑動切換效果"""
        self.sliding_in_progress = True
        original_style = label.styleSheet()
        highlight_style = original_style.replace("border: 2px solid #3498db", "border: 2px solid #2ecc71")
        label.setStyleSheet(highlight_style)
        scale = 1.05
        original_size = label.size()
        scaled_width = int(original_size.width() * scale)
        scaled_height = int(original_size.height() * scale)
        label.resize(scaled_width, scaled_height)
        QTimer.singleShot(150, lambda: label.resize(original_size))
        QTimer.singleShot(300, lambda: self._finish_animation(label, original_style))
    
    def _finish_animation(self, label, original_style):
        """完成動畫後恢復原樣式並釋放滑動鎖"""
        label.setStyleSheet(original_style)
        self.sliding_in_progress = False

    def show_prev_model(self):
        """顯示上一個模型"""
        if not self.sliding_in_progress:
            self.update_model_preview(self.current_model_index - 1, "left")

    def show_next_model(self):
        """顯示下一個模型"""
        if not self.sliding_in_progress:
            self.update_model_preview(self.current_model_index + 1, "right")

    def on_list_selection_changed(self, row):
        """處理列表選擇變更事件"""
        if row >= 0:
            self.update_model_preview(row)

    def update_thread_value(self, value):
        """更新滑桿顯示的線程值"""
        self.thread_value_label.setText(str(value))
    
    def on_tab_changed(self, index):
        """處理標籤頁切換事件"""
        if index == 0:
            self.model_info_group.setVisible(self.model_view_stack.currentWidget() == self.list_view)
        
    def browse_file(self, line_edit):
        """打開文件選擇對話框"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "選擇保存位置", "", "PyTorch Model (*.pth);;All Files (*.*)"
        )
        if file_path:
            line_edit.setText(file_path)
    
    def download(self):
        """開始下載模型"""
        is_official = self.tabs.currentIndex() == 0
        self.is_downloading = True 
        if is_official:
            if not self.official_models:
                QMessageBox.warning(self, "錯誤", "沒有可用的官方模型")
                self.is_downloading = False
                return
            model_ids = list(self.official_models.keys())
            model_id = model_ids[self.current_model_index]
            file_path = self.official_save_path.text()
            num_threads = self.thread_slider.value()
            retry_count = self.retry_spinner.value()
            auto_extract = self.auto_extract.isChecked()
            self.download_button.setEnabled(False)
            self.cancel_button.setText("取消下載") 
            self.model_manager.download_official_model(
                model_id, 
                file_path, 
                num_threads=num_threads, 
                retry_count=retry_count,
                auto_extract=auto_extract
            )
        else:
            url = self.url_edit.text()
            file_path = self.file_edit.text()
            if not url or not file_path:
                QMessageBox.warning(self, "錯誤", "URL和保存位置不能為空")
                self.is_downloading = False
                return
            num_threads = self.thread_slider.value()
            retry_count = self.retry_spinner.value()
            auto_extract = self.auto_extract.isChecked()
            self.download_button.setEnabled(False)
            self.cancel_button.setText("取消下載") 
            self.model_manager.download_model_from_url(
                url, 
                file_path, 
                num_threads=num_threads, 
                retry_count=retry_count,
                auto_extract=auto_extract
            )
    
    def update_progress(self, current, total, speed):
        """更新下載進度顯示"""
        if total > 0:
            percentage = int(current / total * 100)
            self.progress_bar.setValue(percentage)
            self.status_label.setText(f"已下載: {current/1024/1024:.1f} MB / {total/1024/1024:.1f} MB ({percentage}%) - {speed/1024/1024:.2f} MB/s")
    
    def download_finished(self, success, message):
        """處理下載完成事件"""
        self.is_downloading = False
        
        if success:
            self.progress_bar.setValue(100)
            self.status_label.setText("下載完成!")
            QMessageBox.information(self, "下載成功", message)
            self.accept()
        else:
            if "取消" in message or "cancel" in message.lower():
                self.progress_bar.setValue(0)
                self.status_label.setText("下載已取消並清理不完整檔案")
                QMessageBox.information(self, "下載取消", "下載已取消，不完整的檔案已被清理")
            else:
                QMessageBox.critical(self, "下載錯誤", message)
                self.status_label.setText(f"錯誤: {message}")
            
            self.download_button.setEnabled(True)
            self.cancel_button.setText("關閉")
    
    def on_download_retry(self, message):
        """處理備用載點嘗試通知"""
        self.status_label.setText(message)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("QProgressBar { background-color: #f0e68c; }")
        QTimer.singleShot(2000, lambda: self.progress_bar.setStyleSheet(""))