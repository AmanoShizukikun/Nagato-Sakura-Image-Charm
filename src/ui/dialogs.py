import os
import re
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                           QFileDialog, QProgressBar, QMessageBox, QComboBox, QTabWidget,
                           QSpinBox, QCheckBox, QWidget, QGroupBox, QGridLayout, QSlider,
                           QTextEdit, QListWidget, QStackedWidget, QRadioButton, QButtonGroup,
                           QGraphicsDropShadowEffect)
from PyQt6.QtCore import (Qt, QTimer, QPropertyAnimation, QEasingCurve,
                        QParallelAnimationGroup, QPoint, QAbstractAnimation ) 
from PyQt6.QtGui import QPixmap, QCursor, QColor


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
        self.slide_animation_group = None
        self._current_left_preview_label = None
        self._current_right_preview_label = None
        self.center_shadow_effect = None 
        self.swipe_start_pos = None
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
        self.model_manager.download_retry_signal.connect(self.on_download_retry)

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
        # --- 陰影效果 ---
        self.center_shadow_effect = QGraphicsDropShadowEffect()
        self.center_shadow_effect.setBlurRadius(15) 
        self.center_shadow_effect.setColor(QColor(0, 0, 0, 100)) 
        self.center_shadow_effect.setOffset(3, 3) 
        self.preview_area_widget = QWidget()
        self.preview_area_widget.setMinimumHeight(260)
        preview_area_layout = QHBoxLayout(self.preview_area_widget)
        preview_area_layout.setContentsMargins(0, 0, 0, 0)
        preview_area_layout.setSpacing(5)
        preview_area_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- 左預覽圖容器 ---
        self.left_preview_container = QWidget(self.preview_area_widget)
        self.left_preview_container.setFixedSize(180, 240)
        self.left_preview_container.setStyleSheet("background-color: transparent; border: none;")
        self.left_preview_container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.left_preview_container.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.left_preview_container.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.left_preview_container.mousePressEvent = lambda e: self.show_prev_model() 
        self._current_left_preview_label = QLabel(self.left_preview_container)
        self._current_left_preview_label.setGeometry(0, 0, 180, 240)
        self._current_left_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._current_left_preview_label.setStyleSheet("""
            border: 1px solid lightgray;
            background-color: #f8f8f8;
            border-radius: 3px;
            padding: 0px;
        """)
        preview_area_layout.addWidget(self.left_preview_container)

        # --- 中預覽圖容器 ---
        self.preview_container_widget = QWidget(self.preview_area_widget)
        self.preview_container_widget.setFixedSize(192, 256)
        self.preview_container_widget.setStyleSheet("background-color: transparent; border: none;")
        self.preview_container_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.preview_container_widget.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.preview_container_widget.mousePressEvent = self._handle_mouse_press
        self.preview_container_widget.mouseMoveEvent = self._handle_mouse_move
        self.preview_container_widget.mouseReleaseEvent = self._handle_mouse_release
        self.preview_container_widget.setCursor(QCursor(Qt.CursorShape.OpenHandCursor)) 
        self.model_preview_label = QLabel(self.preview_container_widget)
        self.model_preview_label.setGeometry(0, 0, 192, 256)
        self.model_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.model_preview_label.setStyleSheet("""
            border: 3px solid #3498db;
            background-color: white;
            border-radius: 3px;
            padding: 0px;
        """)
        self.model_preview_label.setGraphicsEffect(self.center_shadow_effect)
        preview_area_layout.addWidget(self.preview_container_widget)

        # --- 右預覽圖容器 ---
        self.right_preview_container = QWidget(self.preview_area_widget)
        self.right_preview_container.setFixedSize(180, 240)
        self.right_preview_container.setStyleSheet("background-color: transparent; border: none;")
        self.right_preview_container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.right_preview_container.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.right_preview_container.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.right_preview_container.mousePressEvent = lambda e: self.show_next_model() 
        self._current_right_preview_label = QLabel(self.right_preview_container)
        self._current_right_preview_label.setGeometry(0, 0, 180, 240)
        self._current_right_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._current_right_preview_label.setStyleSheet("""
            border: 1px solid lightgray;
            background-color: #f8f8f8;
            border-radius: 3px;
            padding: 0px;
        """)
        preview_area_layout.addWidget(self.right_preview_container)
        preview_layout.addWidget(self.preview_area_widget)
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
        self.model_info_group.setVisible(False) 

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
        else:
            self.update_model_preview(-1) 

    def toggle_view_mode(self, checked):
        """切換視圖模式"""
        if checked: 
            self.model_view_stack.setCurrentWidget(self.gallery_view)
            self.model_info_group.setVisible(False)
            self.update_model_preview(self.current_model_index, slide_direction=None)
        else:
            self.model_view_stack.setCurrentWidget(self.list_view)
            self.model_info_group.setVisible(True)
            if self.model_list.count() > 0:
                self.model_list.blockSignals(True)
                self.model_list.setCurrentRow(self.current_model_index)
                self.model_list.blockSignals(False)
                self.on_list_selection_changed(self.current_model_index)

    def _scale_and_align_pixmap(self, pixmap, target_width, target_height, alignment=Qt.AlignmentFlag.AlignCenter):
        """縮放圖像並根據對齊方式裁切 (默認居中)"""
        if pixmap.isNull() or target_width <= 0 or target_height <= 0:
            return QPixmap() 
        scaled_pixmap = pixmap.scaled(target_width, target_height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        return scaled_pixmap 

    def _load_pixmap_for_index(self, index):
        """根據索引加載並縮放模型預覽圖"""
        model_ids = list(self.official_models.keys())
        total_models = len(model_ids)
        if not model_ids or index < 0 or index >= total_models:
            return QPixmap(), None, False 
        model_id = model_ids[index]
        model = self.official_models[model_id]
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        preview_path = os.path.join(app_dir, model.get("preview", ""))
        pixmap = QPixmap()
        has_image = False
        if os.path.exists(preview_path):
            pixmap = QPixmap(preview_path)
            has_image = True
        else:
            pixmap = QPixmap(192, 256) 
            pixmap.fill(Qt.GlobalColor.white)
        return pixmap, model['name'], has_image

    def update_model_preview(self, index, slide_direction=None):
        """更新模型預覽信息，準備動畫或直接設置"""
        if self.sliding_in_progress or not self.official_models:
            return
        model_ids = list(self.official_models.keys())
        total_models = len(model_ids)
        if total_models == 0:
            self._current_left_preview_label.clear()
            self._current_left_preview_label.setText("N/A")
            self._current_left_preview_label.setGraphicsEffect(None)
            self.model_preview_label.clear()
            self.model_preview_label.setText("沒有模型")
            if self.model_preview_label.graphicsEffect():
                self.model_preview_label.graphicsEffect().setEnabled(False)
            self._current_right_preview_label.clear()
            self._current_right_preview_label.setText("N/A")
            self._current_right_preview_label.setGraphicsEffect(None)
            self.current_model_name.clear()
            self.model_count_label.clear()
            self.model_usage_label.clear()
            self.model_category_label.clear()
            self.model_date_label.clear()
            self.model_description_label.clear()
            self.model_details_text.clear()
            self.model_category_text.clear()
            self.model_date_text.clear()
            self.official_save_path.clear()
            self.model_info_group.setVisible(False)
            self.current_model_index = -1
            return
        if index < 0: index = total_models - 1
        elif index >= total_models: index = 0

        # --- 判斷滑動方向 ---
        old_index = self.current_model_index
        if slide_direction is None and total_models > 1:
             if index == (old_index + 1) % total_models:
                 slide_direction = "right"
             elif index == (old_index - 1 + total_models) % total_models:
                 slide_direction = "left"
        target_model_index = index 

        # --- 更新文字資訊 ---
        current_model_id = model_ids[target_model_index]
        current_model = self.official_models[current_model_id]
        self.model_count_label.setText(f"模型 {target_model_index + 1} / {total_models}")
        category = current_model.get("category", "未分類")
        added_date = current_model.get("added_date", "未知")
        self.model_category_label.setText(f"類別: {category}")
        self.model_date_label.setText(f"發布日期: {added_date}")
        self.model_category_text.setText(category) 
        self.model_date_text.setText(added_date)  
        if "適用場景" in current_model.get("details", ""):
             usage_match = re.search(r'適用場景：(.+?)($|\n)', current_model["details"])
             usage_info = usage_match.group(1).strip() if usage_match else current_model["description"]
             self.model_usage_label.setText(f"適用場景：{usage_info}")
        else:
             self.model_usage_label.setText(current_model["description"])
        self.current_model_name.setText(current_model["name"])
        self.model_description_label.setText(current_model["description"]) 
        self.model_details_text.setText(current_model["details"]) 
        filename = os.path.basename(current_model.get("url", "unknown_model"))
        self.official_save_path.setText(os.path.join("models", filename))

        # --- 準備圖像 ---
        next_left_idx = (target_model_index - 1 + total_models) % total_models
        next_center_idx = target_model_index
        next_right_idx = (target_model_index + 1) % total_models
        next_left_pixmap, _, has_next_left_img = self._load_pixmap_for_index(next_left_idx)
        next_center_pixmap, next_center_name, has_next_center_img = self._load_pixmap_for_index(next_center_idx)
        next_right_pixmap, _, has_next_right_img = self._load_pixmap_for_index(next_right_idx)
        scaled_next_left = self._scale_and_align_pixmap(next_left_pixmap, self.left_preview_container.width(), self.left_preview_container.height())
        scaled_next_center = self._scale_and_align_pixmap(next_center_pixmap, self.preview_container_widget.width(), self.preview_container_widget.height())
        scaled_next_right = self._scale_and_align_pixmap(next_right_pixmap, self.right_preview_container.width(), self.right_preview_container.height())

        # --- 執行更新 ---
        if slide_direction and total_models > 1:
            self.current_model_index = target_model_index 
            if self.model_preview_label and self.model_preview_label.graphicsEffect():
                 self.model_preview_label.graphicsEffect().setEnabled(False)
            self.animate_slide(
                scaled_next_left, has_next_left_img,
                scaled_next_center, has_next_center_img, next_center_name,
                scaled_next_right, has_next_right_img,
                slide_direction
            )
        else:
            # --- 設置圖像 ---
            self.current_model_index = target_model_index 
            self._current_left_preview_label.setPixmap(scaled_next_left)
            self._current_left_preview_label.setText("" if has_next_left_img else "無預覽")
            self._current_left_preview_label.setGraphicsEffect(None) 

            self.model_preview_label.setPixmap(scaled_next_center)
            self.model_preview_label.setText("" if has_next_center_img else f"無預覽圖\n{next_center_name}")
            if not self.model_preview_label.graphicsEffect() or not self.model_preview_label.graphicsEffect().isEnabled():
                 if not self.center_shadow_effect: 
                     self.center_shadow_effect = QGraphicsDropShadowEffect()
                     self.center_shadow_effect.setBlurRadius(15)
                     self.center_shadow_effect.setColor(QColor(0, 0, 0, 100))
                     self.center_shadow_effect.setOffset(3, 3)
                 self.center_shadow_effect.setEnabled(True) 
                 self.model_preview_label.setGraphicsEffect(self.center_shadow_effect)
            self._current_right_preview_label.setPixmap(scaled_next_right)
            self._current_right_preview_label.setText("" if has_next_right_img else "無預覽")
            self._current_right_preview_label.setGraphicsEffect(None) 
            left_model_id = model_ids[next_left_idx]
            right_model_id = model_ids[next_right_idx]
            self._current_left_preview_label.setToolTip(f"點擊切換至: {self.official_models[left_model_id]['name']}")
            self._current_right_preview_label.setToolTip(f"點擊切換至: {self.official_models[right_model_id]['name']}")
        if self.model_view_stack.currentWidget() == self.list_view and self.model_list.count() > 0:
            self.model_list.blockSignals(True)
            self.model_list.setCurrentRow(self.current_model_index)
            self.model_list.blockSignals(False)
        self.model_info_group.setVisible(self.model_view_stack.currentWidget() == self.list_view)

    def animate_slide(self,
                      next_left_pixmap, has_next_left_img,
                      next_center_pixmap, has_next_center_img, next_center_name,
                      next_right_pixmap, has_next_right_img,
                      direction):
        """為預覽圖添加滑動切換效果 (包含左右)"""
        if self.sliding_in_progress: return
        self.sliding_in_progress = True
        left_container = self.left_preview_container
        center_container = self.preview_container_widget
        right_container = self.right_preview_container
        left_w, left_h = left_container.width(), left_container.height()
        center_w, center_h = center_container.width(), center_container.height()
        right_w, right_h = right_container.width(), right_container.height()
        current_left_label = self._current_left_preview_label
        current_center_label = self.model_preview_label
        current_right_label = self._current_right_preview_label
        if current_left_label and current_left_label.graphicsEffect():
            current_left_label.graphicsEffect().setEnabled(False)
        if current_right_label and current_right_label.graphicsEffect():
            current_right_label.graphicsEffect().setEnabled(False)
        next_left_label = QLabel(left_container)
        next_left_label.setGeometry(0, 0, left_w, left_h)
        next_left_label.setStyleSheet(current_left_label.styleSheet())
        next_left_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        next_left_label.setPixmap(next_left_pixmap)
        next_left_label.setText("" if has_next_left_img else "無預覽")
        next_left_label.setToolTip(current_left_label.toolTip())
        next_left_label.setGraphicsEffect(None)
        next_center_label = QLabel(center_container)
        next_center_label.setGeometry(0, 0, center_w, center_h)
        next_center_label.setStyleSheet(current_center_label.styleSheet())
        next_center_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        next_center_label.setPixmap(next_center_pixmap)
        next_center_label.setText("" if has_next_center_img else f"無預覽圖\n{next_center_name}")
        if not self.center_shadow_effect:
            self.center_shadow_effect = QGraphicsDropShadowEffect()
            self.center_shadow_effect.setBlurRadius(15)
            self.center_shadow_effect.setColor(QColor(0, 0, 0, 100))
            self.center_shadow_effect.setOffset(3, 3)
        self.center_shadow_effect.setEnabled(True)
        next_center_label.setGraphicsEffect(self.center_shadow_effect)
        next_right_label = QLabel(right_container)
        next_right_label.setGeometry(0, 0, right_w, right_h)
        next_right_label.setStyleSheet(current_right_label.styleSheet())
        next_right_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        next_right_label.setPixmap(next_right_pixmap)
        next_right_label.setText("" if has_next_right_img else "無預覽")
        next_right_label.setToolTip(current_right_label.toolTip())
        next_right_label.setGraphicsEffect(None) 
        if direction == "right": # 下一個模型從右邊滑入 (整體向左滑)
            start_pos_next_left = QPoint(left_w, 0)
            start_pos_next_center = QPoint(center_w, 0)
            start_pos_next_right = QPoint(right_w, 0)
            end_pos = QPoint(0, 0)
            end_pos_current_left = QPoint(-left_w, 0)
            end_pos_current_center = QPoint(-center_w, 0)
            end_pos_current_right = QPoint(-right_w, 0)
        else: # "left", 上一個模型從左邊滑入 (整體向右滑)
            start_pos_next_left = QPoint(-left_w, 0)
            start_pos_next_center = QPoint(-center_w, 0)
            start_pos_next_right = QPoint(-right_w, 0)
            end_pos = QPoint(0, 0)
            end_pos_current_left = QPoint(left_w, 0)
            end_pos_current_center = QPoint(center_w, 0)
            end_pos_current_right = QPoint(right_w, 0)
        next_left_label.move(start_pos_next_left)
        next_center_label.move(start_pos_next_center)
        next_right_label.move(start_pos_next_right)
        next_left_label.show()
        next_center_label.show()
        next_right_label.show()
        next_left_label.raise_()
        next_center_label.raise_()
        next_right_label.raise_()
        duration = 300
        easing_curve = QEasingCurve.Type.InOutQuad
        anim_current_left = QPropertyAnimation(current_left_label, b"pos")
        anim_current_left.setDuration(duration)
        anim_current_left.setEndValue(end_pos_current_left)
        anim_current_left.setEasingCurve(easing_curve)
        anim_next_left = QPropertyAnimation(next_left_label, b"pos")
        anim_next_left.setDuration(duration)
        anim_next_left.setStartValue(start_pos_next_left)
        anim_next_left.setEndValue(end_pos)
        anim_next_left.setEasingCurve(easing_curve)
        anim_current_center = QPropertyAnimation(current_center_label, b"pos")
        anim_current_center.setDuration(duration)
        anim_current_center.setEndValue(end_pos_current_center)
        anim_current_center.setEasingCurve(easing_curve)
        anim_next_center = QPropertyAnimation(next_center_label, b"pos")
        anim_next_center.setDuration(duration)
        anim_next_center.setStartValue(start_pos_next_center)
        anim_next_center.setEndValue(end_pos)
        anim_next_center.setEasingCurve(easing_curve)
        anim_current_right = QPropertyAnimation(current_right_label, b"pos")
        anim_current_right.setDuration(duration)
        anim_current_right.setEndValue(end_pos_current_right)
        anim_current_right.setEasingCurve(easing_curve)
        anim_next_right = QPropertyAnimation(next_right_label, b"pos")
        anim_next_right.setDuration(duration)
        anim_next_right.setStartValue(start_pos_next_right)
        anim_next_right.setEndValue(end_pos)
        anim_next_right.setEasingCurve(easing_curve)
        if self.slide_animation_group and self.slide_animation_group.state() == QAbstractAnimation.State.Running:
            self.slide_animation_group.stop()
        self.slide_animation_group = QParallelAnimationGroup(self)
        self.slide_animation_group.addAnimation(anim_current_left)
        self.slide_animation_group.addAnimation(anim_next_left)
        self.slide_animation_group.addAnimation(anim_current_center)
        self.slide_animation_group.addAnimation(anim_next_center)
        self.slide_animation_group.addAnimation(anim_current_right)
        self.slide_animation_group.addAnimation(anim_next_right)
        self.slide_animation_group.finished.connect(
            lambda: self._finish_slide_animation(
                current_left_label, next_left_label,
                current_center_label, next_center_label,
                current_right_label, next_right_label
            )
        )
        self.slide_animation_group.start(QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

    def _finish_slide_animation(self,old_left, new_left, old_center, new_center, old_right, new_right):
        """滑動動畫完成後的清理工作"""
        new_left.move(0, 0)
        new_center.move(0, 0)
        new_right.move(0, 0)
        old_left.deleteLater()
        old_center.deleteLater()
        old_right.deleteLater()
        self._current_left_preview_label = new_left
        self.model_preview_label = new_center
        self._current_right_preview_label = new_right
        if self.model_preview_label and self.model_preview_label.graphicsEffect():
            self.model_preview_label.graphicsEffect().setEnabled(True)
        if self._current_left_preview_label:
            self._current_left_preview_label.setGraphicsEffect(None)
        if self._current_right_preview_label:
            self._current_right_preview_label.setGraphicsEffect(None)
        model_ids = list(self.official_models.keys())
        total_models = len(model_ids)
        if total_models > 0:
            final_left_idx = (self.current_model_index - 1 + total_models) % total_models
            final_right_idx = (self.current_model_index + 1) % total_models
            left_model_id = model_ids[final_left_idx]
            right_model_id = model_ids[final_right_idx]
            self._current_left_preview_label.setToolTip(f"點擊切換至: {self.official_models[left_model_id]['name']}")
            self._current_right_preview_label.setToolTip(f"點擊切換至: {self.official_models[right_model_id]['name']}")
        self.sliding_in_progress = False
        self.slide_animation_group = None

    def show_prev_model(self):
        """顯示上一個模型"""
        if not self.sliding_in_progress and len(self.official_models) > 1:
            self.update_model_preview(self.current_model_index - 1, "left")

    def show_next_model(self):
        """顯示下一個模型"""
        if not self.sliding_in_progress and len(self.official_models) > 1:
            self.update_model_preview(self.current_model_index + 1, "right")

    def on_list_selection_changed(self, row):
        """處理列表選擇變更事件"""
        if row >= 0 and self.model_view_stack.currentWidget() == self.list_view:
            self.update_model_preview(row, slide_direction=None)

    def update_thread_value(self, value):
        """更新滑桿顯示的線程值"""
        self.thread_value_label.setText(str(value))

    def on_tab_changed(self, index):
        """處理標籤頁切換事件"""
        if index == 0:
            self.model_info_group.setVisible(self.model_view_stack.currentWidget() == self.list_view)
        else: 
            self.model_info_group.setVisible(False)

    def browse_file(self, line_edit):
        """打開文件選擇對話框"""
        current_path = line_edit.text()
        start_dir = os.path.dirname(current_path) if os.path.dirname(current_path) else "models"
        if not os.path.exists(start_dir):
            start_dir = "."

        file_path, _ = QFileDialog.getSaveFileName(
            self, "選擇保存位置", start_dir, "模型文件 (*.pth *.pt *.ckpt *.safetensors);;All Files (*.*)"
        )
        if file_path:
            line_edit.setText(file_path)

    def download(self):
        """開始下載模型"""
        is_official = self.tabs.currentIndex() == 0
        if self.is_downloading:
            return
        self.is_downloading = True
        self.download_button.setEnabled(False)
        self.cancel_button.setText("取消下載")
        self.progress_bar.setValue(0)
        if is_official:
            if not self.official_models or self.current_model_index < 0:
                QMessageBox.warning(self, "錯誤", "沒有選中有效的官方模型")
                self._reset_download_state()
                return
            model_ids = list(self.official_models.keys())
            model_id = model_ids[self.current_model_index]
            file_path = self.official_save_path.text()
            if not file_path:
                QMessageBox.warning(self, "錯誤", "請指定保存位置")
                self._reset_download_state()
                return
            num_threads = self.thread_slider.value()
            retry_count = self.retry_spinner.value()
            auto_extract = self.auto_extract.isChecked()
            self.status_label.setText(f"準備下載官方模型: {self.official_models[model_id]['name']}...")
            self.model_manager.download_official_model(
                model_id, file_path, num_threads=num_threads,
                retry_count=retry_count, auto_extract=auto_extract
            )
        else:
            url = self.url_edit.text().strip()
            file_path = self.file_edit.text().strip()
            if not url or not file_path:
                QMessageBox.warning(self, "錯誤", "URL和保存位置不能為空")
                self._reset_download_state()
                return
            num_threads = self.thread_slider.value()
            retry_count = self.retry_spinner.value()
            auto_extract = self.auto_extract.isChecked()
            self.status_label.setText(f"準備從URL下載...")
            self.model_manager.download_model_from_url(
                url, file_path, num_threads=num_threads,
                retry_count=retry_count, auto_extract=auto_extract
            )

    def _reset_download_state(self):
        """重置下載相關的UI狀態"""
        self.is_downloading = False
        self.download_button.setEnabled(True)
        self.cancel_button.setText("關閉")
        self.status_label.setText("準備下載")

    def update_progress(self, current, total, speed):
        """更新下載進度顯示"""
        if total > 0:
            percentage = int(current / total * 100)
            self.progress_bar.setValue(percentage)
            speed_mb = speed / 1024 / 1024
            speed_kb = speed / 1024
            if speed_mb >= 1:
                speed_str = f"{speed_mb:.2f} MB/s"
            elif speed_kb >= 1:
                speed_str = f"{speed_kb:.1f} KB/s"
            else:
                speed_str = f"{speed:.0f} B/s"
            self.status_label.setText(f"已下載: {current/1024/1024:.1f} MB / {total/1024/1024:.1f} MB ({percentage}%) - {speed_str}")
        else:
             speed_mb = speed / 1024 / 1024
             self.status_label.setText(f"已下載: {current/1024/1024:.1f} MB - {speed_mb:.2f} MB/s")

    def download_finished(self, success, message):
        """處理下載完成事件"""
        self.is_downloading = False
        self.download_button.setEnabled(True)
        self.cancel_button.setText("關閉")

        if success:
            self.progress_bar.setValue(100)
            self.status_label.setText("下載完成!")
            QMessageBox.information(self, "下載成功", message)
            self.accept()
        else:
            if "取消" in message or "cancel" in message.lower():
                self.progress_bar.setValue(0)
                self.status_label.setText("下載已取消")
            else:
                QMessageBox.critical(self, "下載錯誤", message)
                self.status_label.setText(f"錯誤: {message}")
                self.progress_bar.setValue(0)

    def on_download_retry(self, message):
        """處理備用載點嘗試通知"""
        self.status_label.setText(message)
        self.progress_bar.setValue(0)
        original_style = self.progress_bar.styleSheet()
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #f0e68c; }") 
        QTimer.singleShot(1500, lambda: self.progress_bar.setStyleSheet(original_style))

    # --- 滑鼠事件 ---
    def _handle_mouse_press(self, event):
        """處理中間預覽圖的滑鼠按下事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.swipe_start_pos = event.position()
            self.preview_container_widget.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            event.accept()
        else:
            event.ignore()

    def _handle_mouse_move(self, event):
        """處理中間預覽圖的滑鼠移動事件 (目前僅接受事件)"""
        if self.swipe_start_pos is not None:
            event.accept()
        else:
            event.ignore()

    def _handle_mouse_release(self, event):
        """處理中間預覽圖的滑鼠釋放事件，判斷是否為滑動"""
        if event.button() == Qt.MouseButton.LeftButton and self.swipe_start_pos is not None:
            current_pos = event.position() 
            delta = current_pos - self.swipe_start_pos
            distance_x = delta.x()
            distance_y = delta.y()
            min_swipe_distance = 30 
            max_vertical_distance = 50 
            self.preview_container_widget.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
            if abs(distance_x) > min_swipe_distance and abs(distance_y) < max_vertical_distance:
                if distance_x < 0: 
                    self.show_next_model()
                else: 
                    self.show_prev_model()
                event.accept()
            else:
                event.ignore() 
            self.swipe_start_pos = None 
        else:
            event.ignore()