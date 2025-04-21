import os
import numpy as np
import logging
from PIL import Image
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QFileDialog, QMessageBox, QLabel, QGridLayout,
    QGroupBox, QFrame, QTabWidget, QSplitter, QScrollArea,
    QToolButton, QSizePolicy
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QImage, QIcon

from src.ui.views import MultiViewWidget
from src.processing.NS_ImageEvaluator import ImageEvaluator


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


class MetricDisplay(QFrame):
    """顯示圖像評估指標的元件"""
    def __init__(self, title="評估指標", parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        layout = QGridLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        self.title_label = QLabel(f"<b>{title}</b>")
        self.title_label.setStyleSheet("font-size: 14px;")
        layout.addWidget(self.title_label, 0, 0, 1, 2)
        self.metrics = {}
        metrics_list = [
            ("psnr", "PSNR:", "峰值信噪比 (dB)"),
            ("ssim", "SSIM:", "結構相似性"),
            ("mse", "MSE:", "均方誤差"),
        ]
        for i, (key, text, tooltip) in enumerate(metrics_list):
            label = QLabel(text)
            value = QLabel("--")
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            value.setToolTip(tooltip)
            value.setStyleSheet("font-weight: bold; padding: 2px;")
            layout.setRowMinimumHeight(i+1, 30)
            layout.addWidget(label, i + 1, 0)
            layout.addWidget(value, i + 1, 1)
            self.metrics[key] = value

    def update_metrics(self, results):
        """更新指標值並使用顏色提示"""
        if "error" in results:
            for key, label in self.metrics.items():
                label.setText("--")
                label.setStyleSheet("font-weight: bold; padding: 2px;")
            return
        psnr_value = results['psnr']
        self.metrics["psnr"].setText(f"{psnr_value:.2f} dB")
        # 根據PSNR值設置顏色 (>30dB好, 20-30dB中等, <20dB差)
        if psnr_value > 30:
            self.metrics["psnr"].setStyleSheet("color: green; font-weight: bold; padding: 2px;")
        elif psnr_value > 20:
            self.metrics["psnr"].setStyleSheet("color: orange; font-weight: bold; padding: 2px;")
        else:
            self.metrics["psnr"].setStyleSheet("color: red; font-weight: bold; padding: 2px;")
        ssim_value = results['ssim']
        self.metrics["ssim"].setText(f"{ssim_value:.4f}")
        # 根據SSIM值設置顏色 (>0.90好, 0.80-0.90中等, <0.80差)
        if ssim_value > 0.90:
            self.metrics["ssim"].setStyleSheet("color: green; font-weight: bold; padding: 2px;")
        elif ssim_value > 0.80:
            self.metrics["ssim"].setStyleSheet("color: orange; font-weight: bold; padding: 2px;")
        else:
            self.metrics["ssim"].setStyleSheet("color: red; font-weight: bold; padding: 2px;")
        mse_value = results['mse']
        self.metrics["mse"].setText(f"{mse_value:.6f}")
        # 根據MSE值設置顏色 (<0.01好, 0.01-0.05中等, >0.05差)
        if mse_value < 0.01:
            self.metrics["mse"].setStyleSheet("color: green; font-weight: bold; padding: 2px;")
        elif mse_value < 0.05:
            self.metrics["mse"].setStyleSheet("color: orange; font-weight: bold; padding: 2px;")
        else:
            self.metrics["mse"].setStyleSheet("color: red; font-weight: bold; padding: 2px;")

    def clear(self):
        """清空所有指標數值"""
        for key, label in self.metrics.items():
            label.setText("--")
            label.setStyleSheet("font-weight: bold; padding: 2px;")


class ResultImageView(QFrame):
    """顯示結果圖像（如差異圖、直方圖）的元件"""
    def __init__(self, title="結果圖像", parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(8, 8, 8, 8)
        title_layout = QHBoxLayout()
        self.title_label = QLabel(f"<b>{title}</b>")
        self.title_label.setStyleSheet("font-size: 13px;")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        title_layout.addWidget(self.title_label)
        title_layout.addStretch()
        self.layout.addLayout(title_layout)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        image_container = QWidget()
        image_layout = QVBoxLayout(image_container)
        image_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(QSize(320, 240))
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        image_layout.addWidget(self.image_label)
        scroll_area.setWidget(image_container)
        self.layout.addWidget(scroll_area)

    def set_image(self, image, title=None):
        """設置要顯示的圖像
        參數:
            image: PIL圖像對象
            title: 可選的標題
        """
        if title:
            self.title_label.setText(f"<b>{title}</b>")
        if image is None:
            self.image_label.clear()
            return
        try:
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                height, width, channels = img_array.shape
            else:
                height, width = img_array.shape
                channels = 1
            bytes_per_line = channels * width
            if channels == 4:
                qimg = QImage(img_array.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888)
            elif channels == 3:
                qimg = QImage(img_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            else:
                qimg = QImage(img_array.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimg)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            self.original_pixmap = pixmap
        except Exception as e:
            logging.error(f"設置圖像時出錯: {e}")
            self.image_label.setText("圖像載入失敗")

    def clear(self):
        """清空圖像顯示"""
        self.image_label.clear()
        self.title_label.setText(f"<b>結果圖像</b>")
        self.original_pixmap = None

    def resizeEvent(self, event):
        """重寫調整大小事件，使圖像能夠適當縮放"""
        super().resizeEvent(event)
        if hasattr(self, 'original_pixmap') and self.original_pixmap:
            self.image_label.setPixmap(self.original_pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

class AssessmentTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.evaluator = ImageEvaluator()
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        upper_widget = QWidget()
        upper_layout = QVBoxLayout(upper_widget)
        upper_layout.setContentsMargins(0, 0, 0, 0)
        self.multi_view = MultiViewWidget(self)
        self.multi_view.image_a_name = "圖像A (基準圖)"
        self.multi_view.image_b_name = "圖像B (比較圖)"
        self.multi_view.image_a_group.setTitle("圖像A (基準圖)")
        self.multi_view.image_b_group.setTitle("圖像B (比較圖)")
        self.multi_view.setMinimumHeight(300) 
        upper_layout.addWidget(self.multi_view)
        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(5, 0, 5, 5)
        image_buttons_layout = QHBoxLayout()
        image_buttons_layout.setSpacing(10)
        self.load_image_a_btn = QPushButton("載入圖A")
        self.load_image_a_btn.setToolTip("載入基準圖像")
        self.load_image_a_btn.setMinimumWidth(100)
        self.load_image_a_btn.clicked.connect(self.load_image_a)
        image_buttons_layout.addWidget(self.load_image_a_btn)
        self.load_image_b_btn = QPushButton("載入圖B")
        self.load_image_b_btn.setToolTip("載入比較圖像")
        self.load_image_b_btn.setMinimumWidth(100)
        self.load_image_b_btn.clicked.connect(self.load_image_b)
        image_buttons_layout.addWidget(self.load_image_b_btn)
        buttons_layout.addLayout(image_buttons_layout)
        buttons_layout.addStretch(1)
        action_buttons_layout = QHBoxLayout()
        action_buttons_layout.setSpacing(10)
        self.swap_btn = QPushButton("交換A與B")
        self.swap_btn.setToolTip("交換圖像A和圖像B的位置")
        self.swap_btn.setMinimumWidth(100)
        self.swap_btn.clicked.connect(self.swap_images)
        action_buttons_layout.addWidget(self.swap_btn)
        self.evaluate_btn = QPushButton("執行評估")
        self.evaluate_btn.setToolTip("計算和顯示評估結果")
        self.evaluate_btn.setMinimumWidth(100)
        self.evaluate_btn.clicked.connect(self.run_assessment)
        self.evaluate_btn.setStyleSheet("QPushButton { background-color: #4a86e8; color: white; font-weight: bold; }")
        action_buttons_layout.addWidget(self.evaluate_btn)
        buttons_layout.addLayout(action_buttons_layout)
        upper_layout.addLayout(buttons_layout)
        main_splitter.addWidget(upper_widget)
        lower_widget = QWidget()
        lower_layout = QVBoxLayout(lower_widget)
        lower_layout.setContentsMargins(0, 0, 0, 0)
        results_splitter = QSplitter(Qt.Orientation.Horizontal)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        metrics_box = CollapsibleBox("評估指標面板")
        metrics_layout = QVBoxLayout()
        self.metric_display = MetricDisplay("圖像質量指標")
        metrics_layout.addWidget(self.metric_display)
        metrics_info = QLabel("指標說明：\n• PSNR: 值越高表示圖像質量越好，通常 >30dB 為優良\n"
                            "• SSIM: 衡量圖像結構相似度，越接近1越好\n"
                            "• MSE: 均方誤差，值越小表示差異越小")
        metrics_info.setWordWrap(True)
        metrics_info.setStyleSheet("color: #666; font-size: 11px; padding: 5px;")
        metrics_layout.addWidget(metrics_info)
        metrics_layout.addStretch(1)
        metrics_box.setContentLayout(metrics_layout)
        left_layout.addWidget(metrics_box)
        results_splitter.addWidget(left_panel)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.result_tabs = QTabWidget()
        self.result_tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.result_tabs.setDocumentMode(True) 
        self.diff_view = ResultImageView()
        self.result_tabs.addTab(self.diff_view, "差異熱力圖")
        hist_tab = QWidget()
        hist_layout = QVBoxLayout(hist_tab)
        hist_layout.setContentsMargins(4, 4, 4, 4)
        hist_splitter = QSplitter(Qt.Orientation.Vertical)
        self.hist_view_a = ResultImageView("圖A - RGB色彩分佈")
        self.hist_view_b = ResultImageView("圖B - RGB色彩分佈")
        hist_splitter.addWidget(self.hist_view_a)
        hist_splitter.addWidget(self.hist_view_b)
        hist_splitter.setSizes([int(hist_tab.height()/2), int(hist_tab.height()/2)])
        hist_layout.addWidget(hist_splitter)
        self.result_tabs.addTab(hist_tab, "色彩直方圖")
        right_layout.addWidget(self.result_tabs)
        results_splitter.addWidget(right_panel)
        results_splitter.setSizes([int(lower_widget.width()/3), int(lower_widget.width()*2/3)])
        lower_layout.addWidget(results_splitter)
        main_splitter.addWidget(lower_widget)
        main_splitter.setSizes([int(self.height()*3/5), int(self.height()*2/5)])
        main_layout.addWidget(main_splitter)
        self.status_label = QLabel("請載入圖A和圖B進行比較和評估")
        self.status_label.setStyleSheet("background-color: #f5f5f5; padding: 5px; border-radius: 3px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)

    def load_image_a(self):
        """載入基準圖A"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "開啟圖A", "", "圖片文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if file_path:
            try:
                image_a = Image.open(file_path).convert("RGB")
                current_image_b = self.multi_view.image_b
                self.multi_view.set_images(
                    image_a,
                    current_image_b,
                    f"圖A: {os.path.basename(file_path)}",
                    self.multi_view.image_b_name if current_image_b else None
                )
                self.multi_view.image_a_group.setTitle(f"圖像A: {os.path.basename(file_path)}")
                self.status_label.setText(f"已載入圖A: {os.path.basename(file_path)} ({image_a.width}x{image_a.height})")
                if self.multi_view.image_b is not None:
                    self.run_assessment()
            except Exception as e:
                QMessageBox.warning(self, "錯誤", f"無法載入圖片: {str(e)}")

    def load_image_b(self):
        """載入比較圖B"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "開啟圖B", "", "圖片文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if file_path:
            try:
                image_b = Image.open(file_path).convert("RGB")
                current_image_a = self.multi_view.image_a
                self.multi_view.set_images(
                    current_image_a,
                    image_b,
                    self.multi_view.image_a_name if current_image_a else None,
                    f"圖B: {os.path.basename(file_path)}"
                )
                self.multi_view.image_b_group.setTitle(f"圖像B: {os.path.basename(file_path)}")
                self.status_label.setText(f"已載入圖B: {os.path.basename(file_path)} ({image_b.width}x{image_b.height})")
                if self.multi_view.image_a is not None:
                    self.run_assessment()
            except Exception as e:
                QMessageBox.warning(self, "錯誤", f"無法載入圖片: {str(e)}")

    def run_assessment(self):
        """執行圖像評估"""
        if self.multi_view.image_a is None or self.multi_view.image_b is None:
            self.status_label.setText("需要同時載入圖A和圖B才能進行評估")
            self.status_label.setStyleSheet("background-color: #fff3cd; padding: 5px; border-radius: 3px; color: #856404;")
            self.metric_display.clear()
            self.diff_view.clear()
            self.hist_view_a.clear()
            self.hist_view_b.clear()
            return
        try:
            self.status_label.setText("正在執行圖像評估...")
            self.status_label.setStyleSheet("background-color: #cce5ff; padding: 5px; border-radius: 3px; color: #004085;")
            results = self.evaluator.evaluate_images(self.multi_view.image_a, self.multi_view.image_b)
            self.metric_display.update_metrics(results)
            if "difference_map" in results:
                self.diff_view.set_image(results["difference_map"], "圖像差異熱力圖")
            if "histogram_img1" in results:
                self.hist_view_a.set_image(results["histogram_img1"], "圖A - RGB色彩分佈")
            if "histogram_img2" in results:
                self.hist_view_b.set_image(results["histogram_img2"], "圖B - RGB色彩分佈")
            self.status_label.setText(
                f"評估完成 (PSNR: {results['psnr']:.2f}dB, SSIM: {results['ssim']:.4f}, MSE: {results['mse']:.6f})"
            )
            self.status_label.setStyleSheet("background-color: #d4edda; padding: 5px; border-radius: 3px; color: #155724;")
        except Exception as e:
            logging.error(f"評估過程中發生錯誤: {e}")
            self.status_label.setText(f"評估過程中發生錯誤: {str(e)}")
            self.status_label.setStyleSheet("background-color: #f8d7da; padding: 5px; border-radius: 3px; color: #721c24;")
            self.metric_display.clear()
            self.diff_view.clear()
            self.hist_view_a.clear()
            self.hist_view_b.clear()

    def swap_images(self):
        """交換圖A和圖B"""
        if self.multi_view.image_a is None or self.multi_view.image_b is None:
            return
        image_a = self.multi_view.image_a
        image_b = self.multi_view.image_b
        name_a = self.multi_view.image_a_name
        name_b = self.multi_view.image_b_name
        title_a = self.multi_view.image_a_group.title()
        title_b = self.multi_view.image_b_group.title()
        self.multi_view.set_images(
            image_b,
            image_a,
            name_b,
            name_a
        )
        self.multi_view.image_a_group.setTitle(title_b)
        self.multi_view.image_b_group.setTitle(title_a)
        self.status_label.setText("已交換圖A和圖B的位置")
        self.status_label.setStyleSheet("background-color: #e2e3e5; padding: 5px; border-radius: 3px; color: #383d41;")
        self.run_assessment()