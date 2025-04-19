import os
import numpy as np
import logging
from PIL import Image
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                           QFileDialog, QMessageBox, QLabel, QGridLayout,
                           QGroupBox, QFrame, QTabWidget, QSplitter, QScrollArea)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QImage

from src.ui.views import MultiViewWidget
from src.processing.NS_ImageEvaluator import ImageEvaluator


class MetricDisplay(QFrame):
    """顯示圖像評估指標的元件"""
    def __init__(self, title="評估指標", parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        layout = QGridLayout(self)
        layout.setContentsMargins(8, 8, 8, 8) 
        layout.setSpacing(4) 
        self.title_label = QLabel(f"<b>{title}</b>")
        layout.addWidget(self.title_label, 0, 0, 1, 2)
        self.metrics = {}
        metrics_list = [
            ("psnr", "PSNR:", "峰值信噪比 (dB)"), 
            ("ssim", "SSIM:", "結構相似性"), 
            ("mse", "MSE:", "均方誤差"),
            ("niqe_img1", "圖A NIQE分:", "無參考圖像品質分數 (越低越好)"),
            ("niqe_img2", "圖B NIQE分:", "無參考圖像品質分數 (越低越好)"),
            ("better_image", "較佳圖像:", "基於NIQE評分"),
            ("quality_diff", "品質差距:", "NIQE分數差距")
        ]
        for i, (key, text, tooltip) in enumerate(metrics_list):
            label = QLabel(text)
            value = QLabel("--")
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            value.setToolTip(tooltip)
            layout.addWidget(label, i+1, 0)
            layout.addWidget(value, i+1, 1)
            self.metrics[key] = value
    
    def update_metrics(self, results):
        """更新指標值"""
        if "error" in results:
            for key, label in self.metrics.items():
                label.setText("--")
            return
        self.metrics["psnr"].setText(f"{results['psnr']:.2f} dB")
        self.metrics["ssim"].setText(f"{results['ssim']:.4f}")
        self.metrics["mse"].setText(f"{results['mse']:.6f}")
        if 'niqe_img1' not in results or np.isnan(results['niqe_img1']):
            self.metrics["niqe_img1"].setText("計算失敗")
        else:
            self.metrics["niqe_img1"].setText(f"{results['niqe_img1']:.2f}")
            
        if 'niqe_img2' not in results or np.isnan(results['niqe_img2']):
            self.metrics["niqe_img2"].setText("計算失敗")
        else:
            self.metrics["niqe_img2"].setText(f"{results['niqe_img2']:.2f}")
        better_image = results.get("better_image", "無法判斷")
        if better_image == "無法判斷":
            self.metrics["better_image"].setText(better_image)
            self.metrics["better_image"].setStyleSheet("")
        else:
            self.metrics["better_image"].setText(f"圖{better_image}")
            if better_image == "A":
                self.metrics["better_image"].setStyleSheet("color: green; font-weight: bold;")
            else:
                self.metrics["better_image"].setStyleSheet("color: blue; font-weight: bold;")
        quality_diff = results.get("quality_diff", float('nan'))
        if np.isnan(quality_diff):
            self.metrics["quality_diff"].setText("無法計算")
        else:
            self.metrics["quality_diff"].setText(f"{quality_diff:.2f}")
    
    def clear(self):
        """清空所有指標數值"""
        for key, label in self.metrics.items():
            label.setText("--")
            label.setStyleSheet("")


class ResultImageView(QFrame):
    """顯示結果圖像（如差異圖、直方圖）的元件"""
    def __init__(self, title="結果圖像", parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(4, 4, 4, 4)
        self.title_label = QLabel(f"<b>{title}</b>")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.title_label)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        image_container = QWidget()
        image_layout = QVBoxLayout(image_container)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(QSize(300, 250)) 
        image_layout.addWidget(self.image_label)
        scroll_area.setWidget(image_container)
        self.layout.addWidget(scroll_area)
    
    def set_image(self, image, title=None):
        """設置要顯示的圖像
        Args:
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
            scaled_pixmap = pixmap.scaled(QSize(600, 450),
                                        Qt.AspectRatioMode.KeepAspectRatio, 
                                        Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
        except Exception as e:
            logging.error(f"設置圖像時出錯: {e}")
            self.image_label.setText("圖像載入失敗")
        
    def clear(self):
        """清空圖像顯示"""
        self.image_label.clear()
        self.title_label.setText(f"<b>結果圖像</b>")

class AssessmentTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.evaluator = ImageEvaluator()
        self.setup_ui()
    
    def setup_ui(self):
        main_scroll = QScrollArea(self)
        main_scroll.setWidgetResizable(True)
        main_scroll.setFrameShape(QFrame.Shape.NoFrame)
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(6, 6, 6, 6) 
        main_layout.setSpacing(6)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(main_scroll)
        self.multi_view = MultiViewWidget(self)
        self.multi_view.image_a_name = "圖像A (基準圖)"
        self.multi_view.image_b_name = "圖像B (比較圖)"
        self.multi_view.image_a_group.setTitle("圖像A (基準圖)")
        self.multi_view.image_b_group.setTitle("圖像B (比較圖)")
        main_layout.addWidget(self.multi_view)
        results_layout = QHBoxLayout()
        left_panel = QVBoxLayout()
        left_panel.setSpacing(4)
        buttons_group = QGroupBox("操作")
        buttons_layout = QHBoxLayout(buttons_group)
        buttons_layout.setContentsMargins(6, 6, 6, 6)
        self.load_image_a_btn = QPushButton("載入圖A (基準圖)")
        self.load_image_a_btn.clicked.connect(self.load_image_a)
        buttons_layout.addWidget(self.load_image_a_btn)
        self.load_image_b_btn = QPushButton("載入圖B (比較圖)")
        self.load_image_b_btn.clicked.connect(self.load_image_b)
        buttons_layout.addWidget(self.load_image_b_btn)
        self.evaluate_btn = QPushButton("執行圖像評估")
        self.evaluate_btn.clicked.connect(self.run_assessment)
        buttons_layout.addWidget(self.evaluate_btn)
        self.swap_btn = QPushButton("交換A與B")
        self.swap_btn.clicked.connect(self.swap_images)
        buttons_layout.addWidget(self.swap_btn)
        left_panel.addWidget(buttons_group)
        self.metric_display = MetricDisplay("圖像評估指標")
        left_panel.addWidget(self.metric_display)
        results_layout.addLayout(left_panel, 1)
        self.result_tabs = QTabWidget()
        self.diff_view = ResultImageView("差異熱力圖")
        self.result_tabs.addTab(self.diff_view, "差異圖")
        self.hist_tab = QWidget()
        hist_layout = QVBoxLayout(self.hist_tab)
        hist_layout.setContentsMargins(4, 4, 4, 4)
        hist_splitter = QSplitter(Qt.Orientation.Vertical)
        self.hist_view_a = ResultImageView("圖A直方圖")
        self.hist_view_b = ResultImageView("圖B直方圖")
        hist_splitter.addWidget(self.hist_view_a)
        hist_splitter.addWidget(self.hist_view_b)
        hist_layout.addWidget(hist_splitter)
        self.result_tabs.addTab(self.hist_tab, "直方圖")
        results_layout.addWidget(self.result_tabs, 2)
        main_layout.addLayout(results_layout)
        self.status_label = QLabel("請載入圖A和圖B進行比較和評估")
        main_layout.addWidget(self.status_label)
        main_scroll.setWidget(main_widget)
    
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
                self.status_label.setText(f"已載入圖A: {os.path.basename(file_path)}")
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
                self.status_label.setText(f"已載入圖B: {os.path.basename(file_path)}")
                if self.multi_view.image_a is not None:
                    self.run_assessment()
            except Exception as e:
                QMessageBox.warning(self, "錯誤", f"無法載入圖片: {str(e)}")
    
    def run_assessment(self):
        """執行圖像評估"""
        if self.multi_view.image_a is None or self.multi_view.image_b is None:
            self.status_label.setText("需要同時載入圖A和圖B才能進行評估")
            self.metric_display.clear()
            self.diff_view.clear()
            self.hist_view_a.clear()
            self.hist_view_b.clear()
            return
        try:
            self.status_label.setText("正在執行圖像評估...")
            results = self.evaluator.evaluate_images(self.multi_view.image_a, self.multi_view.image_b)
            self.metric_display.update_metrics(results)
            if "difference_map" in results:
                self.diff_view.set_image(results["difference_map"], "圖像差異熱力圖")
            if "histogram_img1" in results:
                self.hist_view_a.set_image(results["histogram_img1"], "圖A - RGB色彩分佈")
            if "histogram_img2" in results:
                self.hist_view_b.set_image(results["histogram_img2"], "圖B - RGB色彩分佈")
            better_image = results.get("better_image", "無法判斷")
            if better_image == "A":
                status = f"評估完成: 圖A品質較佳 (PSNR: {results['psnr']:.2f}dB, SSIM: {results['ssim']:.4f})"
            elif better_image == "B":
                status = f"評估完成: 圖B品質較佳 (PSNR: {results['psnr']:.2f}dB, SSIM: {results['ssim']:.4f})"
            else:
                status = f"評估完成: 無法判斷品質優劣 (PSNR: {results['psnr']:.2f}dB, SSIM: {results['ssim']:.4f})"
            self.status_label.setText(status)
        except Exception as e:
            logging.error(f"評估過程中發生錯誤: {e}")
            self.status_label.setText(f"評估過程中發生錯誤: {str(e)}")
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
        self.run_assessment()