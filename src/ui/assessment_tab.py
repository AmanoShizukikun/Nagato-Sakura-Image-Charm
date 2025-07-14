import os
import numpy as np
import logging
from PIL import Image
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QMessageBox, QLabel, QGridLayout,
    QGroupBox, QFrame, QTabWidget, QSplitter, QScrollArea,
    QToolButton, QSizePolicy, QProgressBar, QGraphicsScene, QGraphicsView,
    QApplication
)
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal, pyqtSlot, QRectF, QTimer
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor

from src.ui.views import MultiViewWidget, SynchronizedGraphicsView
from src.processing.NS_ImageEvaluator import ImageEvaluator
from src.processing.NS_ImageQualityScorer import ImageQualityScorer
from src.processing.NS_ImageClassification import ImageClassifier


class ClassificationWorker(QThread):
    """用於處理圖像分類的工作線程"""
    resultReady = pyqtSignal(dict)
    error = pyqtSignal(str)
    def __init__(self, classifier=None, image=None, is_image_a=True):
        super().__init__()
        if classifier is None:
            self.classifier = ImageClassifier()
        else:
            self.classifier = classifier
        self.image = image
        self.is_image_a = is_image_a
        
    def set_image(self, image):
        """設置要分類的圖像"""
        self.image = image
        
    def run(self):
        """執行分類任務"""
        try:
            if self.image is None:
                self.error.emit("未提供圖像進行分類")
                return
            if self.classifier is None:
                logging.warning("分類器為None，創建新的分類器實例")
                self.classifier = ImageClassifier()
            if self.classifier is None:
                self.error.emit("分類器初始化失敗")
                return
            if not self.classifier.is_model_loaded():
                if not self.classifier.load_model():
                    if self.is_image_a:
                        self.resultReady.emit({
                            "model_a": "未偵測到模型",
                            "model_a_acc": None
                        })
                    else:
                        self.resultReady.emit({
                            "model_b": "未偵測到模型",
                            "model_b_acc": None
                        })
                    return
            result = self.classifier.classify_image(self.image)
            if not result["success"]:
                if "model not found" in result.get("error", "").lower():
                    if self.is_image_a:
                        self.resultReady.emit({
                            "model_a": "未偵測到模型",
                            "model_a_acc": None
                        })
                    else:
                        self.resultReady.emit({
                            "model_b": "未偵測到模型",
                            "model_b_acc": None
                        })
                else:
                    self.error.emit(f"分類失敗: {result.get('error', '未知錯誤')}")
                return
            top_class = result["top_class"]
            top_probability = result["top_probability"]
            if top_class is None:
                top_class = "NULL"
            if self.is_image_a:
                self.resultReady.emit({
                    "model_a": top_class,
                    "model_a_acc": top_probability
                })
            else:
                self.resultReady.emit({
                    "model_b": top_class,
                    "model_b_acc": top_probability
                })
        except Exception as e:
            logging.error(f"圖像分類過程出錯: {e}")
            self.error.emit(f"分類失敗: {str(e)}")
        finally:
            try:
                if hasattr(self, 'image'):
                    self.image = None
            except:
                pass

class AssessmentWorker(QThread):
    """用於處理圖像評估的工作線程"""
    progressChanged = pyqtSignal(str) 
    resultReady = pyqtSignal(dict)
    error = pyqtSignal(str) 
    aiScoreReady = pyqtSignal(dict) 

    def __init__(self, evaluator=None, image_a=None, image_b=None):
        super().__init__()
        self.evaluator = evaluator or ImageEvaluator()
        self.image_a = image_a
        self.image_b = image_b
        self.quality_scorer = None
        self.run_ai_scoring = True

    def set_images(self, image_a, image_b):
        """設置要評估的圖像"""
        self.image_a = image_a
        self.image_b = image_b

    def run(self):
        """執行評估任務"""
        try:
            if self.image_a is None or self.image_b is None:
                self.error.emit("需要同時載入圖A和圖B才能進行完整評估")
                return
            self.progressChanged.emit("正在執行圖像評估...")
            results = self.evaluator.evaluate_images(self.image_a, self.image_b, advanced=True)
            self.resultReady.emit(results)
            if self.run_ai_scoring:
                self.progressChanged.emit("正在執行AI品質評估...")
                try:
                    self.quality_scorer = self.get_quality_scorer()
                    if self.quality_scorer:
                        image_a_score = self.quality_scorer.score_image(self.image_a)
                        image_b_score = self.quality_scorer.score_image(self.image_b)
                        ai_results = {
                            'ai_score_a': image_a_score,
                            'ai_score_b': image_b_score
                        }
                        self.aiScoreReady.emit(ai_results)
                        if image_a_score is not None and image_b_score is not None:
                            logging.info(f"AI評分: A={image_a_score:.2f}, B={image_b_score:.2f}")
                        else:
                            logging.warning("AI評分: 未偵測到模型")
                    else:
                        self.error.emit("無法載入AI評分模型")
                except Exception as e:
                    logging.error(f"AI評分過程出錯: {e}")
                    self.error.emit(f"AI評分失敗: {str(e)}")
                finally:
                    if self.quality_scorer:
                        self.quality_scorer.unload_model()
                        logging.info("AI評分完成，已卸載模型")
        except Exception as e:
            logging.error(f"評估過程中發生錯誤: {e}")
            self.error.emit(f"評估失敗: {str(e)}")

    def get_quality_scorer(self):
        """獲取或創建品質評分器實例"""
        if self.quality_scorer is None:
            try:
                self.quality_scorer = ImageQualityScorer()
                logging.info("建立AI圖像品質評分器實例")
            except Exception as e:
                logging.error(f"建立AI圖像品質評分器實例失敗: {e}")
                return None
        return self.quality_scorer

class SingleImageScorerWorker(QThread):
    """用於處理單張圖像AI評分的工作線程"""
    scoreReady = pyqtSignal(object, str, str)
    error = pyqtSignal(str)

    def __init__(self, image=None, file_path="", is_image_a=True):
        super().__init__()
        self.image = image
        self.file_path = file_path
        self.is_image_a = is_image_a
        self.quality_scorer = None

    def run(self):
        try:
            self.quality_scorer = self.get_quality_scorer()
            if self.quality_scorer and self.image:
                score = self.quality_scorer.score_image(self.image)
                file_name = os.path.basename(self.file_path)
                size_info = f"{self.image.width}x{self.image.height}"
                self.scoreReady.emit(score, file_name, size_info)
                if score is not None:
                    logging.info(f"{'圖A' if self.is_image_a else '圖B'} AI評分: {score:.2f}")
                else:
                    logging.warning(f"{'圖A' if self.is_image_a else '圖B'} AI評分: 未偵測到模型")
            else:
                self.error.emit(f"無法評估圖像")
        except Exception as e:
            logging.error(f"{'圖A' if self.is_image_a else '圖B'} AI評分失敗: {e}")
            self.error.emit(f"AI評分失敗: {str(e)}")
        finally:
            if self.quality_scorer:
                self.quality_scorer.unload_model()
                logging.info("AI評分完成，已卸載模型")

    def get_quality_scorer(self):
        """獲取或創建品質評分器實例"""
        if self.quality_scorer is None:
            try:
                self.quality_scorer = ImageQualityScorer()
                logging.info("建立AI圖像品質評分器實例")
            except Exception as e:
                logging.error(f"建立AI圖像品質評分器實例失敗: {e}")
                return None
        return self.quality_scorer

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
        layout.setContentsMargins(8, 8, 8, 8) 
        layout.setSpacing(6)
        self.title_label = QLabel(f"<b>{title}</b>")
        self.title_label.setStyleSheet("font-size: 13px;")
        layout.addWidget(self.title_label, 0, 0, 1, 2)
        self.metrics = {}
        metrics_list = [
            ("psnr", "PSNR:", "峰值信噪比 (dB)"),
            ("ssim", "SSIM:", "結構相似性"),
            ("mse", "MSE:", "均方誤差")
        ]
        for i, (key, text, tooltip) in enumerate(metrics_list):
            label = QLabel(text)
            value = QLabel("--")
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            value.setToolTip(tooltip)
            value.setStyleSheet("font-weight: bold; padding: 2px;")
            layout.setRowMinimumHeight(i+1, 20) 
            layout.addWidget(label, i + 1, 0)
            layout.addWidget(value, i + 1, 1)
            self.metrics[key] = value
        ai_header = QLabel("<b>AI圖像品質評分</b>")
        ai_header.setStyleSheet("font-size: 12px; padding-top: 6px;")
        layout.addWidget(ai_header, len(metrics_list) + 1, 0, 1, 2)
        ai_metrics_list = [
            ("ai_score_a", "圖A評分:", "AI模型評估圖A的品質分數 (0-100)"),
            ("ai_score_b", "圖B評分:", "AI模型評估圖B的品質分數 (0-100)")
        ]
        for i, (key, text, tooltip) in enumerate(ai_metrics_list):
            label = QLabel(text)
            value = QLabel("--")
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            value.setToolTip(tooltip)
            value.setStyleSheet("font-weight: bold; padding: 2px;")
            row_pos = len(metrics_list) + 2 + i
            layout.setRowMinimumHeight(row_pos, 20)
            layout.addWidget(label, row_pos, 0)
            layout.addWidget(value, row_pos, 1)
            self.metrics[key] = value
        model_header = QLabel("<b>AI模型推薦</b>")
        model_header.setStyleSheet("font-size: 12px; padding-top: 6px;")
        layout.addWidget(model_header, len(metrics_list) + 4, 0, 1, 2)
        
        model_metrics_list = [
            ("model_a", "圖A推薦:", "AI推薦用於處理圖A的最佳模型"),
            ("model_a_acc", "準確率:", "推薦模型的準確率"),
            ("model_b", "圖B推薦:", "AI推薦用於處理圖B的最佳模型"),
            ("model_b_acc", "準確率:", "推薦模型的準確率")
        ]
        
        for i, (key, text, tooltip) in enumerate(model_metrics_list):
            label = QLabel(text)
            value = QLabel("--")
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            value.setToolTip(tooltip)
            value.setStyleSheet("font-weight: bold; padding: 2px;")
            row_pos = len(metrics_list) + 5 + i
            layout.setRowMinimumHeight(row_pos, 20)
            layout.addWidget(label, row_pos, 0)
            layout.addWidget(value, row_pos, 1)
            self.metrics[key] = value

    def update_metrics(self, results):
        """更新指標值並使用顏色提示"""
        if "error" in results:
            for key, label in self.metrics.items():
                label.setText("--")
                label.setStyleSheet("font-weight: bold; padding: 2px;")
            return
        
        if 'psnr' in results:
            psnr_value = results['psnr']
            self.metrics["psnr"].setText(f"{psnr_value:.2f} dB")
            # PSNR顏色 (>30dB好, 20-30dB中等, <20dB差)
            if psnr_value > 30:
                self.metrics["psnr"].setStyleSheet("color: green; font-weight: bold; padding: 2px;")
            elif psnr_value > 20:
                self.metrics["psnr"].setStyleSheet("color: orange; font-weight: bold; padding: 2px;")
            else:
                self.metrics["psnr"].setStyleSheet("color: red; font-weight: bold; padding: 2px;")
            
        if 'ssim' in results:
            ssim_value = results['ssim']
            self.metrics["ssim"].setText(f"{ssim_value:.4f}")
            # SSIM顏色 (>0.90好, 0.80-0.90中等, <0.80差)
            if ssim_value > 0.90:
                self.metrics["ssim"].setStyleSheet("color: green; font-weight: bold; padding: 2px;")
            elif ssim_value > 0.80:
                self.metrics["ssim"].setStyleSheet("color: orange; font-weight: bold; padding: 2px;")
            else:
                self.metrics["ssim"].setStyleSheet("color: red; font-weight: bold; padding: 2px;")
            
        if 'mse' in results:
            mse_value = results['mse']
            self.metrics["mse"].setText(f"{mse_value:.6f}")
            # MSE顏色 (<0.01好, 0.01-0.05中等, >0.05差)
            if mse_value < 0.01:
                self.metrics["mse"].setStyleSheet("color: green; font-weight: bold; padding: 2px;")
            elif mse_value < 0.05:
                self.metrics["mse"].setStyleSheet("color: orange; font-weight: bold; padding: 2px;")
            else:
                self.metrics["mse"].setStyleSheet("color: red; font-weight: bold; padding: 2px;")
        
        if 'ai_score_a' in results:
            ai_score = results['ai_score_a']
            if ai_score is None:
                self.metrics["ai_score_a"].setText("未偵測到模型")
                self.metrics["ai_score_a"].setStyleSheet("color: #888; font-weight: bold; padding: 2px;")
            else:
                self.metrics["ai_score_a"].setText(f"{ai_score:.2f}")
                if ai_score > 90:
                    self.metrics["ai_score_a"].setStyleSheet("color: green; font-weight: bold; padding: 2px;")
                elif ai_score > 70:
                    self.metrics["ai_score_a"].setStyleSheet("color: orange; font-weight: bold; padding: 2px;")
                else:
                    self.metrics["ai_score_a"].setStyleSheet("color: red; font-weight: bold; padding: 2px;")
        
        if 'ai_score_b' in results:
            ai_score = results['ai_score_b']
            if ai_score is None:
                self.metrics["ai_score_b"].setText("未偵測到模型")
                self.metrics["ai_score_b"].setStyleSheet("color: #888; font-weight: bold; padding: 2px;")
            else:
                self.metrics["ai_score_b"].setText(f"{ai_score:.2f}")
                if ai_score > 90:
                    self.metrics["ai_score_b"].setStyleSheet("color: green; font-weight: bold; padding: 2px;")
                elif ai_score > 70:
                    self.metrics["ai_score_b"].setStyleSheet("color: orange; font-weight: bold; padding: 2px;")
                else:
                    self.metrics["ai_score_b"].setStyleSheet("color: red; font-weight: bold; padding: 2px;")
                    
        if 'model_a' in results:
            if results['model_a'] is None:
                self.metrics["model_a"].setText("NULL")
                self.metrics["model_a"].setStyleSheet("color: red; font-weight: bold; padding: 2px;")
            elif results['model_a'] == "未偵測到模型":
                self.metrics["model_a"].setText("未偵測到模型")
                self.metrics["model_a"].setStyleSheet("color: #888; font-weight: bold; padding: 2px;")
            else:
                self.metrics["model_a"].setText(results['model_a'])
                self.metrics["model_a"].setStyleSheet("color: white; font-weight: bold; padding: 2px;")
            
        if 'model_a_acc' in results:
            acc = results['model_a_acc']
            if acc is None or results.get('model_a') == "未偵測到模型":
                self.metrics["model_a_acc"].setText("--")
                self.metrics["model_a_acc"].setStyleSheet("font-weight: bold; padding: 2px;")
            else:
                self.metrics["model_a_acc"].setText(f"{acc:.2f}%")
                if acc > 90:
                    self.metrics["model_a_acc"].setStyleSheet("color: green; font-weight: bold; padding: 2px;")
                elif acc > 70:
                    self.metrics["model_a_acc"].setStyleSheet("color: orange; font-weight: bold; padding: 2px;")
                else:
                    self.metrics["model_a_acc"].setStyleSheet("color: red; font-weight: bold; padding: 2px;")
                
        if 'model_b' in results:
            if results['model_b'] is None:
                self.metrics["model_b"].setText("NULL")
                self.metrics["model_b"].setStyleSheet("color: red; font-weight: bold; padding: 2px;")
            elif results['model_b'] == "未偵測到模型":
                self.metrics["model_b"].setText("未偵測到模型")
                self.metrics["model_b"].setStyleSheet("color: #888; font-weight: bold; padding: 2px;")
            else:
                self.metrics["model_b"].setText(results['model_b'])
                self.metrics["model_b"].setStyleSheet("color: white; font-weight: bold; padding: 2px;")
            
        if 'model_b_acc' in results:
            acc = results['model_b_acc']
            if acc is None or results.get('model_b') == "未偵測到模型":
                self.metrics["model_b_acc"].setText("--")
                self.metrics["model_b_acc"].setStyleSheet("font-weight: bold; padding: 2px;")
            else:
                self.metrics["model_b_acc"].setText(f"{acc:.2f}%")
                if acc > 90:
                    self.metrics["model_b_acc"].setStyleSheet("color: green; font-weight: bold; padding: 2px;")
                elif acc > 70:
                    self.metrics["model_b_acc"].setStyleSheet("color: orange; font-weight: bold; padding: 2px;")
                else:
                    self.metrics["model_b_acc"].setStyleSheet("color: red; font-weight: bold; padding: 2px;")
                    
    def clear(self):
        """清空所有指標數值"""
        for key, label in self.metrics.items():
            label.setText("--")
            label.setStyleSheet("font-weight: bold; padding: 2px;")

class ResultImageView(QFrame):
    """顯示結果圖像（如差異圖、直方圖）的元件，支援縮放和平移"""
    def __init__(self, title="結果圖像", parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(4, 4, 4, 4)
        self.layout.setSpacing(2)
        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(4)
        self.title_label = QLabel(f"<b>{title}</b>")
        self.title_label.setStyleSheet("font-size: 12px;")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        title_layout.addWidget(self.title_label)
        self.reset_view_btn = QPushButton("重置縮放")
        self.reset_view_btn.setToolTip("重置圖像縮放比例")
        self.reset_view_btn.clicked.connect(self.reset_view)
        self.reset_view_btn.setMaximumWidth(70)
        self.reset_view_btn.setMaximumHeight(24) 
        self.reset_view_btn.setStyleSheet("font-size: 11px; padding: 2px;") 
        title_layout.addWidget(self.reset_view_btn)
        title_layout.addStretch()
        self.layout.addLayout(title_layout)
        self.scene = QGraphicsScene()
        self.view = SynchronizedGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setBackgroundBrush(Qt.GlobalColor.lightGray)
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.view.setMinimumHeight(60)
        self.setMinimumHeight(90)
        self.layout.addWidget(self.view)
        self.pixmap_item = None
        self.original_pixmap = None

    def set_image(self, image, title=None):
        """設置要顯示的圖像
        參數:
            image: PIL圖像對象
            title: 可選的標題
        """
        if title:
            self.title_label.setText(f"<b>{title}</b>")
        if image is None:
            self.scene.clear()
            self.pixmap_item = None
            self.original_pixmap = None
            return  
        try:
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                height, width, channels = img_array.shape
            else:
                height, width = img_array.shape
                channels = 1
            max_display_size = 800 
            scale_factor = 1.0
            if width > max_display_size or height > max_display_size:
                scale_factor = min(max_display_size / width, max_display_size / height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                from PIL import Image
                pil_img = Image.fromarray(img_array)
                pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                img_array = np.array(pil_img)
                width, height = new_width, new_height
            bytes_per_line = channels * width
            if channels == 4:
                qimg = QImage(img_array.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888)
            elif channels == 3:
                qimg = QImage(img_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            else:
                qimg = QImage(img_array.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimg)
            self.original_pixmap = pixmap
            self.scene.clear()
            self.pixmap_item = self.scene.addPixmap(pixmap)
            self.scene.setSceneRect(QRectF(pixmap.rect()))
            self.view.setHasContent(True)
            QTimer.singleShot(100, self.reset_view)
        except Exception as e:
            logging.error(f"設置圖像時出錯: {e}")
            self.scene.clear()
            text_item = self.scene.addText("圖像載入失敗")
            text_item.setDefaultTextColor(QColor(255, 0, 0))
            self.pixmap_item = None
            self.original_pixmap = None
            self.view.setHasContent(False)

    def reset_view(self):
        """重置視圖縮放和位置"""
        if self.scene and len(self.scene.items()) > 0: 
            self.view.resetTransform()
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.view.ensureVisible(self.scene.sceneRect())

    def clear(self):
        """清空圖像顯示"""
        self.scene.clear()
        self.title_label.setText(f"<b>結果圖像</b>")
        self.pixmap_item = None
        self.original_pixmap = None
        self.view.setHasContent(False)

    def resizeEvent(self, event):
        """重寫調整大小事件，使圖像能夠適當縮放"""
        super().resizeEvent(event)
        if self.scene and len(self.scene.items()) > 0:
            rect = self.scene.sceneRect()
            if not rect.isEmpty():
                self.view.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

class AssessmentTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.evaluator = ImageEvaluator()
        self.image_classifier = ImageClassifier()
        self.assessment_worker = AssessmentWorker(self.evaluator)
        self.classifier_worker_a = ClassificationWorker(self.image_classifier, is_image_a=True) 
        self.classifier_worker_b = ClassificationWorker(self.image_classifier, is_image_a=False)
        self.scorer_worker_a = None
        self.scorer_worker_b = None
        self.load_img_a_path = ""
        self.load_img_b_path = ""
        self.setup_ui()
        self.assessment_worker.progressChanged.connect(self.update_progress)
        self.assessment_worker.resultReady.connect(self.on_assessment_results)
        self.assessment_worker.error.connect(self.on_assessment_error)
        self.assessment_worker.aiScoreReady.connect(self.on_ai_scores)
        self.assessment_worker.finished.connect(self.on_assessment_finished)
        self.classifier_worker_a.resultReady.connect(self.on_classification_results)
        self.classifier_worker_a.error.connect(self.on_classification_error)
        self.classifier_worker_b.resultReady.connect(self.on_classification_results)
        self.classifier_worker_b.error.connect(self.on_classification_error)
        
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(6)
        compare_widget = self.create_single_compare_tab()
        main_layout.addWidget(compare_widget)

    def create_single_compare_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        upper_widget = QWidget()
        upper_layout = QVBoxLayout(upper_widget)
        upper_layout.setContentsMargins(0, 0, 0, 0)
        self.multi_view = MultiViewWidget(self)
        self.multi_view.image_a_name = "圖像A (基準圖)"
        self.multi_view.image_b_name = "圖像B (比較圖)"
        self.multi_view.image_a_group.setTitle("圖像A (基準圖)")
        self.multi_view.image_b_group.setTitle("圖像B (比較圖)")
        self.multi_view.setMinimumHeight(475)
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
        lower_layout.setSpacing(2)
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMaximumHeight(5)
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        lower_layout.addLayout(progress_layout)
        results_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左側：指標
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        metrics_box = CollapsibleBox("評估指標面板")
        metrics_layout = QVBoxLayout()
        self.metric_display = MetricDisplay("圖像評估指標")
        metrics_layout.addWidget(self.metric_display)
        metrics_info = QLabel(
            "指標說明：\n• PSNR: 值越高表示圖像質量越好，通常 >30dB 為優良\n"
            "• SSIM: 衡量圖像結構相似度，越接近1越好\n"
            "• MSE: 均方誤差，值越小表示差異越小\n"
            "• AI評分: AI模型評圖像的品質，分數越高越好"
        )
        metrics_info.setWordWrap(True)
        metrics_info.setStyleSheet("color: #666; font-size: 10px; padding: 4px;")
        metrics_layout.addWidget(metrics_info)
        metrics_layout.addStretch(1)
        metrics_box.setContentLayout(metrics_layout)
        left_layout.addWidget(metrics_box)
        results_splitter.addWidget(left_panel)
        
        # 右側：圖表
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.result_tabs = QTabWidget()
        self.result_tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.result_tabs.setDocumentMode(True)
        self.result_tabs.setMovable(False)
        self.result_tabs.currentChanged.connect(self.on_tab_changed)

        # 差異熱力圖
        self.diff_view = ResultImageView("差異熱力圖")
        self.result_tabs.addTab(self.diff_view, "差異熱力圖")
        
        # 色彩直方圖
        hist_tab = QWidget()
        hist_layout = QVBoxLayout(hist_tab)
        hist_layout.setContentsMargins(2, 2, 2, 2)
        hist_splitter = QSplitter(Qt.Orientation.Vertical)
        hist_splitter.setHandleWidth(4)
        self.hist_view_a = ResultImageView("圖A色彩") 
        self.hist_view_b = ResultImageView("圖B色彩")
        hist_splitter.addWidget(self.hist_view_a)
        hist_splitter.addWidget(self.hist_view_b)
        hist_splitter.setSizes([int(hist_tab.height()/2), int(hist_tab.height()/2)])
        hist_layout.addWidget(hist_splitter)
        self.result_tabs.addTab(hist_tab, "色彩直方圖")
        self.edge_view = ResultImageView("邊緣比較")
        self.result_tabs.addTab(self.edge_view, "邊緣比較")
        right_layout.addWidget(self.result_tabs)
        results_splitter.addWidget(right_panel)
        results_splitter.setSizes([100, 400])
        lower_layout.addWidget(results_splitter)
        main_splitter.addWidget(lower_widget)
        main_splitter.setSizes([int(self.height()*8/10), int(self.height()*2/10)])
        layout.addWidget(main_splitter)
        return tab

    def load_image_a(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "開啟圖A", "", "圖片文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if file_path:
            try:
                image_a = Image.open(file_path).convert("RGB")
                self.load_img_a_path = file_path
                current_image_b = self.multi_view.image_b
                self.multi_view.set_images(
                    image_a,
                    current_image_b,
                    f"圖A: {os.path.basename(file_path)}",
                    self.multi_view.image_b_name if current_image_b else None
                )
                self.multi_view.image_a_group.setTitle(f"圖像A: {os.path.basename(file_path)}")
                self.run_single_image_scoring(image_a, file_path, True)
                self.run_single_image_classification(image_a, True)
                
                if self.multi_view.image_b is not None:
                    self.run_assessment(keep_ai_scores=True)  
            except Exception as e:
                logging.error(f"載入圖A失敗: {e}")
                QMessageBox.warning(self, "錯誤", f"無法載入圖片: {str(e)}")

    def load_image_b(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "開啟圖B", "", "圖片文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if file_path:
            try:
                image_b = Image.open(file_path).convert("RGB")
                self.load_img_b_path = file_path
                current_image_a = self.multi_view.image_a
                self.multi_view.set_images(
                    current_image_a,
                    image_b,
                    self.multi_view.image_a_name if current_image_a else None,
                    f"圖B: {os.path.basename(file_path)}"
                )
                self.multi_view.image_b_group.setTitle(f"圖像B: {os.path.basename(file_path)}")
                self.run_single_image_scoring(image_b, file_path, False)
                self.run_single_image_classification(image_b, False)
                
                if self.multi_view.image_a is not None:
                    self.run_assessment(keep_ai_scores=True)   
            except Exception as e:
                logging.error(f"載入圖B失敗: {e}")
                QMessageBox.warning(self, "錯誤", f"無法載入圖片: {str(e)}")

    def run_single_image_scoring(self, image, file_path, is_image_a):
        """在單獨線程中運行單張圖像的AI評分"""
        if is_image_a:
            if self.scorer_worker_a and self.scorer_worker_a.isRunning():
                self.scorer_worker_a.terminate()
                self.scorer_worker_a.wait()
            self.scorer_worker_a = SingleImageScorerWorker(image, file_path, True)
            self.scorer_worker_a.scoreReady.connect(self.on_image_a_score_ready)
            self.scorer_worker_a.error.connect(lambda error: self.on_single_image_error(error, True))
            self.scorer_worker_a.start()
        else:
            if self.scorer_worker_b and self.scorer_worker_b.isRunning():
                self.scorer_worker_b.terminate()
                self.scorer_worker_b.wait()
            self.scorer_worker_b = SingleImageScorerWorker(image, file_path, False)
            self.scorer_worker_b.scoreReady.connect(self.on_image_b_score_ready)
            self.scorer_worker_b.error.connect(lambda error: self.on_single_image_error(error, False))
            self.scorer_worker_b.start()

    def run_single_image_classification(self, image, is_image_a):
        """在單獨線程中運行單張圖像的分類"""
        if is_image_a:
            if self.classifier_worker_a.isRunning():
                self.classifier_worker_a.terminate()
                self.classifier_worker_a.wait()
            self.classifier_worker_a.set_image(image)
            self.classifier_worker_a.start()
        else:
            if self.classifier_worker_b.isRunning():
                self.classifier_worker_b.terminate()
                self.classifier_worker_b.wait()
            self.classifier_worker_b.set_image(image)
            self.classifier_worker_b.start()

    @pyqtSlot(dict)
    def on_classification_results(self, results):
        """處理分類結果"""
        if results:
            self.metric_display.update_metrics(results)
            if "model_a" in results:
                model_a = results.get('model_a')
                acc_a = results.get('model_a_acc')
                if model_a == "未偵測到模型":
                    logging.info("圖A分類完成: 未偵測到模型")
                elif model_a == "NULL":
                    logging.info("圖A分類完成: 推薦模型為 NULL")
                elif acc_a is not None:
                    logging.info(f"圖A分類完成: 推薦使用 {model_a} 模型, 準確率: {acc_a:.2f}%")
                else:
                    logging.info(f"圖A分類完成: 推薦使用 {model_a} 模型, 準確率: 未知")
            elif "model_b" in results:
                model_b = results.get('model_b')
                acc_b = results.get('model_b_acc')
                if model_b == "未偵測到模型":
                    logging.info("圖B分類完成: 未偵測到模型")
                elif model_b == "NULL":
                    logging.info("圖B分類完成: 推薦模型為 NULL")
                elif acc_b is not None:
                    logging.info(f"圖B分類完成: 推薦使用 {model_b} 模型, 準確率: {acc_b:.2f}%")
                else:
                    logging.info(f"圖B分類完成: 推薦使用 {model_b} 模型, 準確率: 未知")
        self._check_and_unload_classification_model()

    def _check_and_unload_classification_model(self):
        """檢查並卸載分類模型，只有在兩個工作線程都不活躍時才卸載"""
        try:
            a_running = self.classifier_worker_a and self.classifier_worker_a.isRunning()
            b_running = self.classifier_worker_b and self.classifier_worker_b.isRunning()
            if not a_running and not b_running:
                if hasattr(self, 'image_classifier'):
                    self.image_classifier.unload_model()
                    logging.info("圖像分類模型已自動卸載以釋放資源")
        except Exception as e:
            logging.error(f"自動卸載分類模型時出錯: {str(e)}")

    @pyqtSlot(str)
    def on_classification_error(self, error_msg):
        """處理分類錯誤"""
        logging.error(f"圖像分類錯誤: {error_msg}")
        self._check_and_unload_classification_model()

    @pyqtSlot(object, str, str)
    def on_image_a_score_ready(self, score, file_name, size_info):
        """處理圖A評分結果"""
        results = {'ai_score_a': score}
        self.metric_display.update_metrics(results)
        if score is not None:
            logging.info(f"圖A 評分完成")
        else:
            logging.warning("圖A評分: 未偵測到模型")

    @pyqtSlot(object, str, str)
    def on_image_b_score_ready(self, score, file_name, size_info):
        """處理圖B評分結果"""
        results = {'ai_score_b': score}
        self.metric_display.update_metrics(results)
        if score is not None:
            logging.info(f"圖B 評分完成")
        else:
            logging.warning("圖B評分: 未偵測到模型")

    @pyqtSlot(str, bool)
    def on_single_image_error(self, error_msg, is_image_a):
        """處理單張圖像評分錯誤"""
        img_type = "圖A" if is_image_a else "圖B"
        logging.error(f"{img_type}評分錯誤: {error_msg}")

    def run_assessment(self, keep_ai_scores=False):
        """開始執行圖像評估
        參數:
            keep_ai_scores: 如果為True，則保留已有的AI評分結果，不重新計算
        """
        if self.multi_view.image_a is None or self.multi_view.image_b is None:
            QMessageBox.warning(self, "評估錯誤", "需要同時載入圖A和圖B才能進行完整評估")
            return
        self.evaluate_btn.setEnabled(False)
        self.load_image_a_btn.setEnabled(False)
        self.load_image_b_btn.setEnabled(False)
        self.swap_btn.setEnabled(False)
        self.clear_comparison_results() if keep_ai_scores else self.clear_results()
        self.progress_bar.setVisible(True)
        if self.assessment_worker.isRunning():
            self.assessment_worker.terminate()
            self.assessment_worker.wait()
        self.assessment_worker.set_images(self.multi_view.image_a, self.multi_view.image_b)
        self.assessment_worker.run_ai_scoring = not keep_ai_scores
        self.assessment_worker.start()

    def clear_comparison_results(self):
        """清空比較結果顯示，但保留AI評分"""
        self.diff_view.clear()
        self.hist_view_a.clear() 
        self.hist_view_b.clear()
        self.edge_view.clear()
        for key in ["psnr", "ssim", "mse"]:
            if key in self.metric_display.metrics:
                self.metric_display.metrics[key].setText("--")
                self.metric_display.metrics[key].setStyleSheet("font-weight: bold; padding: 2px;")

    @pyqtSlot(str)
    def update_progress(self, message):
        """更新UI顯示進度信息"""
        logging.info(message)

    @pyqtSlot(dict)
    def on_assessment_results(self, results):
        """處理評估結果"""
        self.metric_display.update_metrics(results)
        if "difference_map" in results:
            self.diff_view.set_image(results["difference_map"], "圖像差異熱力圖")
        if "histogram_img1" in results:
            self.hist_view_a.set_image(results["histogram_img1"], "圖A - RGB色彩分佈")
        if "histogram_img2" in results:
            self.hist_view_b.set_image(results["histogram_img2"], "圖B - RGB色彩分佈")
        if "edge_comparison" in results:
            self.edge_view.set_image(results["edge_comparison"], "邊緣比較")

    @pyqtSlot(dict)
    def on_ai_scores(self, ai_results):
        """處理AI評分結果"""
        self.metric_display.update_metrics(ai_results)
        a_score = ai_results.get('ai_score_a')
        b_score = ai_results.get('ai_score_b')
        if a_score is None or b_score is None:
            logging.warning("AI評分: 未偵測到模型")
        else:
            logging.info(f"AI評分完成: A={a_score:.2f}, B={b_score:.2f}")

    @pyqtSlot(str)
    def on_assessment_error(self, error_message):
        """處理評估過程中的錯誤"""
        logging.error(f"評估錯誤: {error_message}")
        QMessageBox.warning(self, "評估錯誤", error_message)
        self.clear_results()
        self.progress_bar.setVisible(False)
        self.evaluate_btn.setEnabled(True)
        self.load_image_a_btn.setEnabled(True)
        self.load_image_b_btn.setEnabled(True)
        self.swap_btn.setEnabled(True)

    @pyqtSlot()
    def on_assessment_finished(self):
        """評估工作完成後恢復UI狀態"""
        self.progress_bar.setVisible(False)
        self.evaluate_btn.setEnabled(True)
        self.load_image_a_btn.setEnabled(True)
        self.load_image_b_btn.setEnabled(True)
        self.swap_btn.setEnabled(True)

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
        self.load_img_a_path, self.load_img_b_path = self.load_img_b_path, self.load_img_a_path
        self.multi_view.set_images(
            image_b,
            image_a,
            name_b,
            name_a
        )
        self.multi_view.image_a_group.setTitle(title_b)
        self.multi_view.image_b_group.setTitle(title_a)
        score_a_text = self.metric_display.metrics["ai_score_a"].text()
        score_b_text = self.metric_display.metrics["ai_score_b"].text()
        if score_a_text != "--" and score_b_text != "--":
            if score_a_text == "未偵測到模型" and score_b_text == "未偵測到模型":
                self.metric_display.update_metrics({
                    'ai_score_a': None,
                    'ai_score_b': None
                })
            elif score_a_text == "未偵測到模型":
                self.metric_display.update_metrics({
                    'ai_score_a': float(score_b_text),
                    'ai_score_b': None
                })
            elif score_b_text == "未偵測到模型":
                self.metric_display.update_metrics({
                    'ai_score_a': None,
                    'ai_score_b': float(score_a_text)
                })
            else:
                self.metric_display.update_metrics({
                    'ai_score_a': float(score_b_text),
                    'ai_score_b': float(score_a_text)
                })
        model_a_text = self.metric_display.metrics["model_a"].text()
        model_b_text = self.metric_display.metrics["model_b"].text()
        model_a_acc_text = self.metric_display.metrics["model_a_acc"].text()
        model_b_acc_text = self.metric_display.metrics["model_b_acc"].text()
        if model_a_text != "--" and model_b_text != "--":
            try:
                model_a_acc = float(model_a_acc_text.replace("%", ""))
                model_b_acc = float(model_b_acc_text.replace("%", ""))
                self.metric_display.update_metrics({
                    'model_a': model_b_text,
                    'model_a_acc': model_b_acc,
                    'model_b': model_a_text,
                    'model_b_acc': model_a_acc
                })
            except ValueError:
                pass
        logging.info("已交換圖A和圖B的位置")
        self.run_assessment(keep_ai_scores=True)
        if image_a and image_b:
            self.run_single_image_classification(image_b, True)
            self.run_single_image_classification(image_a, False)

    def clear_results(self):
        """清空所有結果顯示，包括AI評分和模型推薦"""
        self.metric_display.clear()
        self.diff_view.clear()
        self.hist_view_a.clear() 
        self.hist_view_b.clear()
        self.edge_view.clear()

    def clear_comparison_results(self):
        """清空比較結果顯示，但保留AI評分和模型推薦"""
        self.diff_view.clear()
        self.hist_view_a.clear() 
        self.hist_view_b.clear()
        self.edge_view.clear()
        for key in ["psnr", "ssim", "mse"]:
            if key in self.metric_display.metrics:
                self.metric_display.metrics[key].setText("--")
                self.metric_display.metrics[key].setStyleSheet("font-weight: bold; padding: 2px;")
    
    @pyqtSlot(int)
    def on_tab_changed(self, index):
        """處理結果標籤頁切換事件，重新調整當前顯示的圖片大小"""
        QTimer.singleShot(100, lambda: self.adjust_current_tab_view(index))
    
    def adjust_current_tab_view(self, index):
        """根據當前標籤頁調整對應視圖"""
        if index == 0:
            if self.diff_view.pixmap_item is not None:
                self.diff_view.reset_view()
        elif index == 1:
            if self.hist_view_a.pixmap_item is not None:
                self.hist_view_a.reset_view()
            if self.hist_view_b.pixmap_item is not None:
                self.hist_view_b.reset_view()
        elif index == 2:
            if self.edge_view.pixmap_item is not None:
                self.edge_view.reset_view()
        QApplication.processEvents()
        
    def closeEvent(self, event):
        """處理視窗關閉事件，確保清理線程"""
        if self.assessment_worker and self.assessment_worker.isRunning():
            self.assessment_worker.terminate()
            self.assessment_worker.wait()
        if self.scorer_worker_a and self.scorer_worker_a.isRunning():
            self.scorer_worker_a.terminate() 
            self.scorer_worker_a.wait()
        if self.scorer_worker_b and self.scorer_worker_b.isRunning():
            self.scorer_worker_b.terminate()
            self.scorer_worker_b.wait() 
        if self.classifier_worker_a and self.classifier_worker_a.isRunning():
            self.classifier_worker_a.terminate()
            self.classifier_worker_a.wait()
        if self.classifier_worker_b and self.classifier_worker_b.isRunning():
            self.classifier_worker_b.terminate()
            self.classifier_worker_b.wait()
        if hasattr(self, 'image_classifier'):
            self.image_classifier.unload_model()
        event.accept()