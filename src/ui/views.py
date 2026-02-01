from PyQt6.QtWidgets import QGraphicsView, QSizePolicy, QGraphicsScene, QVBoxLayout, QHBoxLayout, QWidget, QGroupBox, QSlider, QButtonGroup, QRadioButton, QPushButton, QStackedWidget, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsLineItem
from PyQt6.QtGui import QPainter, QWheelEvent, QMouseEvent, QTransform, QColor, QPen, QImage, QPixmap
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QRect
from PIL import Image


class EnhancedGraphicsPixmapItem(QGraphicsPixmapItem):
    """增強的圖片項目，提供更好的縮放抗鋸齒效果"""
    def __init__(self, pixmap=None, parent=None):
        super().__init__(pixmap, parent)
        self.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        
    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        super().paint(painter, option, widget)


class SynchronizedGraphicsView(QGraphicsView):
    viewChanged = pyqtSignal(QRectF, QTransform)
    def __init__(self, scene=None, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, False)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontSavePainterState, False)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setBackgroundBrush(Qt.GlobalColor.gray)
        self.setMinimumSize(100, 100)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.scale_factor = 1.0
        self.setInteractive(True)
        self.synchronized = True
        self._updating = False
        self._has_content = False 
        
    def wheelEvent(self, event: QWheelEvent):
        if self.synchronized and not self._updating and self._has_content:
            zoom_factor = 1.1
            if event.angleDelta().y() < 0:
                zoom_factor = 1.0 / zoom_factor
            self.scale(zoom_factor, zoom_factor)
            self.scale_factor *= zoom_factor
            self._updating = True
            self.viewChanged.emit(self.mapToScene(self.viewport().rect()).boundingRect(), self.transform())
            self._updating = False
        else:
            super().wheelEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        super().mouseMoveEvent(event)
        if (self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag and 
            self.synchronized and not self._updating and self._has_content):
            self._updating = True
            self.viewChanged.emit(self.mapToScene(self.viewport().rect()).boundingRect(), self.transform())
            self._updating = False
            
    def resetView(self):
        if not self._updating:
            self.resetTransform()
            self.scale_factor = 1.0
            if self._has_content and self.synchronized:
                self._updating = True
                self.viewChanged.emit(self.mapToScene(self.viewport().rect()).boundingRect(), self.transform())
                self._updating = False
        
    def setViewportFromOther(self, view_rect: QRectF, transform: QTransform):
        if self.synchronized and not self._updating and self._has_content:
            self._updating = True
            self.setTransform(transform)
            self.centerOn(view_rect.center())
            self.scale_factor = transform.m11()
            self._updating = False
    
    def setHasContent(self, has_content):
        self._has_content = has_content
            
class ImageCompareView(SynchronizedGraphicsView):
    """可以使用滑動分割線比較兩張圖片的畫面"""
    splitPositionChanged = pyqtSignal(float)
    
    def __init__(self, scene=None, parent=None):
        super().__init__(scene, parent)
        self.original_pixmap = None
        self.enhanced_pixmap = None
        self.split_position = 0.5 
        self.dragging_split = False
        self.split_line_width = 2
        self.split_line_color = QColor(255, 255, 255)
        self.split_handle_radius = 15
        self.image_rect = QRectF()  
        self.setMouseTracking(True)
        self._updating_from_external = False 
        
    def set_images(self, original_pixmap, enhanced_pixmap, split_position=None):
        self.original_pixmap = original_pixmap
        self.enhanced_pixmap = enhanced_pixmap
        if split_position is not None:
            self.split_position = split_position
        if self.scene() is None:
            self.setScene(QGraphicsScene())
        self.scene().clear()
        if not self.original_pixmap:
            self.setHasContent(False)
            return  
        if self.original_pixmap and self.enhanced_pixmap:
            original_size = self.original_pixmap.size()
            enhanced_size = self.enhanced_pixmap.size()
            if original_size != enhanced_size:
                self.enhanced_pixmap = self.enhanced_pixmap.scaled(
                    original_size, 
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
        pixmap_item = EnhancedGraphicsPixmapItem(self.original_pixmap)
        self.scene().addItem(pixmap_item)
        self.scene().setSceneRect(QRectF(self.original_pixmap.rect()))
        self.image_rect = QRectF(self.original_pixmap.rect())
        self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.setHasContent(True)
        self.viewport().update()
        
    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.original_pixmap or not self.enhanced_pixmap:
            return
        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        img_rect = self.mapFromScene(self.image_rect).boundingRect()
        split_x = int(img_rect.left() + img_rect.width() * self.split_position)
        painter.save()
        clip_rect = QRect(split_x, int(img_rect.top()), int(img_rect.right() - split_x), int(img_rect.height()))
        painter.setClipRect(clip_rect)
        target_rect = QRectF(img_rect)
        source_rect = QRectF(0, 0, self.enhanced_pixmap.width(), self.enhanced_pixmap.height())
        painter.drawPixmap(target_rect, self.enhanced_pixmap, source_rect)
        painter.restore()
        painter.setPen(QPen(self.split_line_color, self.split_line_width))
        painter.drawLine(split_x, int(img_rect.top()), split_x, int(img_rect.bottom()))
        handle_y = img_rect.top() + img_rect.height() / 2
        painter.setBrush(QColor(255, 255, 255, 200))
        painter.setPen(QPen(QColor(180, 180, 180), 1))
        painter.drawEllipse(QRectF(
            split_x - self.split_handle_radius / 2,
            handle_y - self.split_handle_radius / 2,
            self.split_handle_radius,
            self.split_handle_radius
        ))
        painter.setPen(QPen(QColor(100, 100, 100), 2))
        arrow_size = 4
        painter.drawLine(split_x - arrow_size, int(handle_y), 
                        split_x + arrow_size, int(handle_y))
        painter.drawLine(split_x - arrow_size, int(handle_y) - 2, 
                        split_x - arrow_size + 2, int(handle_y))
        painter.drawLine(split_x - arrow_size, int(handle_y) + 2, 
                        split_x - arrow_size + 2, int(handle_y))
        painter.drawLine(split_x + arrow_size, int(handle_y) - 2, 
                        split_x + arrow_size - 2, int(handle_y))
        painter.drawLine(split_x + arrow_size, int(handle_y) + 2, 
                        split_x + arrow_size - 2, int(handle_y))
        
    def mousePressEvent(self, event):
        if not self.original_pixmap or not self.enhanced_pixmap:
            super().mousePressEvent(event)
            return
        img_rect = self.mapFromScene(self.image_rect).boundingRect()
        split_x = int(img_rect.left() + img_rect.width() * self.split_position)
        hit_area = 15 
        if (abs(event.position().x() - split_x) <= hit_area and 
            event.position().y() >= img_rect.top() and 
            event.position().y() <= img_rect.bottom()):
            self.dragging_split = True
            self.setCursor(Qt.CursorShape.SplitHCursor)
            event.accept()
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        if not self.original_pixmap or not self.enhanced_pixmap:
            super().mouseMoveEvent(event)
            return
        img_rect = self.mapFromScene(self.image_rect).boundingRect()
        if self.dragging_split:
            if img_rect.width() > 0:
                pos_in_rect = (event.position().x() - img_rect.left()) / img_rect.width()
                new_position = max(0.0, min(1.0, pos_in_rect))
                
                if new_position != self.split_position:
                    self.split_position = new_position
                    if not self._updating_from_external:
                        self.splitPositionChanged.emit(self.split_position)
                    self.viewport().update()
                event.accept()
                return
        else:
            split_x = int(img_rect.left() + img_rect.width() * self.split_position)
            hit_area = 15
            if (abs(event.position().x() - split_x) <= hit_area and 
                event.position().y() >= img_rect.top() and 
                event.position().y() <= img_rect.bottom()):
                self.setCursor(Qt.CursorShape.SplitHCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        if self.dragging_split:
            self.dragging_split = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.viewport().update()
        
    def wheelEvent(self, event):
        super().wheelEvent(event)
        self.viewport().update()
        
    def resetView(self):
        super().resetView()
        if self.scene() and len(self.scene().items()) > 0:
            self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.viewport().update()
        
    def set_split_position_externally(self, position):
        """從外部設置分割位置，不會觸發信號（用於滑動條同步）"""
        self._updating_from_external = True
        self.split_position = max(0.0, min(1.0, position))
        self.viewport().update()
        self._updating_from_external = False
        
class MultiViewWidget(QWidget):
    """
    整合三種顯示模式的畫面組件：
    1. 並排顯示 - 左右兩張圖片對比
    2. 分割顯示 - 在一個畫面內用滑動分隔線比較兩張圖片
    3. 單獨顯示 - 只顯示一張圖片，可切換
    """
    viewSwitched = pyqtSignal(int)  
    imageToggled = pyqtSignal(bool)  
    
    def __init__(self, parent=None, show_tool_bar=True):
        super().__init__(parent)
        self.image_a = None 
        self.image_b = None 
        self.showing_a = True 
        self.image_a_name = "圖A" 
        self.image_b_name = "圖B" 
        self.normalize_sizes = True
        self._updating_slider = False
        self._single_view_state_saved = False
        self._single_view_transform = None
        self._single_view_center = None
        self.setup_ui(show_tool_bar)
    
    def addPixmapWithAntialiasing(self, scene, pixmap):
        """添加具有抗鋸齒功能的圖片到場景"""
        if scene and pixmap:
            scene.clear()
            pixmap_item = EnhancedGraphicsPixmapItem(pixmap)
            scene.addItem(pixmap_item)
            scene.setSceneRect(QRectF(pixmap.rect()))
            return pixmap_item
        return None
    
    def setup_ui(self, show_tool_bar=True):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        if show_tool_bar:
            tool_layout = QHBoxLayout()
            self.view_mode_group = QButtonGroup(self)
            self.side_by_side_btn = QRadioButton("並排顯示")
            self.side_by_side_btn.setChecked(True)
            self.view_mode_group.addButton(self.side_by_side_btn, 0)
            tool_layout.addWidget(self.side_by_side_btn)
            self.split_view_btn = QRadioButton("分割顯示")
            self.view_mode_group.addButton(self.split_view_btn, 1)
            tool_layout.addWidget(self.split_view_btn)
            self.single_view_btn = QRadioButton("單獨顯示")
            self.view_mode_group.addButton(self.single_view_btn, 2)
            tool_layout.addWidget(self.single_view_btn)
            tool_layout.addStretch(1)
            self.reset_view_btn = QPushButton("重置畫面")
            self.reset_view_btn.clicked.connect(self.reset_views)
            tool_layout.addWidget(self.reset_view_btn)
            self.sync_views_btn = QPushButton("同步畫面")
            self.sync_views_btn.setCheckable(True)
            self.sync_views_btn.setChecked(True)
            self.sync_views_btn.clicked.connect(self.toggle_sync_views)
            tool_layout.addWidget(self.sync_views_btn)
            self.toggle_display_btn = QPushButton("切換顯示")
            self.toggle_display_btn.clicked.connect(self.toggle_single_display)
            self.toggle_display_btn.setEnabled(False)
            tool_layout.addWidget(self.toggle_display_btn)
            self.normalize_sizes_btn = QPushButton("標準化尺寸")
            self.normalize_sizes_btn.setCheckable(True)
            self.normalize_sizes_btn.setChecked(True)
            self.normalize_sizes_btn.clicked.connect(self.toggle_normalize_sizes)
            tool_layout.addWidget(self.normalize_sizes_btn)
            main_layout.addLayout(tool_layout)
        self.view_stack = QStackedWidget()
        main_layout.addWidget(self.view_stack)
        self.side_by_side_widget = QWidget()
        side_by_side_layout = QHBoxLayout(self.side_by_side_widget)
        side_by_side_layout.setContentsMargins(0, 0, 0, 0)
        self.image_a_group = QGroupBox(self.image_a_name)
        image_a_layout = QVBoxLayout(self.image_a_group)
        self.image_a_scene = QGraphicsScene()
        self.image_a_view = SynchronizedGraphicsView(self.image_a_scene)
        image_a_layout.addWidget(self.image_a_view)
        self.image_b_group = QGroupBox(self.image_b_name)
        image_b_layout = QVBoxLayout(self.image_b_group)
        self.image_b_scene = QGraphicsScene()
        self.image_b_view = SynchronizedGraphicsView(self.image_b_scene)
        image_b_layout.addWidget(self.image_b_view)
        self.image_a_view.viewChanged.connect(self.sync_view_a_to_b)
        self.image_b_view.viewChanged.connect(self.sync_view_b_to_a)
        side_by_side_layout.addWidget(self.image_a_group)
        side_by_side_layout.addWidget(self.image_b_group)
        self.split_view_widget = QWidget()
        split_layout = QVBoxLayout(self.split_view_widget)
        split_layout.setContentsMargins(0, 0, 0, 0)
        self.split_group = QGroupBox("分割比較")
        split_inner_layout = QVBoxLayout(self.split_group)
        self.split_scene = QGraphicsScene()
        self.split_view = ImageCompareView(self.split_scene)
        split_inner_layout.addWidget(self.split_view)
        self.split_slider = QSlider(Qt.Orientation.Horizontal)
        self.split_slider.setRange(0, 100)
        self.split_slider.setValue(50)
        self.split_slider.valueChanged.connect(self.update_split_position)
        self.split_view.splitPositionChanged.connect(self.update_split_slider)
        split_inner_layout.addWidget(self.split_slider)
        split_layout.addWidget(self.split_group)
        self.single_view_widget = QWidget()
        single_layout = QVBoxLayout(self.single_view_widget)
        single_layout.setContentsMargins(0, 0, 0, 0)
        self.single_group = QGroupBox("單獨顯示")
        single_inner_layout = QVBoxLayout(self.single_group)
        self.single_scene = QGraphicsScene()
        self.single_view = SynchronizedGraphicsView(self.single_scene)
        single_inner_layout.addWidget(self.single_view)
        single_layout.addWidget(self.single_group)
        self.view_stack.addWidget(self.side_by_side_widget)
        self.view_stack.addWidget(self.split_view_widget)
        self.view_stack.addWidget(self.single_view_widget)
        if show_tool_bar:
            self.view_mode_group.buttonClicked.connect(self.change_view_mode)
        self.view_stack.setCurrentIndex(0)
    
    def toggle_normalize_sizes(self, checked):
        """切換尺寸標準化功能"""
        self.normalize_sizes = checked
        if checked:
            self.normalize_sizes_btn.setText("標準化尺寸")
        else:
            self.normalize_sizes_btn.setText("原始尺寸")
        self.set_images(self.image_a, self.image_b, self.image_a_name, self.image_b_name)
    
    def normalize_image_sizes(self, image_a, image_b):
        """標準化兩張圖片的尺寸以便於對比"""
        if not image_a or not image_b:
            return image_a, image_b 
        if not self.normalize_sizes:
            return image_a, image_b
        size_a = image_a.size
        size_b = image_b.size
        if size_a == size_b:
            return image_a, image_b
        area_a = size_a[0] * size_a[1]
        area_b = size_b[0] * size_b[1]
        if area_a == 0 or area_b == 0:
            return image_a, image_b
        if area_b > area_a * 2:
            try:
                image_b_resized = image_b.resize(size_a, Image.Resampling.LANCZOS)
                return image_a, image_b_resized
            except Exception:
                return image_a, image_b
        elif area_a > area_b * 2:
            try:
                image_a_resized = image_a.resize(size_b, Image.Resampling.LANCZOS)
                return image_a_resized, image_b
            except Exception:
                return image_a, image_b
        else:
            if area_a < area_b:
                target_size = size_a
                try:
                    image_b_resized = image_b.resize(target_size, Image.Resampling.LANCZOS)
                    return image_a, image_b_resized
                except Exception:
                    return image_a, image_b
            else:
                target_size = size_b
                try:
                    image_a_resized = image_a.resize(target_size, Image.Resampling.LANCZOS)
                    return image_a_resized, image_b
                except Exception:
                    return image_a, image_b
    
    def sync_view_a_to_b(self, view_rect, transform):
        if self.image_b and self.image_b_view._has_content:
            self.image_b_view.setViewportFromOther(view_rect, transform)
    
    def sync_view_b_to_a(self, view_rect, transform):
        if self.image_a and self.image_a_view._has_content:
            self.image_a_view.setViewportFromOther(view_rect, transform)
    
    def set_images(self, image_a, image_b, image_a_name=None, image_b_name=None):
        changed = (self.image_a != image_a or self.image_b != image_b)
        self.image_a = image_a
        self.image_b = image_b
        if changed:
            self._reset_single_view_state()
        
        self.image_a_view.setHasContent(image_a is not None)
        self.image_b_view.setHasContent(image_b is not None)
        self.single_view.setHasContent(image_a is not None or image_b is not None)
        if image_a_name:
            self.image_a_name = image_a_name
        if image_b_name:
            self.image_b_name = image_b_name
        if self.normalize_sizes and image_a and image_b and image_a.size != image_b.size:
            area_a = image_a.size[0] * image_a.size[1]
            area_b = image_b.size[0] * image_b.size[1]
            if area_b > area_a * 2:
                self.image_a_group.setTitle(f"{self.image_a_name}")
                self.image_b_group.setTitle(f"{self.image_b_name} (已縮放至原圖尺寸)")
            elif area_a > area_b * 2:
                self.image_a_group.setTitle(f"{self.image_a_name} (已縮放至增強圖尺寸)")
                self.image_b_group.setTitle(f"{self.image_b_name}")
            else:
                self.image_a_group.setTitle(f"{self.image_a_name} (已標準化)")
                self.image_b_group.setTitle(f"{self.image_b_name} (已標準化)")
        else:
            self.image_a_group.setTitle(self.image_a_name)
            self.image_b_group.setTitle(self.image_b_name)
        if image_a and not image_b:
            self.showing_a = True
        elif not image_a and image_b:
            self.showing_a = False
        self.update_views()
        if changed:
            self.reset_views()
    
    def change_view_mode(self, button=None):
        if button:
            mode = self.view_mode_group.id(button)
        else:
            mode = self.view_stack.currentIndex()
        
        self.view_stack.setCurrentIndex(mode)
        
        self.toggle_display_btn.setEnabled(
            mode == 2 and self.image_a is not None and self.image_b is not None
        )
        self.viewSwitched.emit(mode)
        self.update_views()
    
    def set_view_mode(self, mode):
        if 0 <= mode <= 2:
            self.view_stack.setCurrentIndex(mode)
            button = self.view_mode_group.button(mode)
            if button:
                button.setChecked(True)
                
            self.toggle_display_btn.setEnabled(
                mode == 2 and self.image_a is not None and self.image_b is not None
            )
            self.viewSwitched.emit(mode)
            self.update_views()
    
    def toggle_sync_views(self, checked):
        self.image_a_view.synchronized = checked
        self.image_b_view.synchronized = checked
        self.single_view.synchronized = checked
        self.split_view.synchronized = checked
        if checked:
            self.sync_views_btn.setText("同步畫面")
        else:
            self.sync_views_btn.setText("獨立畫面")
    
    def reset_views(self):
        """重置所有畫面"""
        was_synchronized = self.image_a_view.synchronized
        self._reset_single_view_state()
        try:
            self.image_a_view.synchronized = False
            self.image_b_view.synchronized = False
            self.split_view.synchronized = False
            self.single_view.synchronized = False
            self.image_a_view.resetTransform()
            self.image_b_view.resetTransform()
            self.split_view.resetTransform()
            self.single_view.resetTransform()
            if self.image_a and len(self.image_a_scene.items()) > 0:
                self.image_a_view.fitInView(self.image_a_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            if self.image_b and len(self.image_b_scene.items()) > 0:
                self.image_b_view.fitInView(self.image_b_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            if len(self.single_scene.items()) > 0:
                self.single_view.fitInView(self.single_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.update_split_view()
        finally:
            self.image_a_view.synchronized = was_synchronized
            self.image_b_view.synchronized = was_synchronized
            self.split_view.synchronized = was_synchronized
            self.single_view.synchronized = was_synchronized
    
    def toggle_single_display(self):
        if not self.image_a or not self.image_b:
            return
            
        if self.view_stack.currentIndex() == 2:
            self._save_single_view_state()
            self.showing_a = not self.showing_a
            self.update_single_view()
            self.imageToggled.emit(self.showing_a)
            if self.showing_a:
                self.single_group.setTitle(f"單獨顯示 - {self.image_a_name}")
            else:
                self.single_group.setTitle(f"單獨顯示 - {self.image_b_name}")
    
    def _save_single_view_state(self):
        """保存單獨顯示模式的視圖狀態"""
        if self.single_view and self.single_view._has_content:
            self._single_view_transform = self.single_view.transform()
            self._single_view_center = self.single_view.mapToScene(self.single_view.viewport().rect()).boundingRect().center()
            self._single_view_state_saved = True
    
    def _restore_single_view_state(self):
        """恢復單獨顯示模式的視圖狀態"""
        if (self._single_view_state_saved and 
            self._single_view_transform is not None and 
            self._single_view_center is not None and
            self.single_view and 
            self.single_view._has_content):
            self.single_view.setTransform(self._single_view_transform)
            self.single_view.centerOn(self._single_view_center)
            self.single_view.scale_factor = self._single_view_transform.m11()
    
    def _reset_single_view_state(self):
        """重置單獨顯示模式的視圖狀態記錄"""
        self._single_view_state_saved = False
        self._single_view_transform = None
        self._single_view_center = None
    
    def update_views(self):
        self.update_side_by_side_view()
        self.update_split_view()
        self.update_single_view()
    
    def update_side_by_side_view(self):
        image_a_display, image_b_display = self.normalize_image_sizes(self.image_a, self.image_b)
        self.image_a_scene.clear()
        if image_a_display:
            pixmap = self.pil_to_pixmap(image_a_display)
            self.addPixmapWithAntialiasing(self.image_a_scene, pixmap)
            self.image_a_view.fitInView(self.image_a_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.image_a_view._has_content = True
        else:
            self.image_a_view._has_content = False
        self.image_b_scene.clear()
        if image_b_display:
            pixmap = self.pil_to_pixmap(image_b_display)
            self.addPixmapWithAntialiasing(self.image_b_scene, pixmap)
            self.image_b_view.fitInView(self.image_b_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.image_b_view._has_content = True
        else:
            self.image_b_view._has_content = False
    
    def update_split_view(self):
        if self.image_a and self.image_b:
            image_a_display, image_b_display = self.normalize_image_sizes(self.image_a, self.image_b)
            pixmap_a = self.pil_to_pixmap(image_a_display)
            pixmap_b = self.pil_to_pixmap(image_b_display)
            if pixmap_a.size() != pixmap_b.size():
                target_size = pixmap_a.size()
                pixmap_b = pixmap_b.scaled(
                    target_size, 
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            self._updating_slider = True
            self.split_view.set_images(pixmap_a, pixmap_b, self.split_slider.value() / 100.0)
            self._updating_slider = False
            self.split_group.setTitle("分割比較")
            self.split_view._has_content = True
        elif self.image_a:
            pixmap_a = self.pil_to_pixmap(self.image_a)
            self.addPixmapWithAntialiasing(self.split_scene, pixmap_a)
            self.split_view.fitInView(self.split_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.split_group.setTitle(f"分割比較 - 僅有{self.image_a_name}")
            self.split_view._has_content = True 
        elif self.image_b:
            pixmap_b = self.pil_to_pixmap(self.image_b)
            self.addPixmapWithAntialiasing(self.split_scene, pixmap_b)
            self.split_view.fitInView(self.split_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.split_group.setTitle(f"分割比較 - 僅有{self.image_b_name}")
            self.split_view._has_content = True
        else:
            self.split_scene.clear()
            self.split_group.setTitle("分割比較 - 請載入圖片")
            self.split_view._has_content = False
    
    def update_single_view(self):
        should_preserve_view = (self.single_view._has_content and 
                               self._single_view_state_saved and
                               (self.image_a is not None and self.image_b is not None))
        self.single_scene.clear()
        if self.showing_a and self.image_a:
            if self.image_b and self.normalize_sizes:
                image_a_display, _ = self.normalize_image_sizes(self.image_a, self.image_b)
                pixmap = self.pil_to_pixmap(image_a_display)
            else:
                pixmap = self.pil_to_pixmap(self.image_a)
            self.addPixmapWithAntialiasing(self.single_scene, pixmap)
            self.single_group.setTitle(f"單獨顯示 - {self.image_a_name}")
            self.single_view._has_content = True
            if should_preserve_view:
                self._restore_single_view_state()
            else:
                self.single_view.fitInView(self.single_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
                if not self._single_view_state_saved:
                    self._reset_single_view_state()
        elif not self.showing_a and self.image_b:
            if self.image_a and self.normalize_sizes:
                _, image_b_display = self.normalize_image_sizes(self.image_a, self.image_b)
                pixmap = self.pil_to_pixmap(image_b_display)
            else:
                pixmap = self.pil_to_pixmap(self.image_b)
            self.addPixmapWithAntialiasing(self.single_scene, pixmap)
            self.single_group.setTitle(f"單獨顯示 - {self.image_b_name}")
            self.single_view._has_content = True
            if should_preserve_view:
                self._restore_single_view_state()
            else:
                self.single_view.fitInView(self.single_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
                if not self._single_view_state_saved:
                    self._reset_single_view_state()
        elif self.image_a and not self.image_b:
            pixmap = self.pil_to_pixmap(self.image_a)
            self.addPixmapWithAntialiasing(self.single_scene, pixmap)
            self.single_view.fitInView(self.single_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.single_group.setTitle(f"單獨顯示 - {self.image_a_name}")
            self.showing_a = True
            self.single_view._has_content = True
            self._reset_single_view_state()
        elif not self.image_a and self.image_b:
            pixmap = self.pil_to_pixmap(self.image_b)
            self.addPixmapWithAntialiasing(self.single_scene, pixmap)
            self.single_view.fitInView(self.single_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.single_group.setTitle(f"單獨顯示 - {self.image_b_name}")
            self.showing_a = False
            self.single_view._has_content = True
            self._reset_single_view_state()
        else:
            self.single_group.setTitle("單獨顯示 - 請載入圖片")
            self.single_view._has_content = False
            self._reset_single_view_state()
    
    def update_split_position(self, value):
        """從滑動條更新分割位置"""
        if self.split_view and self.image_a and self.image_b and not self._updating_slider:
            split_position = value / 100.0
            self.split_view.set_split_position_externally(split_position)
    
    def update_split_slider(self, position):
        """從分割線更新滑動條位置"""
        if not self._updating_slider:
            self._updating_slider = True
            self.split_slider.setValue(int(position * 100))
            self._updating_slider = False
    
    def pil_to_pixmap(self, pil_img):
        """將PIL圖片轉換為QPixmap，優化小圖縮放品質"""
        if pil_img:
            img = pil_img.convert("RGB")
            data = img.tobytes("raw", "RGB")
            qimage = QImage(data, img.width, img.height, img.width * 3, QImage.Format.Format_RGB888)
            if qimage.width() < 200 or qimage.height() < 200:
                qimage = qimage.scaled(qimage.width() * 2, qimage.height() * 2, 
                                     Qt.AspectRatioMode.KeepAspectRatio, 
                                     Qt.TransformationMode.SmoothTransformation)
            pixmap = QPixmap.fromImage(qimage)
            pixmap.setDevicePixelRatio(1.0)
            return pixmap
        return QPixmap()