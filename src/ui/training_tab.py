import os
import time
import logging
from datetime import datetime, timedelta
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                          QFileDialog, QProgressBar, QSpinBox, QDoubleSpinBox, 
                          QTabWidget, QTextEdit, QGroupBox, QFormLayout, QLineEdit,
                          QCheckBox, QComboBox, QStyle)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QThread, QTimer

from src.processing.NS_DataProcessor import DataProcessor
from src.training.NS_Trainer import Trainer


# 訓練工作執行緒
class TrainingWorker(QThread):
    progress_updated = pyqtSignal(int, int, int, float, float)
    epoch_completed = pyqtSignal(int, float, float, float)
    training_completed = pyqtSignal()
    training_error = pyqtSignal(str)
    
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer
        self.start_time = None
        self.is_running = True
        
    def run(self):
        try:
            self.start_time = time.time()
            self.is_running = True
            self.trainer.train(
                progress_callback=self.update_progress,
                epoch_callback=self.epoch_complete,
                stop_check_callback=self.check_stop
            )
            if self.is_running:
                self.training_completed.emit()
            else:
                self.training_error.emit("訓練已中止")
        except Exception as e:
            self.training_error.emit(str(e))
            logging.error(f"訓練錯誤: {e}", exc_info=True)
    
    def update_progress(self, epoch, batch, total_batches, g_loss, d_loss):
        self.progress_updated.emit(epoch, batch, total_batches, g_loss, d_loss)
        
    def epoch_complete(self, epoch, g_loss, d_loss, psnr):
        self.epoch_completed.emit(epoch, g_loss, d_loss, psnr)
    
    def check_stop(self):
        """檢查是否中止訓練的回調函數"""
        return not self.is_running
        
    def stop(self):
        """中止訓練"""
        self.is_running = False

# 資料處理工作執行緒
class DataProcessingWorker(QThread):
    progress_updated = pyqtSignal(int, int)
    processing_completed = pyqtSignal(int)
    processing_error = pyqtSignal(str)
    
    def __init__(self, processor, input_dir, output_dir):
        super().__init__()
        self.processor = processor
        self.input_dir = input_dir
        self.output_dir = output_dir
        
    def run(self):
        try:
            processed_count = self.processor.process_images(
                self.input_dir, 
                self.output_dir,
                callback=self.update_progress
            )
            self.processing_completed.emit(processed_count)
        except Exception as e:
            self.processing_error.emit(str(e))
            logging.error(f"資料處理錯誤: {e}", exc_info=True)
    
    def update_progress(self, current, total):
        self.progress_updated.emit(current, total)


class TrainingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger("TrainingTab")
        self.training_start_time = None
        self.init_ui()
        
    def init_ui(self):
        """初始化使用者介面"""
        main_layout = QVBoxLayout()
        
        # 建立標籤頁容器
        tab_widget = QTabWidget()
        
        # 資料處理標籤頁
        data_tab = self.create_data_processing_tab()
        tab_widget.addTab(data_tab, "資料處理")
        
        # 模型訓練標籤頁
        training_tab = self.create_model_training_tab()
        tab_widget.addTab(training_tab, "模型訓練")
        
        main_layout.addWidget(tab_widget)
        self.setLayout(main_layout)
        
    def create_data_processing_tab(self):
        """建立資料處理標籤頁"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # 資料處理設定區塊
        settings_group = QGroupBox("資料處理設定")
        form_layout = QFormLayout()
        
        # 輸入目錄 - 預設為 /data/input_images
        input_layout = QHBoxLayout()
        self.input_dir_edit = QLineEdit("/data/input_images")
        self.browse_input_btn = QPushButton("瀏覽...")
        self.browse_input_btn.clicked.connect(self.browse_input_directory)
        input_layout.addWidget(self.input_dir_edit)
        input_layout.addWidget(self.browse_input_btn)
        form_layout.addRow("輸入目錄:", input_layout)
        
        # 輸出目錄
        output_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("請選擇輸出目錄...")
        self.output_dir_edit.setReadOnly(True)
        self.browse_output_btn = QPushButton("瀏覽...")
        self.browse_output_btn.clicked.connect(self.browse_output_directory)
        output_layout.addWidget(self.output_dir_edit)
        output_layout.addWidget(self.browse_output_btn)
        form_layout.addRow("輸出目錄:", output_layout)
        
        # 品質設定
        quality_layout = QHBoxLayout()
        self.min_quality_spin = QSpinBox()
        self.min_quality_spin.setRange(1, 100)
        self.min_quality_spin.setValue(10)
        self.max_quality_spin = QSpinBox()
        self.max_quality_spin.setRange(2, 101)
        self.max_quality_spin.setValue(101)
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(5, 30)
        self.interval_spin.setValue(10)
        quality_layout.addWidget(QLabel("最小:"))
        quality_layout.addWidget(self.min_quality_spin)
        quality_layout.addWidget(QLabel("最大:"))
        quality_layout.addWidget(self.max_quality_spin)
        quality_layout.addWidget(QLabel("間隔:"))
        quality_layout.addWidget(self.interval_spin)
        form_layout.addRow("品質範圍:", quality_layout)
        settings_group.setLayout(form_layout)
        layout.addWidget(settings_group)
        
        # 處理進度
        progress_group = QGroupBox("處理進度")
        progress_layout = QVBoxLayout()
        self.data_progress_bar = QProgressBar()
        self.data_progress_bar.setRange(0, 100)
        self.data_progress_bar.setTextVisible(True)
        self.data_progress_bar.setFormat("%p% (%v/%m)")
        progress_status_layout = QHBoxLayout()
        progress_status_layout.addWidget(QLabel("狀態:"))
        self.data_status_label = QLabel("準備就緒")
        self.data_status_label.setStyleSheet("font-weight: bold;")
        progress_status_layout.addWidget(self.data_status_label)
        progress_status_layout.addStretch()
        progress_layout.addWidget(self.data_progress_bar)
        progress_layout.addLayout(progress_status_layout)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # 處理按鈕
        button_layout = QHBoxLayout()
        self.start_processing_btn = QPushButton("開始處理")
        self.start_processing_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.start_processing_btn.clicked.connect(self.start_data_processing)
        button_layout.addWidget(self.start_processing_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        layout.addStretch()
        tab.setLayout(layout)
        return tab
        
    def create_model_training_tab(self):
        """建立模型訓練標籤頁"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # 訓練設定區塊
        settings_group = QGroupBox("訓練設定")
        form_layout = QFormLayout()
        
        # 資料集目錄
        dataset_layout = QHBoxLayout()
        self.dataset_dir_edit = QLineEdit()
        self.dataset_dir_edit.setReadOnly(True)
        self.browse_dataset_btn = QPushButton("瀏覽...")
        self.browse_dataset_btn.clicked.connect(self.browse_dataset_directory)
        dataset_layout.addWidget(self.dataset_dir_edit)
        dataset_layout.addWidget(self.browse_dataset_btn)
        form_layout.addRow("資料集目錄:", dataset_layout)
        
        # 模型保存目錄
        model_dir_layout = QHBoxLayout()
        self.model_dir_edit = QLineEdit("./models")
        self.browse_model_dir_btn = QPushButton("瀏覽...")
        self.browse_model_dir_btn.clicked.connect(self.browse_model_directory)
        model_dir_layout.addWidget(self.model_dir_edit)
        model_dir_layout.addWidget(self.browse_model_dir_btn)
        form_layout.addRow("模型儲存目錄:", model_dir_layout)
        
        # 模型保存選項
        save_options_layout = QVBoxLayout()
        
        # 添加模型保存選擇
        self.save_best_loss_cb = QCheckBox("保存最佳損失模型")
        self.save_best_loss_cb.setChecked(True)
        self.save_best_psnr_cb = QCheckBox("保存最佳PSNR模型")
        self.save_best_psnr_cb.setChecked(True)
        self.save_final_cb = QCheckBox("保存最終模型")
        self.save_final_cb.setChecked(True)
        self.save_checkpoint_cb = QCheckBox("保存週期檢查點")
        
        # 檢查點週期設定
        checkpoint_layout = QHBoxLayout()
        self.checkpoint_interval_spin = QSpinBox()
        self.checkpoint_interval_spin.setRange(1, 100)
        self.checkpoint_interval_spin.setValue(10)
        self.checkpoint_interval_spin.setEnabled(False)
        checkpoint_layout.addWidget(QLabel("每隔"))
        checkpoint_layout.addWidget(self.checkpoint_interval_spin)
        checkpoint_layout.addWidget(QLabel("個週期保存一次"))
        checkpoint_layout.addStretch()
        
        # 連接檢查點選項與間隔選擇器的狀態
        self.save_checkpoint_cb.toggled.connect(self.checkpoint_interval_spin.setEnabled)
        save_options_layout.addWidget(self.save_best_loss_cb)
        save_options_layout.addWidget(self.save_best_psnr_cb)
        save_options_layout.addWidget(self.save_final_cb)
        save_options_layout.addWidget(self.save_checkpoint_cb)
        save_options_layout.addLayout(checkpoint_layout)
        form_layout.addRow("模型保存選項:", save_options_layout)
        
        # 批次大小
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 32)
        self.batch_size_spin.setValue(8)
        form_layout.addRow("批次大小:", self.batch_size_spin)
        
        # 學習率
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.000001, 0.01)
        self.learning_rate_spin.setValue(0.0001)
        self.learning_rate_spin.setDecimals(6)
        self.learning_rate_spin.setSingleStep(0.0001)
        form_layout.addRow("學習率:", self.learning_rate_spin)
        
        # 訓練週期
        self.num_epochs_spin = QSpinBox()
        self.num_epochs_spin.setRange(1, 1000)
        self.num_epochs_spin.setValue(50)
        form_layout.addRow("訓練週期:", self.num_epochs_spin)
        settings_group.setLayout(form_layout)
        layout.addWidget(settings_group)
        
        # 訓練進度
        progress_group = QGroupBox("訓練進度")
        progress_layout = QVBoxLayout()
        self.epoch_progress_bar = QProgressBar()
        self.epoch_progress_bar.setRange(0, 100)
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setRange(0, 100)
        
        # 添加預計完成時間標籤
        self.eta_label = QLabel("預計完成時間: --")
        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        progress_layout.addWidget(QLabel("週期進度:"))
        progress_layout.addWidget(self.epoch_progress_bar)
        progress_layout.addWidget(QLabel("批次進度:"))
        progress_layout.addWidget(self.batch_progress_bar)
        progress_layout.addWidget(self.eta_label)
        progress_layout.addWidget(QLabel("訓練日誌:"))
        progress_layout.addWidget(self.training_log)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # 訓練按鈕
        button_layout = QHBoxLayout()
        self.start_training_btn = QPushButton("開始訓練")
        self.start_training_btn.clicked.connect(self.start_model_training)
        
        # 中止按鈕
        self.stop_training_btn = QPushButton("中止訓練")
        self.stop_training_btn.clicked.connect(self.stop_model_training)
        self.stop_training_btn.setEnabled(False)
        button_layout.addWidget(self.start_training_btn)
        button_layout.addWidget(self.stop_training_btn)
        layout.addLayout(button_layout)
        tab.setLayout(layout)
        return tab
    
    def browse_input_directory(self):
        """瀏覽輸入目錄"""
        directory = QFileDialog.getExistingDirectory(self, "選擇輸入圖片目錄")
        if directory:
            self.input_dir_edit.setText(directory)
    
    def browse_output_directory(self):
        """瀏覽輸出目錄"""
        directory = QFileDialog.getExistingDirectory(self, "選擇輸出資料集目錄")
        if directory:
            self.output_dir_edit.setText(directory)
    
    def browse_dataset_directory(self):
        """瀏覽資料集目錄"""
        directory = QFileDialog.getExistingDirectory(self, "選擇訓練資料集目錄")
        if directory:
            self.dataset_dir_edit.setText(directory)
    
    def browse_model_directory(self):
        """瀏覽模型儲存目錄"""
        directory = QFileDialog.getExistingDirectory(self, "選擇模型儲存目錄")
        if directory:
            self.model_dir_edit.setText(directory)
    
    def start_data_processing(self):
        """開始資料處理程序"""
        input_dir = self.input_dir_edit.text()
        output_dir = self.output_dir_edit.text()
        min_quality = self.min_quality_spin.value()
        max_quality = self.max_quality_spin.value()
        interval = self.interval_spin.value()
        if not input_dir or not output_dir:
            self.data_status_label.setText("請選擇輸入和輸出目錄")
            return
        if not os.path.exists(input_dir):
            self.data_status_label.setText(f"輸入目錄不存在: {input_dir}")
            return
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            self.data_status_label.setText(f"無法建立輸出目錄: {str(e)}")
            return
        self.start_processing_btn.setEnabled(False)
        self.data_status_label.setText("正在處理圖片...")
        self.data_progress_bar.setValue(0)
        self.data_progress_bar.setStyleSheet("")
        processor = DataProcessor(min_quality, max_quality, interval)
        self.data_worker = DataProcessingWorker(processor, input_dir, output_dir)
        self.data_worker.progress_updated.connect(self.update_data_progress)
        self.data_worker.processing_completed.connect(self.data_processing_completed)
        self.data_worker.processing_error.connect(self.data_processing_error)
        self.data_worker.start()
    
    @pyqtSlot(int, int)
    def update_data_progress(self, current, total):
        """更新資料處理進度"""
        percentage = int(current / total * 100) if total > 0 else 0
        self.data_progress_bar.setValue(percentage)
        self.data_status_label.setText(f"正在處理 {current}/{total} 張圖片 ({percentage}%)")
    
    @pyqtSlot(int)
    def data_processing_completed(self, count):
        """資料處理完成"""
        self.data_progress_bar.setValue(100)
        self.data_progress_bar.setStyleSheet("QProgressBar::chunk {background-color: #55AA55;}")
        self.data_status_label.setText(f"處理完成! 已處理 {count} 張圖片")
        self.start_processing_btn.setEnabled(True)
        self.training_log.append(f"\n資料處理完成，已處理 {count} 張圖片")
        self.training_log.append("您現在可以進行模型訓練")
        QTimer.singleShot(5000, lambda: self.data_progress_bar.setStyleSheet(""))
    
    @pyqtSlot(str)
    def data_processing_error(self, error_message):
        """資料處理錯誤"""
        self.data_status_label.setText(f"處理錯誤: {error_message}")
        self.data_progress_bar.setStyleSheet("QProgressBar::chunk {background-color: #FF5555;}")
        self.start_processing_btn.setEnabled(True)
        self.logger.error(f"資料處理發生錯誤: {error_message}")
        QTimer.singleShot(5000, lambda: self.data_progress_bar.setStyleSheet(""))
    
    def start_model_training(self):
        """開始模型訓練程序"""
        dataset_dir = self.dataset_dir_edit.text()
        model_dir = self.model_dir_edit.text()
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        log_dir = os.path.join(base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        batch_size = self.batch_size_spin.value()
        learning_rate = self.learning_rate_spin.value()
        num_epochs = self.num_epochs_spin.value()
        if not dataset_dir:
            self.training_log.append("錯誤: 請選擇資料集目錄")
            return
        save_options = {
            'save_best_loss': self.save_best_loss_cb.isChecked(),
            'save_best_psnr': self.save_best_psnr_cb.isChecked(),
            'save_final': self.save_final_cb.isChecked(),
            'save_checkpoint': self.save_checkpoint_cb.isChecked(),
            'checkpoint_interval': self.checkpoint_interval_spin.value() if self.save_checkpoint_cb.isChecked() else 0
        }
        self.start_training_btn.setEnabled(False)
        self.stop_training_btn.setEnabled(True)
        self.training_log.append("正在初始化訓練...")
        self.epoch_progress_bar.setValue(0)
        self.epoch_progress_bar.setMaximum(num_epochs)
        self.training_start_time = time.time()
        trainer = Trainer(
            data_dir=dataset_dir,
            save_dir=model_dir,
            log_dir=log_dir,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            save_options=save_options
        )
        self.training_worker = TrainingWorker(trainer)
        self.training_worker.progress_updated.connect(self.update_training_progress)
        self.training_worker.epoch_completed.connect(self.epoch_completed)
        self.training_worker.training_completed.connect(self.training_completed)
        self.training_worker.training_error.connect(self.training_error)
        self.training_worker.start()
    
    def stop_model_training(self):
        """中止模型訓練"""
        if hasattr(self, 'training_worker') and self.training_worker.isRunning():
            self.training_log.append("\n正在中止訓練...")
            self.stop_training_btn.setEnabled(False)
            self.training_worker.stop()
    
    @pyqtSlot(int, int, int, float, float)
    def update_training_progress(self, epoch, batch, total_batches, g_loss, d_loss):
        """更新訓練進度"""
        percentage = int(batch / total_batches * 100) if total_batches > 0 else 0
        self.batch_progress_bar.setValue(percentage)
        if self.training_start_time:
            elapsed_seconds = time.time() - self.training_start_time
            total_batches_all_epochs = total_batches * self.num_epochs_spin.value()
            completed_batches = epoch * total_batches + batch
            if completed_batches > 0:
                seconds_per_batch = elapsed_seconds / completed_batches
                remaining_batches = total_batches_all_epochs - completed_batches
                remaining_seconds = seconds_per_batch * remaining_batches
                eta_datetime = datetime.now() + timedelta(seconds=remaining_seconds)
                eta_str = eta_datetime.strftime("%Y-%m-%d %H:%M:%S")
                self.eta_label.setText(f"預計完成時間: {eta_str}")
        if batch % 10 == 0: 
            self.training_log.append(f"Epoch {epoch+1}, 批次 {batch}/{total_batches}, G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}")
            self.training_log.verticalScrollBar().setValue(self.training_log.verticalScrollBar().maximum())
    
    @pyqtSlot(int, float, float, float)
    def epoch_completed(self, epoch, g_loss, d_loss, psnr):
        """週期完成"""
        self.epoch_progress_bar.setValue(epoch)
        self.training_log.append(f"\n===== Epoch {epoch}/{self.num_epochs_spin.value()} 完成 =====")
        self.training_log.append(f"生成器損失: {g_loss:.4f}")
        self.training_log.append(f"判別器損失: {d_loss:.4f}")
        self.training_log.append(f"PSNR: {psnr:.2f} dB\n")
        self.training_log.verticalScrollBar().setValue(self.training_log.verticalScrollBar().maximum())
    
    @pyqtSlot()
    def training_completed(self):
        """訓練完成"""
        self.training_log.append("\n訓練完成！")
        self.eta_label.setText("預計完成時間: 已完成")
        self.start_training_btn.setEnabled(True)
        self.stop_training_btn.setEnabled(False)
    
    @pyqtSlot(str)
    def training_error(self, error_message):
        """訓練錯誤"""
        self.training_log.append(f"\n錯誤: {error_message}")
        self.eta_label.setText("預計完成時間: 訓練中斷")
        self.start_training_btn.setEnabled(True)
        self.stop_training_btn.setEnabled(False)