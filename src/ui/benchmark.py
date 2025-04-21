import os
import platform
import time
import torch
import psutil
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
                           QPushButton, QGroupBox, QTextEdit, QProgressBar,
                           QFormLayout, QSpinBox, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QGuiApplication

from src.ui.main_window import ImageEnhancerApp


class BenchmarkWorker(QThread):
    """執行基準測試的工作執行緒"""
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal(dict)
    
    def __init__(self, model_manager, device, width, height, iterations=10):
        super().__init__()
        self.model_manager = model_manager
        self.device = device
        self.width = width
        self.height = height
        self.iterations = iterations
        
    def run(self):
        """執行基準測試"""
        try:
            results = {}
            if not self.model_manager.prepare_model_for_inference():
                self.finished_signal.emit({"error": "模型載入失敗"})
                return
            model = self.model_manager.get_current_model()
            if model is None:
                self.finished_signal.emit({"error": "無法獲取模型"})
                return
            test_input = torch.rand(1, 3, self.height, self.width).to(self.device)
            for _ in range(3):
                with torch.no_grad():
                    model(test_input)
            total_time = 0
            times = []
            for i in range(self.iterations):
                start = time.time()
                with torch.no_grad():
                    model(test_input)
                end = time.time()
                iteration_time = end - start
                times.append(iteration_time)
                total_time += iteration_time
                self.progress_signal.emit(i + 1, self.iterations)
            avg_time = total_time / self.iterations
            fps = 1.0 / avg_time
            pixels_per_second = self.width * self.height * fps
            megapixels_per_second = pixels_per_second / 1000000
            score = int(megapixels_per_second * 100)
            results = {
                "average_time": avg_time,
                "fps": fps,
                "megapixels_per_second": megapixels_per_second,
                "resolution": f"{self.width}x{self.height}",
                "total_time": total_time,
                "iterations": self.iterations,
                "device": self.device,
                "times": times,
                "score": score
            }
            self.finished_signal.emit(results)
        except Exception as e:
            self.finished_signal.emit({"error": str(e)})

class BenchmarkDialog(QDialog):
    """基準測試對話框"""
    def __init__(self, model_manager, parent=None):
        super().__init__(parent)
        self.model_manager = model_manager
        self.setWindowTitle("基準測試")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        self.benchmark_worker = None
        self.results = {}
        self.init_ui()
        self.populate_system_info()
        
    def init_ui(self):
        """初始化UI元素"""
        layout = QVBoxLayout(self)
        build_group = QGroupBox("建構資訊")
        build_layout = QFormLayout()
        self.version_label = QLabel(f"程式版本: {ImageEnhancerApp.version}")
        build_layout.addRow(self.version_label)
        build_group.setLayout(build_layout)
        layout.addWidget(build_group)
        system_group = QGroupBox("系統資訊")
        system_layout = QFormLayout()
        self.os_label = QLabel("載入中...")
        self.cpu_label = QLabel("載入中...")
        self.gpu_label = QLabel("載入中...")
        system_layout.addRow("作業系統:", self.os_label)
        system_layout.addRow("CPU:", self.cpu_label)
        system_layout.addRow("GPU:", self.gpu_label)
        system_group.setLayout(system_layout)
        layout.addWidget(system_group)
        settings_group = QGroupBox("處理設定")
        settings_layout = QFormLayout()
        self.device_combo = QComboBox()
        self.device_combo.addItem("自動選擇", "auto")
        if torch.cuda.is_available():
            self.device_combo.addItem("CUDA", "cuda")
        self.device_combo.addItem("CPU", "cpu")
        settings_layout.addRow("計算設備:", self.device_combo)
        res_layout = QHBoxLayout()
        self.width_spin = QSpinBox()
        self.width_spin.setRange(256, 4096)
        self.width_spin.setValue(1280)
        self.width_spin.setSingleStep(128)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(256, 4096)
        self.height_spin.setValue(720)
        self.height_spin.setSingleStep(128)
        res_layout.addWidget(QLabel("寬:"))
        res_layout.addWidget(self.width_spin)
        res_layout.addWidget(QLabel("高:"))
        res_layout.addWidget(self.height_spin)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItem("自訂", "custom")
        self.resolution_combo.addItem("HD (1280x720)", "1280,720")
        self.resolution_combo.addItem("FHD (1920x1080)", "1920,1080")
        self.resolution_combo.addItem("2K (2560x1440)", "2560,1440")
        self.resolution_combo.addItem("4K (3840x2160)", "3840,2160")
        self.resolution_combo.currentIndexChanged.connect(self.on_resolution_changed)
        res_layout.addWidget(QLabel("預設:"))
        res_layout.addWidget(self.resolution_combo)
        settings_layout.addRow("輸入解析度:", res_layout)
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(5, 100)
        self.iterations_spin.setValue(10)
        settings_layout.addRow("測試次數:", self.iterations_spin)
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        result_group = QGroupBox("基準測試結果")
        result_layout = QVBoxLayout()
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setText("尚未執行基準測試")
        result_layout.addWidget(self.result_text)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        result_layout.addWidget(self.progress_bar)
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        button_layout = QHBoxLayout()
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.reject)
        self.copy_button = QPushButton("複製結果")
        self.copy_button.clicked.connect(self.copy_results)
        self.copy_button.setEnabled(False)
        self.benchmark_button = QPushButton("開始跑分")
        self.benchmark_button.clicked.connect(self.start_benchmark)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.copy_button)
        button_layout.addWidget(self.benchmark_button)
        layout.addLayout(button_layout)
    
    def populate_system_info(self):
        """獲取並顯示系統信息"""
        try:
            os_info = f"{platform.system()} {platform.release()} {platform.architecture()[0]}"
            self.os_label.setText(os_info)
            cpu_info = platform.processor()
            cpu_count = psutil.cpu_count(logical=True)
            physical_cpu_count = psutil.cpu_count(logical=False)
            if not cpu_info:
                cpu_info = f"Unknown CPU, {physical_cpu_count} 核心, {cpu_count} 執行緒"
            else:
                cpu_info = f"{cpu_info}, {physical_cpu_count} 核心, {cpu_count} 執行緒"
            self.cpu_label.setText(cpu_info)
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_names = []
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_names.append(f"{gpu_name}")
                
                gpu_info = ", ".join(gpu_names)
                self.gpu_label.setText(gpu_info)
            else:
                self.gpu_label.setText("未檢測到支援的GPU")
        
        except Exception as e:
            self.os_label.setText("獲取失敗")
            self.cpu_label.setText("獲取失敗")
            self.gpu_label.setText("獲取失敗")
    
    def on_resolution_changed(self, index):
        """處理解析度下拉選單變更事件"""
        if index == 0: 
            return
        resolution_data = self.resolution_combo.currentData()
        if resolution_data != "custom":
            width, height = map(int, resolution_data.split(','))
            self.width_spin.setValue(width)
            self.height_spin.setValue(height)
    
    def start_benchmark(self):
        """開始執行基準測試"""
        try:
            device_selection = self.device_combo.currentData()
            if device_selection == "auto":
                device = self.model_manager.get_device()
            else:
                device = torch.device(device_selection)
            if not self.model_manager.get_registered_model_path():
                QMessageBox.warning(self, "警告", "請先選擇一個模型再執行基準測試")
                return
            self.benchmark_button.setEnabled(False)
            self.cancel_button.setEnabled(False)
            self.progress_bar.setValue(0)
            self.result_text.setText("基準測試執行中，請稍候...")
            self.benchmark_worker = BenchmarkWorker(
                self.model_manager,
                device,
                self.width_spin.value(),
                self.height_spin.value(),
                self.iterations_spin.value()
            )
            self.benchmark_worker.progress_signal.connect(self.update_progress)
            self.benchmark_worker.finished_signal.connect(self.on_benchmark_finished)
            self.benchmark_worker.start()
            
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"執行基準測試時出錯: {str(e)}")
            self.benchmark_button.setEnabled(True)
            self.cancel_button.setEnabled(True)
    
    def update_progress(self, current, total):
        """更新進度提示"""
        progress_percent = int(current / total * 100)
        self.progress_bar.setValue(progress_percent)
        self.result_text.setText(f"執行測試中: {current}/{total} ({progress_percent}%)")
    
    def on_benchmark_finished(self, results):
        """處理基準測試完成事件"""
        self.benchmark_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setValue(100)
        if "error" in results:
            QMessageBox.critical(self, "錯誤", f"基準測試失敗: {results['error']}")
            return
        self.results = results
        self.display_results()
        self.copy_button.setEnabled(True)
    
    def display_results(self):
        """顯示測試結果"""
        if not self.results:
            return
        device_name = "CPU"
        if str(self.results["device"]) == "cuda":
            gpu_id = 0 
            device_name = f"GPU ({torch.cuda.get_device_name(gpu_id)})"
        times = self.results.get("times", [])
        min_time = min(times) if times else 0
        max_time = max(times) if times else 0
        std_dev = np.std(times) if times else 0
        result_text = f"""=== Nagato-Sakura-Image-Charm 基準測試結果 ===

建構資訊:
  程式版本: {ImageEnhancerApp.version}

系統資訊:
  作業系統: {self.os_label.text()}
  處理器: {self.cpu_label.text()}
  顯示卡: {self.gpu_label.text()}

處理設定:
  計算設備: {device_name}
  輸入解析度: {self.results['resolution']}
  測試次數: {self.results['iterations']}

基準測試結果:
  性能分數: {self.results['score']}
  平均處理時間: {self.results['average_time']:.4f} 秒
  最快處理時間: {min_time:.4f} 秒
  最慢處理時間: {max_time:.4f} 秒
  處理時間標準差: {std_dev:.6f}
  每秒處理幀數: {self.results['fps']:.2f} FPS
  每秒處理像素: {self.results['megapixels_per_second']:.2f} MP/s
  總測試時間: {self.results['total_time']:.2f} 秒

測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.result_text.setText(result_text)
    
    def copy_results(self):
        """複製結果到剪貼板"""
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(self.result_text.toPlainText())
        QMessageBox.information(self, "已複製", "測試結果已複製到剪貼板")