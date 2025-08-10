import sys
import os
import logging
from PyQt6.QtWidgets import QApplication

from src.ui.main_window import ImageEnhancerApp


class ColoredFormatter(logging.Formatter):
    """彩色日誌格式器"""
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 綠色
        'WARNING': '\033[33m',  # 黃色
        'ERROR': '\033[31m',    # 紅色 
        'CRITICAL': '\033[35m', # 紫色
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)

def setup_logging():
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_file_path = os.path.join(logs_dir, "app.log")
    for handler in logging.root.handlers[:]: 
        logging.root.removeHandler(handler)
    colored_formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(colored_formatter)
    console_handler.setLevel(logging.INFO)
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[file_handler, console_handler],
        force=True
    )
    for handler in logging.root.handlers:
        if hasattr(handler, 'flush'):
            handler.flush()
    logging.info("啟動長門櫻-影像魅影")
    logging.info(f"日誌存放位置: {log_file_path}")
    for handler in logging.root.handlers:
        if hasattr(handler, 'flush'):
            handler.flush()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    setup_logging()
    app = QApplication(sys.argv)
    window = ImageEnhancerApp()
    window.show()
    exit_code = app.exec()
    logging.info("關閉長門櫻-影像魅影")
    logging.shutdown()
    sys.exit(exit_code)