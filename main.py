import sys
import os
import logging
from PyQt6.QtWidgets import QApplication

from src.ui.main_window import ImageEnhancerApp


# 設定日誌
def setup_logging():
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_file_path = os.path.join(logs_dir, "app.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
    logging.info("啟動長門櫻-影像魅影")
    logging.info(f"日誌存放位置: {log_file_path}")

if __name__ == "__main__":
    setup_logging()
    app = QApplication(sys.argv)
    window = ImageEnhancerApp()
    window.show()
    sys.exit(app.exec())