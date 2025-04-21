import sys
import os
import logging
from PyQt6.QtWidgets import QApplication

from src.ui.main_window import ImageEnhancerApp


def setup_logging():
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_file_path = os.path.join(logs_dir, "app.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w', encoding='utf-8'),
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
    exit_code = app.exec()
    logging.info("關閉長門櫻-影像魅影")
    logging.shutdown()
    sys.exit(exit_code)