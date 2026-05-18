import logging
import os
from logging.handlers import RotatingFileHandler



def setup_logging():
    # Tự động xác định đường dẫn động đến thư mục logs/ bên trong project root
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, '..'))
    LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

    # Tạo thư mục logs nếu chưa tồn tại
    os.makedirs(LOGS_DIR, exist_ok=True)

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    log_file_path = os.path.join(LOGS_DIR, 'trading.log')
    file_handler = RotatingFileHandler(log_file_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8')
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Setup Logger
    logger = logging.getLogger("TradingBot")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
