import logging
import os
from logging.handlers import RotatingFileHandler



def setup_logging():
    if not os.path.exists("logs"):
        os.makedirs("logs")

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    file_handler = RotatingFileHandler("../logs/trading.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8')
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Setup Logger
    logger = logging.getLogger("TradingBot")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
