import logging
from logging.config import dictConfig
import psutil
import time
from threading import Thread

def setup_logging():
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "filename": "fastapi_app.log",
                "formatter": "default",
            },
        },
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s: %(message)s",
            },
        },
        "root": {
            "level": "INFO",
            "handlers": ["file"]
        },
    }
    dictConfig(logging_config)

    # Start a separate thread for memory logging
    memory_logging_thread = Thread(target=log_memory_usage_periodically, daemon=True)
    memory_logging_thread.start()

def log_memory_usage_periodically():
    """Logs memory usage at regular intervals."""
    logger = logging.getLogger()  # Root logger
    process = psutil.Process()

    while True:
        memory_info = process.memory_info()
        rss_in_gb = memory_info.rss / (1024 * 1024 * 1024)  # Convert to GB
 
        logger.info(f"Memory Usage : {rss_in_gb:.2f} GB")
        time.sleep(60)  # Log memory usage every 60 seconds
